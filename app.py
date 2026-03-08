"""
OmniTry Virtual Try-On — fal.ai Deployment
Weights are cached to fal's persistent /data volume after the first cold boot.
"""
from __future__ import annotations

import os
import sys
import math
import random
import io
import base64
import urllib.request

import fal
from pydantic import BaseModel, Field

# ─── Paths (persisted on fal's /data volume — survives between cold starts) ──
FLUX_DIR    = "/data/omnitry/flux"
LORA_DIR    = "/data/omnitry/checkpoints"
REPO_DIR    = "/data/omnitry/repo"

OBJECT_MAP = {
    "upper_body":  "top clothes",
    "lower_body":  "bottom clothes",
    "dresses":     "dresses",
    "accessories": "accessories",
    "shoes":       "shoes",
}

# ─── I/O Schemas ──────────────────────────────────────────────────────────────

class Input(BaseModel):
    human_img:           str   = Field(description="Person image — base64 or URL")
    garm_img:            str   = Field(description="Garment image — base64 or URL")
    garment_category:    str   = Field(default="upper_body",
                                       description="upper_body | lower_body | dresses | accessories | shoes")
    num_inference_steps: int   = Field(default=30)
    guidance_scale:      float = Field(default=7.5)
    seed:                int   = Field(default=-1)

class Output(BaseModel):
    image: str = Field(description="Result image as base64-encoded PNG")

# ─── Helper: image loading ────────────────────────────────────────────────────

def load_image(src: str):
    from PIL import Image
    if src.startswith("http://") or src.startswith("https://"):
        req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    return Image.open(io.BytesIO(base64.b64decode(src))).convert("RGB")

# ─── LoRA forward hack from OmniTry gradio_demo.py ───────────────────────────

def create_hacked_forward(module):
    def lora_fwd(self, adapter, x, *a, **kw):
        out = self.base_layer(x, *a, **kw)
        if adapter:
            x = x.to(self.lora_A[adapter].weight.dtype)
            out = out + self.lora_B[adapter](self.lora_A[adapter](
                         self.lora_dropout[adapter](x))) * self.scaling[adapter]
        return out

    def hacked(self, x, *a, **kw):
        import torch
        return torch.cat((
            lora_fwd(self, "vtryon_lora",  x[:1], *a, **kw),
            lora_fwd(self, "garment_lora", x[1:], *a, **kw),
        ), dim=0)

    return hacked.__get__(module, type(module))

# ─── fal App ──────────────────────────────────────────────────────────────────

class OmniTry(fal.App):
    # Class-level attributes for fal.ai configuration
    requirements = [
        "torch>=2.0.1",
        "torchvision",
        "diffusers>=0.25.0",
        "transformers>=4.36.2",
        "accelerate",
        "peft",
        "omegaconf",
        "safetensors",
        "huggingface_hub",
        "opencv-python-headless",
        "pillow",
        "git+https://github.com/Kunbyte-AI/OmniTry.git",
    ]
    machine_type = "GPU-H100"  # H100 = 80GB VRAM, required for FLUX.1-Fill-dev
    keep_alive = 60  # seconds to keep worker warm between requests
    def setup(self):
        """
        Runs once per worker lifecycle — NOT per request.
        /data is fal's persistent distributed volume, so weights downloaded on
        the first cold boot are immediately available on every subsequent one.
        """
        import torch
        from huggingface_hub import snapshot_download
        from peft import LoraConfig
        from safetensors import safe_open

        # Importing from OmniTry package installed via requirements above
        from omnitry.models.transformer_flux import FluxTransformer2DModel
        from omnitry.pipelines.pipeline_flux_fill import FluxFillPipeline

        os.makedirs(FLUX_DIR, exist_ok=True)
        os.makedirs(LORA_DIR, exist_ok=True)

        # Download FLUX backbone once, then it lives in /data forever
        if not os.path.isdir(os.path.join(FLUX_DIR, "transformer")):
            print("First boot: downloading FLUX.1-Fill-dev backbone (~35 GB)…")
            snapshot_download(
                repo_id="camenduru/FLUX.1-Fill-dev-ungated",
                local_dir=FLUX_DIR,
            )
        else:
            print("FLUX backbone found in /data cache ✓")

        # Download OmniTry LoRA weights once
        if not os.path.isfile(os.path.join(LORA_DIR, "omnitry_v1_unified.safetensors")):
            print("First boot: downloading OmniTry LoRA weights…")
            snapshot_download(repo_id="Kunbyte/OmniTry", local_dir=LORA_DIR)
        else:
            print("OmniTry LoRA found in /data cache ✓")

        self.device      = torch.device("cuda:0")
        self.weight_dtype = torch.bfloat16

        print("Loading FluxTransformer2DModel…")
        self.transformer = (
            FluxTransformer2DModel
            .from_pretrained(FLUX_DIR, subfolder="transformer")
            .requires_grad_(False)
            .to(dtype=self.weight_dtype)
        )

        print("Loading FluxFillPipeline…")
        self.pipeline = FluxFillPipeline.from_pretrained(
            FLUX_DIR,
            transformer=self.transformer.eval(),
            torch_dtype=self.weight_dtype,
        )
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.vae.enable_tiling()

        # Inject LoRA
        print("Injecting OmniTry LoRA adapters…")
        lora_cfg = LoraConfig(
            r=16, lora_alpha=16,
            init_lora_weights="gaussian",
            target_modules=[
                "x_embedder",
                "attn.to_k","attn.to_q","attn.to_v","attn.to_out.0",
                "attn.add_k_proj","attn.add_q_proj","attn.add_v_proj","attn.to_add_out",
                "ff.net.0.proj","ff.net.2","ff_context.net.0.proj","ff_context.net.2",
                "norm1_context.linear","norm1.linear","norm.linear","proj_mlp","proj_out",
            ],
        )
        self.transformer.add_adapter(lora_cfg, adapter_name="vtryon_lora")
        self.transformer.add_adapter(lora_cfg, adapter_name="garment_lora")

        lora_path = os.path.join(LORA_DIR, "omnitry_v1_unified.safetensors")
        with safe_open(lora_path, framework="pt") as f:
            weights = {k: f.get_tensor(k) for k in f.keys()}
        self.transformer.load_state_dict(weights, strict=False)

        import peft
        for _, m in self.transformer.named_modules():
            if isinstance(m, peft.tuners.lora.layer.Linear):
                m.forward = create_hacked_forward(m)

        print("OmniTry setup complete ✓")

    @fal.endpoint("/predict")
    def predict(self, request: Input) -> Output:
        import torch, torchvision.transforms as T, numpy as np

        seed = request.seed if request.seed != -1 else random.randint(0, 2**32 - 1)
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        person  = load_image(request.human_img)
        garment = load_image(request.garm_img)

        # Resize person image, keep multiple of 16, max 1024×1024
        r = min(1.0, math.sqrt(1024*1024 / (person.width * person.height)))
        tW, tH = int(person.width*r)//16*16, int(person.height*r)//16*16

        person_t = T.Compose([T.Resize((tH, tW)), T.ToTensor()])(person)

        # Letterbox garment to same canvas
        gr = min(tW / garment.width, tH / garment.height)
        garm_t = T.Compose([
            T.Resize((int(garment.height*gr), int(garment.width*gr))),
            T.ToTensor()
        ])(garment)
        padded = torch.ones_like(person_t)
        gh, gw = garm_t.shape[1], garm_t.shape[2]
        padded[:, (tH-gh)//2:(tH-gh)//2+gh, (tW-gw)//2:(tW-gw)//2+gw] = garm_t

        prompt   = OBJECT_MAP.get(request.garment_category, "top clothes")
        img_cond = torch.stack([person_t, padded]).to(dtype=self.weight_dtype, device=self.device)
        mask     = torch.zeros_like(img_cond)

        with torch.no_grad():
            result = self.pipeline(
                prompt=[prompt]*2,
                height=tH, width=tW,
                img_cond=img_cond, mask=mask,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                generator=torch.Generator(self.device).manual_seed(seed),
            ).images[0]

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return Output(image=base64.b64encode(buf.getvalue()).decode())
