# OmniTry on fal.ai — Deployment Guide

> **Note:** The `fal` CLI uses Unix-only system calls (`fcntl`) and **cannot run directly on Windows**.
> We deploy via **GitHub Actions** (Ubuntu Linux runner) instead.

---

## Step 1: Push omnitry-fal to its own GitHub repo

Create a new repo e.g. `nutshellagency/omnitry-fal` and push this folder.

```bash
cd C:\Users\Hassan\Apps\Virtual-Try-On\omnitry-fal
git init
git add .
git commit -m "Initial fal.ai OmniTry deployment"
git remote add origin https://github.com/nutshellagency/omnitry-fal.git
git push -u origin main
```

## Step 2: Add your fal.ai API key as a GitHub Secret

1. Go to fal.ai → Dashboard → API Keys → create a key
2. Go to your GitHub repo → Settings → Secrets and variables → Actions
3. Add secret: **Name** = `FAL_KEY`, **Value** = your fal.ai key

## Step 3: GitHub Actions deploys automatically

On every push to `main`, the included `.github/workflows/deploy.yml` will:
- Install the fal CLI on Ubuntu
- Run `fal deploy app.py --app-name omnitry-app`
- Print your endpoint URL in the workflow logs

The endpoint will be something like:
```
https://nutshellagency-omnitry-app.fal.run/predict
```

## Step 4: Add to server .env

```
FAL_KEY=your_fal_key_here
FAL_OMNITRY_ENDPOINT=https://nutshellagency-omnitry-app.fal.run/predict
```

Then restart the dev server.

## Step 5: Test

Open `local-test.html`, select **Standard** tier and run a try-on.

---

## Cold Boot Notes

| Boot # | What happens | Time |
|--------|-------------|------|
| 1st    | Downloads 35GB FLUX + LoRA to `/data` cache | ~3-5 min |
| 2nd+   | Loads from `/data` NVMe cache | ~60 sec |

**Zero idle cost** — billing only when the model is actively running.

## Cost Estimate

- A100 GPU: ~$1.89/hour  
- OmniTry inference: ~60-90 sec GPU time  
- **Cost per try-on: ~$0.03-$0.05**
