# Deploying Spotter to Render

This guide walks through deploying Spotter to Render using the Blueprint (`render.yaml`).

## Prerequisites

- A Render account (you have $50 credit)
- GitHub/GitLab repo connected to Render
- API keys ready (see Environment Variables below)

## Deploy Steps

### 1. Connect Repository

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New** → **Blueprint**
3. Connect your GitHub/GitLab account if not already connected
4. Select the `spotter` repository
5. Render will detect `render.yaml` at the root

### 2. Apply Blueprint

1. Review the Blueprint preview (shows the `spotter` web service)
2. Click **Apply**
3. Render will:
   - Create the web service on the `1c-2g` plan (1 CPU, 2 GB RAM)
   - Provision a 1 GB persistent disk mounted at `/opt/render/project/src/runs`
   - Start the build using `pip install -r requirements.txt`

### 3. Set Environment Variables

After the first deploy (or before), go to **Environment** tab and add these **Secret Files/Variables** (`sync: false` means they're not in the blueprint):

| Variable | Required | Description |
|----------|----------|-------------|
| `ELEVENLABS_API_KEY` | No | Voice coach TTS (optional) |
| `BACKBOARD_API_KEY` | No | Backboard integration (optional) |
| `TINKER_API_KEY` | No | Tinker fine-tuned progress plans (optional) |
| `SPOTTER_COACH_SUMMARY_PROVIDER` | No | `hf_inference`, `llama_cpp`, `local_transformers`, or `tinker` |
| `SPOTTER_COACH_SUMMARY_MODEL` | No | Model ID for coach summary |
| `HF_TOKEN` | No | Hugging Face token for gated models |

**Optional voice coach:**
| Variable | Description |
|----------|-------------|
| `ELEVENLABS_VOICE_ID` | Voice ID |
| `ELEVENLABS_MODEL_ID` | e.g., `eleven_flash_v2_5` |
| `ELEVENLABS_OUTPUT_FORMAT` | e.g., `mp3_44100_128` |
| `SPOTTER_TTS_LANGUAGE` | `en`, `hi`, `mr` |
| `SPOTTER_TTS_MAX_CHARS_PER_RUN` | Default `600` |

### 4. Verify Deployment

1. Wait for build to complete (2-5 minutes first time)
2. Check **Logs** for "=== Total warmup: X.XXs ==="
3. Visit `https://<your-service>.onrender.com/healthz`
   - Should return `{"status": "ok", "service": "spotter", "warmed": "true"}`
   - During warmup: `{"status": "warming", "service": "spotter", "warmed": "false"}`

### 5. Test the App

1. Open the service URL in browser
2. Upload a workout clip (or use demo clips)
3. Verify annotated video and report render correctly

## Instance Choice: `1c-2g` (1 CPU, 2 GB RAM)

**Why this plan:**
- MediaPipe + OpenCV: ~200 MB RAM
- PyTorch CPU (router + transformers): ~600 MB RAM
- Gradio + FastAPI + Python overhead: ~300 MB
- Model weights (Nemotron 4B GGUF or HF cache): ~1-2 GB disk
- **Total working set: ~1.2 GB RAM** → fits comfortably in 2 GB with headroom
- The `0.5c-512mb` plan (512 MB) is too small for torch + mediapipe warmup
- Cost: ~$7/month on Render (within $50 credit)

## Persistent Disk

The 1 GB disk at `/opt/render/project/src/runs` stores:
- SQLite history database (`history.db`)
- Per-run artifacts (JSON, annotated videos, thumbnails)
- Survives deploys and restarts

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Build fails on torch | Ensure `runtime: python-3.10` matches `.python-version` |
| OOM on warmup | Upgrade to `2c-4g` or enable `SPOTTER_COACH_SUMMARY_DISABLE_REMOTE=true` to skip HF model |
| Port not binding | Ensure `startCommand` uses `$PORT` and `0.0.0.0` |
| Health check fails | Check `/healthz` returns 200; warmup may still be running |

## Updating

Push to `main` (or your deploy branch) → Render auto-deploys via Blueprint.