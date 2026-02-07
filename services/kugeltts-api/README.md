# kugeltts-api

FastAPI service that exposes KugelAudio TTS on `/v1/audio/speech` with GPU support and Evido network integration.

## Architecture notes

- **Local-first model loading**: the service defaults to `local_files_only=True` and expects a model directory mounted at `/app/models`. This avoids any Hugging Face downloads and allows offline startup. If `KUGEL_ALLOW_HF=true`, the service can fall back to a Hugging Face repo when the local path is missing.
- **GPU pinning**: CUDA runtime is enabled and `NVIDIA_VISIBLE_DEVICES` controls which GPU(s) are visible to the container.
- **Security**: the container runs as a non-root user. The default compose file publishes the API on `127.0.0.1:8020` for local testing; adjust or remove the port mapping if you only want internal network access. Hugging Face cache paths must be writable by the container user.

## Model layout (local, offline)

Recommended layout inside the repo (use Git LFS for large files):

```
models/
  kugelaudio-0-open/
    config.json
    model.safetensors
    preprocessor_config.json
    voices/
      voices.json
      *.pt
```

Set `KUGEL_MODEL_ID=/app/models/kugelaudio-0-open` (default). If the tokenizer relies on a language model, ensure the tokenizer files are also stored locally and referenced in `preprocessor_config.json` via `language_model_pretrained_name`.

If you need Hugging Face fallback, set `KUGEL_ALLOW_HF=true` and optionally mount a cache.

## Environment variables

Copy `.env.example` to `.env` and adjust values as needed:

```bash
cp .env.example .env
```

- `KUGEL_MODEL_ID` (default: `/app/models/kugelaudio-0-open`)
- `KUGEL_MODEL_DIR` (default: `../../models` on host, mounted to `/app/models`)
- `KUGEL_ALLOW_HF` (default: `false`)
- `KUGEL_HF_REPO_ID` (default: `kugelaudio/kugelaudio-0-open`)
- `KUGEL_HF_REVISION` (default: empty, uses default branch)
- `HF_HOME` (default: `/app/hf-cache` in container)
- `HF_HOME_HOST` (default: `hf-cache` named volume on host)
- `HF_HUB_CACHE` (default: `/app/hf-cache/hub`)
- `TRANSFORMERS_CACHE` (default: `/app/hf-cache/transformers`)
- `TORCH_DTYPE` (default: `bfloat16`, falls back to `float32` on CPU)
- `NVIDIA_VISIBLE_DEVICES` (default: `all`)
- `KUGEL_MAX_NEW_TOKENS` (default: `4096`)

## Endpoints

- `GET /health` → `{ "status": "ok" }`
- `POST /v1/audio/speech`

Request:
```json
{
  "input": "text",
  "response_format": "wav",
  "cfg_scale": 3.0
}
```

Response: `audio/wav` bytes.

## Compose (Evido network)

This compose file connects to the external `evido-live-translate` network and publishes the API to `127.0.0.1:8020` on the host for local testing.

```bash
cd services/kugeltts-api
docker compose build --no-cache --pull
docker compose up -d
```

From another container on the same network:

```bash
curl http://kugeltts-api:8000/health
curl -X POST http://kugeltts-api:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hallo Welt","response_format":"wav","cfg_scale":3.0}' \
  --output out.wav
```

## Local build & run (PEP 668-safe)

The image installs dependencies into a dedicated virtual environment (`/opt/venv`) to avoid the Ubuntu 24.04 PEP 668 restriction on system Python packages.

```bash
cd services/kugeltts-api
docker compose build --no-cache --pull
docker compose up -d
```

### GPU pinning examples

Pin to the first GPU:

```bash
NVIDIA_VISIBLE_DEVICES=0 docker compose up -d
```

Pin to a specific GPU UUID:

```bash
NVIDIA_VISIBLE_DEVICES=GPU-12345678-1234-1234-1234-123456789abc docker compose up -d
```

### Offline model mounts & HF fallback

By default the container expects a local model mount at `/app/models` via `KUGEL_MODEL_DIR`:

```bash
KUGEL_MODEL_DIR=../../models \
KUGEL_MODEL_ID=/app/models/kugelaudio-0-open \
docker compose up -d
```

If you need Hugging Face fallback, allow it explicitly and mount a cache:

```bash
KUGEL_ALLOW_HF=true \
KUGEL_HF_REPO_ID=kugelaudio/kugelaudio-0-open \
KUGEL_HF_REVISION=main \
HF_HOME_HOST=./hf-cache \
docker compose up -d
```

The service will create `HF_HOME` automatically and verify it is writable. If the configured cache path is not writable (for example due to a root-owned bind mount), it falls back to `/tmp/hf-cache` with a warning. For persistent cache, prefer a named volume or ensure the host path is owned by the container user.

When using bind mounts for `HF_HOME`, ensure the host path is owned by the container UID/GID (appuser). On startup the container attempts `chown -R appuser:appuser /app/hf-cache`; if ownership cannot be changed, cache initialization either falls back to `/tmp/hf-cache` or fails fast when neither location is writable.

## Logs / stop

```bash
docker compose logs -f --tail=200
docker compose down
```

## Risks & mitigation

- **Tokenizer assets**: ensure the tokenizer files for the language model are stored locally (or allow HF fallback). Missing files will prevent startup.
- **Large model checkout**: use Git LFS to avoid partial clone issues and ensure reproducible builds.
- **GPU availability**: verify `nvidia-smi` works inside the container and that `NVIDIA_VISIBLE_DEVICES` targets the desired GPU UUID or index.
