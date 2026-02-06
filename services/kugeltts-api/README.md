# kugeltts-api

FastAPI service that exposes KugelAudio TTS on `/v1/audio/speech` with GPU support and Evido network integration.

## Architecture notes

- **Local-first model loading**: the service defaults to `local_files_only=True` and expects a model directory mounted at `/app/models`. This avoids any Hugging Face downloads and allows offline startup.
- **GPU pinning**: CUDA runtime is enabled and `NVIDIA_VISIBLE_DEVICES` controls which GPU(s) are visible to the container.
- **Security**: the container runs as a non-root user and does not publish ports by default (internal network only). If you need host access, add a `ports` mapping explicitly.

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
- `HF_HOME` (default: `/app/hf-cache` in container)
- `HF_HOME_HOST` (default: `./hf-cache` on host)
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

This compose file connects to the external `evido-live-translate` network and exposes port 8000 only inside the Docker network.

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

## Logs / stop

```bash
docker compose logs -f --tail=200
docker compose down
```

## Risks & mitigation

- **Tokenizer assets**: ensure the tokenizer files for the language model are stored locally (or allow HF fallback). Missing files will prevent startup.
- **Large model checkout**: use Git LFS to avoid partial clone issues and ensure reproducible builds.
- **GPU availability**: verify `nvidia-smi` works inside the container and that `NVIDIA_VISIBLE_DEVICES` targets the desired GPU UUID or index.
