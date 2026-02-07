FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/kugeltts-api/requirements.txt /app/requirements.txt
RUN python3 -m venv /opt/venv \
    && pip install --upgrade pip \
    && pip install -r /app/requirements.txt

COPY pyproject.toml README.md /app/
COPY src/ /app/src/
COPY services/kugeltts-api/scripts/patch_tokenizer_meta_tensor.py /app/scripts/patch_tokenizer_meta_tensor.py
COPY services/kugeltts-api/scripts/entrypoint.sh /app/scripts/entrypoint.sh
RUN python /app/scripts/patch_tokenizer_meta_tensor.py /app/src/kugelaudio_open/models/tokenizer.py \
    && pip install --no-deps -e /app

COPY services/kugeltts-api/app /app/services/kugeltts-api/app
COPY services/kugeltts-api/config /app/services/kugeltts-api/config

RUN useradd -m appuser \
    && mkdir -p /app/hf-cache /app/hf-cache/hub /app/hf-cache/transformers \
    && chown -R appuser:appuser /app

EXPOSE 8000

ENTRYPOINT ["/app/scripts/entrypoint.sh"]

CMD ["uvicorn", "app.main:app", "--app-dir", "/app/services/kugeltts-api", "--host", "0.0.0.0", "--port", "8000"]
