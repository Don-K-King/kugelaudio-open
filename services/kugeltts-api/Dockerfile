FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/kugeltts-api/requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /app/requirements.txt

COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN python3 -m pip install --no-deps -e /app

COPY services/kugeltts-api/app /app/services/kugeltts-api/app

RUN useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--app-dir", "/app/services/kugeltts-api", "--host", "0.0.0.0", "--port", "8000"]
