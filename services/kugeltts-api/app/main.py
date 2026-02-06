"""FastAPI entrypoint for KugelTTS service."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import Response

from .kugel_engine import KugelEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="kugeltts-api")
engine = KugelEngine()


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1)
    response_format: str = Field("wav", description="Only wav is supported")
    cfg_scale: float = Field(3.0, ge=0.0)


@app.on_event("startup")
def _startup() -> None:
    engine.load()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/audio/speech")
def speech(request: SpeechRequest) -> Response:
    if request.response_format.lower() != "wav":
        raise HTTPException(status_code=400, detail="Only response_format=wav supported")

    audio_bytes = engine.synthesize(request.input, cfg_scale=request.cfg_scale)
    return Response(content=audio_bytes, media_type="audio/wav")
