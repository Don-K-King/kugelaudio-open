"""FastAPI entrypoint for KugelTTS service."""

from __future__ import annotations

import logging
import os
import re

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from starlette.responses import JSONResponse, Response

from .kugel_engine import KugelEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="kugeltts-api")
engine = KugelEngine()

LANGUAGE_TAG_REGEX = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
MAX_INPUT_CHARS = int(os.getenv("KUGEL_MAX_INPUT_CHARS", "1000"))
MAX_REQUEST_BODY_BYTES = int(os.getenv("KUGEL_MAX_REQUEST_BODY_BYTES", "0"))


@app.middleware("http")
async def _limit_request_body(request: Request, call_next):
    if MAX_REQUEST_BODY_BYTES <= 0:
        return await call_next(request)

    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            f"Request body too large. Limit is {MAX_REQUEST_BODY_BYTES} bytes."
                        )
                    },
                )
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header."},
            )

    body = await request.body()
    if len(body) > MAX_REQUEST_BODY_BYTES:
        return JSONResponse(
            status_code=413,
            content={
                "detail": f"Request body too large. Limit is {MAX_REQUEST_BODY_BYTES} bytes."
            },
        )

    return await call_next(request)


class SpeechRequest(BaseModel):
    input: str = Field(
        ..., min_length=1, max_length=MAX_INPUT_CHARS, description="Text to synthesize."
    )
    response_format: str = Field("wav", description="Only wav is supported")
    cfg_scale: float = Field(3.0, ge=0.0)
    language: str | None = Field(
        default=None,
        description="Optional BCP-47 language tag (e.g., en, de-DE).",
    )

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str | None) -> str | None:
        if value is None:
            return value
        candidate = value.strip()
        if not candidate:
            return None
        if not LANGUAGE_TAG_REGEX.match(candidate):
            raise ValueError("language must be a valid BCP-47 tag")
        return candidate

    @field_validator("input", mode="before")
    @classmethod
    def _validate_input_length(cls, value: str) -> str:
        if isinstance(value, str) and len(value) > MAX_INPUT_CHARS:
            raise ValueError(f"input must be at most {MAX_INPUT_CHARS} characters")
        return value


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

    audio_bytes = engine.synthesize(
        request.input,
        cfg_scale=request.cfg_scale,
        language=request.language,
    )
    return Response(content=audio_bytes, media_type="audio/wav")
