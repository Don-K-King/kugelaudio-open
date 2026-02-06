"""KugelTTS inference engine for FastAPI service."""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

import soundfile as sf
import torch

logger = logging.getLogger(__name__)


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_dtype(device: str, requested: Optional[str]) -> torch.dtype:
    if device != "cuda":
        return torch.float32

    if not requested:
        return torch.bfloat16

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(requested.lower(), torch.bfloat16)


class KugelEngine:
    """Loads KugelAudio model + processor and runs synthesis."""

    def __init__(self) -> None:
        self.model_id = os.getenv("KUGEL_MODEL_ID", "/app/models/kugelaudio-0-open")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = _resolve_dtype(self.device, os.getenv("TORCH_DTYPE"))
        self.local_files_only = not _bool_env("KUGEL_ALLOW_HF", False)
        self.max_new_tokens = int(os.getenv("KUGEL_MAX_NEW_TOKENS", "4096"))
        self.model = None
        self.processor = None

    def load(self) -> None:
        from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
        from kugelaudio_open.processors import KugelAudioProcessor

        logger.info("Loading KugelAudio model from %s", self.model_id)
        model_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": False,
            "_fast_init": False,
            "device_map": None,
            "local_files_only": self.local_files_only,
        }

        if self.device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            model = KugelAudioForConditionalGenerationInference.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except Exception as exc:
            if model_kwargs.get("attn_implementation"):
                logger.warning("Flash attention load failed, falling back: %s", exc)
                model_kwargs.pop("attn_implementation", None)
                model = KugelAudioForConditionalGenerationInference.from_pretrained(
                    self.model_id,
                    **model_kwargs,
                )
            else:
                raise

        model.to(self.device)
        model.eval()
        model.model.strip_encoders()

        processor = KugelAudioProcessor.from_pretrained(
            self.model_id,
            local_files_only=self.local_files_only,
        )

        self.model = model
        self.processor = processor
        logger.info("Model loaded on %s with dtype %s", self.device, self.dtype)

    def synthesize(self, text: str, cfg_scale: float) -> bytes:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded")

        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                cfg_scale=cfg_scale,
                max_new_tokens=self.max_new_tokens,
            )

        if not outputs.speech_outputs:
            raise RuntimeError("Generation failed - no audio output")

        audio = outputs.speech_outputs[0]
        audio_np = audio.float().detach().cpu().numpy().squeeze()
        sampling_rate = getattr(self.processor.audio_processor, "sampling_rate", 24000)

        buffer = io.BytesIO()
        sf.write(buffer, audio_np, sampling_rate, format="WAV")
        return buffer.getvalue()
