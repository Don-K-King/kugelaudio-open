"""KugelTTS inference engine for FastAPI service."""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
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
        self.hf_repo_id = os.getenv("KUGEL_HF_REPO_ID", "kugelaudio/kugelaudio-0-open")
        self.hf_home = os.getenv("HF_HOME", "/app/hf-cache")
        self.hf_revision = os.getenv("KUGEL_HF_REVISION") or None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = _resolve_dtype(self.device, os.getenv("TORCH_DTYPE"))
        self.allow_hf = _bool_env("KUGEL_ALLOW_HF", False)
        self.max_new_tokens = int(os.getenv("KUGEL_MAX_NEW_TOKENS", "4096"))
        self.model = None
        self.processor = None

    def _configure_hf_env(self, hf_home: Path) -> tuple[Path, Path]:
        hub_dir = hf_home / "hub"
        transformers_dir = hf_home / "transformers"
        os.environ["HF_HOME"] = str(hf_home)
        os.environ["HF_HUB_CACHE"] = str(hub_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(transformers_dir)
        self.hf_home = str(hf_home)
        return hub_dir, transformers_dir

    def _try_prepare_hf_cache(self, hf_home: Path) -> bool:
        hub_dir, transformers_dir = self._configure_hf_env(hf_home)
        try:
            hub_dir.mkdir(parents=True, exist_ok=True)
            transformers_dir.mkdir(parents=True, exist_ok=True)
            test_file = hub_dir / ".write_test"
            test_file.write_text("ok")
            test_file.unlink(missing_ok=True)
            test_file = transformers_dir / ".write_test"
            test_file.write_text("ok")
            test_file.unlink(missing_ok=True)
        except OSError:
            return False
        return True

    def _ensure_hf_cache(self) -> None:
        primary = Path(self.hf_home).expanduser()
        if self._try_prepare_hf_cache(primary):
            logger.info(
                "HF cache ready at %s (hub=%s, transformers=%s)",
                primary,
                primary / "hub",
                primary / "transformers",
            )
            return

        fallback = Path("/tmp/hf-cache")
        logger.warning(
            "HF cache directory %s is not writable; falling back to %s",
            primary,
            fallback,
        )
        if not self._try_prepare_hf_cache(fallback):
            raise RuntimeError(
                f"HF cache directory is not writable: {primary} or {fallback}"
            )
        logger.info(
            "HF cache ready at %s (hub=%s, transformers=%s)",
            fallback,
            fallback / "hub",
            fallback / "transformers",
        )

    def _looks_like_local_path(self, model_id: str) -> bool:
        return os.path.isabs(model_id) or model_id.startswith(("./", "../"))

    def _resolve_model_source(self) -> tuple[str, bool, str]:
        if self._looks_like_local_path(self.model_id):
            candidate = Path(self.model_id).expanduser()
            if not candidate.exists():
                raise RuntimeError(
                    f"Local model path not found: {candidate}. "
                    "Mount the model directory or update KUGEL_MODEL_ID."
                )
            logger.info("Local model path exists: %s", candidate)
            return str(candidate), True, "local"

        if not self.allow_hf:
            raise RuntimeError(
                f"Model '{self.model_id}' is not a local path and HF fallback is disabled. "
                "Set KUGEL_ALLOW_HF=true or mount models."
            )

        repo_id = self.model_id
        if "/" not in repo_id:
            repo_id = self.hf_repo_id
            logger.warning(
                "KUGEL_MODEL_ID=%s does not look like a repo id; using KUGEL_HF_REPO_ID=%s",
                self.model_id,
                repo_id,
            )
        return repo_id, False, "hf"

    def load(self) -> None:
        from kugelaudio_open.models import KugelAudioForConditionalGenerationInference
        from kugelaudio_open.processors import KugelAudioProcessor

        self._ensure_hf_cache()
        model_source, local_files_only, source_label = self._resolve_model_source()
        logger.info(
            "KugelAudio model id=%s, resolved_source=%s (%s)",
            self.model_id,
            model_source,
            source_label,
        )
        logger.info(
            "HF allow=%s, HF_HOME=%s, HF_HUB_CACHE=%s, TRANSFORMERS_CACHE=%s, HF revision=%s",
            self.allow_hf,
            os.environ.get("HF_HOME"),
            os.environ.get("HF_HUB_CACHE"),
            os.environ.get("TRANSFORMERS_CACHE"),
            self.hf_revision or "default",
        )
        logger.info(
            "CUDA available=%s, device=%s", torch.cuda.is_available(), self.device
        )
        if self.device == "cuda":
            try:
                logger.info("CUDA device name=%s", torch.cuda.get_device_name(0))
            except Exception as exc:
                logger.warning("Failed to read CUDA device name: %s", exc)

        model_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": False,
            "_fast_init": False,
            "device_map": None,
            "local_files_only": local_files_only,
        }
        if self.hf_revision and not local_files_only:
            model_kwargs["revision"] = self.hf_revision

        if self.device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"

        try:
            model = KugelAudioForConditionalGenerationInference.from_pretrained(
                model_source,
                **model_kwargs,
            )
        except Exception as exc:
            if model_kwargs.get("attn_implementation"):
                logger.warning("Flash attention load failed, falling back: %s", exc)
                model_kwargs.pop("attn_implementation", None)
                model = KugelAudioForConditionalGenerationInference.from_pretrained(
                    model_source,
                    **model_kwargs,
                )
            else:
                raise

        model.to(self.device)
        model.eval()
        model.model.strip_encoders()

        processor_kwargs = {"local_files_only": local_files_only}
        if self.hf_revision and not local_files_only:
            processor_kwargs["revision"] = self.hf_revision
        processor = KugelAudioProcessor.from_pretrained(
            model_source,
            **processor_kwargs,
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
