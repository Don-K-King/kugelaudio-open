"""Language registry helpers for KugelTTS."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "language_registry.json"
)


@dataclass(frozen=True)
class LanguageRegistry:
    default: str
    supported: frozenset[str]

    def supports(self, tag: str) -> bool:
        return tag in self.supported


@dataclass(frozen=True)
class VoiceRegistry:
    mapping: dict[str, str]

    def get(self, tag: str) -> Optional[str]:
        return self.mapping.get(tag)


def _normalize_registry_tag(tag: str) -> str:
    normalized = normalize_language_tag(tag)
    if normalized is None:
        raise ValueError("language tags must be non-empty strings")
    return normalized


def _coerce_supported(values: Iterable[str]) -> frozenset[str]:
    normalized = {_normalize_registry_tag(value) for value in values}
    return frozenset(normalized)


def load_language_registry(path: Optional[Path] = None) -> LanguageRegistry:
    registry_path = path or Path(
        os.getenv("KUGEL_LANGUAGE_REGISTRY_PATH", DEFAULT_LANGUAGE_REGISTRY_PATH)
    )
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Language registry not found at {registry_path}. "
            "Set KUGEL_LANGUAGE_REGISTRY_PATH or provide the default file."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Language registry JSON is invalid: {registry_path}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Language registry must be a JSON object.")

    default = payload.get("default")
    supported = payload.get("supported")
    if not isinstance(default, str) or not default.strip():
        raise RuntimeError("Language registry requires a non-empty 'default' value.")
    if not isinstance(supported, list) or not all(
        isinstance(value, str) for value in supported
    ):
        raise RuntimeError("Language registry requires 'supported' to be a list of strings.")

    normalized_default = _normalize_registry_tag(default)
    normalized_supported = _coerce_supported(supported)
    if normalized_default not in normalized_supported:
        logger.warning(
            "Language registry default '%s' not in supported list; adding it.",
            normalized_default,
        )
        normalized_supported = frozenset({*normalized_supported, normalized_default})

    return LanguageRegistry(default=normalized_default, supported=normalized_supported)


def normalize_language_tag(value: Optional[str]) -> Optional[str]:
    """Normalize BCP-47 tags: trim, lowercase language, uppercase region."""
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None

    parts = stripped.split("-")
    if not parts:
        return None

    parts[0] = parts[0].lower()
    if len(parts) > 1:
        region = parts[1]
        if region.isalpha() and len(region) == 2:
            parts[1] = region.upper()
        elif region.isdigit() and len(region) == 3:
            parts[1] = region
    return "-".join(parts)


def resolve_language_tag(
    value: Optional[str],
    registry: LanguageRegistry,
    *,
    allow_default_fallback: bool = True,
) -> Optional[str]:
    normalized = normalize_language_tag(value)
    if normalized is None:
        return None
    if registry.supports(normalized):
        return normalized

    parts = normalized.split("-")
    if len(parts) > 1:
        base = parts[0]
        if registry.supports(base):
            return base

    if allow_default_fallback and registry.supports(registry.default):
        return registry.default

    return None


def load_voice_registry(path: Optional[Path] = None) -> VoiceRegistry:
    registry_path = path or Path(
        os.getenv(
            "KUGEL_VOICE_REGISTRY_PATH",
            Path(__file__).resolve().parent / "voice_registry.json",
        )
    )
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Voice registry not found at {registry_path}. "
            "Set KUGEL_VOICE_REGISTRY_PATH or provide the default file."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Voice registry JSON is invalid: {registry_path}") from exc

    if not isinstance(payload, dict) or not payload:
        raise RuntimeError("Voice registry must be a non-empty JSON object.")

    normalized: dict[str, str] = {}
    for language, voice in payload.items():
        if not isinstance(language, str) or not language.strip():
            raise RuntimeError("Voice registry languages must be non-empty strings.")
        if not isinstance(voice, str) or not voice.strip():
            raise RuntimeError("Voice registry voice ids must be non-empty strings.")
        normalized_language = normalize_language_tag(language)
        if normalized_language is None:
            raise RuntimeError(
                "Voice registry languages must be valid BCP-47 tags or 'default'."
            )
        normalized[normalized_language] = voice.strip()

    return VoiceRegistry(mapping=normalized)
