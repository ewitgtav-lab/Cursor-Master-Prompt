from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OcrSpaceConfig:
    api_key: str
    language: str = "eng"


def load_ocrspace_config(*, secrets: dict[str, object] | None = None) -> OcrSpaceConfig:
    secrets = secrets or {}
    api_key = str(secrets.get("OCRSPACE_API_KEY") or os.getenv("OCRSPACE_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing OCRSPACE_API_KEY. Set it in Streamlit Secrets.")

    language = str(secrets.get("OCRSPACE_LANGUAGE") or os.getenv("OCRSPACE_LANGUAGE") or "eng").strip()
    return OcrSpaceConfig(api_key=api_key, language=language)


def ocrspace_extract_text(
    *,
    config: OcrSpaceConfig,
    file_bytes: bytes,
    filename: str,
    mime_type: str,
    timeout_seconds: int = 60,
) -> str:
    """
    OCR.Space endpoint supports images and PDFs via multipart upload.
    Docs: https://ocr.space/ocrapi
    """
    try:
        import requests
    except Exception as e:
        raise RuntimeError("Missing dependency: requests") from e

    url = "https://api.ocr.space/parse/image"
    data = {
        "apikey": config.api_key,
        "language": config.language,
        "isOverlayRequired": "false",
        "detectOrientation": "true",
        "scale": "true",
        "OCREngine": "2",
    }

    files = {
        "file": (filename, file_bytes, mime_type or "application/octet-stream"),
    }

    try:
        r = requests.post(url, data=data, files=files, timeout=timeout_seconds)
    except Exception as e:
        raise RuntimeError("OCR.Space request failed (network/timeout).") from e

    try:
        payload = r.json()
    except Exception as e:
        raise RuntimeError(f"OCR.Space returned non-JSON response (status {r.status_code}).") from e

    if r.status_code != 200:
        raise RuntimeError(f"OCR.Space error (status {r.status_code}): {payload!r}")

    if payload.get("IsErroredOnProcessing"):
        msg = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "Unknown OCR.Space error"
        raise RuntimeError(f"OCR.Space processing error: {msg}")

    parsed_results = payload.get("ParsedResults") or []
    texts: list[str] = []
    for item in parsed_results:
        t = (item or {}).get("ParsedText") or ""
        if t.strip():
            texts.append(t)

    return "\n".join(texts).strip()

