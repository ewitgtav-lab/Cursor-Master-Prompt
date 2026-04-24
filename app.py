from __future__ import annotations

import os
import re
import shutil
import traceback
from dataclasses import dataclass
from typing import Final, Iterable

import streamlit as st


APP_TITLE: Final = "The Clarity Bridge"
APP_TAGLINE: Final = "From Complexity to Clarity."


DOCUMENT_TYPES: Final[list[str]] = ["Legal", "Medical", "Insurance", "Technical"]
PERSONAS: Final[list[str]] = ["A 5th Grader", "A Busy Parent", "A Non-native Speaker"]

DEFAULT_TEXT_MODEL_CANDIDATES: Final[list[str]] = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]

DEFAULT_VISION_MODEL_CANDIDATES: Final[list[str]] = [
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]


@dataclass(frozen=True)
class ExtractedDocument:
    filename: str
    mime_type: str
    text: str


def _soft_tech_css() -> str:
    # Streamlit doesn't expose full theming from app.py; this nudges layout toward the requested aesthetic.
    return """
<style>
  .block-container { padding-top: 2.25rem; padding-bottom: 2.5rem; max-width: 1100px; }
  .clarity-hero { text-align:center; margin: 0.25rem 0 1.25rem; }
  .clarity-hero h1 { margin-bottom: 0.25rem; letter-spacing: -0.02em; color: #262730; }
  .clarity-hero p { margin-top: 0; color: #4b5563; }
  .privacy-banner {
    border: 1px solid rgba(38,39,48,0.12);
    background: rgba(240,242,246,0.75);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    color: #262730;
    margin: 0.75rem 0 1rem;
  }
  .scroll-box {
    border: 1px solid rgba(38,39,48,0.12);
    background: rgba(240,242,246,0.65);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    height: 420px;
    overflow: auto;
    white-space: pre-wrap;
    color: #262730;
  }
  .result-box {
    border: 1px solid rgba(38,39,48,0.12);
    background: rgba(255,255,255,0.9);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    min-height: 420px;
    overflow: auto;
    white-space: pre-wrap;
    color: #262730;
  }
  .muted { color: #6b7280; font-size: 0.92rem; }
</style>
"""


def build_gemini_system_prompt(document_type: str, persona: str) -> str:
    return f"""You are a Plain Language Advocate. Your task is to protect the user from "Fine Print" and "Jargon".

Document type context: {document_type}
Explain it to me like: {persona}

Requirements:
- Tone: calm, supportive, and 100% jargon-free.
- Be concrete. If the document is missing key details, say so plainly.
- Do not invent facts not present in the document.

Output format (use these exact section headers):

TL;DR (one sentence):
<exactly one sentence>

Checklist (3–5 action items):
- <action item>

Gotchas / Red Flags:
- <red flag>

Jargon Decoder (2-column table):
| The Scary Word | What it actually means |
|---|---|
| ... | ... |
"""


def build_gemini_user_prompt(extracted_text: str) -> str:
    return f"""Simplify the following document for an everyday person.

Document text:
{extracted_text}
"""


def get_gemini_api_key() -> str | None:
    # Streamlit Cloud-friendly: support st.secrets and env vars.
    # Prefer secrets if available.
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        api_key = None
    return api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def get_gemini_model_text() -> str | None:
    try:
        v = st.secrets.get("GEMINI_MODEL_TEXT")  # type: ignore[attr-defined]
        return str(v) if v else None
    except Exception:
        return os.getenv("GEMINI_MODEL_TEXT")

def get_gemini_model_vision() -> str | None:
    try:
        v = st.secrets.get("GEMINI_MODEL_VISION")  # type: ignore[attr-defined]
        return str(v) if v else None
    except Exception:
        return os.getenv("GEMINI_MODEL_VISION")

def can_use_gemini() -> bool:
    return bool(get_gemini_api_key())

def _set_last_extraction_error(message: str) -> None:
    try:
        st.session_state["_clarity_last_extraction_error"] = message
    except Exception:
        pass

def _get_last_extraction_error() -> str:
    try:
        return str(st.session_state.get("_clarity_last_extraction_error", "") or "")
    except Exception:
        return ""


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_join(parts: Iterable[str]) -> str:
    joined = "\n".join(p for p in parts if p and p.strip())
    return _normalize_whitespace(joined)

def is_tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    import pdfplumber

    chunks: list[str] = []
    # pdfplumber expects a file path or a file-like object, not raw bytes.
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                chunks.append(txt)
    return _safe_join(chunks)

def _looks_like_model_not_found(err: Exception) -> bool:
    msg = f"{err!r}".lower()
    return ("not_found" in msg) or ("is not found" in msg) or ("listmodels" in msg) or ("404" in msg)

def _gemini_generate_text(*, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
    """
    Generate text with Gemini.
    Prefers `google-genai` (new SDK) and falls back to `google-generativeai` (legacy SDK).
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY in Streamlit Secrets or as an environment variable.")

    model_override = get_gemini_model_text()
    model_candidates = [model_override] if model_override else DEFAULT_TEXT_MODEL_CANDIDATES

    # New SDK: google-genai
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        last_err: Exception | None = None
        for model_name in model_candidates:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                return _normalize_whitespace(getattr(resp, "text", "") or "")
            except Exception as e:
                last_err = e
                if _looks_like_model_not_found(e) and not model_override:
                    continue
                raise
        if last_err is not None:
            raise last_err
    except Exception:
        pass

    # Legacy SDK: google-generativeai
    try:
        import google.generativeai as genai  # type: ignore[import-not-found]

        genai.configure(api_key=api_key)
        last_err: Exception | None = None
        for model_name in model_candidates:
            try:
                model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
                resp = model.generate_content(
                    user_prompt,
                    generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens},
                )
                return _normalize_whitespace(getattr(resp, "text", "") or "")
            except Exception as e:
                last_err = e
                if _looks_like_model_not_found(e) and not model_override:
                    continue
                raise
        if last_err is not None:
            raise last_err
    except Exception as e:
        raise RuntimeError("Gemini request failed (API error). Please try again.") from e


def _gemini_vision_ocr(images: list["Image.Image"]) -> str:
    """
    Cloud-friendly OCR fallback using Gemini Vision:
    - Works on Streamlit Community Cloud without system packages (no Tesseract needed).
    - Returns extracted text only (no summarization).
    """
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError("Missing Gemini API key for vision OCR.")

    model_override = get_gemini_model_vision()
    model_candidates = [model_override] if model_override else DEFAULT_VISION_MODEL_CANDIDATES

    prompt = (
        "Extract ALL readable text from these images.\n"
        "Return only the text, preserving line breaks where possible.\n"
        "Do not add commentary, headings, or summaries."
    )

    # New SDK: google-genai
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        parts: list[types.Part] = []
        for img in images:
            buf = BytesIO()
            img.save(buf, format="PNG")
            parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))

        last_err: Exception | None = None
        for model_name in model_candidates:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[prompt, *parts],
                    config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=4096),
                )
                return _normalize_whitespace(getattr(resp, "text", "") or "")
            except Exception as e:
                last_err = e
                if _looks_like_model_not_found(e) and not model_override:
                    continue
                raise
        if last_err is not None:
            raise last_err
    except Exception as e:
        # Don't lose the root cause; surface it to the UI via session_state.
        _set_last_extraction_error("Gemini Vision OCR (google-genai) traceback:\n" + traceback.format_exc())
        last = repr(e)

    # Legacy SDK: google-generativeai
    try:
        import google.generativeai as genai  # type: ignore[import-not-found]

        genai.configure(api_key=api_key)
        last_err: Exception | None = None
        for model_name in model_candidates:
            try:
                model = genai.GenerativeModel(model_name=model_name)
                resp = model.generate_content(
                    [prompt, *images],
                    generation_config={"temperature": 0.0, "max_output_tokens": 4096},
                )
                return _normalize_whitespace(getattr(resp, "text", "") or "")
            except Exception as e:
                last_err = e
                if _looks_like_model_not_found(e) and not model_override:
                    continue
                raise
        if last_err is not None:
            raise last_err
    except Exception as e:
        _set_last_extraction_error("Gemini Vision OCR (google-generativeai) traceback:\n" + traceback.format_exc())
        raise RuntimeError(f"Gemini Vision OCR failed. Last error: {last if 'last' in locals() else 'n/a'}; then {e!r}") from e

def _pdf_pages_to_images(pdf_bytes: bytes, *, max_pages: int = 6, resolution: int = 175) -> list["Image.Image"]:
    import pdfplumber
    from PIL import Image

    images: list[Image.Image] = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            try:
                pil_img: Image.Image = page.to_image(resolution=resolution).original
                images.append(pil_img.convert("RGB"))
            except Exception:
                continue
    return images


def _try_ocr_image_bytes(image_bytes: bytes) -> str:
    """
    OCR is optional because Streamlit Cloud environments vary.
    - If pytesseract is available AND a Tesseract binary exists, use it.
    - Otherwise, return empty string and let the UI show a helpful message.
    """
    try:
        from PIL import Image
    except Exception:
        return ""

    try:
        import pytesseract
    except Exception:
        return ""

    if not is_tesseract_available():
        return ""

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return ""

    try:
        return _normalize_whitespace(pytesseract.image_to_string(img))
    except Exception:
        return ""


def extract_text_from_image(image_bytes: bytes) -> str:
    text = _try_ocr_image_bytes(image_bytes)
    return _normalize_whitespace(text)

def extract_text_from_image_via_gemini(image_bytes: bytes) -> str:
    from PIL import Image

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return _gemini_vision_ocr([img])


def extract_document(file_name: str, mime_type: str, data: bytes) -> ExtractedDocument:
    if mime_type == "application/pdf" or file_name.lower().endswith(".pdf"):
        text = extract_text_from_pdf(data)
        if not text.strip():
            # Optional OCR fallback for scanned PDFs
            try:
                import pdfplumber
                from PIL import Image
            except Exception:
                return ExtractedDocument(filename=file_name, mime_type=mime_type, text="")

            ocr_chunks: list[str] = []
            try:
                with pdfplumber.open(BytesIO(data)) as pdf:
                    for page in pdf.pages:
                        try:
                            pil_img: Image.Image = page.to_image(resolution=200).original
                            buf = BytesIO()
                            pil_img.save(buf, format="PNG")
                            ocr_txt = _try_ocr_image_bytes(buf.getvalue())
                            if ocr_txt.strip():
                                ocr_chunks.append(ocr_txt)
                        except Exception:
                            continue
            except Exception:
                ocr_chunks = []

            text = _safe_join(ocr_chunks)

        # Cloud-friendly OCR fallback (Gemini Vision) when Tesseract isn't available.
        if not text.strip() and can_use_gemini():
            try:
                images = _pdf_pages_to_images(data, max_pages=6, resolution=175)
                if images:
                    text = _gemini_vision_ocr(images)
            except Exception as e:
                _set_last_extraction_error(f"Gemini Vision OCR (PDF) error: {e!r}")

        return ExtractedDocument(filename=file_name, mime_type=mime_type, text=text)

    if mime_type in {"image/png", "image/jpeg"} or file_name.lower().endswith((".png", ".jpg", ".jpeg")):
        text = extract_text_from_image(data)
        if not text.strip() and can_use_gemini():
            try:
                text = extract_text_from_image_via_gemini(data)
            except Exception as e:
                _set_last_extraction_error(f"Gemini Vision OCR (image) error: {e!r}")
        return ExtractedDocument(filename=file_name, mime_type=mime_type, text=text)

    return ExtractedDocument(filename=file_name, mime_type=mime_type, text="")


def run_gemini(system_prompt: str, user_prompt: str, *, timeout_hint_seconds: int = 40) -> str:
    # timeout_hint_seconds is best-effort; SDK support varies.
    _ = timeout_hint_seconds
    return _gemini_generate_text(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=0.3,
        max_output_tokens=1200,
    )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Tool Suite")
        # `st.page_link` can crash on some Streamlit versions if multipage metadata isn't present.
        # Use a simple label/button instead (this is a single-page app).
        st.markdown("**The Clarity Bridge**")
        st.markdown(
            "Looking for your other tool?\n\n- [Duplicate Detective](https://share.streamlit.io/)",
        )
        st.markdown("---")
        st.markdown("### Settings")
        st.caption("Tip: Add your API key via Streamlit Secrets.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🌉", layout="wide")
    st.markdown(_soft_tech_css(), unsafe_allow_html=True)

    _render_sidebar()

    st.markdown(
        f"""
<div class="clarity-hero">
  <h1>{APP_TITLE}</h1>
  <p>{APP_TAGLINE}</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="privacy-banner"><b>Privacy Mode:</b> Files are processed in RAM and never stored on a database. Your privacy is protected.</div>',
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([1.1, 1.0, 1.0])
    with col_a:
        document_type = st.selectbox("Document Type", DOCUMENT_TYPES, index=0)
    with col_b:
        persona = st.selectbox("Explain it to me like", PERSONAS, index=0)
    with col_c:
        st.write("")
        st.write("")
        st.caption("Best results with clear, text-heavy documents.")

    uploaded = st.file_uploader("Upload a PDF or image (PDF, JPG, PNG)", type=["pdf", "jpg", "jpeg", "png"])

    if not uploaded:
        st.info("Upload a document to see a clean Before vs. After simplification.")
        return

    file_bytes = uploaded.getvalue()
    mime_type = uploaded.type or ""

    with st.spinner("Reading your document…"):
        try:
            extracted = extract_document(uploaded.name, mime_type, file_bytes)
        except Exception:
            extracted = ExtractedDocument(filename=uploaded.name, mime_type=mime_type, text="")

    if not extracted.text.strip():
        ocr_ready = is_tesseract_available()
        st.error("I couldn't extract readable text from that file.")
        st.caption(
            "Tip: Streamlit Community Cloud usually can’t install system packages like Tesseract, "
            "so this app will try a Gemini Vision OCR fallback when an API key is set."
        )
        st.caption(
            f"File: {uploaded.name} • Type: {mime_type or 'unknown'} • Tesseract available: {'yes' if ocr_ready else 'no'} • Gemini key set: {'yes' if can_use_gemini() else 'no'}"
        )
        st.markdown(
            "- If this is a **scanned** document (image-only PDF / photo), OCR is required.\n"
            "- On **Streamlit Cloud**, the easiest path is setting your **Gemini API key** (Secrets) so Vision OCR can run.\n"
            "- On **local Windows**, you can also install the **Tesseract** app and add it to PATH."
        )
        last_err = _get_last_extraction_error()
        if last_err:
            with st.expander("Show technical details"):
                st.code(last_err)
        st.stop()

    system_prompt = build_gemini_system_prompt(document_type=document_type, persona=persona)
    user_prompt = build_gemini_user_prompt(extracted.text)

    with st.spinner("Creating your simplified explanation…"):
        try:
            simplified = run_gemini(system_prompt, user_prompt)
        except Exception as e:
            st.error(str(e))
            st.stop()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("Original document text")
        st.markdown(f'<div class="scroll-box">{st._utils.escape_markdown(extracted.text)}</div>', unsafe_allow_html=True)
        st.caption(f"Source: {extracted.filename}")

    with right:
        st.subheader("Simplified Truth")
        st.markdown(f'<div class="result-box">{st._utils.escape_markdown(simplified)}</div>', unsafe_allow_html=True)

        st.download_button(
            "Download simplified version (.txt)",
            data=simplified.encode("utf-8"),
            file_name="clarity_bridge_simplified.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.write("")
        st.caption("Was this helpful?")
        try:
            st.feedback("thumbs")
        except Exception:
            # Backwards compatibility with older Streamlit versions.
            st.button("👍 Helpful")
            st.button("👎 Not helpful")


# BytesIO is used in several extract paths; keep import after Streamlit for quicker startup.
from io import BytesIO  # noqa: E402  (intentional late import)


if __name__ == "__main__":
    main()

