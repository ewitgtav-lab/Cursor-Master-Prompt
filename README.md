# The Clarity Bridge

**From Complexity to Clarity.**

The Clarity Bridge is a Streamlit app that turns confusing documents (legal, medical, insurance, technical) into plain language so everyday people can understand what matters, what to do next, and what to watch out for.

## What it does

- **Upload** a PDF or an image (JPG/PNG).
- **Extract** the text (PDF text extraction + optional OCR fallback for scanned docs).
- **Simplify** with an AI “Plain Language Advocate” that produces:
  - **TL;DR**: exactly one sentence
  - **Checklist**: 3–5 action items
  - **Gotchas / Red Flags**
  - **Jargon Decoder**: “Scary word” vs “what it actually means”
- **Export** the simplified version as a `.txt` file.

## Privacy

The app processes your uploaded file **in memory (RAM)** and does **not** store it in a database.

## Setup

### 1) Install

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Add your API keys (Streamlit Cloud-friendly)

Set these in **Streamlit Secrets** (recommended on Streamlit Cloud):

- `GROQ_API_KEY` (LLM for the simplification)
- `OCRSPACE_API_KEY` (OCR fallback for scanned PDFs/images)

Optional:

- `GROQ_MODEL` (default: `llama-3.1-8b-instant`)

### 3) Run

```bash
streamlit run app.py
```

## Notes on OCR (scanned documents)

- On Streamlit Cloud, system installs (like Tesseract) are not reliable.
- This app uses **OCR.Space** as the OCR fallback when `OCRSPACE_API_KEY` is set.

## Customize the AI prompt

The prompt is intentionally modular:

- `build_gemini_system_prompt(...)` in `app.py`
- `build_gemini_user_prompt(...)` in `app.py`

Edit those functions to change the tone, sections, or formatting without touching the rest of the app.

