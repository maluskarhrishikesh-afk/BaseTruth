"""Record raw VLM responses for PAN and Payslip extraction from both
gemma-4-31b-it (Google AI Studio) and gemma4:e4b (Ollama local).

Usage:
    python scripts/record_vlm_responses.py

Outputs (written to logs/vlm_responses/):
    pan_google_gemma-4-31b-it.txt
    pan_ollama_gemma4-e4b.txt
    payslip_google_gemma-4-31b-it.txt
    payslip_ollama_gemma4-e4b.txt
"""

import base64
import json
import os
import re
import sys

import requests

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

PAN_IMAGE_PATH   = os.path.join(ROOT, "artifacts", "debug", "pan_sig_annotated.jpg")
PAYSLIP_PDF_PATH = os.path.join(ROOT, "tests", "sample", "Payslip_2025_Dec.pdf")
SETTINGS_PATH    = os.path.join(ROOT, "artifacts", "config", "settings.json")
OUT_DIR          = os.path.join(ROOT, "logs", "vlm_responses")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load settings ─────────────────────────────────────────────────────────────
with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
    settings = json.load(fh)

providers   = settings.get("providers", {})
google_cfg  = providers.get("google", {})
ollama_cfg  = providers.get("ollama", {})

GOOGLE_BASE = google_cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta")
GOOGLE_KEY  = google_cfg.get("api_key", "")
GOOGLE_MODEL = "gemma-4-31b-it"

OLLAMA_BASE  = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:e4b"

CONNECT_TIMEOUT = 10   # seconds to wait for connection
READ_TIMEOUT    = 240  # seconds to wait for response body


# ── Helper: import real prompts from codebase ─────────────────────────────────
from basetruth.integrations import ollama as _ollama_mod
from basetruth.integrations import document_extract as _doc_mod

PAN_SYSTEM_PROMPT = _ollama_mod.PAN_COMBINED_EXTRACTION_SYSTEM_PROMPT
PAN_USER_PROMPT   = _ollama_mod.PAN_COMBINED_EXTRACTION_PROMPT

PAYSLIP_SYSTEM_PROMPT = _doc_mod._get_prompt("system")
PAYSLIP_USER_PROMPT   = _doc_mod._get_prompt("financial")   # payslip maps to the 'financial' prompt category


# ── Helper: write result to a file ───────────────────────────────────────────
def save(name: str, content: str) -> None:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"  Saved → {path}")


# ── Helper: encode image to base64 ───────────────────────────────────────────
def load_image_b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode()


# ── Helper: extract text from a PDF (first few KB of embedded text) ───────────
def extract_pdf_text(pdf_path: str) -> str:
    """Try to get embedded text from a PDF using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(pdf_path)
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
        combined = "\n".join(texts).strip()
        print(f"  PDF text extracted: {len(combined)} chars, {len(reader.pages)} pages")
        return combined
    except Exception as e:
        print(f"  PDF text extraction failed: {e} — will use image fallback")
        return ""


# ── Helper: render first PDF page to PNG ─────────────────────────────────────
def pdf_first_page_b64(pdf_path: str) -> str | None:
    """Render first PDF page to a JPEG and return base64."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(2, 2)  # 2x zoom for clarity
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("jpeg")
        return base64.b64encode(img_bytes).decode()
    except Exception as e:
        print(f"  PDF→image render failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Google AI Studio calls
# ─────────────────────────────────────────────────────────────────────────────

def google_call_with_image(system_prompt: str, user_prompt: str, image_b64: str, label: str) -> str:
    """Send system+user prompt + image to Google generateContent. Returns raw JSON string."""
    url = f"{GOOGLE_BASE}/models/{GOOGLE_MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": GOOGLE_KEY}

    # Google uses system_instruction + contents (no 'system' role in contents)
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": user_prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                ],
            }
        ],
    }

    print(f"  [{label}] POST {url}")
    try:
        r = requests.post(url, headers=headers, json=payload,
                          timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except Exception as e:
        return f"REQUEST ERROR: {e}"

    print(f"  [{label}] Status {r.status_code}, body {len(r.text)} chars")
    return r.text


def google_call_text_only(system_prompt: str, user_prompt: str, label: str) -> str:
    """Send system+user (text-only) to Google generateContent. Returns raw JSON string."""
    url = f"{GOOGLE_BASE}/models/{GOOGLE_MODEL}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": GOOGLE_KEY}

    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]},
        ],
    }

    print(f"  [{label}] POST {url}")
    try:
        r = requests.post(url, headers=headers, json=payload,
                          timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except Exception as e:
        return f"REQUEST ERROR: {e}"

    print(f"  [{label}] Status {r.status_code}, body {len(r.text)} chars")
    return r.text


# ─────────────────────────────────────────────────────────────────────────────
# Ollama calls
# ─────────────────────────────────────────────────────────────────────────────

def ollama_call_with_image(system_prompt: str, user_prompt: str, image_b64: str, label: str) -> str:
    """Send system+user+image to Ollama /api/chat. Returns raw JSON string."""
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt, "images": [image_b64]},
        ],
    }

    print(f"  [{label}] POST {url}")
    try:
        r = requests.post(url, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except Exception as e:
        return f"REQUEST ERROR (Ollama unreachable?): {e}"

    print(f"  [{label}] Status {r.status_code}, body {len(r.text)} chars")
    return r.text


def ollama_call_text_only(system_prompt: str, user_prompt: str, label: str) -> str:
    """Send system+user (text-only) to Ollama /api/chat. Returns raw JSON string."""
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    print(f"  [{label}] POST {url}")
    try:
        r = requests.post(url, json=payload, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    except Exception as e:
        return f"REQUEST ERROR (Ollama unreachable?): {e}"

    print(f"  [{label}] Status {r.status_code}, body {len(r.text)} chars")
    return r.text


# ─────────────────────────────────────────────────────────────────────────────
# Format a pretty summary of a raw response
# ─────────────────────────────────────────────────────────────────────────────

def format_report(label: str, raw: str) -> str:
    """Build a human-readable report: header + raw JSON + extracted model text."""
    sep = "=" * 72
    lines = [
        sep,
        f"LABEL  : {label}",
        f"CHARS  : {len(raw)}",
        sep,
        "--- RAW API RESPONSE ---",
        raw,
        "",
        "--- EXTRACTED MODEL TEXT ---",
    ]
    # Try to pull the text out of candidates
    try:
        j = json.loads(raw)
        candidates = j.get("candidates") or []
        if candidates:
            for part in candidates[0].get("content", {}).get("parts", []):
                thought = part.get("thought", False)
                text = part.get("text", "")
                lines.append(f"[{'THOUGHT' if thought else 'RESPONSE'}]\n{text}")
        else:
            # Ollama format
            msg = j.get("message", {})
            lines.append(msg.get("content", "<no content>"))
    except Exception:
        lines.append("<could not parse>")

    lines.append(sep)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 72)
    print("Step 1/4: PAN card extraction — Google gemma-4-31b-it")
    print("=" * 72)
    pan_image_b64 = load_image_b64(PAN_IMAGE_PATH)
    raw_pan_google = google_call_with_image(
        PAN_SYSTEM_PROMPT, PAN_USER_PROMPT, pan_image_b64,
        label="PAN / Google"
    )
    save(f"pan_google_{GOOGLE_MODEL}.txt", format_report(f"PAN / Google / {GOOGLE_MODEL}", raw_pan_google))

    print("\n" + "=" * 72)
    print("Step 2/4: PAN card extraction — Ollama gemma4:e4b")
    print("=" * 72)
    raw_pan_ollama = ollama_call_with_image(
        PAN_SYSTEM_PROMPT, PAN_USER_PROMPT, pan_image_b64,
        label="PAN / Ollama"
    )
    save("pan_ollama_gemma4-e4b.txt", format_report(f"PAN / Ollama / {OLLAMA_MODEL}", raw_pan_ollama))

    print("\n" + "=" * 72)
    print("Step 3/4: Payslip extraction — Google gemma-4-31b-it")
    print("=" * 72)
    pdf_text = extract_pdf_text(PAYSLIP_PDF_PATH)
    if pdf_text:
        # Text-based PDF — same path the real code takes (no image needed)
        payslip_user = PAYSLIP_USER_PROMPT + f"\n\n--- Extracted PDF text ---\n{pdf_text[:4000]}"
        raw_payslip_google = google_call_text_only(
            PAYSLIP_SYSTEM_PROMPT, payslip_user,
            label="Payslip / Google (text)"
        )
    else:
        # Render PDF to image as fallback
        img_b64 = pdf_first_page_b64(PAYSLIP_PDF_PATH)
        if img_b64:
            raw_payslip_google = google_call_with_image(
                PAYSLIP_SYSTEM_PROMPT, PAYSLIP_USER_PROMPT, img_b64,
                label="Payslip / Google (image)"
            )
        else:
            raw_payslip_google = "ERROR: could not render PDF to image"
    save(f"payslip_google_{GOOGLE_MODEL}.txt", format_report(f"Payslip / Google / {GOOGLE_MODEL}", raw_payslip_google))

    print("\n" + "=" * 72)
    print("Step 4/4: Payslip extraction — Ollama gemma4:e4b")
    print("=" * 72)
    if pdf_text:
        payslip_user_ollama = PAYSLIP_USER_PROMPT + f"\n\n--- Extracted PDF text ---\n{pdf_text[:4000]}"
        raw_payslip_ollama = ollama_call_text_only(
            PAYSLIP_SYSTEM_PROMPT, payslip_user_ollama,
            label="Payslip / Ollama (text)"
        )
    else:
        img_b64 = pdf_first_page_b64(PAYSLIP_PDF_PATH)
        if img_b64:
            raw_payslip_ollama = ollama_call_with_image(
                PAYSLIP_SYSTEM_PROMPT, PAYSLIP_USER_PROMPT, img_b64,
                label="Payslip / Ollama (image)"
            )
        else:
            raw_payslip_ollama = "ERROR: could not render PDF to image"
    save("payslip_ollama_gemma4-e4b.txt", format_report(f"Payslip / Ollama / {OLLAMA_MODEL}", raw_payslip_ollama))

    print(f"\nAll done! 4 response files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
