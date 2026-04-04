"""Gemma / Gemini VLM integration for OCR fallback.

When classical OCR (PaddleOCR / Tesseract) returns low-confidence text — or
fails to find a PAN regex — this module calls a Vision-Language Model (VLM)
to extract structured information directly from the image.

Two backends are supported (tried in order):
  1. Local Gemma model via 🤗 Transformers (requires ``transformers`` +
     ``torch`` / ``accelerate`` and the model already downloaded).
  2. Google Gemini API (requires the ``google-generativeai`` package and the
     ``GEMINI_API_KEY`` environment variable to be set).

Both backends degrade gracefully: if neither is available the function
returns ('', 'unavailable') and the pipeline falls back to plain Tesseract
text without crashing.

Public API
----------
  extract_text_with_vlm(path: Path, doc_hint: str = "") -> Tuple[str, str]
      Returns (extracted_text, engine_name).
      engine_name is one of: 'gemma_local', 'gemini_api', 'unavailable', 'error'
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prompt template
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT = (
    "You are an OCR assistant specialized in Indian identity and financial documents. "
    "Extract ALL text from this image exactly as it appears. "
    "Preserve the original spelling, capitalization, and spacing. "
    "Output only the extracted text, nothing else. "
    "If you see a PAN card, make sure to extract: Name, Father's Name, Date of Birth, "
    "and PAN Number (format: 5 letters, 4 digits, 1 letter). "
    "If you see an Aadhaar card, extract: Name, Date of Birth, Gender, Address, "
    "and last 4 digits of Aadhaar number."
)

_STRUCTURED_PROMPT = (
    "You are an OCR assistant specialized in Indian identity documents. "
    "Extract all text from this document image. "
    "Output only the raw text as seen on the document, line by line."
)


# ---------------------------------------------------------------------------
# Backend 1: Local Gemma via Hugging Face Transformers
# ---------------------------------------------------------------------------

def _extract_with_local_gemma(path: Path, prompt: str) -> Tuple[str, str]:
    """Attempt OCR using a locally-downloaded Gemma or Gemini model.

    Looks for the model at the path set by the ``GEMMA_MODEL_PATH`` environment
    variable, or falls back to ``google/gemma-3-4b-it`` from the Transformers
    model hub.

    Returns ('', 'unavailable') if transformers or torch are not installed.
    """
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText  # type: ignore
        import torch  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return "", "unavailable"

    model_id = os.environ.get("GEMMA_MODEL_PATH", "google/gemma-3-4b-it")

    try:
        log.debug("Loading local Gemma model: %s", model_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )
        model.eval()

        with Image.open(str(path)) as img:
            pil_img = img.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text_input = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=text_input, images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        decoded = processor.decode(output[0], skip_special_tokens=True)
        # Strip the prompt echo that some models include
        if prompt[:30] in decoded:
            decoded = decoded[decoded.find(prompt[:30]) + len(prompt):]

        result = decoded.strip()
        log.debug(
            "Local Gemma OCR extracted %d chars from %s", len(result), path.name
        )
        return result, "gemma_local"

    except Exception as exc:  # noqa: BLE001
        log.debug("Local Gemma OCR failed for %s: %s", path.name, exc)
        return "", "error"


# ---------------------------------------------------------------------------
# Backend 2: Google Gemini API
# ---------------------------------------------------------------------------

def _extract_with_gemini_api(path: Path, prompt: str) -> Tuple[str, str]:
    """Call the Gemini Vision API to extract text from *path*.

    Requires:
      - ``pip install google-generativeai``
      - ``GEMINI_API_KEY`` environment variable
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "", "unavailable"

    try:
        import google.generativeai as genai  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return "", "unavailable"

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        with Image.open(str(path)) as img:
            pil_img = img.convert("RGB")

        response = model.generate_content([prompt, pil_img])
        result = (response.text or "").strip()
        log.debug(
            "Gemini API OCR extracted %d chars from %s", len(result), path.name
        )
        return result, "gemini_api"

    except Exception as exc:  # noqa: BLE001
        log.debug("Gemini API OCR failed for %s: %s", path.name, exc)
        return "", "error"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_with_vlm(
    path: Path,
    doc_hint: str = "",
    prompt: str | None = None,
) -> Tuple[str, str]:
    """Extract text from an image using the best available VLM backend.

    Parameters
    ----------
    path      : path to the image file
    doc_hint  : optional label ('pan', 'aadhaar', etc.) to specialise the prompt
    prompt    : override prompt; uses sensible default if None

    Returns
    -------
    (text, engine) where engine is one of:
      'gemma_local'   — local Hugging Face model
      'gemini_api'    — Google Gemini Vision API
      'unavailable'   — no VLM backend is set up
      'error'         — backend present but call failed
    """
    if not path.exists():
        return "", "unavailable"

    effective_prompt = prompt or _DEFAULT_PROMPT

    # Backend 1: local model
    text, engine = _extract_with_local_gemma(path, effective_prompt)
    if text and engine not in ("unavailable", "error"):
        return text, engine

    # Backend 2: Gemini API
    text, engine = _extract_with_gemini_api(path, effective_prompt)
    if text and engine not in ("unavailable", "error"):
        return text, engine

    return "", "unavailable"


def vlm_available() -> bool:
    """Return True if at least one VLM backend is configured."""
    if os.environ.get("GEMINI_API_KEY"):
        try:
            import google.generativeai  # type: ignore  # noqa: F401
            return True
        except ImportError:
            pass

    gemma_path = os.environ.get("GEMMA_MODEL_PATH", "")
    if gemma_path:
        try:
            from transformers import AutoProcessor  # type: ignore  # noqa: F401
            return True
        except ImportError:
            pass

    return False
