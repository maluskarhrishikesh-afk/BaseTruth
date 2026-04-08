"""
Temporary accuracy test: extract data from Psychology_Marksheet.jpg and SSC-Marksheet.pdf
using three methods — LiteParse, PaddleOCR/pytesseract, and Gemma4.

Run from BaseTruth root:
    python tests/extract_accuracy_test.py

Output files written next to this script:
    tests/output_liteparse.json
    tests/output_paddleocr.json
    tests/output_gemma4.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# ── Ensure the src package is importable ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TESTS_DIR = Path(__file__).resolve().parent
FILES = {
    "Psychology_Marksheet.jpg": TESTS_DIR / "Psychology_Marksheet.jpg",
    "SSC-Marksheet.pdf": TESTS_DIR / "SSC-Marksheet.pdf",
}


# ────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────

def _write(name: str, data: dict) -> None:
    out = TESTS_DIR / name
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓  Written → {out}")


# ────────────────────────────────────────────────────────────────────────────
# Method 1 — LiteParse (Node.js CLI)
# ────────────────────────────────────────────────────────────────────────────

def run_liteparse() -> dict:
    from basetruth.integrations.liteparse import check_liteparse_available, parse_document_to_json
    from basetruth.integrations.pdf import build_liteparse_json_from_text, extract_text_from_pdf

    status = check_liteparse_available()
    results: dict = {}

    for label, path in FILES.items():
        print(f"    LiteParse ← {label}")
        if not path.exists():
            results[label] = {"error": f"File not found: {path}"}
            continue

        if not status["available"]:
            # LiteParse unavailable — fall back to raw text extraction for PDFs
            results[label] = {
                "method": "liteparse_unavailable_fallback",
                "message": status.get("message", "LiteParse not available"),
            }
            if path.suffix.lower() == ".pdf":
                raw_text = extract_text_from_pdf(path)
                parsed = build_liteparse_json_from_text(raw_text, path.name)
                results[label]["raw_text_chars"] = len(raw_text)
                results[label]["structured"] = parsed
            continue

        with tempfile.NamedTemporaryFile(suffix="_liteparse_out.json", delete=False) as tmp:
            out_path = Path(tmp.name)

        result = parse_document_to_json(path, out_path)
        if result.get("status") == "success" and out_path.exists():
            try:
                liteparse_json = json.loads(out_path.read_text(encoding="utf-8"))
                results[label] = {
                    "method": "liteparse",
                    "command_source": result.get("command_source"),
                    "parsed": liteparse_json,
                }
            except json.JSONDecodeError as exc:
                results[label] = {
                    "method": "liteparse",
                    "error": f"JSON decode error: {exc}",
                    "raw": out_path.read_text(encoding="utf-8")[:2000],
                }
        else:
            results[label] = {
                "method": "liteparse",
                "status": result.get("status"),
                "error": result.get("message", "Unknown error"),
            }

        try:
            out_path.unlink()
        except OSError:
            pass

    return results


# ────────────────────────────────────────────────────────────────────────────
# Method 2 — PaddleOCR + pytesseract
# ────────────────────────────────────────────────────────────────────────────

def run_paddleocr() -> dict:
    from basetruth.integrations.pdf import (
        _ocr_confidence_score,
        _ocr_with_paddle,
        ocr_image_directly,
    )

    results: dict = {}

    for label, path in FILES.items():
        print(f"    PaddleOCR ← {label}")
        if not path.exists():
            results[label] = {"error": f"File not found: {path}"}
            continue

        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
            # Direct image OCR
            ocr_text, engine = ocr_image_directly(path)
            results[label] = {
                "method": engine,
                "chars_extracted": len(ocr_text),
                "coherence_score": round(_ocr_confidence_score(ocr_text), 3),
                "text": ocr_text,
            }
        else:
            # PDF — rasterise first page then OCR
            from basetruth.integrations.pdf import get_document_image_bytes
            from PIL import Image as _PIL
            import io

            img_bytes = get_document_image_bytes(path)
            if img_bytes:
                pil_img = _PIL.open(io.BytesIO(img_bytes)).convert("RGB")
                paddle_text, paddle_conf = _ocr_with_paddle(pil_img)
                coherence = _ocr_confidence_score(paddle_text)

                # Also try pytesseract
                tess_text = ""
                try:
                    import pytesseract  # type: ignore
                    tess_text = pytesseract.image_to_string(pil_img, lang="eng") or ""
                except Exception as exc:
                    tess_text = f"(pytesseract unavailable: {exc})"

                results[label] = {
                    "method": "paddleocr+pytesseract",
                    "paddleocr": {
                        "mean_confidence": round(paddle_conf, 3),
                        "coherence_score": round(coherence, 3),
                        "chars_extracted": len(paddle_text),
                        "text": paddle_text,
                    },
                    "pytesseract": {
                        "chars_extracted": len(tess_text),
                        "text": tess_text,
                    },
                }
            else:
                results[label] = {
                    "method": "paddleocr",
                    "error": "Could not rasterise PDF page",
                }

    return results


# ────────────────────────────────────────────────────────────────────────────
# Method 3 — Gemma4 via Ollama (vision LLM)
# ────────────────────────────────────────────────────────────────────────────

def run_gemma4() -> dict:
    from basetruth.integrations.ollama import analyze_document_with_ollama, probe_ollama
    from basetruth.integrations.pdf import get_document_image_bytes

    # Check if Ollama is up before we try each file
    base_url, models, raw = probe_ollama()
    if not base_url:
        print("    ⚠  Ollama not reachable — Gemma4 results will be empty")

    results: dict = {
        "_ollama_available": bool(base_url),
        "_ollama_base": base_url or "",
        "_available_models": models or [],
    }

    for label, path in FILES.items():
        print(f"    Gemma4 ← {label}")
        if not path.exists():
            results[label] = {"error": f"File not found: {path}"}
            continue

        img_bytes = get_document_image_bytes(path)
        if not img_bytes:
            results[label] = {"error": "Could not produce image bytes for Gemma4"}
            continue

        analysis = analyze_document_with_ollama(img_bytes)
        if analysis:
            results[label] = {
                "method": "gemma4_ollama",
                "model": analysis.get("model", "?"),
                "document_type": analysis.get("document_type"),
                "confidence": analysis.get("confidence"),
                "extracted_fields": analysis.get("extracted_fields", {}),
                "fraud_signals": analysis.get("fraud_signals", []),
                "authenticity_assessment": analysis.get("authenticity_assessment", {}),
                "summary": analysis.get("summary", ""),
                # Include raw response so we can judge quality manually
                "raw_response": analysis.get("raw_response", ""),
            }
        else:
            results[label] = {
                "method": "gemma4_ollama",
                "error": "Ollama unavailable or returned empty response",
            }

    return results


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Method 1: LiteParse ===")
    liteparse_results = run_liteparse()
    _write("output_liteparse.json", liteparse_results)

    print("\n=== Method 2: PaddleOCR / pytesseract ===")
    paddle_results = run_paddleocr()
    _write("output_paddleocr.json", paddle_results)

    print("\n=== Method 3: Gemma4 (Ollama) ===")
    gemma4_results = run_gemma4()
    _write("output_gemma4.json", gemma4_results)

    print("\nDone. Review:")
    print(f"  tests/output_liteparse.json")
    print(f"  tests/output_paddleocr.json")
    print(f"  tests/output_gemma4.json")
