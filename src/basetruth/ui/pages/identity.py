"""Identity Verification page — Aadhaar QR, PAN extraction, layered fraud detection, ArcFace face match."""
from __future__ import annotations

import concurrent.futures as _cf
import re as _re
import xml.etree.ElementTree as _ET
from typing import Any, Dict

import streamlit as st

from basetruth.analysis.identity_checks import compare_dob_values, compare_first_last_names
from basetruth.analysis.upload_authenticity import analyse_upload_authenticity, build_format_check
from basetruth.integrations.pdf import ocr_image_bytes_directly
from basetruth.integrations.ollama import (
    extract_pan_details_and_signature_with_ollama,
    extract_aadhaar_details_with_ollama,
    classify_document_type_with_ollama,
    probe_ollama,
    select_ollama_model,
    _extract_json_object,
    OLLAMA_CONNECT_TIMEOUT_SEC,
    OLLAMA_READ_TIMEOUT_SEC,
)
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _page_title,
    get_entity_identity_checks,
    save_identity_check,
    search_entities,
)
from basetruth.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# PAN constants
# ---------------------------------------------------------------------------

_PAN_RE = _re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")
_PAN_ENTITY_TYPES = {
    "P": "Individual",
    "C": "Company",
    "H": "Hindu Undivided Family",
    "F": "Firm",
    "A": "Association of Persons",
    "T": "Trust / AOP",
    "B": "Body of Individuals",
    "L": "Local Authority",
    "J": "Artificial Juridical Person",
    "G": "Government",
}

# ---------------------------------------------------------------------------
# Aadhaar QR decoder
# ---------------------------------------------------------------------------


def _parse_aadhaar_qr(img_bytes: bytes) -> Dict[str, Any]:
    """Decode the QR code on an Aadhaar card and return extracted fields.

    Strategy (in order of robustness):
    1. WeChatQRCode detector (opencv-contrib, deep-learning based — best for
       blurry / low-resolution camera captures)
    2. Standard cv2.QRCodeDetector with a full preprocessing cascade tried at
       original size then at 2×, 3×, 4× upscale
    """
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {}

        def _parse_data(data: str) -> Dict[str, Any]:
            """Turn raw QR string into a structured dict."""
            try:
                root = _ET.fromstring(data)
                a = root.attrib
                return {
                    "qr_found": True,
                    "qr_type": "xml",
                    "name": a.get("name", ""),
                    "dob": a.get("dob", ""),
                    "yob": a.get("yob", ""),
                    "gender": a.get("gender", ""),
                    "uid": a.get("uid", ""),
                    "co": a.get("co", ""),
                    "vtc": a.get("vtc", ""),
                    "dist": a.get("dist", ""),
                    "state": a.get("state", ""),
                    "pc": a.get("pc", ""),
                }
            except _ET.ParseError:
                return {
                    "qr_found": True,
                    "qr_type": "secure",
                    "note": "Secure Aadhaar QR detected (2018+). Demographic data is "
                    "cryptographically signed and cannot be displayed offline.",
                }

        # ── Strategy 1: WeChatQRCode (deep-learning, handles blur/perspective) ──
        try:
            wechat = cv2.wechat_qrcode_WeChatQRCode()
            decoded_list, _ = wechat.detectAndDecode(img)
            if decoded_list:
                for d in decoded_list:
                    if d:
                        return _parse_data(d)
        except Exception:  # noqa: BLE001
            pass  # opencv-contrib not available in this build — fall through

        # ── Strategy 2: Classic QRCodeDetector with preprocessing cascade ────
        detector = cv2.QRCodeDetector()
        h, w = img.shape[:2]

        def _variants(src_bgr: "np.ndarray") -> "list[np.ndarray]":
            gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            eq = clahe.apply(denoised)
            adapt = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 4,
            )
            adapt2 = cv2.adaptiveThreshold(
                eq, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY, 11, 2,
            )
            _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharp = cv2.filter2D(denoised, -1, kernel)
            return [src_bgr, gray, denoised, eq, adapt, adapt2, otsu, sharp]

        data = ""
        for variant in _variants(img):
            data, _, _ = detector.detectAndDecode(variant)
            if data:
                break

        if not data:
            for scale in (2, 3, 4):
                big = cv2.resize(
                    img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC
                )
                # Also try WeChatQRCode on upscaled variants
                try:
                    wechat = cv2.wechat_qrcode_WeChatQRCode()
                    decoded_list, _ = wechat.detectAndDecode(big)
                    if decoded_list:
                        for d in decoded_list:
                            if d:
                                return _parse_data(d)
                except Exception:  # noqa: BLE001
                    pass
                for variant in _variants(big):
                    data, _, _ = detector.detectAndDecode(variant)
                    if data:
                        break
                if data:
                    break

        if not data:
            return {"qr_found": False}

        return _parse_data(data)
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# PAN card OCR helper — shared PaddleOCR path
# ---------------------------------------------------------------------------


def _extract_pan_info_ocr(img_bytes: bytes) -> Dict[str, Any]:
    """Read PAN text from the uploaded image using PaddleOCR only.

    The identity flow must stay on the same OCR engine as the rest of
    BaseTruth, so this helper delegates to the shared PaddleOCR pipeline.
    """
    try:
        text, engine = ocr_image_bytes_directly(img_bytes)
        if engine != "paddleocr" or not text.strip():
            log.warning("_extract_pan_info_ocr: PaddleOCR returned no text — engine=%s", engine)
            return {}

        log.debug(
            "_extract_pan_info_ocr: PaddleOCR raw text (%d chars): %s",
            len(text), text[:300],
        )

        _pan_re_global = _re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")
        _skip_words = {
            "INCOME", "DEPARTMENT", "GOVT", "INDIA", "TAX", "PERMANENT",
            "ACCOUNT", "NUMBER", "CARD", "OF", "SIGNATURE", "FATHER",
        }

        result: Dict[str, Any] = {}
        m = _pan_re_global.search(text.upper())
        if m:
            result["pan_number"] = m.group()

        for line in text.splitlines():
            clean = _re.sub(r"[^A-Z\s]", "", line.strip().upper()).strip()
            words = [w for w in clean.split() if len(w) >= 2]
            if (
                len(words) >= 2
                and not any(kw in clean for kw in _skip_words)
                and not _pan_re_global.search(clean)
            ):
                result["name"] = " ".join(words)
                break

        if not result.get("pan_number"):
            m = _pan_re_global.search(text.upper())
            if m:
                result["pan_number"] = m.group()

        if result.get("name") and not result.get("full_name"):
            result["full_name"] = result["name"]
        if result:
            result["engine"] = "paddleocr"
            log.info(
                "_extract_pan_info_ocr: PaddleOCR extraction complete — "
                "pan=%s name=%s",
                result.get("pan_number", ""),
                result.get("full_name", ""),
            )
        else:
            log.warning("_extract_pan_info_ocr: PaddleOCR found no recognisable PAN fields")
        return result
    except Exception:  # noqa: BLE001
        log.warning("_extract_pan_info_ocr: unexpected error during PaddleOCR extraction", exc_info=True)
        return {}


def _extract_pan_info(img_bytes: bytes) -> Dict[str, Any]:
    """Extract PAN fields using Gemma4 first, then PaddleOCR recovery when needed.

    Uses the combined VLM call that returns both PAN text fields and the
    signature bounding-box in a single request, halving the LLM round-trips
    compared to the previous two-call approach.
    """
    log.info(
        "_extract_pan_info: initiating combined PAN+signature VLM extraction "
        "(provider resolved from settings.json feature_models.pan_extraction)",
    )
    # One VLM call returns both PAN fields AND sig_box — no second call needed
    gemma_data = extract_pan_details_and_signature_with_ollama(img_bytes)
    vlm_provider = gemma_data.get("engine", "vlm")
    vlm_model = gemma_data.get("model", "unknown")
    log.info(
        "_extract_pan_info: VLM call returned — engine=%s model=%s fields_found=%d",
        vlm_provider, vlm_model,
        sum(1 for k in ("pan_number", "full_name", "father_name", "date_of_birth") if gemma_data.get(k)),
    )

    log.info("_extract_pan_info: running PaddleOCR as fallback / cross-check")
    ocr_data = _extract_pan_info_ocr(img_bytes)

    merged: Dict[str, Any] = {}
    field_sources: Dict[str, str] = {}

    def _pick_value(field: str) -> str:
        gemma_value = str(gemma_data.get(field) or "").strip()
        ocr_value = str(ocr_data.get(field) or "").strip()
        if gemma_value:
            field_sources[field] = "gemma4"
            return gemma_value
        if ocr_value:
            field_sources[field] = "ocr"
            return ocr_value
        return ""

    pan_number = _pick_value("pan_number").upper().replace(" ", "")
    if pan_number:
        match = _re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", pan_number)
        if match:
            merged["pan_number"] = match.group(0)

    full_name = _pick_value("full_name") or _pick_value("name")
    if full_name:
        merged["full_name"] = full_name
        merged["name"] = full_name

    father_name = _pick_value("father_name")
    if father_name:
        merged["father_name"] = father_name

    date_of_birth = _pick_value("date_of_birth")
    if date_of_birth:
        merged["date_of_birth"] = date_of_birth

    if merged:
        sources = []
        if any(source == "gemma4" for source in field_sources.values()):
            sources.append("gemma4")
        if any(source == "ocr" for source in field_sources.values()):
            sources.append("ocr")
        merged["field_sources"] = field_sources
        merged["extraction_source"] = " + ".join(sources)
        merged["engine"] = "gemma4_ollama" if sources and sources[0] == "gemma4" else "ocr"

    
    # Forward the signature bounding-box supplied by the combined Gemma4 call
    # so callers can pass it straight to _crop_pan_signature and skip a second
    # Gemma4 round-trip for signature detection.
    if gemma_data.get("sig_box"):
        merged["sig_box"] = gemma_data["sig_box"]

    if gemma_data.get("model"):
        merged["gemma_model"] = gemma_data["model"]
    if gemma_data.get("raw_response"):
        merged["gemma_raw_response"] = gemma_data["raw_response"]

    log.info(
        "_extract_pan_info: merge complete — provider=%s model=%s source=%s "
        "pan=%s name=%s father=%s dob=%s sig_box=%s field_sources=%s",
        gemma_data.get("engine", "vlm"),
        gemma_data.get("model", "unknown"),
        merged.get("extraction_source", "none"),
        merged.get("pan_number", ""),
        merged.get("full_name", ""),
        merged.get("father_name", ""),
        merged.get("date_of_birth", ""),
        "yes" if merged.get("sig_box") else "not found",
        merged.get("field_sources", {}),
    )
    return merged


# ---------------------------------------------------------------------------
# PAN card Image Tampering — Error Level Analysis (ELA)
# ---------------------------------------------------------------------------


def _pan_tampering_check(img_bytes: bytes) -> Dict[str, Any]:
    """Detect editing artefacts via Error Level Analysis (ELA).

    Saves the image at JPEG quality 75, then measures the residual difference.
    A high mean ELA residual suggests regions were edited at a different
    compression quality (copy-paste forgery indicator).
    """
    try:
        import io  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        orig_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf = io.BytesIO()
        orig_img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        recomp = Image.open(buf).convert("RGB")
        ela = abs(
            np.array(orig_img, dtype=np.float32)
            - np.array(recomp, dtype=np.float32)
        )
        ela_mean = float(ela.mean())
        if ela_mean > 15:
            return {
                "passed": False,
                "score": max(0, 100 - int(ela_mean * 2)),
                "reason": "High ELA residuals — localised editing artefacts detected.",
            }
        if ela_mean > 8:
            return {
                "passed": None,
                "score": max(0, 100 - int(ela_mean * 3)),
                "reason": "Moderate ELA residuals — image may have been processed.",
            }
        return {
            "passed": True,
            "score": min(100, 100 - int(ela_mean)),
            "reason": "ELA residuals within normal range for an unedited image.",
        }
    except Exception as exc:  # noqa: BLE001
        return {"passed": None, "score": None, "reason": f"Tampering check unavailable: {exc}"}


# ---------------------------------------------------------------------------
# PAN signature crop
# ---------------------------------------------------------------------------

# System + user prompts for Gemma4 signature bounding-box detection.
# Uses full-image-fraction coordinates so results work regardless of
# image resolution or how much background surrounds the card.
_SIG_SYSTEM_PROMPT = (
    "You are a precise document layout analyser. "
    "Return strict JSON only. No commentary, no markdown."
)
_SIG_USER_PROMPT = (
    "Look at this document image. There is a handwritten signature somewhere "
    "on the document (often a cursive ink stroke below printed text).\n\n"
    "Locate ONLY the handwritten signature ink strokes — do NOT include:\n"
    "  - The PAN number (a 10-character code like ABCDE1234F printed ABOVE the signature)\n"
    "  - The printed word 'Signature' (label below the strokes)\n"
    "  - Any printed text, logos, numbers, or government seals\n"
    "  - The hologram or photo area\n\n"
    "CRITICAL y1 rule: The PAN number is printed text that always appears ABOVE the\n"
    "handwritten signature. Your y1 value MUST be strictly below the bottom edge of the\n"
    "PAN number — never set y1 high enough to include the PAN number line itself.\n\n"
    "IMPORTANT: Signatures often start very close to the left margin. "
    "Make sure x1 captures the very beginning of the LEFTMOST ink stroke. "
    "When in doubt, bias x1 slightly to the left.\n\n"
    "Return a JSON object with the bounding box as fractions of the "
    "FULL image dimensions (0.0 = left/top edge, 1.0 = right/bottom edge):\n\n"
    "{\n"
    '  "x1": <left edge of ink strokes / image width>,\n'
    '  "y1": <top edge of ink strokes / image height — MUST be below the PAN number>,\n'
    '  "x2": <right edge of ink strokes / image width>,\n'
    '  "y2": <bottom edge of ink strokes / image height>,\n'
    '  "confidence": <0.0-1.0>\n'
    "}\n\n"
    "Rules:\n"
    "- All values must be between 0.0 and 1.0.\n"
    "- x1 < x2 and y1 < y2.\n"
    "- Max 3% padding around ink strokes — tight fit, ink strokes only.\n"
    "- If no signature found, set all coordinates and confidence to 0.0.\n"
    "- Output JSON only — no extra text."
)

# Fixed ratios used as fallback when Gemma4 is unavailable.
# Calibrated against a camera-captured PAN card photo WITH background margin
# (the card occupies roughly the central 60% of a typical phone shot).
_SIG_FALLBACK_BOX: Dict[str, float] = {"x1": 0.10, "y1": 0.77, "x2": 0.37, "y2": 0.88}

# Safety margin added to the Gemma4 box before cropping.
# We add extra on the LEFT (x1) because Gemma4 tends to start boxes
# slightly right of the first ink stroke.
_SIG_GEMMA4_EXPAND = 0.03        # 3% on right/top/bottom
_SIG_GEMMA4_EXPAND_LEFT = 0.07   # 7% extra on left to capture first strokes


def _crop_pan_signature(
    img_bytes: bytes,
    *,
    precomputed_box: Dict[str, float] | None = None,
) -> bytes | None:
    """Crop the applicant's handwritten signature from a PAN card image.

    Strategy
    --------
    1. If ``precomputed_box`` is provided (e.g. from the combined Gemma4 call
       in ``_extract_pan_info``), use it directly and skip the Gemma4 network
       call entirely — this is the normal fast path when PAN extraction and
       signature detection were already done in one shot.
    2. Otherwise, ask Gemma4 (via Ollama) for the signature bounding box as
       normalised fractions {x1, y1, x2, y2}.
    3. Expand the chosen box by a 5 % safety margin on every side — Gemma4 can
       underestimate by a few percent, and the cleanup step will retighten.
    4. Coarse-crop the expanded region from the original image.
    5. Clean up with OpenCV:
       a. Grayscale + light denoise.
       b. Adaptive Gaussian threshold (C=12) to isolate ink strokes regardless
          of varying local brightness.
       c. Dilate to merge nearby stroke segments into contour blobs.
       d. Filter contours by minimum area (noise) AND by Y centroid position
          (ignore anything in the lower 35 % of the crop — card edge / beige
          background).
       e. Tighten to the union bounding rect of valid ink contours.
       f. Resize to ≤ 300 px wide so the displayed image matches a realistic
          human signature size — real signatures are small (3–5 cm).
       g. Whiten background, encode as high-quality JPEG.
    6. Fallback: if Gemma4 is unreachable or returns bad coordinates, use
       fixed ratios known to work for standard camera-captured PAN cards.

    Returns ``None`` on any failure so callers treat a missing signature as a
    non-fatal condition without crashing the identity-verification flow.
    """
    import base64
    import json

    import requests
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]

        # ── Step 1: Determine signature bounding box ──────────────────────
        box: Dict[str, float] | None = None

        # Fast path: use the box already returned by the combined Gemma4 call
        # so we avoid a second network round-trip to Ollama.
        if precomputed_box is not None:
            pb = precomputed_box
            width_f  = pb.get("x2", 0) - pb.get("x1", 0)
            height_f = pb.get("y2", 0) - pb.get("y1", 0)
            if (
                0 <= pb.get("x1", 0) < pb.get("x2", 0) <= 1
                and 0 <= pb.get("y1", 0) < pb.get("y2", 0) <= 1
                and width_f >= 0.03
                and height_f >= 0.02
            ):
                # Accept the box and apply asymmetric expand:
                # Extra left margin because Gemma4 often starts too far right,
                # clipping the beginning of the first stroke.
                box = {
                    "x1": max(0.0, pb["x1"] - _SIG_GEMMA4_EXPAND - _SIG_GEMMA4_EXPAND_LEFT),
                    "y1": max(0.0, pb["y1"] - _SIG_GEMMA4_EXPAND),
                    "x2": min(1.0, pb["x2"] + _SIG_GEMMA4_EXPAND),
                    "y2": min(1.0, pb["y2"] + _SIG_GEMMA4_EXPAND),
                }

        # Slow path: ask Gemma4 when no precomputed box is available
        if box is None:
            base_url, models, _ = probe_ollama()
            if base_url:
                model = select_ollama_model(models)
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SIG_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _SIG_USER_PROMPT,
                            "images": [base64.b64encode(img_bytes).decode("ascii")],
                        },
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                }
                log.info(
                    "_crop_pan_signature: sending signature-box VLM request | model=%s base_url=%s image_bytes=%d",
                    model,
                    base_url,
                    len(img_bytes),
                )
                log.info("_crop_pan_signature: exact system prompt follows:\n%s", _SIG_SYSTEM_PROMPT)
                log.info("_crop_pan_signature: exact user prompt follows:\n%s", _SIG_USER_PROMPT)
                try:
                    resp = requests.post(  # nosemgrep: basetruth-ssrf
                        f"{base_url}/api/chat",
                        json=payload,
                        timeout=(OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_READ_TIMEOUT_SEC),
                    )
                    resp.raise_for_status()
                    raw  = str(resp.json().get("message", {}).get("content", "")).strip()
                    log.info(
                        "_crop_pan_signature: exact raw VLM response follows:\n%s",
                        raw,
                    )
                    text = _extract_json_object(raw)
                    data = json.loads(text) if text else {}
                    x1, y1, x2, y2 = (
                        float(data.get("x1", 0)),
                        float(data.get("y1", 0)),
                        float(data.get("x2", 0)),
                        float(data.get("y2", 0)),
                    )
                    # Validate: in range, non-degenerate, not unreasonably large
                    width_f  = x2 - x1
                    height_f = y2 - y1
                    if (
                        0 <= x1 < x2 <= 1
                        and 0 <= y1 < y2 <= 1
                        and width_f >= 0.03
                        and height_f >= 0.02
                        and width_f <= 0.7
                        and height_f <= 0.5
                    ):
                        # Asymmetric expand: extra left to avoid clipping first stroke
                        box = {
                            "x1": max(0.0, x1 - _SIG_GEMMA4_EXPAND - _SIG_GEMMA4_EXPAND_LEFT),
                            "y1": max(0.0, y1 - _SIG_GEMMA4_EXPAND),
                            "x2": min(1.0, x2 + _SIG_GEMMA4_EXPAND),
                            "y2": min(1.0, y2 + _SIG_GEMMA4_EXPAND),
                        }
                except Exception:  # noqa: BLE001
                    box = None  # fall back silently

        if box is None:
            box = _SIG_FALLBACK_BOX

        # ── Step 2: Coarse crop ────────────────────────────────────────────
        px0 = int(w * box["x1"]); py0 = int(h * box["y1"])
        px1 = int(w * box["x2"]); py1 = int(h * box["y2"])
        if px1 <= px0 or py1 <= py0:
            return None
        coarse = img[py0:py1, px0:px1]

        # -- Step 3: Adaptive threshold + contour tightening ----------------
        gray     = cv2.cvtColor(coarse, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

        # Adaptive threshold isolates dark ink regardless of local brightness.
        # blockSize=31 gives finer local contrast for camera-captured cards.
        binary = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=10,
        )
        # Morphological close merges nearby stroke segments without inflating
        # blob boundaries as aggressively as the previous dilate-only approach.
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        coarse_area = coarse.shape[0] * coarse.shape[1]
        coarse_h    = coarse.shape[0]
        min_area    = max(20, coarse_area * 0.0008)  # lower floor catches thin strokes
        y_cutoff    = coarse_h * 0.85  # reject bottom 15% (card edge / 'Signature' label)

        # Safety top-cutoff: when Gemma4 sets y1 too high and the bounding box
        # accidentally captures the PAN number (printed text above the signature),
        # the top rows of the crop contain those uniform text glyphs. Filtering
        # contours whose centroid is in the top 20% of a suspiciously tall crop
        # removes the PAN number blobs without touching the actual signature strokes
        # which sit in the lower portion. We only apply this cutoff when the crop is
        # taller than 15% of the full image — a realistic handwritten signature
        # never needs that much vertical space on a PAN card.
        crop_to_img_ratio = (py1 - py0) / max(h, 1)
        y_top_cutoff = coarse_h * 0.20 if crop_to_img_ratio > 0.15 else 0

        valid: list = []
        for c in contours:
            if cv2.contourArea(c) < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cy = M["m01"] / M["m00"]
            if cy > y_cutoff:
                continue  # card edge / background noise
            if y_top_cutoff and cy < y_top_cutoff:
                continue  # skip top portion — likely PAN number text, not signature ink
            valid.append(c)

        if valid:
            all_pts        = np.concatenate(valid, axis=0)
            bx, by, bw, bh = cv2.boundingRect(all_pts)
            pad_x = 10; pad_y = 10
            tx  = max(0, bx - pad_x)
            ty  = max(0, by - pad_y)
            tx2 = min(coarse.shape[1], bx + bw + pad_x)
            ty2 = min(coarse.shape[0], by + bh + pad_y)
            tight = coarse[ty:ty2, tx:tx2] if (tx2 - tx) >= 20 and (ty2 - ty) >= 6 else coarse
        else:
            tight = coarse

        # ── Step 4: Produce clean white-background grayscale JPEG ─────────
        tight_gray     = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
        denoised_tight = cv2.fastNlMeansDenoising(tight_gray, h=8)
        _, mask        = cv2.threshold(denoised_tight, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result         = np.where(mask == 255, np.uint8(255), denoised_tight)

        # Resize to a realistic signature display size (<= 400 px wide).
        # Real handwritten signatures are 3-5 cm wide.
        _MAX_SIG_W = 400
        sig_h, sig_w = result.shape[:2]
        if sig_w > _MAX_SIG_W:
            scale    = _MAX_SIG_W / sig_w
            new_w    = _MAX_SIG_W
            new_h    = max(1, int(sig_h * scale))
            result   = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return bytes(buf) if ok else None

    except Exception:  # noqa: BLE001
        # Signature crop is best-effort — never let it crash the main flow
        return None


# ---------------------------------------------------------------------------
# PAN format validator
# ---------------------------------------------------------------------------


def _validate_pan(pan: str) -> Dict[str, Any]:
    pan = pan.strip().upper()
    if not pan:
        return {"valid": False, "error": "PAN is empty"}
    if not _PAN_RE.match(pan):
        return {
            "valid": False,
            "error": f"Invalid format (expected ABCDE1234F, got '{pan}')",
        }
    return {
        "valid": True,
        "pan": pan,
        "entity_type": _PAN_ENTITY_TYPES.get(pan[3], f"Unknown ({pan[3]})"),
        "surname_initial": pan[4],
    }


# ---------------------------------------------------------------------------
# Layered PAN Fraud Analysis payload
# ---------------------------------------------------------------------------


def _build_pan_layer_analysis(
    pan_data: Dict[str, Any],
    pan_validation: Dict[str, Any],
    pan_img_bytes: bytes | None,
) -> Dict[str, Any]:
    """Build the PAN analysis payload for explainability storage."""
    layers = []

    # Layer 1 — Format
    if pan_validation.get("valid"):
        layers.append(
            {
                "icon": "✅",
                "title": "Layer 1 — PAN Format Check",
                "status": "PASS",
                "status_color": "#16a34a",
                "detail": (
                    f"PAN `{pan_validation['pan']}` is syntactically valid.  \n"
                    f"Entity type: **{pan_validation['entity_type']}**  \n"
                    f"Surname initial: **{pan_validation['surname_initial']}**"
                ),
            }
        )
    elif pan_data.get("pan_number"):
        layers.append(
            {
                "icon": "❌",
                "title": "Layer 1 — PAN Format Check",
                "status": "FAIL",
                "status_color": "#dc2626",
                "detail": (
                    f"Read `{pan_data['pan_number']}` — "
                    f"{pan_validation.get('error', 'invalid format')}."
                ),
            }
        )
    else:
        layers.append(
            {
                "icon": "⚪",
                "title": "Layer 1 — PAN Format Check",
                "status": "N/A",
                "status_color": "#94a3b8",
                "detail": "PAN number could not be read from the image.",
            }
        )

    # Layer 2 — Government API (skipped)
    layers.append(
        {
            "icon": "ℹ️",
            "title": "Layer 2 — Government API (NSDL / Karza / Signzy)",
            "status": "SKIPPED",
            "status_color": "#2563eb",
            "detail": (
                "Real-time PAN verification via NSDL or Karza requires a paid API subscription.  \n"
                "Not configured — set `KARZA_API_KEY` or `NSDL_API_KEY` in your `.env` to enable."
            ),
        }
    )

    # Layer 4 — Tampering (ELA)
    tampering_score_val = None
    if pan_img_bytes:
        t_result = _pan_tampering_check(pan_img_bytes)
        tampering_score_val = t_result.get("score")
        if t_result.get("passed") is True:
            layers.append(
                {
                    "icon": "✅",
                    "title": "Layer 4 — Image Tampering (ELA)",
                    "status": f"CLEAN  ({tampering_score_val}/100)",
                    "status_color": "#16a34a",
                    "detail": t_result["reason"],
                }
            )
        elif t_result.get("passed") is False:
            layers.append(
                {
                    "icon": "🚨",
                    "title": "Layer 4 — Image Tampering (ELA)",
                    "status": f"SUSPECT  ({tampering_score_val}/100)",
                    "status_color": "#dc2626",
                    "detail": t_result["reason"],
                }
            )
        else:
            layers.append(
                {
                    "icon": "⚠️",
                    "title": "Layer 4 — Image Tampering (ELA)",
                    "status": "INCONCLUSIVE",
                    "status_color": "#d97706",
                    "detail": t_result["reason"],
                }
            )
    else:
        layers.append(
            {
                "icon": "⚪",
                "title": "Layer 4 — Image Tampering (ELA)",
                "status": "N/A",
                "status_color": "#94a3b8",
                "detail": "Upload a PAN card image to run tampering detection.",
            }
        )

    scores = [s for s in [tampering_score_val] if s is not None]
    if pan_validation.get("valid"):
        scores.append(100)
    if pan_data.get("pan_number"):
        scores.append(80)
    if scores:
        overall = sum(scores) // len(scores)
        risk_label = (
            "LOW RISK" if overall >= 75 else "MEDIUM RISK" if overall >= 50 else "HIGH RISK"
        )
        risk_color = "#16a34a" if overall >= 75 else "#d97706" if overall >= 50 else "#dc2626"
    else:
        overall = None
        risk_label = "UNKNOWN"
        risk_color = "#94a3b8"

    return {
        "layers": layers,
        "overall_score": overall,
        "risk_label": risk_label,
        "risk_color": risk_color,
    }


# ---------------------------------------------------------------------------
# Document-type pre-validation
# ---------------------------------------------------------------------------

# These sets say which Gemma4 document_type values are acceptable for each slot.
# If Gemma4 finds something NOT in the accepted set (and is confident enough),
# we reject the upload and explain what was wrong.
_SLOT_ACCEPTED_TYPES: Dict[str, set] = {
    "aadhaar": {"aadhaar_card"},
    "pan":     {"pan_card"},
    "selfie":  {"photograph_selfie"},
}

# Human-readable slot names used in error messages shown to the user.
_SLOT_DISPLAY_NAMES: Dict[str, str] = {
    "aadhaar": "Aadhaar card",
    "pan":     "PAN card",
    "selfie":  "selfie / portrait photo",
}

# We only block an upload when Gemma4 is THIS confident that it is the wrong
# type.  Below this threshold we let it through — the classifier might be wrong
# and we don't want to frustrate users with false rejections.
_DOC_CHECK_CONFIDENCE_THRESHOLD = 0.65


def _check_document_type(img_bytes: bytes, slot: str) -> Dict[str, Any]:
    """Run a fast Gemma4 check to confirm the uploaded image belongs to the right slot.

    This runs BEFORE the heavy QR decode / OCR so a wrong document is caught
    early, without wasting time on extraction.

    How it works
    ------------
    1. Call Gemma4 with a short, focused prompt: "what type of document is this?"
    2. If Gemma4 says it is the EXPECTED type (e.g. aadhaar_card for the Aadhaar
       slot) → return ok=True.
    3. If Gemma4 says it is a DIFFERENT known type with high confidence
       (e.g. pan_card uploaded to the Aadhaar slot) → return ok=False with an
       explanation message so the user knows exactly what to fix.
    4. If Ollama is not running, or confidence is low, or the type is "other"
       → return ok=True (skipped) so we never block a legitimate upload just
       because the classifier was uncertain.

    Parameters
    ----------
    img_bytes : raw image bytes to check
    slot      : one of "aadhaar", "pan", "selfie"

    Returns a dict:
        ok         — True = proceed with extraction; False = wrong document
        skipped    — True = Ollama unavailable, check was not run
        detected   — what Gemma4 thinks the document is
        confidence — how sure Gemma4 was (0.0–1.0)
        reason     — human-readable explanation for the user
    """
    log.info(
        f"Identity Verification: Checking if uploaded image matches the expected slot '{slot}' ({_SLOT_DISPLAY_NAMES.get(slot, slot)}).",
        extra={"slot": slot}
    )
    # Map the slot name to the matching feature key so the classify call uses
    # the same provider as the full extraction that will follow.  This ensures
    # that when active_provider=feature_models, the pre-flight check doesn't
    # fall back to the global Ollama default (which may be unavailable or slow).
    _SLOT_FEATURE_MAP: Dict[str, str] = {
        "aadhaar": "aadhaar_extraction",
        "pan":     "pan_extraction",
    }
    classify_feature = _SLOT_FEATURE_MAP.get(slot)  # None for "selfie" slot
    result = classify_document_type_with_ollama(img_bytes, feature=classify_feature)

    # Ollama is not reachable — skip silently so we never block the user
    if not result or not result.get("available"):
        return {
            "ok":         True,
            "skipped":    True,
            "detected":   "unknown",
            "confidence": 0.0,
            "reason":     "Ollama unavailable — document type check skipped.",
        }

    doc_type   = result.get("document_type", "other")
    confidence = result.get("confidence", 0.0)
    reason     = result.get("reason", "")

    accepted   = _SLOT_ACCEPTED_TYPES.get(slot, set())
    slot_label = _SLOT_DISPLAY_NAMES.get(slot, slot)

    # "other" means Gemma4 could not confidently identify the document type
    # → do not block; the user might be uploading something valid but unusual
    if doc_type == "other":
        return {
            "ok":         True,
            "skipped":    False,
            "detected":   doc_type,
            "confidence": confidence,
            "reason":     reason or "Document type could not be identified — proceeding anyway.",
        }

    # Document type matches what is expected for this slot → all good
    if doc_type in accepted:
        return {
            "ok":         True,
            "skipped":    False,
            "detected":   doc_type,
            "confidence": confidence,
            "reason":     reason,
        }

    # Gemma4 detected a DIFFERENT known document type with high confidence → block
    if confidence >= _DOC_CHECK_CONFIDENCE_THRESHOLD:
        # Map the detected type to a friendly display name for the error message
        detected_label = {
            "aadhaar_card":      "an Aadhaar card",
            "pan_card":          "a PAN card",
            "photograph_selfie": "a selfie / portrait photo",
        }.get(doc_type, f"a '{doc_type}'")
        return {
            "ok":         False,
            "skipped":    False,
            "detected":   doc_type,
            "confidence": confidence,
            "reason": (
                f"This image looks like {detected_label}, but this slot expects "
                f"a {slot_label}. Please upload the correct document."
            ),
        }

    # Confidence is below the threshold — we are not sure enough to block the upload
    return {
        "ok":         True,
        "skipped":    False,
        "detected":   doc_type,
        "confidence": confidence,
        "reason":     reason,
    }


# ---------------------------------------------------------------------------
# Parallel upload prefetch
# ---------------------------------------------------------------------------


def _prefetch_upload_results(up_aadhaar, up_pan, up_selfie) -> None:
    """Run all pending document-analysis work concurrently before rendering results.

    When the user uploads all three documents at the same time we would normally
    process them one after another (Aadhaar → PAN → Selfie), forcing the user to
    wait for the sum of all three analysis times.

    This function detects which computations are NOT yet cached in session state,
    fires them all at the same time with a thread pool, and writes every result
    back to session state in the main thread once all threads are done.  A single
    "Analysing documents…" spinner replaces the three individual spinners — the
    total wait becomes the time of the SLOWEST task, not the sum.

    Thread safety
    -------------
    - All session-state READS happen here in the main thread, BEFORE any thread
      is launched.  We pass captured values as closure defaults so threads never
      touch session_state directly.
    - All session-state WRITES happen here in the main thread, AFTER all threads
      have finished.  This avoids any concurrent mutation of session state.
    - Streamlit widgets are never called from inside threads.
    """
    # Each entry in `tasks` is a callable that returns a dict of {cache_key: result}.
    # active_slots tracks which document types have actual pending work so we can
    # choose the right spinner message (specific for one doc, generic for many).
    tasks: list = []
    active_slots: list = []  # e.g. ["aadhaar"] or ["pan", "selfie"]

    # ── Aadhaar pipeline: doc check → QR decode → Gemma4 fallback ───────
    if up_aadhaar:
        _ab    = up_aadhaar.getvalue()                              # capture bytes in main thread
        _adc_k = f"_idv_doccheck_aadhaar_{up_aadhaar.size}"
        _aqr_k = f"_idv_qr_{up_aadhaar.size}"
        _adc_miss = _adc_k not in st.session_state                  # True = needs computing
        _aqr_miss = _aqr_k not in st.session_state
        # Snapshot existing values so threads never touch session_state
        _existing_adc = st.session_state.get(_adc_k)

        if _adc_miss or _aqr_miss:
            def _do_aadhaar(
                ab=_ab, adc_k=_adc_k, aqr_k=_aqr_k,
                adc_miss=_adc_miss, aqr_miss=_aqr_miss,
                existing_adc=_existing_adc,
            ):
                """Thread worker for Aadhaar: doc check + QR decode."""
                out: Dict[str, Any] = {}
                # Doc check: use the captured value if already computed, else run it
                dc = _check_document_type(ab, "aadhaar") if adc_miss else existing_adc
                if adc_miss:
                    out[adc_k] = dc
                # QR decode only runs if the doc check passed
                if dc and dc.get("ok") and aqr_miss:
                    pqr = _parse_aadhaar_qr(ab)
                    # Fall back to Gemma4 OCR if the standard QR scanner found nothing
                    if pqr.get("qr_found") is False or not pqr:
                        fb = extract_aadhaar_details_with_ollama(ab)
                        if fb:
                            pqr = fb
                    out[aqr_k] = pqr
                return out
            active_slots.append("aadhaar")
            tasks.append(_do_aadhaar)

    # ── PAN pipeline: doc check → OCR/Gemma4 extraction → signature crop ─
    if up_pan:
        _pb     = up_pan.getvalue()
        _pdc_k  = f"_idv_doccheck_pan_{up_pan.size}"
        _pocr_k = f"_idv_pan_{up_pan.size}"
        _psig_k = f"_idv_pan_sig_v3_{up_pan.size}"
        _pdc_miss  = _pdc_k  not in st.session_state
        _pocr_miss = _pocr_k not in st.session_state
        _psig_miss = _psig_k not in st.session_state
        # Snapshot existing values so the thread never touches session_state
        _existing_pdc  = st.session_state.get(_pdc_k)
        _existing_pocr = st.session_state.get(_pocr_k)  # may be {} when already cached

        if _pdc_miss or _pocr_miss or _psig_miss:
            def _do_pan(
                pb=_pb, pdc_k=_pdc_k, pocr_k=_pocr_k, psig_k=_psig_k,
                pdc_miss=_pdc_miss, pocr_miss=_pocr_miss, psig_miss=_psig_miss,
                existing_pdc=_existing_pdc, existing_pocr=_existing_pocr,
            ):
                """Thread worker for PAN: doc check + combined OCR extraction + sig crop."""
                out: Dict[str, Any] = {}
                dc = _check_document_type(pb, "pan") if pdc_miss else existing_pdc
                if pdc_miss:
                    out[pdc_k] = dc
                if dc and dc.get("ok"):
                    # Combined Gemma4 call returns PAN fields AND the signature
                    # bounding-box in a single response to avoid two round-trips
                    pan_data = _extract_pan_info(pb) if pocr_miss else (existing_pocr or {})
                    if pocr_miss:
                        out[pocr_k] = pan_data
                    # Crop the signature strip using the bounding box Gemma4 returned
                    if psig_miss:
                        sig = _crop_pan_signature(pb, precomputed_box=pan_data.get("sig_box"))
                        out[psig_k] = sig
                return out
            active_slots.append("pan")
            tasks.append(_do_pan)

    # ── Selfie pipeline: doc check only (no extraction needed) ───────────
    if up_selfie:
        _sb    = up_selfie.getvalue()
        _sdc_k = f"_idv_doccheck_selfie_{up_selfie.size}"
        if _sdc_k not in st.session_state:
            def _do_selfie(sb=_sb, sdc_k=_sdc_k):
                """Thread worker for Selfie: just confirm it is a portrait photo."""
                dc = _check_document_type(sb, "selfie")
                return {sdc_k: dc}
            tasks.append(_do_selfie)
            active_slots.append("selfie")

    if not tasks:
        return  # every result is already cached — nothing to compute

    # Choose the right spinner message:
    # - single slot pending: show the document-specific message the user already knows
    # - two or more slots pending: show the combined parallel message
    if len(active_slots) == 1:
        _spin_msg = {
            "aadhaar": "Scanning Aadhaar QR code...",
            "pan":     "Reading PAN card...",
            "selfie":  "Verifying selfie...",
        }.get(active_slots[0], "Verifying document type...")
    else:
        _spin_msg = "Analysing documents... (running in parallel)"

    # Fire all pending tasks at the same time and wait for the last one to finish.
    # One spinner covers the whole batch; total wait = slowest pipeline, not the sum.
    with st.spinner(_spin_msg):
        with _cf.ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            futures = [pool.submit(fn) for fn in tasks]
            for fut in _cf.as_completed(futures):
                try:
                    # Write each result back to session state in the main thread
                    for cache_key, result in fut.result().items():
                        st.session_state[cache_key] = result
                except Exception:  # noqa: BLE001
                    # Individual pipeline failures are surfaced in the display
                    # columns below — don't crash the whole page here
                    pass


# ---------------------------------------------------------------------------
# Identity Verification page
# ---------------------------------------------------------------------------


def _page_identity_verification() -> None:
    st.markdown(_page_title("🧑‍💻", "Identity Verification"), unsafe_allow_html=True)
    st.caption(
        "Upload Aadhaar + PAN card + Selfie to verify identity offline using "
        "QR parsing, PAN validation, name cross-check, and ArcFace face matching."
    )

    with st.expander("ℹ️ How it works", expanded=False):
        st.markdown(
            "**Step 1** — Upload your Aadhaar card, PAN card, and a selfie photo.  \n"
            "**Step 2** — The system reads the Aadhaar QR code to extract your name, DOB, and "
            "other details, then validates the PAN format and checks for tampering.  \n"
            "**Step 3** — ArcFace compares the face on your Aadhaar card with your selfie.  \n"
            "**Step 4** — Results are saved to the database and linked to your profile.  \n\n"
            "Everything runs 100% locally — no data leaves this server."
        )

    IMG_TYPES = ["jpg", "jpeg", "png", "webp"]

    # ── Step 1: Document Input ────────────────────────────────────────────
    # Tiny wrapper so camera-captured bytes behave like an UploadedFile
    class _DocumentCapture:
        """Wraps raw bytes from st.camera_input to match UploadedFile API."""
        def __init__(self, data: bytes, name: str) -> None:
            self._data = data
            self.size = len(data)
            self.name = name

        def getvalue(self) -> bytes:
            return self._data

    st.subheader("Provide Documents")

    tab_upload, tab_camera = st.tabs(["📁 Upload Documents", "📷 Capture with Camera"])

    # Effective document sources — populated by whichever tab the user interacts with
    aadhaar_file: _DocumentCapture | None = None  # type: ignore[assignment]
    pan_file: _DocumentCapture | None = None       # type: ignore[assignment]
    selfie_bytes: bytes | None = None
    selfie_name = "selfie.jpg"

    # ---- Upload tab -------------------------------------------------------
    with tab_upload:
        col_a, col_p, col_s = st.columns(3)

        # ── Pass 1: render all three file uploaders side by side ─────────
        # We only render the widgets here and collect which files the user has
        # provided.  No analysis happens yet — that comes next in the parallel
        # prefetch block so all three pipelines can run at the same time.
        with col_a:
            st.markdown("**📄 Aadhaar Card**")
            _up_aadhaar = st.file_uploader(
                "Drag and drop file here",
                type=IMG_TYPES,
                key="idv_aadhaar",
                label_visibility="visible",
            )
            st.caption("Limit 200MB per file • JPG, JPEG, PNG, WEBP")

        with col_p:
            st.markdown("**💳 PAN Card**")
            _up_pan = st.file_uploader(
                "Drag and drop file here",
                type=IMG_TYPES,
                key="idv_pan",
                label_visibility="visible",
            )
            st.caption("Limit 200MB per file • JPG, JPEG, PNG, WEBP")

        with col_s:
            st.markdown("**🤳 Selfie Photo**")
            _up_selfie = st.file_uploader(
                "Drag and drop file here",
                type=IMG_TYPES,
                key="idv_selfie",
                label_visibility="visible",
            )
            st.caption("Limit 200MB per file • JPG, JPEG, PNG, WEBP")

        # ── Pass 2: run all pending analysis jobs in parallel ─────────────
        # _prefetch_upload_results checks which computations are not yet
        # cached in session state and fires them all at the same time via a
        # thread pool.  On subsequent page re-renders everything is already
        # cached, so this call returns immediately with no spinner shown.
        _prefetch_upload_results(_up_aadhaar, _up_pan, _up_selfie)

        # ── Pass 3: display results from session state ────────────────────
        # All heavy Gemma4 / QR / OCR work is now done (or was already cached
        # from a previous render).  These blocks only READ from session state
        # — no blocking calls happen during rendering.

        with col_a:
            if _up_aadhaar:
                # Read the doc-type check result cached by the prefetch above
                _a_doccheck = st.session_state.get(
                    f"_idv_doccheck_aadhaar_{_up_aadhaar.size}", {}
                )
                # Treat an explicit False as "wrong type"; missing key means
                # the prefetch had an error, so we allow the file through rather
                # than blocking a legitimate upload
                _a_check_failed = _a_doccheck.get("ok") is False
                aadhaar_file = _up_aadhaar if not _a_check_failed else None  # type: ignore[assignment]
                # Always show a preview so the user can confirm what they uploaded
                st.image(_up_aadhaar.getvalue(), caption="Aadhaar card", use_container_width=True)
                # Read QR decode result cached by the prefetch (empty if blocked)
                _aq = st.session_state.get(f"_idv_qr_{_up_aadhaar.size}", {})
                if _a_check_failed:
                    # Wrong document type — show the classifier's reason message
                    st.error(f"\u26a0\ufe0f {_a_doccheck.get('reason', 'Wrong document type uploaded. Please upload an Aadhaar card here.')}")
                elif _aq.get("qr_found") is False:
                    st.warning("No QR code detected. Ensure the QR code is visible.")
                elif _aq.get("qr_type") in ("xml", "gemma4"):
                    if _aq.get("qr_type") == "gemma4":
                        st.success("Aadhaar details extracted via Gamma4 fallback")
                    else:
                        st.success("QR decoded successfully")
                    st.markdown(
                        f"**UID:** {_aq.get('uid', '—')}  \n"
                        f"**Full Name:** {_aq.get('name', '—')}  \n"
                        f"**DOB/YOB:** {_aq.get('dob') or _aq.get('yob', '—')}  \n"
                        f"**Gender (M/F/T):** {_aq.get('gender', '—')}  \n"
                        f"**Care of (typically Father or Husband's name):** {_aq.get('co', '—')}  \n"
                        f"**VTC (Village/Town/City):** {_aq.get('vtc', '—')}  \n"
                        f"**District:** {_aq.get('dist', '—')}  \n"
                        f"**State:** {_aq.get('state', '—')}  \n"
                        f"**PIN Code:** {_aq.get('pc', '—')}"
                    )
                    st.caption("Aadhaar looks good")
                elif _aq.get("qr_type") == "secure":
                    st.info(_aq.get("note", "Secure QR detected."))
                elif not _aq:
                    st.warning("QR scan error — check that OpenCV is available.")

        with col_p:
            if _up_pan:
                # Read the doc-type check result cached by the prefetch
                _p_doccheck = st.session_state.get(
                    f"_idv_doccheck_pan_{_up_pan.size}", {}
                )
                _p_check_failed = _p_doccheck.get("ok") is False
                pan_file = _up_pan if not _p_check_failed else None  # type: ignore[assignment]
                # Always show the uploaded image first
                st.image(_up_pan.getvalue(), caption="PAN card", use_container_width=True)
                if _p_check_failed:
                    # Wrong document type — stop here, no extraction to display
                    st.error(f"\u26a0\ufe0f {_p_doccheck.get('reason', 'Wrong document type uploaded. Please upload a PAN card here.')}")
                else:
                    # Read the OCR + Gemma4 extraction result cached by the prefetch
                    _pd = st.session_state.get(f"_idv_pan_{_up_pan.size}", {})
                    _extracted_pan = _pd.get("pan_number", "")
                    if _extracted_pan or _pd.get("full_name") or _pd.get("date_of_birth"):
                        st.success("PAN decoded successfully")
                    if _extracted_pan:
                        _pv = _validate_pan(_extracted_pan)
                        if _pv["valid"]:
                            st.markdown(
                                f"**PAN:** {_extracted_pan}  \n"
                                f"**Entity type:** {_pv['entity_type']}"
                            )
                        else:
                            st.warning(f"PAN read: `{_extracted_pan}` — {_pv.get('error', '')}")
                    else:
                        st.info("PAN number not detected. Enter it manually below if Gemma4 and OCR could not read it.")
                    if _pd.get("full_name") or _pd.get("name"):
                        st.markdown(f"**Full Name:** {_pd.get('full_name') or _pd.get('name')}")
                    if _pd.get("father_name"):
                        st.markdown(
                            f"**Care of (typically Father or Husband's name): S/O:** "
                            f"{_pd['father_name']}"
                        )
                    if _pd.get("date_of_birth"):
                        st.markdown(f"**DOB/YOB::** {_pd['date_of_birth']}")
                    # Read the signature crop bytes (None if crop failed or blocked)
                    _sig_bytes = st.session_state.get(f"_idv_pan_sig_v3_{_up_pan.size}")
                    if _sig_bytes:
                        st.markdown("**✍️ Signature (extracted):**")
                        # use_container_width=False preserves realistic signature size
                        st.image(_sig_bytes, caption="PAN card signature", use_container_width=False)
                    else:
                        st.caption("Signature could not be extracted from this image.")

        with col_s:
            if _up_selfie:
                # Read the doc-type check result cached by the prefetch
                _s_doccheck = st.session_state.get(
                    f"_idv_doccheck_selfie_{_up_selfie.size}", {}
                )
                _s_check_failed = _s_doccheck.get("ok") is False
                # Always show what the user uploaded so they can confirm visually
                st.image(_up_selfie.getvalue(), caption="Selfie", use_container_width=True)
                if _s_check_failed:
                    # Looks like an ID document, not a portrait — tell the user
                    st.error(f"\u26a0\ufe0f {_s_doccheck.get('reason', 'Wrong document type uploaded. Please upload a selfie or portrait photo here.')}")
                else:
                    # Confirmed portrait photo — accept for face-matching
                    selfie_bytes = _up_selfie.getvalue()
                    selfie_name = _up_selfie.name

    # ---- Camera tab -------------------------------------------------------
    with tab_camera:
        st.info(
            "Click **Open Camera** for each document, then use the shutter button "
            "inside the camera view to capture the photo.",
            icon="📷",
        )
        st.markdown(
            """
            <div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;
                        padding:0.6rem 0.9rem;font-size:0.85rem;margin-bottom:0.75rem;">
            📸 <strong>Tips for best results:</strong>
            Place the document on a flat surface in good light.
            Hold the camera directly above — avoid tilting.
            For Aadhaar, ensure the entire QR code is visible and in focus.
            For PAN card, make sure all text is sharp and not in shadow.
            </div>
            """,
            unsafe_allow_html=True,
        )
        cam_col_a, cam_col_p, cam_col_s = st.columns(3)

        # ---- Aadhaar camera
        with cam_col_a:
            st.markdown("**📄 Aadhaar Card**")
            if not st.session_state.get("idv_cam_a_open"):
                if st.button("📷 Open Camera", key="btn_cam_a_open", use_container_width=True):
                    st.session_state["idv_cam_a_open"] = True
                    st.rerun()
            else:
                _cam_a = st.camera_input(
                    "Take Photo",
                    key="cam_aadhaar_input",
                    label_visibility="collapsed",
                )
                if _cam_a:
                    # New photo just taken — clear old doc check so it reruns for the new image
                    _cam_a_new = _cam_a.getvalue()
                    if st.session_state.get("idv_cam_a_bytes") != _cam_a_new:
                        st.session_state.pop("idv_cam_a_doccheck", None)
                    st.session_state["idv_cam_a_bytes"] = _cam_a_new
                if st.session_state.get("idv_cam_a_bytes"):
                    # Run document type check once per captured image (cached in session state)
                    if "idv_cam_a_doccheck" not in st.session_state:
                        with st.spinner("Verifying document type..."):
                            st.session_state["idv_cam_a_doccheck"] = _check_document_type(
                                st.session_state["idv_cam_a_bytes"], "aadhaar"
                            )
                    _cam_a_check = st.session_state.get("idv_cam_a_doccheck", {})
                    # Show error if we are confident this is the wrong document type
                    if not _cam_a_check.get("ok") and not _cam_a_check.get("skipped"):
                        st.error(f"⚠️ {_cam_a_check.get('reason', 'Wrong document type.')}")
                    else:
                        st.success("✅ Photo captured")
                    st.image(
                        st.session_state["idv_cam_a_bytes"],
                        caption="Aadhaar — captured",
                        use_container_width=True,
                    )
                if st.button("✖ Close Camera", key="btn_cam_a_close", use_container_width=True):
                    st.session_state["idv_cam_a_open"] = False
                    st.rerun()
            # Show thumbnail if captured but camera closed
            if (
                not st.session_state.get("idv_cam_a_open")
                and st.session_state.get("idv_cam_a_bytes")
            ):
                _cam_a_check = st.session_state.get("idv_cam_a_doccheck", {})
                # Show error banner if the captured image failed the document type check
                if not _cam_a_check.get("ok") and not _cam_a_check.get("skipped"):
                    st.error(f"⚠️ {_cam_a_check.get('reason', 'Wrong document type.')}")
                st.image(
                    st.session_state["idv_cam_a_bytes"],
                    caption="Aadhaar — captured",
                    use_container_width=True,
                )

        # ---- PAN camera
        with cam_col_p:
            st.markdown("**💳 PAN Card**")
            if not st.session_state.get("idv_cam_p_open"):
                if st.button("📷 Open Camera", key="btn_cam_p_open", use_container_width=True):
                    st.session_state["idv_cam_p_open"] = True
                    st.rerun()
            else:
                _cam_p = st.camera_input(
                    "Take Photo",
                    key="cam_pan_input",
                    label_visibility="collapsed",
                )
                if _cam_p:
                    # New photo just taken — clear old doc check so it reruns for the new image
                    _cam_p_new = _cam_p.getvalue()
                    if st.session_state.get("idv_cam_p_bytes") != _cam_p_new:
                        st.session_state.pop("idv_cam_p_doccheck", None)
                    st.session_state["idv_cam_p_bytes"] = _cam_p_new
                if st.session_state.get("idv_cam_p_bytes"):
                    # Run document type check once per captured image (cached in session state)
                    if "idv_cam_p_doccheck" not in st.session_state:
                        with st.spinner("Verifying document type..."):
                            st.session_state["idv_cam_p_doccheck"] = _check_document_type(
                                st.session_state["idv_cam_p_bytes"], "pan"
                            )
                    _cam_p_check = st.session_state.get("idv_cam_p_doccheck", {})
                    # Show error if we are confident this is the wrong document type
                    if not _cam_p_check.get("ok") and not _cam_p_check.get("skipped"):
                        st.error(f"⚠️ {_cam_p_check.get('reason', 'Wrong document type.')}")
                    else:
                        st.success("✅ Photo captured")
                    st.image(
                        st.session_state["idv_cam_p_bytes"],
                        caption="PAN — captured",
                        use_container_width=True,
                    )
                if st.button("✖ Close Camera", key="btn_cam_p_close", use_container_width=True):
                    st.session_state["idv_cam_p_open"] = False
                    st.rerun()
            if (
                not st.session_state.get("idv_cam_p_open")
                and st.session_state.get("idv_cam_p_bytes")
            ):
                _cam_p_check = st.session_state.get("idv_cam_p_doccheck", {})
                # Show error banner if the captured image failed the document type check
                if not _cam_p_check.get("ok") and not _cam_p_check.get("skipped"):
                    st.error(f"⚠️ {_cam_p_check.get('reason', 'Wrong document type.')}")
                st.image(
                    st.session_state["idv_cam_p_bytes"],
                    caption="PAN — captured",
                    use_container_width=True,
                )

        # ---- Selfie camera
        with cam_col_s:
            st.markdown("**🤳 Selfie Photo**")
            if not st.session_state.get("idv_cam_s_open"):
                if st.button("📷 Open Camera", key="btn_cam_s_open", use_container_width=True):
                    st.session_state["idv_cam_s_open"] = True
                    st.rerun()
            else:
                _cam_s = st.camera_input(
                    "Take Photo",
                    key="cam_selfie_input",
                    label_visibility="collapsed",
                )
                if _cam_s:
                    # New photo just taken — clear old doc check so it reruns for the new image
                    _cam_s_new = _cam_s.getvalue()
                    if st.session_state.get("idv_cam_s_bytes") != _cam_s_new:
                        st.session_state.pop("idv_cam_s_doccheck", None)
                    st.session_state["idv_cam_s_bytes"] = _cam_s_new
                if st.session_state.get("idv_cam_s_bytes"):
                    # Run document type check once per captured image (cached in session state)
                    if "idv_cam_s_doccheck" not in st.session_state:
                        with st.spinner("Verifying document type..."):
                            st.session_state["idv_cam_s_doccheck"] = _check_document_type(
                                st.session_state["idv_cam_s_bytes"], "selfie"
                            )
                    _cam_s_check = st.session_state.get("idv_cam_s_doccheck", {})
                    # Show error if we are confident this is the wrong document type
                    if not _cam_s_check.get("ok") and not _cam_s_check.get("skipped"):
                        st.error(f"⚠️ {_cam_s_check.get('reason', 'Wrong document type.')}")
                    else:
                        st.success("✅ Selfie captured")
                    st.image(
                        st.session_state["idv_cam_s_bytes"],
                        caption="Selfie — captured",
                        use_container_width=True,
                    )
                if st.button("✖ Close Camera", key="btn_cam_s_close", use_container_width=True):
                    st.session_state["idv_cam_s_open"] = False
                    st.rerun()
            if (
                not st.session_state.get("idv_cam_s_open")
                and st.session_state.get("idv_cam_s_bytes")
            ):
                _cam_s_check = st.session_state.get("idv_cam_s_doccheck", {})
                # Show error banner if the captured image failed the document type check
                if not _cam_s_check.get("ok") and not _cam_s_check.get("skipped"):
                    st.error(f"⚠️ {_cam_s_check.get('reason', 'Wrong document type.')}")
                st.image(
                    st.session_state["idv_cam_s_bytes"],
                    caption="Selfie — captured",
                    use_container_width=True,
                )

    # ---- Resolve effective sources (uploaded files take priority) ---------
    # For each camera slot, only use the captured bytes if the doc type check
    # passed (or was skipped because Ollama was unreachable). If the check
    # found the wrong document type with high confidence, we block the image
    # so the face match and extraction steps don't receive garbage input.
    if aadhaar_file is None and st.session_state.get("idv_cam_a_bytes"):
        _cam_a_check = st.session_state.get("idv_cam_a_doccheck", {})
        # ok=True (correct doc or skipped) — allow through; ok=False — blocked
        if _cam_a_check.get("ok", True) or _cam_a_check.get("skipped", False):
            aadhaar_file = _DocumentCapture(
                st.session_state["idv_cam_a_bytes"], "aadhaar_camera.jpg"
            )
    if pan_file is None and st.session_state.get("idv_cam_p_bytes"):
        _cam_p_check = st.session_state.get("idv_cam_p_doccheck", {})
        # ok=True (correct doc or skipped) — allow through; ok=False — blocked
        if _cam_p_check.get("ok", True) or _cam_p_check.get("skipped", False):
            pan_file = _DocumentCapture(
                st.session_state["idv_cam_p_bytes"], "pan_camera.jpg"
            )
    if selfie_bytes is None and st.session_state.get("idv_cam_s_bytes"):
        _cam_s_check = st.session_state.get("idv_cam_s_doccheck", {})
        # ok=True (correct doc or skipped) — allow through; ok=False — blocked
        if _cam_s_check.get("ok", True) or _cam_s_check.get("skipped", False):
            selfie_bytes = st.session_state["idv_cam_s_bytes"]
            selfie_name = "camera_selfie.jpg"

    # ── Parse & validate documents ─────────────────────────────────────────
    aadhaar_qr: Dict[str, Any] = {}
    pan_data: Dict[str, Any] = {}
    pan_validation: Dict[str, Any] = {}

    # For camera-captured Aadhaar, run QR parse and show results below tabs
    if aadhaar_file is not None:
        _a_key = f"_idv_qr_{aadhaar_file.size}"
        if _a_key not in st.session_state:
            with st.spinner("Scanning Aadhaar QR code..."):
                parsed_qr = _parse_aadhaar_qr(aadhaar_file.getvalue())
                if parsed_qr.get("qr_found") is False or not parsed_qr:
                    gemma_fallback = extract_aadhaar_details_with_ollama(aadhaar_file.getvalue())
                    if gemma_fallback:
                        parsed_qr = gemma_fallback
                st.session_state[_a_key] = parsed_qr

        aadhaar_qr = st.session_state[_a_key]

    # For camera-captured PAN, run OCR and show validation below tabs
    if pan_file is not None:
        _p_key = f"_idv_pan_{pan_file.size}"
        if _p_key not in st.session_state:
            with st.spinner("Reading PAN card..."):
                st.session_state[_p_key] = _extract_pan_info(pan_file.getvalue())
        pan_data = st.session_state[_p_key]
        _extracted_pan = pan_data.get("pan_number", "")
        if _extracted_pan:
            pan_validation = _validate_pan(_extracted_pan)

    # Show camera-source parse results below the tabs (upload-source ones shown inside tab)
    _is_cam_aadhaar = isinstance(aadhaar_file, _DocumentCapture) and aadhaar_file is not None
    _is_cam_pan = isinstance(pan_file, _DocumentCapture) and pan_file is not None
    if _is_cam_aadhaar and aadhaar_qr:
        if aadhaar_qr.get("qr_type") in ("xml", "gemma4"):
            if aadhaar_qr.get("qr_type") == "gemma4":
                st.success("Aadhaar details extracted via Gamma4 fallback")
            else:
                st.success("QR decoded successfully")
            st.markdown(
                f"**UID:** {aadhaar_qr.get('uid', '—')}  \n"
                f"**Full Name:** {aadhaar_qr.get('name', '—')}  \n"
                f"**DOB/YOB:** {aadhaar_qr.get('dob') or aadhaar_qr.get('yob', '—')}  \n"
                f"**Gender:** {aadhaar_qr.get('gender', '—')}  \n"
                f"**Care of (C/O):** {aadhaar_qr.get('co', '—')}  \n"
                f"**VTC:** {aadhaar_qr.get('vtc', '—')}  \n"
                f"**District:** {aadhaar_qr.get('dist', '—')}  \n"
                f"**State:** {aadhaar_qr.get('state', '—')}  \n"
                f"**PIN Code:** {aadhaar_qr.get('pc', '—')}"
            )
            st.caption("Aadhaar looks good")
        elif aadhaar_qr.get("qr_type") == "secure":
            st.info(aadhaar_qr.get("note", "Secure Aadhaar QR detected."))
        elif aadhaar_qr.get("qr_found") is False:
            st.warning("No Aadhaar QR code detected in the captured photo.")
    if _is_cam_pan:
        if pan_data.get("pan_number"):
            if pan_validation.get("valid"):
                st.success("PAN decoded successfully")
                st.markdown(
                    f"**PAN:** {pan_data['pan_number']}  \n"
                    f"**Entity type:** {pan_validation.get('entity_type', '')}"
                )
            else:
                st.warning(f"PAN read: `{pan_data['pan_number']}` — {pan_validation.get('error', '')}")
        if pan_data.get("full_name") or pan_data.get("name"):
            st.markdown(f"**Full name on PAN:** {pan_data.get('full_name') or pan_data.get('name')}")
        if pan_data.get("father_name"):
            st.markdown(
                f"**Care of (typically Father or Husband's name): S/O:** "
                f"{pan_data['father_name']}"
            )
        if pan_data.get("date_of_birth"):
            st.markdown(f"**Date of birth:** {pan_data['date_of_birth']}")
        elif any(pan_data.get(field) for field in ("full_name", "father_name", "date_of_birth")):
            st.info("PAN details were partially extracted. Review the fields below and enter PAN manually if needed.")

    # ── Step 2 — Document Cross-Checks and PAN Layers ─────────────────────
    if aadhaar_file or pan_file:
        st.divider()
        st.subheader("Document Cross-Checks")
        chk_cols = st.columns(2)

        aadhaar_name = (
            aadhaar_qr.get("name", "") if aadhaar_qr.get("qr_type") in ("xml", "gemma4") else ""
        )
        pan_name = pan_data.get("full_name") or pan_data.get("name", "")
        aadhaar_dob = aadhaar_qr.get("dob") or aadhaar_qr.get("yob", "")
        pan_dob = pan_data.get("date_of_birth", "")
        _name_check = compare_first_last_names(aadhaar_name, pan_name)
        _dob_check = compare_dob_values(aadhaar_dob, pan_dob)

        with chk_cols[0]:
            if _name_check.get("passed") is True:
                st.success(
                    "**First Name & Last Name Match: PASS**  \n"
                    f"Aadhaar: *{_name_check['aadhaar_first_name']} {_name_check['aadhaar_last_name']}*  \n"
                    f"PAN: *{_name_check['pan_first_name']} {_name_check['pan_last_name']}*"
                )
            elif _name_check.get("passed") is False:
                st.error(
                    "**First Name & Last Name Match: FAIL**  \n"
                    f"{_name_check['message']}"
                )
            elif aadhaar_name or pan_name:
                st.info(_name_check.get("message", "Name comparison unavailable."))
            else:
                st.caption(
                    "Upload both Aadhaar and PAN cards to compare first and last names."
                )

        with chk_cols[1]:
            if _dob_check.get("passed") is True:
                st.success(
                    "**DOB Match: PASS**  \n"
                    f"{_dob_check['message']}"
                )
            elif _dob_check.get("passed") is False:
                st.error(
                    "**DOB Match: FAIL**  \n"
                    f"{_dob_check['message']}"
                )
            elif aadhaar_dob or pan_dob:
                st.info(_dob_check.get("message", "DOB comparison unavailable."))
            elif pan_validation.get("valid"):
                st.info(
                    f"**PAN Format: VALID**  \nEntity type: {pan_validation['entity_type']}"
                )
            elif pan_file and not pan_validation.get("valid") and pan_data.get("pan_number"):
                pv = _validate_pan(pan_data.get("pan_number", ""))
                if not pv.get("valid"):
                    st.warning(f"**PAN Format: INVALID** — {pv.get('error', '')}")

    _stored_name_check = compare_first_last_names(
        aadhaar_qr.get("name", "") if aadhaar_qr.get("qr_type") in ("xml", "gemma4") else "",
        pan_data.get("full_name") or pan_data.get("name", ""),
    )
    _stored_dob_check = compare_dob_values(
        aadhaar_qr.get("dob") or aadhaar_qr.get("yob", ""),
        pan_data.get("date_of_birth", ""),
    )
    _stored_pan_layers = _build_pan_layer_analysis(
        pan_data,
        pan_validation,
        pan_file.getvalue() if pan_file else None,
    )

    def _aadhaar_format_check_payload() -> Dict[str, Any]:
        if aadhaar_qr.get("qr_type") in ("xml", "gemma4"):
            return build_format_check(
                "Aadhaar details decoded successfully and exposed applicant identity fields.",
                True,
            )
        if aadhaar_qr.get("qr_type") == "secure":
            return build_format_check(
                "A secure Aadhaar QR was detected. Offline validation can confirm its presence but cannot decrypt the payload.",
                None,
            )
        if aadhaar_qr.get("qr_found") is False:
            return build_format_check(
                "No Aadhaar QR code was detected in the uploaded image.",
                False,
            )
        return build_format_check(
            "Aadhaar format validation could not run because no usable QR payload was found.",
            None,
        )

    def _selfie_format_check_payload() -> Dict[str, Any]:
        return build_format_check(
            "Selfie image was uploaded successfully and decoded for face verification.",
            True if selfie_bytes else None,
        )

    # ── Mark page dirty when files are uploaded (for unsaved-changes guard) ────
    if aadhaar_file or pan_file or selfie_bytes:
        st.session_state["_idv_has_uploads"] = True

    # ── Step 3: Applicant Details form (auto-filled from documents) ────────
    st.divider()
    st.subheader("Applicant Details")
    st.info(
        "Fields marked **auto-filled** are extracted from the documents. "
        "Please provide Phone and Email manually.",
        icon="ℹ️",
    )

    _qr_name = (
        aadhaar_qr.get("name", "") if aadhaar_qr.get("qr_type") in ("xml", "gemma4") else ""
    )
    _pan_name_for_form = (pan_data.get("full_name") or pan_data.get("name") or "").strip()
    _preferred_name = _qr_name or _pan_name_for_form
    _name_parts = _preferred_name.split(maxsplit=1)
    _default_fn = _name_parts[0] if _name_parts else ""
    _default_ln = _name_parts[1] if len(_name_parts) > 1 else ""
    _default_pan = pan_data.get("pan_number", "")
    _default_aadh = aadhaar_qr.get("uid", "")

    if pan_data.get("father_name") or pan_data.get("date_of_birth"):
        meta_bits = []
        if pan_data.get("father_name"):
            meta_bits.append(f"Father's name: **{pan_data['father_name']}**")
        if pan_data.get("date_of_birth"):
            meta_bits.append(f"DOB: **{pan_data['date_of_birth']}**")
        st.caption("  |  ".join(meta_bits))

    _auto_key = (
        f"_idv_auto_{getattr(aadhaar_file, 'size', 0)}_"
        f"{getattr(pan_file, 'size', 0)}"
    )
    if st.session_state.get("_idv_auto_key") != _auto_key:
        if _default_fn:
            st.session_state["idv_ei_fn"] = _default_fn
        if _default_ln:
            st.session_state["idv_ei_ln"] = _default_ln
        if _default_pan:
            st.session_state["idv_ei_pan"] = _default_pan
        if _default_aadh:
            st.session_state["idv_ei_aadh"] = _default_aadh
        st.session_state["_idv_auto_key"] = _auto_key

    mc1, mc2 = st.columns(2)
    e_fn = mc1.text_input(
        "First name \u00a0★ required",
        key="idv_ei_fn",
        disabled=True,
        help="Auto-filled from Aadhaar QR / PAN. Upload documents to populate.",
    )
    e_ln = mc2.text_input(
        "Last name \u00a0★ required",
        key="idv_ei_ln",
        disabled=True,
        help="Auto-filled from Aadhaar QR / PAN. Upload documents to populate.",
    )
    mc3, mc4 = st.columns(2)
    e_pan = mc3.text_input(
        "PAN number \u00a0★ required",
        key="idv_ei_pan",
        placeholder="ABCDE1234F",
        disabled=True,
        help="Auto-filled from PAN card OCR. Upload PAN card to populate.",
    )
    e_aadh = mc4.text_input(
        "Aadhaar number \u00a0★ required",
        key="idv_ei_aadh",
        placeholder="1234 5678 9012",
        disabled=True,
        help="Auto-filled from Aadhaar QR code. Upload Aadhaar card to populate.",
    )
    mc5, mc6 = st.columns(2)
    e_email = mc5.text_input(
        "Email  *(enter manually)*",
        key="idv_ei_email",
        placeholder="applicant@email.com",
    )
    e_phone = mc6.text_input(
        "Phone  *(enter manually)*",
        key="idv_ei_phone",
        placeholder="+91 98765 43210",
    )

    # Required-field validation — blocks Save to Database
    _required_missing = []
    if not str(st.session_state.get("idv_ei_fn", "")).strip():
        _required_missing.append("First name")
    if not str(st.session_state.get("idv_ei_ln", "")).strip():
        _required_missing.append("Last name")
    if not str(st.session_state.get("idv_ei_pan", "")).strip():
        _required_missing.append("PAN number")
    if not str(st.session_state.get("idv_ei_aadh", "")).strip():
        _required_missing.append("Aadhaar number")
    if _required_missing:
        st.caption(
            f"⚠️ The following fields are required and will be auto-filled when you upload the "
            f"corresponding documents: **{', '.join(_required_missing)}**"
        )

    forced_ref: str | None = None
    extra_identity: dict | None = None
    if _DB_IMPORTS_OK and _db_available_cached():
        with st.expander(
            "🔗 Link to an existing entity record (optional)", expanded=False
        ):
            search_q = st.text_input(
                "Search by name / PAN / Aadhaar / email / BT-ref",
                key="idv_entity_search",
                placeholder="e.g. BT-000003, MVWNV2212G…",
            )
            if search_q.strip():
                matches = search_entities(search_q.strip(), "all", limit=8)
                if matches:
                    opts = {
                        f"{m['entity_ref']}  —  {m['first_name']} {m['last_name']}  "
                        f"({m.get('pan_number') or m.get('email') or 'no id'})": m[
                            "entity_ref"
                        ]
                        for m in matches
                    }
                    chosen_label = st.selectbox(
                        "Select person", list(opts.keys()), key="idv_entity_select"
                    )
                    forced_ref = opts[chosen_label]
                    st.success(
                        f"Will link to **{chosen_label.split('—')[0].strip()}**"
                    )
                else:
                    st.info(
                        "No match found. A new entity will be created from the details above."
                    )

    if any([e_fn, e_ln, e_pan, e_aadh, e_email, e_phone]):
        extra_identity = {
            "first_name": e_fn.strip(),
            "last_name": e_ln.strip(),
            "pan_number": e_pan.strip().upper(),
            "aadhar_number": e_aadh.replace(" ", "").strip(),
            "email": e_email.strip().lower(),
            "phone": e_phone.strip(),
        }

    # ── Step 4: Run identity verification ──────────────────────────────────
    st.divider()
    _can_run = aadhaar_file is not None and selfie_bytes is not None
    if not _can_run:
        st.info(
            "Upload the Aadhaar card and provide a selfie (or take one with the camera) "
            "to run verification."
        )

    if _can_run:
        if st.button(
            "Run Identity Verification  🔍",
            type="primary",
            use_container_width=True,
        ):
            if not forced_ref and not extra_identity:
                st.warning(
                    "Please fill in at least the applicant's name or PAN to link the result."
                )
                st.stop()

            with st.spinner("Running face detection and ArcFace matching..."):
                from basetruth.vision.face import compare_faces  # noqa: PLC0415

                assert aadhaar_file is not None
                assert selfie_bytes is not None
                face_result = compare_faces(aadhaar_file.getvalue(), selfie_bytes)

            # Store result and all inputs in session state for the explicit save step
            st.session_state["idv_face_result"] = face_result
            st.session_state["idv_face_doc_bytes"] = aadhaar_file.getvalue()
            st.session_state["idv_face_doc_name"] = aadhaar_file.name
            st.session_state["idv_face_selfie_bytes"] = selfie_bytes
            st.session_state["idv_face_selfie_name"] = selfie_name
            st.session_state["idv_face_forced_ref"] = forced_ref
            st.session_state["idv_face_extra_identity"] = extra_identity
            st.session_state["idv_face_pan_bytes"] = pan_file.getvalue() if pan_file else None
            st.session_state["idv_face_pan_name"] = getattr(pan_file, "name", "pan_card.jpg") if pan_file else ""
            # Retrieve the already-computed signature crop from the cached session key.
            # If the pan_file size matches a cached crop, reuse it; otherwise run the crop now.
            _pan_sig_key = f"_idv_pan_sig_v3_{pan_file.size}" if pan_file else None
            st.session_state["idv_face_pan_signature_bytes"] = (
                st.session_state.get(_pan_sig_key)
                if _pan_sig_key
                else None
            )
            st.session_state["idv_face_pan_data"] = {
                key: pan_data.get(key)
                for key in (
                    "pan_number",
                    "full_name",
                    "father_name",
                    "date_of_birth",
                    "extraction_source",
                    "engine",
                )
                if pan_data.get(key)
            }
            st.session_state["idv_face_cross_checks"] = {
                "first_last_name_match": _stored_name_check,
                "dob_match": _stored_dob_check,
                "pan_format": {
                    "passed": pan_validation.get("valid"),
                    "message": (
                        f"PAN format is valid for entity type {pan_validation.get('entity_type', 'Unknown')}."
                        if pan_validation.get("valid")
                        else pan_validation.get("error", "PAN format check could not be completed.")
                    ),
                    "entity_type": pan_validation.get("entity_type", ""),
                    "pan_number": pan_validation.get("pan") or pan_data.get("pan_number", ""),
                },
            }
            st.session_state["idv_face_layered_analysis"] = {
                "pan_layers": _stored_pan_layers,
                "upload_authenticity": {
                    "aadhaar": analyse_upload_authenticity(
                        aadhaar_file.getvalue(),
                        getattr(aadhaar_file, "name", "aadhaar_upload"),
                        format_check=_aadhaar_format_check_payload(),
                    ),
                    "photo": analyse_upload_authenticity(
                        selfie_bytes,
                        selfie_name,
                        format_check=_selfie_format_check_payload(),
                    ),
                },
            }
            st.session_state["idv_face_aadhaar_qr"] = {
                key: aadhaar_qr.get(key)
                for key in ("name", "uid", "dob", "yob", "gender", "dist", "state", "qr_type")
                if aadhaar_qr.get(key)
            }
            st.session_state["idv_face_saved"] = False
            st.session_state.pop("idv_face_saved_ref", None)
            st.session_state.pop("idv_face_saved_pdf", None)

        # ── Display result whenever session state holds one ───────────────
        _face_result = st.session_state.get("idv_face_result")
        if _face_result is not None:
            st.subheader("Verification Result")

            if "error" in _face_result:
                st.error(f"Face matching failed: {_face_result['error']}")
            else:
                score = _face_result["display_score"]
                is_match = _face_result["match"]

                r1, r2 = st.columns(2)
                with r1:
                    st.image(
                        _face_result["doc_annotated_rgb"],
                        caption="Face detected on Aadhaar",
                        use_container_width=True,
                    )
                with r2:
                    st.image(
                        _face_result["selfie_annotated_rgb"],
                        caption="Face detected in selfie",
                        use_container_width=True,
                    )

                if is_match:
                    st.success(
                        f"### ✅ IDENTITY MATCH — {score:.1f}% confidence\n"
                        "The face on the Aadhaar card matches the provided selfie."
                    )
                else:
                    st.error(
                        f"### 🚨 IDENTITY MISMATCH — {score:.1f}% confidence\n"
                        "The faces DO NOT match. Possible fraud risk."
                    )
                st.caption(
                    f"Cosine similarity: {_face_result['confidence']:.3f} "
                    f"(threshold: {_face_result['threshold']:.2f})"
                )
                if is_match:
                    st.success("**Photo Match: PASS** — Aadhaar photo and selfie match.")
                else:
                    st.error("**Photo Match: FAIL** — Aadhaar photo and selfie do not match.")

                # ── Save section ──────────────────────────────────────────
                st.divider()
                _already_saved = st.session_state.get("idv_face_saved", False)
                _saved_ref = st.session_state.get("idv_face_saved_ref")
                _saved_pdf = st.session_state.get("idv_face_saved_pdf")

                if _already_saved:
                    st.success(
                        f"✅ Saved to database — Entity: **{_saved_ref or 'unlinked'}**"
                    )
                    if _saved_pdf:
                        st.download_button(
                            "Download Identity Check Report (PDF)",
                            data=_saved_pdf,
                            file_name=f"identity_check_{_saved_ref or 'unlinked'}.pdf",
                            mime="application/pdf",
                            key="idv_pdf_dl",
                        )
                else:
                    if _DB_IMPORTS_OK and _db_available_cached():
                        _save_blocked = bool(_required_missing)
                        if _save_blocked:
                            st.warning(
                                f"⚠️ Cannot save — required fields missing: "
                                f"**{', '.join(_required_missing)}**. "
                                "Upload the corresponding documents above to auto-fill them."
                            )
                        if st.button(
                            "💾 Save to Database",
                            type="secondary",
                            use_container_width=True,
                            key="idv_save_btn",
                            disabled=_save_blocked,
                        ):
                            _s_doc_name = st.session_state.get("idv_face_doc_name", "")
                            _s_selfie_name = st.session_state.get("idv_face_selfie_name", "")
                            _s_forced_ref = st.session_state.get("idv_face_forced_ref")
                            _s_extra_identity = st.session_state.get("idv_face_extra_identity")
                            _s_doc_bytes = st.session_state.get("idv_face_doc_bytes")
                            _s_selfie_bytes = st.session_state.get("idv_face_selfie_bytes")

                            db_payload = {
                                k: v
                                for k, v in _face_result.items()
                                if k not in ("doc_annotated_rgb", "selfie_annotated_rgb")
                            }
                            _pan_context = st.session_state.get("idv_face_pan_data") or {}
                            _cross_checks = st.session_state.get("idv_face_cross_checks") or {}
                            _layered_analysis = st.session_state.get("idv_face_layered_analysis") or {}
                            _aadhaar_context = st.session_state.get("idv_face_aadhaar_qr") or {}
                            _cross_checks["photo_match"] = {
                                "passed": bool(_face_result.get("match")),
                                "message": (
                                    "Aadhaar photo and selfie match."
                                    if _face_result.get("match")
                                    else "Aadhaar photo and selfie do not match."
                                ),
                                "display_score": _face_result.get("display_score"),
                                "threshold": _face_result.get("threshold"),
                            }
                            if _pan_context:
                                db_payload["pan_extraction"] = _pan_context
                            if _cross_checks:
                                db_payload["cross_checks"] = _cross_checks
                            if _layered_analysis:
                                db_payload["layered_analysis"] = _layered_analysis
                            if _aadhaar_context:
                                db_payload["aadhaar_qr"] = _aadhaar_context
                            for k, v in list(db_payload.items()):
                                if hasattr(v, "item"):
                                    db_payload[k] = v.item()

                            _ref_for_pdf = _s_forced_ref or ""
                            _name_for_pdf = (
                                f"{_s_extra_identity.get('first_name', '')} "
                                f"{_s_extra_identity.get('last_name', '')}".strip()
                                if _s_extra_identity
                                else ""
                            )

                            try:
                                from basetruth.reporting.pdf import (  # noqa: PLC0415
                                    render_identity_check_pdf,
                                )

                                pdf_bytes = render_identity_check_pdf(
                                    check_type="face_match",
                                    result=db_payload,
                                    entity_ref=_ref_for_pdf,
                                    entity_name=_name_for_pdf,
                                    doc_filename=_s_doc_name,
                                    selfie_filename=_s_selfie_name,
                                    doc_image_bytes=_s_doc_bytes,
                                    selfie_image_bytes=_s_selfie_bytes,
                                )
                            except Exception:  # noqa: BLE001
                                pdf_bytes = None

                            _s_pan_name = st.session_state.get("idv_face_pan_name", "")
                            _s_pan_bytes = st.session_state.get("idv_face_pan_bytes")
                            _s_pan_signature_bytes = st.session_state.get("idv_face_pan_signature_bytes")

                            saved = save_identity_check(
                                check_type="face_match",
                                result=db_payload,
                                forced_entity_ref=_s_forced_ref,
                                extra_identity=_s_extra_identity,
                                doc_filename=_s_doc_name,
                                pdf_bytes=pdf_bytes,
                                doc_bytes=_s_doc_bytes,
                                selfie_bytes=_s_selfie_bytes,
                                pan_filename=_s_pan_name,
                                pan_bytes=_s_pan_bytes,
                                pan_signature_bytes=_s_pan_signature_bytes,
                            )
                            if saved:
                                st.session_state["idv_face_saved"] = True
                                st.session_state["idv_face_saved_ref"] = (
                                    saved.get("entity_ref") or _s_forced_ref
                                )
                                st.session_state["idv_face_saved_pdf"] = pdf_bytes
                                # Clear the dirty flag — work is now persisted
                                st.session_state.pop("_idv_has_uploads", None)
                                st.rerun()
                            else:
                                st.error(
                                    "⚠️ Result could not be saved to the database. "
                                    "The identity check ran successfully but the record was not "
                                    "persisted. Check the Logs screen for details."
                                )
                    else:
                        st.warning(
                            "Database is offline — connect PostgreSQL to save results."
                        )

                # ── History ───────────────────────────────────────────────
                _display_ref = (
                    st.session_state.get("idv_face_saved_ref")
                    or st.session_state.get("idv_face_forced_ref")
                )
                if _display_ref and _DB_IMPORTS_OK and _db_available_cached():
                    st.divider()
                    st.subheader(f"Previous Identity Checks for {_display_ref}")
                    checks = get_entity_identity_checks(_display_ref)
                    face_checks = [
                        c for c in checks if c["check_type"] == "face_match"
                    ]
                    if face_checks:
                        try:
                            import pandas as pd  # noqa: PLC0415

                            df = pd.DataFrame(
                                [
                                    {
                                        "Date": c["created_at"][:19],
                                        "Verdict": c["verdict"],
                                        "Score": (
                                            f"{c['display_score']:.1f}%"
                                            if c["display_score"]
                                            else "-"
                                        ),
                                        "Match": "Yes" if c["is_match"] else "No",
                                    }
                                    for c in face_checks
                                ]
                            )
                            st.dataframe(df, hide_index=True, use_container_width=True)
                        except ImportError:
                            for c in face_checks:
                                st.write(f"{c['created_at'][:10]} — {c['verdict']}")
                    else:
                        st.caption("No previous checks found for this entity.")
