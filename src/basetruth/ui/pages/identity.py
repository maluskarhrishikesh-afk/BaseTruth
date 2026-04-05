"""Identity Verification page — Aadhaar QR, PAN OCR (capped upscale), layered fraud detection, ArcFace face match."""
from __future__ import annotations

import re as _re
import xml.etree.ElementTree as _ET
from typing import Any, Dict

import streamlit as st

from basetruth.analysis.identity_checks import compare_dob_values, compare_first_last_names
from basetruth.analysis.upload_authenticity import analyse_upload_authenticity, build_format_check
from basetruth.integrations.ollama import extract_pan_details_with_ollama, extract_aadhaar_details_with_ollama
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _page_title,
    get_entity_identity_checks,
    save_identity_check,
    search_entities,
)

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
# PAN card OCR fallback — FIXED: capped upscale (no 3× on large images)
# ---------------------------------------------------------------------------


def _extract_pan_info_ocr(img_bytes: bytes) -> Dict[str, Any]:
    """OCR a PAN card image and extract PAN number + name.

    Preprocessing caps the image at 1 200 px wide (still upscales small images
    up to 1.5×) rather than blindly applying a 3× scale. This prevents
    pytesseract from timing out on high-resolution scans.
    """
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        nparr = np.frombuffer(img_bytes, np.uint8)
        orig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if orig is None:
            return {}

        def _try_ocr(gray_img) -> str:  # type: ignore[no-untyped-def]
            try:
                import pytesseract  # noqa: PLC0415
            except ImportError:
                return ""
            best = ""
            for psm in (6, 11, 3, 8, 7):
                try:
                    t = pytesseract.image_to_string(
                        gray_img,
                        config=(
                            f"--psm {psm} --oem 3 "
                            "-c tessedit_char_whitelist="
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
                        ),
                        timeout=15,  # prevent hangs on large images
                    )
                    if len(t) > len(best):
                        best = t
                    if _re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", t.upper()):
                        return t
                except Exception:  # noqa: BLE001
                    pass
            return best

        def _preprocess(img_bgr):  # type: ignore[no-untyped-def]
            """Return a list of preprocessed grayscale variants for OCR.

            For camera-captured images the cap is raised to 2400 px and the
            max upscale to 2.5× so small/low-res captures still get useful
            text extraction. Adaptive threshold and denoising are added to
            handle uneven lighting and camera noise.
            """
            _, w_orig = img_bgr.shape[:2]
            max_w = 2400  # raised from 1200 — camera images need more detail
            scale = min(max_w / max(w_orig, 1), 2.5)  # raised from 1.5×
            resized = cv2.resize(
                img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Denoise first — camera sensors add noise that hurts OCR
            denoised = cv2.fastNlMeansDenoising(gray, h=12)
            variants = [gray, denoised]
            _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(otsu)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            variants.append(clahe.apply(denoised))
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            variants.append(cv2.filter2D(denoised, -1, kernel))
            # Adaptive threshold — best for uneven camera lighting / shadows
            adapt = cv2.adaptiveThreshold(
                denoised, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 4,
            )
            variants.append(adapt)
            return variants

        _pan_re_global = _re.compile(r"[A-Z]{5}[0-9]{4}[A-Z]")
        _skip_words = {
            "INCOME", "DEPARTMENT", "GOVT", "INDIA", "TAX", "PERMANENT",
            "ACCOUNT", "NUMBER", "CARD", "OF", "SIGNATURE", "FATHER",
        }

        result: Dict[str, Any] = {}
        all_text = ""

        for variant in _preprocess(orig):
            text = _try_ocr(variant)
            if not text.strip():
                continue
            all_text += "\n" + text
            if not result.get("pan_number"):
                m = _pan_re_global.search(text.upper())
                if m:
                    result["pan_number"] = m.group()
            if not result.get("name"):
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
            if result.get("pan_number") and result.get("name"):
                break

        if not result.get("pan_number") and all_text:
            m = _pan_re_global.search(all_text.upper())
            if m:
                result["pan_number"] = m.group()

        if result.get("name") and not result.get("full_name"):
            result["full_name"] = result["name"]
        if result:
            result["engine"] = "ocr"
        return result
    except Exception:  # noqa: BLE001
        return {}


def _extract_pan_info(img_bytes: bytes) -> Dict[str, Any]:
    """Extract PAN fields using Gemma4 first, then OCR fallback when needed."""
    gemma_data = extract_pan_details_with_ollama(img_bytes)
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

    if gemma_data.get("model"):
        merged["gemma_model"] = gemma_data["model"]
    if gemma_data.get("raw_response"):
        merged["gemma_raw_response"] = gemma_data["raw_response"]

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

    st.subheader("Step 1 — Provide Documents")

    tab_upload, tab_camera = st.tabs(["📁 Upload Documents", "📷 Capture with Camera"])

    # Effective document sources — populated by whichever tab the user interacts with
    aadhaar_file: _DocumentCapture | None = None  # type: ignore[assignment]
    pan_file: _DocumentCapture | None = None       # type: ignore[assignment]
    selfie_bytes: bytes | None = None
    selfie_name = "selfie.jpg"

    # ---- Upload tab -------------------------------------------------------
    with tab_upload:
        col_a, col_p, col_s = st.columns(3)

        with col_a:
            st.markdown("**📄 Aadhaar Card**")
            _up_aadhaar = st.file_uploader(
                "Drag and drop file here",
                type=IMG_TYPES,
                key="idv_aadhaar",
                label_visibility="visible",
            )
            st.caption("Limit 200MB per file • JPG, JPEG, PNG, WEBP")
            if _up_aadhaar:
                aadhaar_file = _up_aadhaar  # type: ignore[assignment]
                # Preview + QR inline in the upload tab
                _a_key = f"_idv_qr_{_up_aadhaar.size}"
                if _a_key not in st.session_state:
                    with st.spinner("Scanning Aadhaar QR code..."):
                        parsed_qr = _parse_aadhaar_qr(_up_aadhaar.getvalue())
                        if parsed_qr.get("qr_found") is False or not parsed_qr:
                            gemma_fallback = extract_aadhaar_details_with_ollama(_up_aadhaar.getvalue())
                            if gemma_fallback:
                                parsed_qr = gemma_fallback
                        st.session_state[_a_key] = parsed_qr
                _aq = st.session_state[_a_key]
                st.image(_up_aadhaar.getvalue(), caption="Aadhaar card", use_container_width=True)
                if _aq.get("qr_found") is False:
                    st.warning("No QR code detected. Ensure the QR code is visible.")
                elif _aq.get("qr_type") in ("xml", "gemma4"):
                    if _aq.get("qr_type") == "gemma4":
                        st.success("Aadhaar details extracted via Gamma4 fallback")
                    else:
                        st.success("QR decoded successfully")
                    st.markdown(
                        f"**Name:** {_aq.get('name', '—')}  \n"
                        f"**DOB/YOB:** {_aq.get('dob') or _aq.get('yob', '—')}  \n"
                        f"**Gender:** {_aq.get('gender', '—')}  \n"
                        f"**District:** {_aq.get('dist', '—')}, {_aq.get('state', '—')}"
                    )
                    st.caption("Aadhaar looks good")
                elif _aq.get("qr_type") == "secure":
                    st.info(_aq.get("note", "Secure QR detected."))
                elif not _aq:
                    st.warning("QR scan error — check that OpenCV is available.")

        with col_p:
            st.markdown("**💳 PAN Card**")
            _up_pan = st.file_uploader(
                "Drag and drop file here",
                type=IMG_TYPES,
                key="idv_pan",
                label_visibility="visible",
            )
            st.caption("Limit 200MB per file • JPG, JPEG, PNG, WEBP")
            if _up_pan:
                pan_file = _up_pan  # type: ignore[assignment]
                # Preview + OCR inline in the upload tab
                _p_key = f"_idv_pan_{_up_pan.size}"
                if _p_key not in st.session_state:
                    with st.spinner("Reading PAN card..."):
                        st.session_state[_p_key] = _extract_pan_info(_up_pan.getvalue())
                _pd = st.session_state[_p_key]
                st.image(_up_pan.getvalue(), caption="PAN card", use_container_width=True)
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
                    st.markdown(f"**Full name on PAN:** {_pd.get('full_name') or _pd.get('name')}")
                if _pd.get("date_of_birth"):
                    st.markdown(f"**Date of birth:** {_pd['date_of_birth']}")

        with col_s:
            st.markdown("**🤳 Selfie Photo**")
            _up_selfie = st.file_uploader(
                "Drag and drop file here",
                type=IMG_TYPES,
                key="idv_selfie",
                label_visibility="visible",
            )
            st.caption("Limit 200MB per file • JPG, JPEG, PNG, WEBP")
            if _up_selfie:
                selfie_bytes = _up_selfie.getvalue()
                selfie_name = _up_selfie.name
                st.image(selfie_bytes, caption="Selfie", use_container_width=True)

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
                    st.session_state["idv_cam_a_bytes"] = _cam_a.getvalue()
                if st.session_state.get("idv_cam_a_bytes"):
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
                    st.session_state["idv_cam_p_bytes"] = _cam_p.getvalue()
                if st.session_state.get("idv_cam_p_bytes"):
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
                    st.session_state["idv_cam_s_bytes"] = _cam_s.getvalue()
                if st.session_state.get("idv_cam_s_bytes"):
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
                st.image(
                    st.session_state["idv_cam_s_bytes"],
                    caption="Selfie — captured",
                    use_container_width=True,
                )

    # ---- Resolve effective sources (uploaded files take priority) ---------
    if aadhaar_file is None and st.session_state.get("idv_cam_a_bytes"):
        aadhaar_file = _DocumentCapture(
            st.session_state["idv_cam_a_bytes"], "aadhaar_camera.jpg"
        )
    if pan_file is None and st.session_state.get("idv_cam_p_bytes"):
        pan_file = _DocumentCapture(
            st.session_state["idv_cam_p_bytes"], "pan_camera.jpg"
        )
    if selfie_bytes is None and st.session_state.get("idv_cam_s_bytes"):
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
                f"**Name:** {aadhaar_qr.get('name', '—')}  \n"
                f"**DOB/YOB:** {aadhaar_qr.get('dob') or aadhaar_qr.get('yob', '—')}  \n"
                f"**Gender:** {aadhaar_qr.get('gender', '—')}  \n"
                f"**District:** {aadhaar_qr.get('dist', '—')}, {aadhaar_qr.get('state', '—')}"
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
        if pan_data.get("date_of_birth"):
            st.markdown(f"**Date of birth:** {pan_data['date_of_birth']}")
        elif any(pan_data.get(field) for field in ("full_name", "father_name", "date_of_birth")):
            st.info("PAN details were partially extracted. Review the fields below and enter PAN manually if needed.")

    # ── Step 2 — Document Cross-Checks and PAN Layers ─────────────────────
    if aadhaar_file or pan_file:
        st.divider()
        st.subheader("Step 2 — Document Cross-Checks")
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

    # ── Step 3: Applicant Details form (auto-filled from documents) ────────
    st.divider()
    st.subheader("Step 3 — Applicant Details")
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
        "First name  *(auto-filled)*" if _default_fn else "First name",
        key="idv_ei_fn",
    )
    e_ln = mc2.text_input(
        "Last name  *(auto-filled)*" if _default_ln else "Last name",
        key="idv_ei_ln",
    )
    mc3, mc4 = st.columns(2)
    e_pan = mc3.text_input(
        "PAN number  *(auto-filled)*" if _default_pan else "PAN number",
        key="idv_ei_pan",
        placeholder="ABCDE1234F",
    )
    e_aadh = mc4.text_input(
        "Aadhaar number  *(from QR)*" if _default_aadh else "Aadhaar number",
        key="idv_ei_aadh",
        placeholder="1234 5678 9012",
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
                        if st.button(
                            "💾 Save to Database",
                            type="secondary",
                            use_container_width=True,
                            key="idv_save_btn",
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

                            saved = save_identity_check(
                                check_type="face_match",
                                result=db_payload,
                                forced_entity_ref=_s_forced_ref,
                                extra_identity=_s_extra_identity,
                                doc_filename=_s_doc_name,
                                selfie_filename=_s_selfie_name,
                                pdf_bytes=pdf_bytes,
                                doc_bytes=_s_doc_bytes,
                                selfie_bytes=_s_selfie_bytes,
                                pan_filename=_s_pan_name,
                                pan_bytes=_s_pan_bytes,
                            )
                            if saved:
                                st.session_state["idv_face_saved"] = True
                                st.session_state["idv_face_saved_ref"] = (
                                    saved.get("entity_ref") or _s_forced_ref
                                )
                                st.session_state["idv_face_saved_pdf"] = pdf_bytes
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
                                        "Document": c["doc_filename"],
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
