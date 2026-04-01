"""Identity Verification page — Aadhaar QR, PAN OCR (capped upscale), layered fraud detection, ArcFace face match."""
from __future__ import annotations

import re as _re
import xml.etree.ElementTree as _ET
from typing import Any, Dict

import streamlit as st

from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _render_entity_link_widget,
    db_available,
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
    """Decode the QR code on an Aadhaar card and return extracted fields."""
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {}
        detector = cv2.QRCodeDetector()
        data, _pts, _ = detector.detectAndDecode(img)
        if not data:
            h, w = img.shape[:2]
            big = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            data, _pts, _ = detector.detectAndDecode(big)
        if not data:
            return {"qr_found": False}
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
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# PAN card OCR — FIXED: capped upscale (no 3× on large images)
# ---------------------------------------------------------------------------


def _extract_pan_info(img_bytes: bytes) -> Dict[str, Any]:
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
            """Return a list of preprocessed grayscale variants.

            Cap the image at max_w=1200 so pytesseract never receives a
            5000×7000 px bitmap. Small images are still upscaled up to 1.5×.
            """
            h_orig, w_orig = img_bgr.shape[:2]
            max_w = 1200
            scale = min(max_w / max(w_orig, 1), 1.5)
            resized = cv2.resize(
                img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            variants = [gray]
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            variants.append(otsu)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            variants.append(clahe.apply(gray))
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            variants.append(cv2.filter2D(gray, -1, kernel))
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

        return result
    except Exception:  # noqa: BLE001
        return {}


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
# Name similarity
# ---------------------------------------------------------------------------


def _names_match(name1: str, name2: str) -> tuple:
    """Return (is_match, similarity_float) for two name strings."""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if not n1 or not n2:
        return False, 0.0
    if n1 == n2:
        return True, 1.0
    words1 = set(n1.split())
    words2 = set(n2.split())
    intersection = words1 & words2
    union = words1 | words2
    sim = len(intersection) / len(union) if union else 0.0
    return sim >= 0.5, sim


# ---------------------------------------------------------------------------
# Layered PAN Fraud Analysis UI
# ---------------------------------------------------------------------------


def _render_pan_layers(
    pan_data: Dict[str, Any],
    pan_validation: Dict[str, Any],
    pan_img_bytes: bytes | None,
    selfie_bytes: bytes | None,
    aadhaar_name: str,
) -> None:
    """Render the 5-layer PAN fraud dashboard."""
    st.markdown("#### 🛡️ PAN Fraud Detection — Layered Analysis")
    st.caption(
        "Each layer applies a different verification technique. "
        "Only paid government APIs (Layer 2) are skipped."
    )

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
                    f"OCR read `{pan_data['pan_number']}` — "
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

    # Layer 3 — OCR extraction
    ocr_pan = pan_data.get("pan_number", "")
    ocr_name = pan_data.get("name", "")
    if ocr_pan or ocr_name:
        ocr_detail = ""
        if ocr_pan:
            ocr_detail += f"PAN extracted: **`{ocr_pan}`**  \n"
        if ocr_name:
            ocr_detail += f"Name on card: **{ocr_name}**"
        layers.append(
            {
                "icon": "✅",
                "title": "Layer 3 — OCR Text Extraction",
                "status": "EXTRACTED",
                "status_color": "#16a34a",
                "detail": ocr_detail.strip(),
            }
        )
    else:
        layers.append(
            {
                "icon": "⚠️",
                "title": "Layer 3 — OCR Text Extraction",
                "status": "NOT FOUND",
                "status_color": "#d97706",
                "detail": (
                    "No PAN number or name could be extracted via OCR.  \n"
                    "Try a clearer, higher-contrast scan."
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

    # Layer 5 — Face match (deferred to Step 4)
    if selfie_bytes:
        layers.append(
            {
                "icon": "🔄",
                "title": "Layer 5 — Face Match (ArcFace)",
                "status": "PENDING",
                "status_color": "#7c3aed",
                "detail": "Selfie uploaded. Face match runs in **Step 4 — Run Identity Verification** below.",
            }
        )
    else:
        layers.append(
            {
                "icon": "⚪",
                "title": "Layer 5 — Face Match (ArcFace)",
                "status": "N/A",
                "status_color": "#94a3b8",
                "detail": "Upload a selfie or use the camera below to enable face matching.",
            }
        )

    # Render layers
    for layer in layers:
        layer_html = (
            f'<div style="display:flex;align-items:flex-start;gap:12px;'
            f'padding:14px 16px;border-radius:12px;margin-bottom:8px;'
            f'background:var(--secondary-background-color,#f8fafc);'
            f'border:1px solid rgba(0,0,0,0.06);border-left:4px solid {layer["status_color"]};">'
            f'<span style="font-size:1.1rem;flex-shrink:0;margin-top:1px;">{layer["icon"]}</span>'
            f'<div style="flex:1;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
            f'<strong style="font-size:.9rem;">{layer["title"]}</strong>'
            f'<span style="font-size:.75rem;font-weight:700;padding:2px 9px;border-radius:6px;'
            f'background:{layer["status_color"]}1a;color:{layer["status_color"]};">'
            f'{layer["status"]}</span>'
            f"</div>"
            f'<div style="font-size:.82rem;color:var(--text-color,#475569);">{layer["detail"]}</div>'
            f"</div></div>"
        )
        st.markdown(layer_html, unsafe_allow_html=True)

    # Overall fraud score
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
        risk_color = (
            "#16a34a" if overall >= 75 else "#d97706" if overall >= 50 else "#dc2626"
        )
        st.markdown(
            f'<div style="margin-top:12px;padding:14px 20px;border-radius:12px;'
            f'background:{risk_color}1a;border:1px solid {risk_color}44;">'
            f'<strong style="font-size:.9rem;color:{risk_color};">Overall PAN Fraud Score: '
            f'<span style="font-size:1.2rem;">{overall}/100</span> — {risk_label}</strong>'
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Identity Verification page
# ---------------------------------------------------------------------------


def _page_identity_verification() -> None:
    st.title("🧑‍💻 Identity Verification")
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

    # ── Step 1: Document uploads ───────────────────────────────────────────
    st.subheader("Step 1 — Upload Documents")
    col_a, col_p, col_s = st.columns(3)

    with col_a:
        st.markdown("**📄 Aadhaar Card**")
        aadhaar_file = st.file_uploader(
            "Upload Aadhaar card photo",
            type=IMG_TYPES,
            key="idv_aadhaar",
            label_visibility="collapsed",
        )

    with col_p:
        st.markdown("**💳 PAN Card**")
        pan_file = st.file_uploader(
            "Upload PAN card photo",
            type=IMG_TYPES,
            key="idv_pan",
            label_visibility="collapsed",
        )

    with col_s:
        st.markdown("**🤳 Selfie Photo**")
        selfie_file = st.file_uploader(
            "Upload selfie",
            type=IMG_TYPES,
            key="idv_selfie",
            label_visibility="collapsed",
        )

    selfie_bytes: bytes | None = None
    selfie_name = "selfie.jpg"
    if selfie_file:
        selfie_bytes = selfie_file.getvalue()
        selfie_name = selfie_file.name
    else:
        with col_s:
            st.caption("No selfie uploaded — take a photo with your camera:")
            cam = st.camera_input(
                "Take photo", key="idv_camera", label_visibility="collapsed"
            )
            if cam:
                selfie_bytes = cam.getvalue()
                selfie_name = "camera_capture.jpg"

    # ── Step 2: Parse & validate uploaded documents ────────────────────────
    aadhaar_qr: Dict[str, Any] = {}
    pan_data: Dict[str, Any] = {}
    pan_validation: Dict[str, Any] = {}

    if aadhaar_file:
        _a_key = f"_idv_qr_{aadhaar_file.size}"
        if _a_key not in st.session_state:
            with st.spinner("Scanning Aadhaar QR code..."):
                st.session_state[_a_key] = _parse_aadhaar_qr(aadhaar_file.getvalue())
        aadhaar_qr = st.session_state[_a_key]

        with col_a:
            st.image(
                aadhaar_file.getvalue(),
                caption="Aadhaar card preview",
                use_container_width=True,
            )
            if aadhaar_qr.get("qr_found") is False:
                st.warning(
                    "No QR code detected. Ensure the QR code is visible and the image is clear."
                )
            elif aadhaar_qr.get("qr_type") == "xml":
                st.success("QR decoded successfully")
                st.markdown(
                    f"**Name:** {aadhaar_qr.get('name', '—')}  \n"
                    f"**DOB/YOB:** {aadhaar_qr.get('dob') or aadhaar_qr.get('yob', '—')}  \n"
                    f"**Gender:** {aadhaar_qr.get('gender', '—')}  \n"
                    f"**District:** {aadhaar_qr.get('dist', '—')}, "
                    f"{aadhaar_qr.get('state', '—')}"
                )
            elif aadhaar_qr.get("qr_type") == "secure":
                st.info(aadhaar_qr.get("note", "Secure QR detected."))
            elif not aadhaar_qr:
                st.warning("QR scan error — check that OpenCV is available.")

    if pan_file:
        _p_key = f"_idv_pan_{pan_file.size}"
        if _p_key not in st.session_state:
            with st.spinner("Reading PAN card..."):
                st.session_state[_p_key] = _extract_pan_info(pan_file.getvalue())
        pan_data = st.session_state[_p_key]

        with col_p:
            st.image(
                pan_file.getvalue(),
                caption="PAN card preview",
                use_container_width=True,
            )
            extracted_pan = pan_data.get("pan_number", "")
            if extracted_pan:
                pan_validation = _validate_pan(extracted_pan)
                if pan_validation["valid"]:
                    st.success(
                        f"PAN: **{extracted_pan}**  \n"
                        f"Entity type: {pan_validation['entity_type']}"
                    )
                else:
                    st.warning(
                        f"PAN read: `{extracted_pan}` — "
                        f"{pan_validation.get('error', '')}"
                    )
            else:
                st.info(
                    "PAN number not detected via OCR. You can enter it manually below."
                )
            if pan_data.get("name"):
                st.caption(f"Name on PAN card: **{pan_data['name']}**")

    if selfie_bytes:
        with col_s:
            st.image(selfie_bytes, caption="Selfie preview", use_container_width=True)

    # ── Step 2 — Document Cross-Checks and PAN Layers ─────────────────────
    if aadhaar_file or pan_file:
        st.divider()
        st.subheader("Step 2 — Document Cross-Checks")
        chk_cols = st.columns(2)

        aadhaar_name = (
            aadhaar_qr.get("name", "") if aadhaar_qr.get("qr_type") == "xml" else ""
        )
        pan_name = pan_data.get("name", "")

        with chk_cols[0]:
            if aadhaar_name and pan_name:
                matched, sim = _names_match(aadhaar_name, pan_name)
                if matched:
                    st.success(
                        f"**Name Match: PASS** ({sim * 100:.0f}%)  \n"
                        f"Aadhaar QR: *{aadhaar_name}*  \n"
                        f"PAN card: *{pan_name}*"
                    )
                else:
                    st.error(
                        f"**Name Mismatch: FAIL** ({sim * 100:.0f}% overlap)  \n"
                        f"Aadhaar QR: *{aadhaar_name}*  \n"
                        f"PAN card: *{pan_name}*  \n"
                        "Names on Aadhaar and PAN card do not match."
                    )
            elif aadhaar_name:
                st.info(
                    f"Name from Aadhaar QR: **{aadhaar_name}**  \n"
                    "(Upload PAN card to cross-check name)"
                )
            elif pan_name:
                st.info(
                    f"Name from PAN card: **{pan_name}**  \n"
                    "(Aadhaar QR needed to cross-check name)"
                )
            else:
                st.caption(
                    "Upload both Aadhaar and PAN cards to check name consistency."
                )

        with chk_cols[1]:
            if pan_validation.get("valid") and aadhaar_name:
                surname_initial = pan_validation.get("surname_initial", "")
                aadhaar_surname_initial = (
                    aadhaar_name.strip().split()[-1][0].upper()
                    if aadhaar_name.strip().split()
                    else ""
                )
                if surname_initial and aadhaar_surname_initial:
                    if surname_initial == aadhaar_surname_initial:
                        st.success(
                            f"**PAN Surname Check: PASS**  \n"
                            f"PAN 5th char '{surname_initial}' matches Aadhaar "
                            f"surname initial '{aadhaar_surname_initial}'"
                        )
                    else:
                        st.warning(
                            f"**PAN Surname Check: MISMATCH**  \n"
                            f"PAN 5th char '{surname_initial}' vs Aadhaar "
                            f"surname initial '{aadhaar_surname_initial}'  \n"
                            "This may indicate the PAN belongs to a different person."
                        )
            elif pan_validation.get("valid"):
                st.info(
                    f"**PAN Format: VALID**  \nEntity type: {pan_validation['entity_type']}"
                )
            elif pan_file and not pan_validation.get("valid") and pan_data.get("pan_number"):
                pv = _validate_pan(pan_data.get("pan_number", ""))
                if not pv.get("valid"):
                    st.warning(f"**PAN Format: INVALID** — {pv.get('error', '')}")

        # Layered PAN fraud analysis
        if pan_file:
            st.divider()
            _render_pan_layers(
                pan_data,
                pan_validation,
                pan_file.getvalue(),
                selfie_bytes,
                aadhaar_name,
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
        aadhaar_qr.get("name", "") if aadhaar_qr.get("qr_type") == "xml" else ""
    )
    _qr_name_parts = _qr_name.strip().split(maxsplit=1)
    _default_fn = _qr_name_parts[0] if _qr_name_parts else ""
    _default_ln = _qr_name_parts[1] if len(_qr_name_parts) > 1 else ""
    _default_pan = pan_data.get("pan_number", "")
    _default_aadh = aadhaar_qr.get("uid", "")

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
    if _DB_IMPORTS_OK and db_available():
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

                face_result = compare_faces(aadhaar_file.getvalue(), selfie_bytes)

            st.subheader("Verification Result")

            if "error" in face_result:
                st.error(f"Face matching failed: {face_result['error']}")
            else:
                score = face_result["display_score"]
                is_match = face_result["match"]
                selected_entity_ref = forced_ref

                r1, r2 = st.columns(2)
                with r1:
                    st.image(
                        face_result["doc_annotated_rgb"],
                        caption="Face detected on Aadhaar",
                        use_container_width=True,
                    )
                with r2:
                    st.image(
                        face_result["selfie_annotated_rgb"],
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
                    f"Cosine similarity: {face_result['confidence']:.3f} "
                    f"(threshold: {face_result['threshold']:.2f})"
                )

                if _DB_IMPORTS_OK and db_available():
                    db_payload = {
                        k: v
                        for k, v in face_result.items()
                        if k not in ("doc_annotated_rgb", "selfie_annotated_rgb")
                    }
                    for k, v in list(db_payload.items()):
                        if hasattr(v, "item"):
                            db_payload[k] = v.item()

                    _ref_for_pdf = forced_ref or ""
                    _name_for_pdf = (
                        f"{extra_identity.get('first_name', '')} "
                        f"{extra_identity.get('last_name', '')}".strip()
                        if extra_identity
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
                            doc_filename=aadhaar_file.name,
                            selfie_filename=selfie_name,
                        )
                    except Exception:  # noqa: BLE001
                        pdf_bytes = None

                    saved = save_identity_check(
                        check_type="face_match",
                        result=db_payload,
                        forced_entity_ref=forced_ref,
                        extra_identity=extra_identity,
                        doc_filename=aadhaar_file.name,
                        selfie_filename=selfie_name,
                        pdf_bytes=pdf_bytes,
                    )
                    if saved:
                        selected_entity_ref = saved.get("entity_ref") or forced_ref
                        st.success(
                            f"Saved to database — Entity: **{saved.get('entity_ref', 'unlinked')}**, "
                            f"Record ID: {saved['id']}"
                        )
                    if pdf_bytes:
                        st.download_button(
                            "Download Identity Check Report (PDF)",
                            data=pdf_bytes,
                            file_name=f"identity_check_{selected_entity_ref or 'unlinked'}.pdf",
                            mime="application/pdf",
                            key="idv_pdf_dl",
                        )
                else:
                    st.warning(
                        "Database is offline — result not persisted. "
                        "Connect PostgreSQL to save results."
                    )

                if selected_entity_ref and _DB_IMPORTS_OK and db_available():
                    st.divider()
                    st.subheader(
                        f"Previous Identity Checks for {selected_entity_ref}"
                    )
                    checks = get_entity_identity_checks(selected_entity_ref)
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
