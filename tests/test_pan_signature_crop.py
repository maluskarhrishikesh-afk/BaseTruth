"""
Standalone test: crop the handwritten signature from a PAN card image.

Run from the project root:
    python tests/test_pan_signature_crop.py

What this script does
---------------------
1. Loads tests/sample/pan_card.jpg (or any path you pass via --image).
2. Sends the full image to Gemma4 (via Ollama) and asks it to return the
   pixel-precise bounding box of ONLY the handwritten signature strokes.
3. Crops that region, then tightens further to actual ink contours using
   OpenCV so there is zero extra text / background captured.
4. Saves two files next to the source image:
     <name>_sig_raw.jpg   -- the region Gemma4 picked (expanded slightly)
     <name>_sig_tight.jpg -- the final tight-cropped signature (white bg)
5. Prints the bounding-box coordinates Gemma4 returned for inspection.

Works for PAN cards, signed cheques, and any other document -- Gemma4
figures out the layout automatically from the image.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys

# ---------------------------------------------------------------------------
# Allow running directly without installing the package
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

# ---------------------------------------------------------------------------
# Gemma4 / Ollama helpers (re-use your existing probe/select helpers)
# ---------------------------------------------------------------------------

try:
    from basetruth.integrations.ollama import (
        probe_ollama,
        select_ollama_model,
        _extract_json_object,
        OLLAMA_CONNECT_TIMEOUT_SEC,
        OLLAMA_READ_TIMEOUT_SEC,
    )
    _OLLAMA_HELPERS_OK = True
except ImportError:
    _OLLAMA_HELPERS_OK = False

import requests  # noqa: E402 -- always available


# ---------------------------------------------------------------------------
# Gemma4 prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a precise document layout analyser. "
    "Return strict JSON only. No commentary, no markdown."
)

# Key design decisions in this prompt:
#   1. Asks for coordinates relative to the FULL image (including any background
#      around the card), so the model doesn't assume the card fills the frame.
#   2. Explicitly warns against clipping the left edge of the signature.
#   3. Only 3% padding allowed -- we rely on OpenCV to add pixel-precise padding.
#   4. Includes a "note" field so we can debug model reasoning.
_USER_PROMPT = (
    "Look at this document image. There is a handwritten signature somewhere "
    "on the document (often a cursive ink stroke below printed text).\n\n"
    "Locate ONLY the handwritten signature strokes -- do NOT include:\n"
    "  - the printed word 'Signature' (label below the strokes)\n"
    "  - the PAN number or any other printed text\n"
    "  - the hologram or photo area\n\n"
    "IMPORTANT: Signatures often start very close to the left margin. "
    "Make sure x1 captures the very beginning of the LEFTMOST ink stroke. "
    "When in doubt, bias x1 slightly to the left.\n\n"
    "Return a JSON object with the bounding box as fractions of the "
    "FULL image dimensions (0.0 = left/top edge, 1.0 = right/bottom edge):\n\n"
    "{\n"
    '  "x1": <left edge of ink strokes / image width>,\n'
    '  "y1": <top edge of ink strokes / image height>,\n'
    '  "x2": <right edge of ink strokes / image width>,\n'
    '  "y2": <bottom edge of ink strokes / image height>,\n'
    '  "confidence": <0.0-1.0>,\n'
    '  "note": "<one sentence: where signature is and what it looks like>"\n'
    "}\n\n"
    "Rules:\n"
    "- All values must be between 0.0 and 1.0.\n"
    "- x1 < x2 and y1 < y2.\n"
    "- Max 3% padding around ink strokes -- tight fit only.\n"
    "- If no signature found, set all coordinates and confidence to 0.0.\n"
    "- Output JSON only -- no extra text."
)


# ---------------------------------------------------------------------------
# Gemma4 bounding-box query
# ---------------------------------------------------------------------------

def ask_gemma4_for_sig_box(img_bytes: bytes) -> dict:
    """
    Send the image to Gemma4 and get back the signature bounding box.

    Returns a dict with keys x1, y1, x2, y2 (all 0.0-1.0) on success,
    or an empty dict on failure.
    """
    if not _OLLAMA_HELPERS_OK:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = "gemma4:latest"
        timeout = (10, 120)
    else:
        base_url, models, attempted = probe_ollama()
        if not base_url:
            print(f"[ERROR] Ollama not reachable.  Tried: {attempted}")
            return {}
        model = select_ollama_model(models)
        timeout = (OLLAMA_CONNECT_TIMEOUT_SEC, OLLAMA_READ_TIMEOUT_SEC)
        print(f"[INFO]  Using Ollama model: {model}  at {base_url}")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _USER_PROMPT,
                "images": [base64.b64encode(img_bytes).decode("ascii")],
            },
        ],
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        resp = requests.post(  # nosemgrep: basetruth-ssrf
            f"{base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ERROR] Ollama request failed: {exc}")
        return {}

    raw = str(resp.json().get("message", {}).get("content", "")).strip()
    print(f"\n[GEMMA4 RAW RESPONSE]\n{raw}\n")

    json_text = (_extract_json_object(raw) if _OLLAMA_HELPERS_OK else _simple_extract_json(raw))
    if not json_text:
        print("[WARN]  No JSON found in Gemma4 response.")
        return {}

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        print(f"[WARN]  JSON parse error: {exc}")
        return {}

    x1 = float(data.get("x1", 0))
    y1 = float(data.get("y1", 0))
    x2 = float(data.get("x2", 0))
    y2 = float(data.get("y2", 0))
    conf = float(data.get("confidence", 0))
    note = str(data.get("note", ""))

    print(f"[GEMMA4] box = x1={x1:.3f} y1={y1:.3f} x2={x2:.3f} y2={y2:.3f}  conf={conf:.2f}")
    if note:
        print(f"[GEMMA4] note: {note}")

    width_f  = x2 - x1
    height_f = y2 - y1

    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        print("[WARN]  Gemma4 returned out-of-range or degenerate coordinates.")
        return {}

    if width_f < 0.01 or height_f < 0.005:
        print("[WARN]  Gemma4 box is too small -- likely a failed detection.")
        return {}

    # Reject boxes that span most of the image (model confused)
    if width_f > 0.65 or height_f > 0.45:
        print(f"[WARN]  Gemma4 box is too large (w={width_f:.2f}, h={height_f:.2f}) -- ignoring.")
        return {}

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf}


def _simple_extract_json(text: str) -> str:
    """Minimal JSON extractor when the ollama helpers aren't available."""
    s = text.find("{")
    e = text.rfind("}")
    return text[s:e + 1] if s != -1 and e != -1 and e >= s else ""


# ---------------------------------------------------------------------------
# Core cropping logic
# ---------------------------------------------------------------------------

def crop_signature(
    img_bytes: bytes,
    box: dict,            # normalised {x1, y1, x2, y2}
    *,
    expand: float = 0.03,              # safety margin: 3% on each side
    min_contour_area_frac: float = 0.0008,  # noise floor
    pad_px: int = 10,                  # pixel padding around tight contour rect
    max_output_width: int = 400,       # max display width of output image
) -> tuple[bytes | None, bytes | None]:
    """
    Crop the signature from the image using the given normalised bounding box.

    Returns (raw_crop_bytes, tight_crop_bytes).
    raw_crop_bytes   -- the coarse region expanded by `expand` margin.
    tight_crop_bytes -- tightened to actual ink contours, white background.
    Either can be None on failure.
    """
    import cv2
    import numpy as np

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    h, w = img.shape[:2]
    print(f"[INFO]  Image: {w}x{h} px")

    # -- Step 1: Coarse crop (Gemma4 box + safety margin) -------------------
    # Bias x1 extra to the left (4% extra) to compensate for the common
    # Gemma4 tendency to start the box slightly right of the first stroke.
    x1 = max(0.0, box["x1"] - expand - 0.04)
    y1 = max(0.0, box["y1"] - expand)
    x2 = min(1.0, box["x2"] + expand)
    y2 = min(1.0, box["y2"] + expand)

    px0 = int(w * x1); py0 = int(h * y1)
    px1 = int(w * x2); py1 = int(h * y2)

    print(f"[INFO]  Coarse crop region: x={px0}-{px1}, y={py0}-{py1} px")

    if px1 <= px0 or py1 <= py0:
        return None, None

    coarse = img[py0:py1, px0:px1].copy()
    ok_raw, buf_raw = cv2.imencode(".jpg", coarse, [cv2.IMWRITE_JPEG_QUALITY, 92])
    raw_bytes = bytes(buf_raw) if ok_raw else None

    # -- Step 2: Grayscale + denoise ----------------------------------------
    gray = cv2.cvtColor(coarse, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # -- Step 3: Adaptive threshold to isolate dark ink strokes --------------
    # blockSize=31 works well for camera-captured PAN cards.
    # BINARY_INV = ink becomes white (foreground) for contour detection.
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=10,
    )

    # -- Step 4: Close small gaps in stroke segments -------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # -- Step 5: Find contours and reject noise / card-edge artefacts --------
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ch, cw = coarse.shape[:2]
    crop_area = ch * cw
    min_area  = max(20, crop_area * min_contour_area_frac)

    # Reject contours whose centroid is in the bottom 15% of the crop;
    # those are typically card-edge or "Signature" label artefacts.
    y_cutoff = ch * 0.85

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cy = M["m01"] / M["m00"]
        if cy > y_cutoff:
            continue
        valid.append(c)

    if not valid:
        print("[WARN]  No valid ink contours found; using raw crop.")
        return raw_bytes, raw_bytes

    # -- Step 6: Union bounding rect of all valid contours ------------------
    all_pts        = np.concatenate(valid, axis=0)
    bx, by, bw, bh = cv2.boundingRect(all_pts)

    # Pixel padding around actual ink
    tx  = max(0,  bx - pad_px)
    ty  = max(0,  by - pad_px)
    tx2 = min(cw, bx + bw + pad_px)
    ty2 = min(ch, by + bh + pad_px)

    print(f"[INFO]  Ink contour union: bx={bx},by={by},bw={bw},bh={bh}  "
          f"tight_region: ({tx},{ty})-({tx2},{ty2}) in {cw}x{ch} crop")

    if (tx2 - tx) < 15 or (ty2 - ty) < 6:
        print("[WARN]  Tight crop too small; using raw crop.")
        return raw_bytes, raw_bytes

    tight = coarse[ty:ty2, tx:tx2]

    # -- Step 7: White background, preserve ink tones -----------------------
    tight_gray     = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
    tight_denoised = cv2.fastNlMeansDenoising(tight_gray, h=8)
    _, mask = cv2.threshold(tight_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # White-out background pixels, preserve ink pixel values
    clean = np.where(mask == 255, np.uint8(255), tight_denoised)

    # -- Step 8: Resize to a realistic display size -------------------------
    sig_h, sig_w = clean.shape[:2]
    if sig_w > max_output_width:
        scale = max_output_width / sig_w
        new_w = max_output_width
        new_h = max(1, int(sig_h * scale))
        clean = cv2.resize(clean, (new_w, new_h), interpolation=cv2.INTER_AREA)

    print(f"[INFO]  Final tight crop: {clean.shape[1]}x{clean.shape[0]} px  "
          f"(raw crop was {cw}x{ch} px)")

    ok_tight, buf_tight = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, 95])
    tight_bytes = bytes(buf_tight) if ok_tight else None

    return raw_bytes, tight_bytes


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop a handwritten signature from a document image using Gemma4."
    )
    parser.add_argument(
        "--image",
        default=os.path.join(os.path.dirname(__file__), "sample", "pan_card.jpg"),
        help="Path to the document image (default: tests/sample/pan_card.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save output files (default: same folder as image)",
    )
    parser.add_argument(
        "--no-gemma4",
        action="store_true",
        help="Skip Gemma4 and use the hardcoded fallback box (offline testing)",
    )
    args = parser.parse_args()

    img_path = os.path.abspath(args.image)
    if not os.path.exists(img_path):
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    out_dir   = args.output_dir or os.path.dirname(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    print(f"[INFO]  Loading image: {img_path}")
    with open(img_path, "rb") as fh:
        img_bytes = fh.read()

    # -- Gemma4 bounding-box query ------------------------------------------
    box: dict = {}

    if not args.no_gemma4:
        box = ask_gemma4_for_sig_box(img_bytes)

    if not box:
        print("[INFO]  Using hardcoded fallback box (full-image fractions).")
        print("        Manually calibrated for tests/sample/pan_card.jpg (3120x4160 px).")
        print("        Signature 'Maluskar' occupies approx x=440-1050, y=3510-3700 px.")
        # Measured from the actual image:
        #   PAN card sits at ~x:420-2750, y:510-3310
        #   Signature 'Maluskar' ink: x~440-1050, y~3510-3700  (full image coords)
        box = {"x1": 0.141, "y1": 0.843, "x2": 0.336, "y2": 0.890, "confidence": 0.0}

    # -- Crop ---------------------------------------------------------------
    raw_bytes, tight_bytes = crop_signature(img_bytes, box)

    # -- Save outputs -------------------------------------------------------
    raw_path   = os.path.join(out_dir, f"{base_name}_sig_raw.jpg")
    tight_path = os.path.join(out_dir, f"{base_name}_sig_tight.jpg")

    if raw_bytes:
        with open(raw_path, "wb") as fh:
            fh.write(raw_bytes)
        print(f"[SAVED] Raw crop   -> {raw_path}")
    else:
        print("[WARN]  Raw crop failed.")

    if tight_bytes:
        with open(tight_path, "wb") as fh:
            fh.write(tight_bytes)
        print(f"[SAVED] Tight crop -> {tight_path}")
    else:
        print("[WARN]  Tight crop failed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
