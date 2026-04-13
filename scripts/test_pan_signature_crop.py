"""Standalone test: Gemma4-guided PAN card signature extraction.

Purpose
-------
Use Gemma4 (via Ollama) to locate the signature region inside any PAN card
image, then crop and clean that region with OpenCV.  Fixed pixel ratios are
NOT used because uploaded images can be any resolution, rotation, or crop.

Strategy
--------
1. Send the full PAN card image to Gemma4 with a prompt that asks for the
   signature bounding box as normalised fractions (0.0–1.0) so the answer is
   resolution-independent.
2. Parse the JSON response to get {x1, y1, x2, y2} fractions.
3. Crop that region from the original image.
4. Clean up with OpenCV (grayscale, denoise, adaptive threshold, tighten to
   ink bounding box) and save the result as JPEG.

Fallback
--------
If Gemma4 is unavailable or returns unusable coordinates, a fixed-ratio region
known to work for standard camera-captured Indian PAN cards is used instead.

Usage
-----
    python scripts/test_pan_signature_crop.py

Outputs (written to artifacts/debug/)
--------------------------------------
    pan_sig_gemma4_box.jpg  — original image with Gemma4-predicted box drawn in green
    pan_sig_coarse.jpg      — raw crop of the located region (pre-cleanup)
    pan_sig_extracted.jpg   — final cleaned signature (what gets saved to MinIO)
"""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

# ─── PATHS ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = ROOT / "tests" / "sample" / "pan_card.jpg"
OUTPUT_DIR   = ROOT / "artifacts" / "debug"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── OLLAMA CONFIG ──────────────────────────────────────────────────────────
OLLAMA_BASE    = "http://localhost:11434"
OLLAMA_MODEL   = "gemma4:latest"
CONNECT_TIMEOUT = 5
READ_TIMEOUT    = 600   # first load of gemma4 can take time; match production setting

# ─── FALLBACK RATIOS (used only if Gemma4 is unavailable) ───────────────────
# These are calibrated against a camera-captured PAN card with ~15 % background
# margin on all sides (image size 3120×4160 or similar).
FALLBACK_BOX = {"x1": 0.04, "y1": 0.62, "x2": 0.40, "y2": 0.73}

# Safety margin added to every edge of the Gemma4 box before cropping.
# Gemma4 can underestimate the bounding box by a few percentage points, so we
# expand each edge outward before feeding the region to the cleanup pipeline.
# The cleanup (contour tightening) will then re-find the tight ink boundary.
GEMMA4_EXPAND_MARGIN = 0.05  # 5 % of image dimension on every side

# ─── CLEANUP PARAMS ─────────────────────────────────────────────────────────
# Adaptive threshold block size — should be odd; larger = handles more
# lighting variation across the crop.
ADAPTIVE_BLOCK = 51
# C constant for adaptive threshold — 12 is sensitive enough to capture faint
# ink without picking up card-border or background texture noise.
ADAPTIVE_C     = 12
# Minimum contour area as a fraction of the coarse crop area (noise filter)
MIN_CONTOUR_FRAC = 0.002
# Padding (pixels) added around the tight ink bounding box
CROP_PAD_X = 16
CROP_PAD_Y = 12
# Only consider contours whose centroid falls in the upper fraction of the
# coarse crop.  The card edge (blue→beige transition) and bottom background
# are in the lower ~35 % of the expanded crop; the signature ink is near the
# top.  This cutoff prevents card-edge noise from widening the tight bbox.
CROP_INK_Y_FRAC = 0.65  # ignore contours with centroid below 65 % of crop height


# ─── GEMMA4 PROMPT ───────────────────────────────────────────────────────────
# Ask Gemma4 to return the signature's bounding box as normalised fractions of
# the full image width and height.  This makes the answer resolution-independent.
_SIG_SYSTEM_PROMPT = (
    "You are a document layout analyser specialising in Indian identity documents. "
    "Return strict JSON only. Do not add commentary or markdown."
)

_SIG_USER_PROMPT = """\
Look at this Indian PAN card image.

Locate the handwritten applicant signature — it appears as cursive/handwritten
ink strokes below the PAN number and above or next to the printed label
"Signature".

Return a JSON object with the normalised bounding box of the signature region
(NOT the "Signature" label text — the actual handwritten strokes):

{
  "x1": <left edge  as fraction of image width,  0.0–1.0>,
  "y1": <top edge   as fraction of image height, 0.0–1.0>,
  "x2": <right edge as fraction of image width,  0.0–1.0>,
  "y2": <bottom edge as fraction of image height,0.0–1.0>,
  "confidence": <0.0–1.0, how confident you are>
}

Rules:
- All four values must be between 0.0 and 1.0.
- x1 < x2 and y1 < y2.
- Include 10–15 % padding around the signature strokes so nothing is clipped.
- Output JSON only — no extra text.
""".strip()


# ────────────────────────────────────────────────────────────────────────────
# Step 1 — Ask Gemma4 for the bounding box
# ────────────────────────────────────────────────────────────────────────────

def locate_signature_with_gemma4(img_bytes: bytes) -> Optional[Dict[str, float]]:
    """Ask Gemma4 to identify the signature bounding box in the PAN card image.

    Returns a dict with keys x1, y1, x2, y2 (all in 0.0–1.0 range) or None
    if Gemma4 is not reachable or returns unusable coordinates.
    """
    payload = {
        "model": OLLAMA_MODEL,
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

    try:
        print("  Sending image to Gemma4 for signature location...")
        resp = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  Gemma4 unreachable: {exc}")
        return None

    raw = str(resp.json().get("message", {}).get("content", "")).strip()
    print(f"  Gemma4 raw response:\n    {raw[:300]}")

    # Extract the first JSON object from the response
    match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not match:
        print("  Could not find JSON object in response.")
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        print(f"  JSON parse error: {exc}")
        return None

    # Validate all four coordinate keys are present and sensible
    try:
        box = {
            "x1": float(data["x1"]),
            "y1": float(data["y1"]),
            "x2": float(data["x2"]),
            "y2": float(data["y2"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  Missing or invalid coordinate key: {exc}")
        return None

    # Sanity-check: all values in [0, 1] and x1<x2, y1<y2
    if not (0 <= box["x1"] < box["x2"] <= 1 and 0 <= box["y1"] < box["y2"] <= 1):
        print(f"  Gemma4 returned invalid box (values out of range or inverted): {box}")
        return None

    # Guard against a box that is clearly too large (whole card) or too small
    width  = box["x2"] - box["x1"]
    height = box["y2"] - box["y1"]
    if width > 0.7 or height > 0.5:
        print(f"  Gemma4 box is suspiciously large (w={width:.2f}, h={height:.2f}); using fallback.")
        return None
    if width < 0.03 or height < 0.02:
        print(f"  Gemma4 box is suspiciously small (w={width:.2f}, h={height:.2f}); using fallback.")
        return None

    conf = data.get("confidence", "?")
    print(f"  Gemma4 box: x1={box['x1']:.3f} y1={box['y1']:.3f} x2={box['x2']:.3f} y2={box['y2']:.3f} (confidence={conf})")

    # Expand every edge outward so minor Gemma4 under-estimates don't clip ink.
    # The cleanup pipeline will re-tighten to actual ink contours, so extra
    # background around the signature is harmless.
    m = GEMMA4_EXPAND_MARGIN
    box = {
        "x1": max(0.0, box["x1"] - m),
        "y1": max(0.0, box["y1"] - m),
        "x2": min(1.0, box["x2"] + m),
        "y2": min(1.0, box["y2"] + m),
    }
    print(f"  Expanded box (+{m:.0%} margin): x1={box['x1']:.3f} y1={box['y1']:.3f} x2={box['x2']:.3f} y2={box['y2']:.3f}")
    return box


# ────────────────────────────────────────────────────────────────────────────
# Step 2 — Crop and clean the located region
# ────────────────────────────────────────────────────────────────────────────

def crop_and_clean(img_bytes: bytes, box: Dict[str, float]) -> Optional[bytes]:
    """Crop the bounding box from the image and produce a clean signature JPEG.

    The cleanup pipeline:
    1. Convert the coarse crop to greyscale.
    2. Denoise lightly (fastNlMeansDenoising) to remove sensor/JPEG noise.
    3. Apply adaptive Gaussian threshold to isolate ink strokes regardless of
       local background brightness — this works even when the crop straddles
       the card edge or has uneven lighting from a camera capture.
    4. Find contour groups; dilate slightly to merge nearby ink strokes.
    5. Compute the tight bounding rectangle around all valid contours, add a
       small margin, and crop again to produce the minimal clean output.
    6. Whiten the background using the same Otsu mask so the final image has
       clean white paper with dark ink (ideal for signature comparison algorithms).
    """
    import cv2
    import numpy as np

    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    px0 = int(w * box["x1"])
    py0 = int(h * box["y1"])
    px1 = int(w * box["x2"])
    py1 = int(h * box["y2"])

    if px1 <= px0 or py1 <= py0:
        return None

    coarse = img[py0:py1, px0:px1]

    # ── Denoise & adaptive threshold ─────────────────────────────────────────
    gray     = cv2.cvtColor(coarse, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold: evaluates small neighbourhoods independently so ink
    # is separated from both card-blue background and surrounding beige mat.
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK,
        C=ADAPTIVE_C,
    )

    # Dilate to merge nearby ink strokes into a single contour blob
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # ── Find contours and filter noise ───────────────────────────────────────
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coarse_area = coarse.shape[0] * coarse.shape[1]
    coarse_h    = coarse.shape[0]
    min_area    = max(30, coarse_area * MIN_CONTOUR_FRAC)
    y_cutoff    = coarse_h * CROP_INK_Y_FRAC  # ignore anything in the lower strip

    valid_contours = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        # Filter out anything in the lower strip (card edge, background noise)
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cy = M["m01"] / M["m00"]  # contour centroid y-coordinate
        if cy > y_cutoff:
            continue
        valid_contours.append(c)

    # ── Tight bounding box around all ink contours ───────────────────────────
    if valid_contours:
        all_pts        = __import__("numpy").concatenate(valid_contours, axis=0)
        bx, by, bw, bh = cv2.boundingRect(all_pts)
        tx  = max(0, bx - CROP_PAD_X)
        ty  = max(0, by - CROP_PAD_Y)
        tx2 = min(coarse.shape[1], bx + bw + CROP_PAD_X)
        ty2 = min(coarse.shape[0], by + bh + CROP_PAD_Y)
        if (tx2 - tx) >= 20 and (ty2 - ty) >= 6:
            tight = coarse[ty:ty2, tx:tx2]
        else:
            tight = coarse
    else:
        tight = coarse  # no ink found — fall back to full coarse region

    # ── Produce clean white-background grayscale output ──────────────────────
    # Re-run the pipeline on the tight crop so the final output is clean.
    tight_gray     = cv2.cvtColor(tight, cv2.COLOR_BGR2GRAY)
    denoised_tight = cv2.fastNlMeansDenoising(tight_gray, h=8)
    _, mask        = cv2.threshold(denoised_tight, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Where the mask says "background" (white), force pure white; keep ink pixels as-is
    result = __import__("numpy").where(mask == 255, __import__("numpy").uint8(255), denoised_tight)

    ok, buf = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return bytes(buf) if ok else None


# ────────────────────────────────────────────────────────────────────────────
# Debug helpers
# ────────────────────────────────────────────────────────────────────────────

def save_annotated_box(img_bytes: bytes, box: Dict[str, float], source: str) -> None:
    """Draw the predicted bounding box on the original image and save it."""
    import cv2
    import numpy as np

    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR).copy()
    h, w  = img.shape[:2]

    x0 = int(w * box["x1"]); y0 = int(h * box["y1"])
    x1 = int(w * box["x2"]); y1 = int(h * box["y2"])

    color     = (0, 200, 0) if source == "gemma4" else (0, 100, 255)
    thickness = max(3, w // 400)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    label = f"[{source}]  ({box['x1']:.2f},{box['y1']:.2f}) -> ({box['x2']:.2f},{box['y2']:.2f})"
    font_scale = max(0.7, w / 2400)
    cv2.putText(img, label, (x0 + 4, max(y0 - 12, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    out = OUTPUT_DIR / "pan_sig_gemma4_box.jpg"
    cv2.imwrite(str(out), img)
    print(f"[saved] {out}")


def save_coarse(img_bytes: bytes, box: Dict[str, float]) -> None:
    """Save just the raw coarse crop for inspection."""
    import cv2
    import numpy as np

    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w  = img.shape[:2]
    coarse = img[int(h * box["y1"]):int(h * box["y2"]),
                 int(w * box["x1"]):int(w * box["x2"])]
    out = OUTPUT_DIR / "pan_sig_coarse.jpg"
    cv2.imwrite(str(out), coarse)
    print(f"[saved] {out}  ({coarse.shape[1]}×{coarse.shape[0]} px)")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not SAMPLE_IMAGE.exists():
        print(f"ERROR: sample image not found at {SAMPLE_IMAGE}")
        sys.exit(1)

    img_bytes = SAMPLE_IMAGE.read_bytes()
    print(f"Loaded: {SAMPLE_IMAGE}  ({len(img_bytes):,} bytes)")

    # ── Step 1: Ask Gemma4 for the signature location ────────────────────────
    print("\n[Step 1] Locating signature with Gemma4...")
    box = locate_signature_with_gemma4(img_bytes)
    source = "gemma4"

    if box is None:
        print("  Gemma4 failed — using fixed FALLBACK_BOX.")
        box = FALLBACK_BOX
        source = "fallback"

    # ── Step 2: Draw annotated preview ──────────────────────────────────────
    print(f"\n[Step 2] Saving annotated preview (source={source})...")
    save_annotated_box(img_bytes, box, source)

    # ── Step 3: Save raw coarse crop ────────────────────────────────────────
    print("\n[Step 3] Saving coarse crop...")
    save_coarse(img_bytes, box)

    # ── Step 4: Crop and clean ───────────────────────────────────────────────
    print("\n[Step 4] Cropping and cleaning signature...")
    sig_bytes = crop_and_clean(img_bytes, box)

    if sig_bytes:
        out = OUTPUT_DIR / "pan_sig_extracted.jpg"
        out.write_bytes(sig_bytes)
        print(f"[saved] {out}  ({len(sig_bytes):,} bytes)  ← this is what gets stored in MinIO")
        print()
        print("SUCCESS — open these files to verify:")
        print("  artifacts/debug/pan_sig_gemma4_box.jpg  : original with bounding box")
        print("  artifacts/debug/pan_sig_coarse.jpg      : raw crop (pre-cleanup)")
        print("  artifacts/debug/pan_sig_extracted.jpg   : final clean signature")
    else:
        print("FAILED — crop_and_clean() returned None.")
        sys.exit(1)


if __name__ == "__main__":
    main()
