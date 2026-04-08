"""image_forensics_detect.py — Full 11-layer image tampering detector.

This module exposes the complete forensic engine originally developed and
validated in ``tests/forensics_detect.py``.  It is the canonical source for
all image forensics in BaseTruth and is imported by the Bulk Scan pipeline
to populate ``scans.layered_analysis_json``.

Supported input formats: JPEG, PNG, TIFF, BMP, WebP.
For PDF inputs the caller must render page 1 to a temporary PNG first
(using PyMuPDF / fitz) before calling :func:`run_forensics`.

Techniques
----------
1  ELA               Error Level Analysis          (JPEG + PNG)
2  Metadata          EXIF + PNG tEXt chunks        (JPEG + PNG)
3  File entropy      Shannon entropy of raw bytes  (all)
4  Noise residual    Gaussian-blur high-freq noise (all)
5  DCT analysis      Double-compression comb       (JPEG only)
6  Clone detection   SIFT copy-move                (all)
7  Color anomaly     HSV chromatic outlier pixels  (all)
8  Edge density      Canny edge density heatmap    (all)
9  Saturation        Over-saturated tile map       (all)
10 Font consistency  Stroke-width / sharpness CV   (all — best on text docs)
11 AI artifact       FFT spectral grid analysis    (all)

Public API
----------
    from basetruth.analysis.image_forensics_detect import run_forensics

    result: dict = run_forensics(path="/abs/path/to/image.png")

The returned dict has the structure described in :func:`run_forensics`.
All functions degrade gracefully when optional libraries are absent.
"""
from __future__ import annotations

import io
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from basetruth.logger import get_logger

log = get_logger(__name__)

warnings.filterwarnings("ignore")

# ── Optional heavy imports — degrade gracefully ───────────────────────────────
# We try to load the image processing libraries. If any are missing the module
# still imports cleanly — forensics just returns a "UNAVAILABLE" result instead
# of crashing the whole application.
try:
    import numpy as np
    import cv2
    from PIL import Image, ImageChops
    from PIL.ExifTags import TAGS
    import exifread
    _FORENSICS_AVAILABLE = True
except ImportError as _imp_err:
    log.warning("image_forensics_detect: missing dependency — %s. Forensics disabled.", _imp_err)
    _FORENSICS_AVAILABLE = False
    np = None  # type: ignore[assignment]
    cv2 = None  # type: ignore[assignment]

# ── Configuration ──────────────────────────────────────────────────────────────
# ELA_QUALITY: the JPEG quality level we re-save at for Error Level Analysis.
# 75 is a good middle ground — low enough to reveal editing artefacts, high enough
# to keep the image readable for comparison.
ELA_QUALITY = 75
# ELA_AMPLIFY is kept for reference but not used in the current pipeline.
ELA_AMPLIFY = 10

# List of software names that indicate a human edited the image in a photo editor.
# If any of these appear in the image's hidden metadata tags, the document is flagged.
_EDIT_SOFTWARE = [
    "photoshop", "gimp", "lightroom", "paint", "snapseed", "affinity",
    "picsart", "pixelmator", "canva", "inkscape", "adobe", "capture one",
    "darktable", "rawtherapee", "medibang", "clip studio", "procreate",
    "preview", "irfanview", "xnview", "imagemagick", "pillow", "opencv",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_format(path: str) -> str:
    try:
        from PIL import Image as _PImage  # noqa: PLC0415
        return _PImage.open(path).format or (
            "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"
        )
    except Exception:
        return "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"


# ── Layer 1: ELA ───────────────────────────────────────────────────────────────

def _ela_analysis(image_path: str) -> Dict[str, Any]:
    """Error Level Analysis — detects regions re-saved at different quality.

    How it works:
    1. Re-save the image at a known lower quality (75%).
    2. Subtract the re-saved version from the original pixel-by-pixel.
    3. Any region that was already compressed before (i.e., pasted from elsewhere)
       produces much smaller differences — it 'glows' differently in the heatmap.
    4. We divide the image into 32x32 blocks. If more than 5% of blocks glow
       brighter than 2.5x the average, we flag the image as suspicious.
    """
    try:
        from PIL import Image, ImageChops  # noqa: PLC0415
        orig = Image.open(image_path).convert("RGB")
        buf = io.BytesIO()
        # Re-save at the reference quality to create a known baseline
        orig.save(buf, format="JPEG", quality=ELA_QUALITY)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")
        # Pixel-by-pixel difference between original and re-compressed version
        ela_arr = np.array(ImageChops.difference(orig, recompressed), dtype=np.float32)

        mean_ela = float(np.mean(ela_arr))
        h, w = ela_arr.shape[:2]
        block = 32  # 32x32 pixel tiles — small enough to catch localised edits
        high = total = 0
        for y in range(0, h - block, block):
            for x in range(0, w - block, block):
                # A block is "hot" if its brightness is more than 2.5x the image average
                if np.mean(ela_arr[y:y + block, x:x + block]) > mean_ela * 2.5:
                    high += 1
                total += 1
        suspicious_ratio = round(high / total, 4) if total else 0.0

        status = "SUSPICIOUS" if suspicious_ratio > 0.05 else "CLEAN"
        interpretation = (
            "HIGH — many blocks have anomalous re-compression levels (tampering likely)"
            if suspicious_ratio > 0.05
            else "LOW — uniform ELA across image (consistent with original)"
        )
        plain_english = (
            "When you save a JPEG image, tiny quality details are lost. If someone pastes "
            "new content onto an old image and saves it again, the pasted area 'glows' in "
            "the ELA heatmap because it hasn't degraded as much as the rest. "
            + ("This image has {:.1f}% of blocks glowing — above the 5% threshold.".format(suspicious_ratio * 100)
               if suspicious_ratio > 0.05
               else "This image has uniform ELA — a good sign of an untouched original.")
        )
        return {
            "name": "Error Level Analysis (ELA)",
            "status": status,
            "plain_english": plain_english,
            "metrics": {
                "mean_ela": round(mean_ela, 3),
                "max_ela": round(float(np.max(ela_arr)), 3),
                "std_ela": round(float(np.std(ela_arr)), 3),
                "suspicious_block_ratio": suspicious_ratio,
                "threshold_for_suspicious": 0.05,
                "interpretation": interpretation,
            },
        }
    except Exception as exc:
        log.warning("ELA analysis failed: %s", exc)
        return {"name": "Error Level Analysis (ELA)", "status": "ERROR", "error": str(exc), "metrics": {}}


# ── Layer 2: Metadata ──────────────────────────────────────────────────────────

def _metadata_analysis(image_path: str, fmt: str) -> Dict[str, Any]:
    """Inspect EXIF / PNG metadata for editing-software fingerprints.

    Every photo taken by a real camera contains hidden metadata tags:
    - Make / Model (which camera took it)
    - DateTimeOriginal (when the shutter was pressed)
    - Software (only present if the image was opened in an editor)

    We flag the image when:
    - A known editing tool name appears in the Software tag
    - There is no camera Make / Model (screenshot or synthetic image)
    - DateTimeOriginal is missing (common after editing tools strip it)
    - DateTimeOriginal differs from ImageDateTime (image was edited and re-saved)
    """
    try:
        from PIL import Image as _PImage  # noqa: PLC0415
        from PIL.ExifTags import TAGS as _TAGS  # noqa: PLC0415
        tags: Dict[str, str] = {}
        suspicious_flags: List[str] = []

        if fmt == "JPEG":
            with open(image_path, "rb") as f:
                raw = exifread.process_file(f, details=False, strict=False)
            for k, v in raw.items():
                tags[k] = str(v)
            software = tags.get("Image Software", "")
            make = tags.get("Image Make", "")
            model = tags.get("Image Model", "")
            datetime_orig = tags.get("EXIF DateTimeOriginal", "")
            datetime_img = tags.get("Image DateTime", "")
            if any(s in software.lower() for s in _EDIT_SOFTWARE):
                suspicious_flags.append(f"Editing software detected: {software!r}")
            if not make and not model:
                suspicious_flags.append("No camera Make/Model in EXIF (re-saved without camera metadata)")
            if not datetime_orig:
                suspicious_flags.append("No DateTimeOriginal (stripped or synthetically created)")
            if datetime_orig and datetime_img and datetime_orig != datetime_img:
                suspicious_flags.append(
                    f"DateTimeOriginal ({datetime_orig}) != ImageDateTime ({datetime_img}) — post-edit save"
                )
            if not raw:
                suspicious_flags.append("EXIF completely absent — metadata stripped (common after editing)")

        pil_img = _PImage.open(image_path)
        for k, v in (pil_img.info or {}).items():
            if isinstance(v, (str, bytes)):
                str_v = v.decode(errors="replace") if isinstance(v, bytes) else v
                if k not in tags:
                    tags[k] = str_v
        exif_raw = pil_img.getexif()
        if exif_raw:
            for tag_id, val in exif_raw.items():
                key = f"EXIF:{_TAGS.get(tag_id, tag_id)}"
                if key not in tags:
                    tags[key] = str(val)

        if fmt != "JPEG":
            for k, v in tags.items():
                if isinstance(v, str) and any(kw in k.lower() for kw in ("software", "creator", "comment")):
                    if any(s in v.lower() for s in _EDIT_SOFTWARE):
                        suspicious_flags.append(f"Editing tool in metadata: {k!r} = {v!r}")
            if not tags:
                suspicious_flags.append("No metadata whatsoever — all metadata stripped (common after editing)")
            elif not any(kw in k.lower() for kw in ("make", "model", "camera") for k in tags):
                suspicious_flags.append("No camera Make/Model — image may be screen-captured or re-exported")

        tamper_risk = "HIGH" if len(suspicious_flags) >= 2 else "MEDIUM" if suspicious_flags else "LOW"
        status = "SUSPICIOUS" if suspicious_flags else "CLEAN"
        plain_english = (
            "Every digital photo carries hidden tags (metadata) recording which camera took it and when. "
            + (f"This image has {len(suspicious_flags)} suspicious flag(s): {'; '.join(suspicious_flags[:2])}."
               if suspicious_flags
               else "No suspicious metadata flags found — looks like an unmodified original.")
        )
        return {
            "name": "Metadata / EXIF Analysis",
            "status": status,
            "plain_english": plain_english,
            "suspicious_flags": suspicious_flags,
            "metrics": {
                "format": fmt,
                "image_size_px": list(pil_img.size),
                "total_tags": len(tags),
                "software": (tags.get("Image Software") or tags.get("Software") or tags.get("software") or None),
                "tamper_risk": tamper_risk,
                "interpretation": f"{tamper_risk} risk — {len(suspicious_flags)} suspicious flag(s) found",
            },
        }
    except Exception as exc:
        log.warning("Metadata analysis failed: %s", exc)
        return {"name": "Metadata / EXIF Analysis", "status": "ERROR", "error": str(exc), "suspicious_flags": [], "metrics": {}}


# ── Layer 3: File Entropy ──────────────────────────────────────────────────────

def _file_entropy(path: str) -> float:
    """Compute Shannon entropy of every raw byte in the file.

    Shannon entropy tells us how 'random' the data is:
    - A perfectly random file (like encrypted data) scores near 8.0 bits.
    - A real unedited JPEG photo typically scores 7.8 – 8.0 because JPEG
      compression produces near-random byte patterns.
    - A file that has been repeatedly edited and re-saved loses some natural
      randomness, pushing the score below 7.8.
    """
    try:
        data = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
        counts = np.bincount(data, minlength=256).astype(np.float64)
        counts = counts[counts > 0]
        p = counts / counts.sum()
        return round(float(-np.sum(p * np.log2(p))), 4)
    except Exception:
        return 0.0


def _entropy_layer(path: str) -> Dict[str, Any]:
    entropy = _file_entropy(path)
    status = "CLEAN" if entropy >= 7.8 else "SUSPICIOUS"
    return {
        "name": "File Entropy (Data Randomness)",
        "status": status,
        "plain_english": (
            "Entropy measures how 'random' a file's raw data is. A perfectly original image "
            "scores near 8.0. Files that have been heavily edited lose some natural randomness. "
            f"This file scores {entropy}/8.0 — {'healthy, consistent with an original.' if entropy >= 7.8 else 'lower than expected, possible repeated re-encoding.'}"
        ),
        "metrics": {
            "file_entropy_bits": entropy,
            "max_possible": 8.0,
            "interpretation": "HEALTHY" if entropy >= 7.8 else "LOW — possible repeated re-encoding",
        },
    }


# ── Layer 4: Noise Residual ────────────────────────────────────────────────────

def _noise_analysis(image_path: str) -> Dict[str, Any]:
    """Check that sensor noise is uniform across the whole image.

    Every real camera sensor adds a faint, invisible 'grain' pattern. Because the
    grain comes from the hardware, it should look the same everywhere in the photo.
    If someone pastes content from a different image, the noise pattern at the
    boundary will be different — like a fingerprint mismatch.

    Algorithm:
    1. Blur the image with a Gaussian filter to remove real structure.
    2. Subtract the blurred version — what remains is the noise residual.
    3. Divide into 64x64 tiles. Compute the CV (ratio of std dev to mean) for each.
    4. If > 10% of tiles have a CV more than 2x the global average, flag it.
    """
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Subtract the smoothed version to isolate the noise signal
        residual = np.abs(gray - cv2.GaussianBlur(gray, (5, 5), 0))
        mean_n = float(np.mean(residual))
        tile = 64  # Each tile is 64x64 pixels
        h, w = residual.shape
        cv_vals = []
        for y in range(0, h - tile, tile):
            for x in range(0, w - tile, tile):
                patch = residual[y:y + tile, x:x + tile]
                # CV = standard deviation / mean — measures how uneven the noise is
                cv_vals.append(np.std(patch) / (np.mean(patch) + 1e-6))
        cv_arr = np.array(cv_vals)
        cv_global = float(np.mean(cv_arr)) if len(cv_arr) else 0.0
        hotspot = round(float(np.sum(cv_arr > cv_global * 2.0)) / len(cv_arr), 4) if len(cv_arr) else 0.0
        status = "SUSPICIOUS" if hotspot > 0.10 else "CLEAN"
        return {
            "name": "Noise Residual Analysis",
            "status": status,
            "plain_english": (
                "Every real camera sensor produces a faint invisible 'static' noise pattern. "
                "If parts of an image were patched together from different photos, the noise won't "
                "match at the join. "
                + (f"This image has {hotspot * 100:.1f}% hotspot tiles — anomalous noise spikes detected."
                   if hotspot > 0.10
                   else "Noise is uniform across the image — consistent with a single original source.")
            ),
            "metrics": {
                "mean_noise": round(mean_n, 4),
                "std_noise": round(float(np.std(residual)), 4),
                "noise_cv_global": round(cv_global, 4),
                "hotspot_tile_ratio": hotspot,
                "threshold_for_suspicious": 0.10,
                "interpretation": (
                    "ANOMALOUS — localised noise spikes detected (possible splice boundary)"
                    if hotspot > 0.10
                    else "UNIFORM — noise residual consistent across entire image"
                ),
            },
        }
    except Exception as exc:
        log.warning("Noise analysis failed: %s", exc)
        return {"name": "Noise Residual Analysis", "status": "ERROR", "error": str(exc), "metrics": {}}


# ── Layer 5: DCT Analysis ──────────────────────────────────────────────────────

def _dct_analysis(image_path: str, fmt: str) -> Dict[str, Any]:
    """Detect double-JPEG compression using DCT coefficient statistics.

    JPEG compression works by slicing the image into 8x8 pixel blocks and
    running a Discrete Cosine Transform (DCT) on each block. When a JPEG is
    edited and re-saved as JPEG, the block grid of the *original* save fights
    with the grid of the *new* save, leaving a characteristic 'comb' pattern
    in the DCT coefficient histogram. We measure the ratio of local minima to
    local maxima — a high ratio means double-compression.

    Note: this check only applies to JPEG files — PNG is lossless so it never
    double-compresses.
    """

    if fmt != "JPEG":
        return {
            "name": "DCT Double-Compression Analysis",
            "status": "N/A",
            "plain_english": (
                "When a JPEG image is edited and saved again, it leaves an invisible repeating "
                "'comb' pattern detectable via DCT analysis. This file is a PNG (lossless), "
                "so DCT double-compression analysis does not apply."
            ),
            "metrics": {"skipped": True, "reason": f"File format is {fmt} — DCT applies to JPEG only"},
        }
    try:
        from scipy.signal import argrelextrema  # noqa: PLC0415
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        h8, w8 = (h // 8) * 8, (w // 8) * 8
        gray = gray[:h8, :w8]
        all_ac: List[float] = []
        for y in range(0, h8, 8):
            for x in range(0, w8, 8):
                dct_block = cv2.dct(gray[y:y + 8, x:x + 8] - 128.0)
                all_ac.extend(dct_block.flatten()[1:11].tolist())
        ac_arr = np.array(all_ac)
        hist, _ = np.histogram(ac_arr, bins=200, range=(-100, 100))
        local_min = argrelextrema(hist, np.less, order=2)[0]
        local_max = argrelextrema(hist, np.greater, order=2)[0]
        comb_ratio = round(len(local_min) / (len(local_max) + 1e-6), 3)
        ac_kurtosis = round(
            float(np.mean((ac_arr - ac_arr.mean()) ** 4) / (ac_arr.std() ** 4 + 1e-10)), 3
        )
        status = "SUSPICIOUS" if comb_ratio > 1.3 else "CLEAN"
        return {
            "name": "DCT Double-Compression Analysis",
            "status": status,
            "plain_english": (
                "When a JPEG is edited and re-saved, it leaves an invisible 'comb' pattern in its "
                "mathematical frequency data. "
                + (f"This image has a comb ratio of {comb_ratio} (threshold 1.3) — double-compression detected."
                   if comb_ratio > 1.3
                   else f"Comb ratio is {comb_ratio} — no double-compression signature found.")
            ),
            "metrics": {
                "dct_ac_mean": round(float(np.mean(ac_arr)), 4),
                "dct_ac_std": round(float(np.std(ac_arr)), 4),
                "dct_ac_kurtosis": ac_kurtosis,
                "histogram_local_minima": int(len(local_min)),
                "histogram_local_maxima": int(len(local_max)),
                "comb_ratio": comb_ratio,
                "threshold_for_suspicious": 1.3,
                "interpretation": (
                    "DOUBLE-COMPRESSED — comb signature detected (image re-saved after editing)"
                    if comb_ratio > 1.3
                    else "SINGLE-COMPRESSED — no double-compression signature found"
                ),
            },
        }
    except Exception as exc:
        log.warning("DCT analysis failed: %s", exc)
        return {"name": "DCT Double-Compression Analysis", "status": "ERROR", "error": str(exc), "metrics": {}}


# ── Layer 6: Clone / Copy-Move ─────────────────────────────────────────────────

def _clone_detection(image_path: str) -> Dict[str, Any]:
    """Detect copy-paste / clone-stamp fraud using SIFT feature matching.

    A clone-stamp is a tool that artists use to copy one area of an image onto
    another area — for example, copying a clean background over a signature or
    removing a date stamp by pasting a matching texture over it.

    Algorithm:
    1. Find up to 3,000 'keypoints' (distinctive visual corners and blobs).
    2. Match every keypoint against every other keypoint in the same image.
    3. If two keypoints are very similar (distance < 120) BUT are far apart
       (more than 50 pixels), they were probably cloned from each other.
    4. If > 25% of keypoints have such a clone match, flag it.
    """
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # SIFT detects scale-invariant corners and blobs — robust descriptors
        sift = cv2.SIFT_create(nfeatures=3000)
        kps, descs = sift.detectAndCompute(gray, None)
        if descs is None or len(descs) < 10:
            return {
                "name": "Clone / Copy-Move Detection",
                "status": "N/A",
                "plain_english": "Too few keypoints to analyse for copy-move patterns.",
                "metrics": {"keypoints": 0, "clone_matches": 0, "clone_ratio": 0.0},
            }
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descs, descs, k=3)
        clone_hits = 0
        for m_list in matches:
            for m in m_list[1:]:
                if m.distance < 120:
                    p1 = np.array(kps[m.queryIdx].pt)
                    p2 = np.array(kps[m.trainIdx].pt)
                    if np.linalg.norm(p1 - p2) > 50:
                        clone_hits += 1
        clone_ratio = round(clone_hits / len(kps), 4) if kps else 0.0
        status = "SUSPICIOUS" if clone_ratio > 0.25 else "CLEAN"
        return {
            "name": "Clone / Copy-Move Detection",
            "status": status,
            "plain_english": (
                "This test looks for areas within the same image that are suspiciously similar "
                "— like someone used a 'clone stamp' to cover something up. "
                + (f"Clone ratio {clone_ratio:.3f} is above the 0.25 threshold — possible copy-move detected."
                   if clone_ratio > 0.25
                   else f"Clone ratio {clone_ratio:.3f} is within normal range — no copy-move pattern found.")
            ),
            "metrics": {
                "keypoints": len(kps),
                "clone_matches": clone_hits,
                "clone_ratio": clone_ratio,
                "threshold_for_suspicious": 0.25,
                "interpretation": (
                    "SUSPICIOUS — high number of spatially-distant self-matches (possible clone stamp)"
                    if clone_ratio > 0.25
                    else "CLEAN — no significant copy-move pattern detected"
                ),
            },
        }
    except Exception as exc:
        log.warning("Clone detection failed: %s", exc)
        return {"name": "Clone / Copy-Move Detection", "status": "ERROR", "error": str(exc), "metrics": {}}


# ── Layer 7: Color Anomaly ─────────────────────────────────────────────────────

def _color_anomaly_analysis(image_path: str) -> Dict[str, Any]:
    """Detect pixels whose colour does not fit the image's natural palette.

    Real photographs have a dominant colour palette (skin tones, sky, grass, etc.).
    A digital paste or colour-fill often introduces colours that don't match the
    rest of the image at all — like a perfectly bright-blue rectangle on a document
    that has no blue anywhere else.

    Algorithm:
    1. Convert the image to HSV (Hue / Saturation / Value).
    2. Build a histogram of hue values for non-dark, non-grey foreground pixels.
    3. Find the top-3 dominant hue bins and allow a ±10-degree tolerance around them.
    4. Any pixel that is vibrant (S > 60) but NOT in the dominant palette is
       classified as a colour anomaly.
    5. Group anomaly pixels into blobs. Large blobs (> 2000 px) almost always
       indicate a paste or annotation.
    """

    try:
        img_bgr = cv2.imread(image_path)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        foreground_mask = (V > 30) & (S > 15)
        h_fg = H[foreground_mask].astype(int)
        hist, _ = np.histogram(h_fg, bins=36, range=(0, 180))
        top3_bins = np.argsort(hist)[-3:]
        dominant_mask = np.zeros(36, dtype=bool)
        for b in top3_bins:
            for delta in range(-2, 3):
                dominant_mask[(b + delta) % 36] = True
        h_bin = (H / 5).astype(int).clip(0, 35)
        is_dominant = dominant_mask[h_bin]
        anomaly_mask = (~is_dominant) & (S > 60) & (V > 40)
        anomaly_pixels = int(np.sum(anomaly_mask))
        total_fg_pixels = int(np.sum(foreground_mask))
        anomaly_ratio = round(anomaly_pixels / (total_fg_pixels + 1), 6)
        anomaly_u8 = anomaly_mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        anomaly_closed = cv2.morphologyEx(anomaly_u8, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(anomaly_closed, connectivity=8)
        blobs = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > 200:
                bx = int(stats[i, cv2.CC_STAT_LEFT])
                by = int(stats[i, cv2.CC_STAT_TOP])
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                blob_px = hsv[labels == i]
                blobs.append({
                    "area_px": area,
                    "bounding_box": [bx, by, bw, bh],
                    "mean_hue_degrees": round(float(np.mean(blob_px[:, 0])) * 2, 1),
                    "mean_saturation": round(float(np.mean(blob_px[:, 1])), 1),
                })
        blobs.sort(key=lambda b: b["area_px"], reverse=True)

        if anomaly_ratio > 0.01 or (blobs and blobs[0]["area_px"] > 2000):
            status = "SUSPICIOUS"
            interpretation = "HIGHLY SUSPICIOUS — large chromatically-anomalous region(s) detected"
        elif anomaly_ratio > 0.003:
            status = "SUSPICIOUS"
            interpretation = f"SUSPICIOUS — {anomaly_ratio * 100:.3f}% of pixels have implausible hue"
        else:
            status = "CLEAN"
            interpretation = "CLEAN — no significant chromatic anomalies detected"

        return {
            "name": "Color Anomaly Detection",
            "status": status,
            "plain_english": (
                "This test looks for pixels whose color doesn't fit naturally with the rest of the "
                "image's color palette — which would suggest a digital paintbrush, paste, or color-splice. "
                + (f"Found {anomaly_pixels} suspicious pixels in {len(blobs[:5])} blob(s)."
                   if status == "SUSPICIOUS"
                   else "No significant color anomalies detected.")
            ),
            "metrics": {
                "anomaly_pixels": anomaly_pixels,
                "anomaly_ratio": anomaly_ratio,
                "threshold_for_suspicious": 0.003,
                "anomaly_blobs": blobs[:5],
                "interpretation": interpretation,
            },
        }
    except Exception as exc:
        log.warning("Color anomaly analysis failed: %s", exc)
        return {"name": "Color Anomaly Detection", "status": "ERROR", "error": str(exc), "metrics": {}}


# ── Layer 8: Edge Density ──────────────────────────────────────────────────────

def _edge_analysis(image_path: str) -> Dict[str, Any]:
    """Detect unnatural sharp edges caused by pasting or digital drawing.

    Natural photographs have smooth, camera-blurred edges. When someone pastes
    a region onto an existing image or draws a digital box/text, the resulting
    edge is laser-sharp and perfectly straight — unlike anything a camera produces.

    We use the Canny edge detector to find edges, divide the result into 32x32
    tiles, and flag tiles with > 3x the average edge density.
    """

    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=50, threshold2=150)
        tile = 32
        h, w = edges.shape
        densities = []
        for y in range(0, h - tile, tile):
            for x in range(0, w - tile, tile):
                densities.append(float(np.mean(edges[y:y + tile, x:x + tile])))
        d_arr = np.array(densities)
        mean_d = float(np.mean(d_arr)) if len(d_arr) else 0.0
        high_d_ratio = round(float(np.sum(d_arr > mean_d * 3.0)) / len(d_arr), 4) if len(d_arr) else 0.0
        status = "SUSPICIOUS" if high_d_ratio > 0.06 else "CLEAN"
        return {
            "name": "Edge Discontinuity / Density Analysis",
            "status": status,
            "plain_english": (
                "Digitally drawn lines or pasted overlays have unnaturally sharp, clean edges "
                "compared to natural photographic edges. "
                + (f"{high_d_ratio * 100:.1f}% of tiles show unnaturally high edge density — above the 6% threshold."
                   if status == "SUSPICIOUS"
                   else f"{high_d_ratio * 100:.1f}% high-density tiles — within normal range.")
            ),
            "metrics": {
                "mean_edge_density": round(mean_d, 4),
                "high_density_tile_ratio": high_d_ratio,
                "threshold_for_suspicious": 0.06,
                "interpretation": (
                    "SUSPICIOUS — elevated localised edge density (drawn line or sharp boundary anomaly)"
                    if status == "SUSPICIOUS"
                    else "NORMAL — edge distribution consistent with natural image"
                ),
            },
        }
    except Exception as exc:
        log.warning("Edge analysis failed: %s", exc)
        return {"name": "Edge Discontinuity / Density Analysis", "status": "ERROR", "error": str(exc), "metrics": {}}


# ── Layer 9: Saturation Anomaly ────────────────────────────────────────────────

def _saturation_anomaly(image_path: str) -> Dict[str, Any]:
    """Detect localised over-saturation — a sign of colour filters or annotations.

    Applying a bright-colour overlay or a vivid annotation (stamp, highlight) to
    just one part of a document makes that region far more saturated than the rest.
    We divide the image into 32x32 tiles and flag tiles whose mean saturation is
    more than 3x the global average.
    """

    try:
        img = cv2.imread(image_path)
        S = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
        mean_s = float(np.mean(S))
        tile = 32
        h, w = S.shape
        high_tiles = total_tiles = 0
        high_coords: List[List[int]] = []
        for y in range(0, h - tile, tile):
            for x in range(0, w - tile, tile):
                if float(np.mean(S[y:y + tile, x:x + tile])) > mean_s * 3.0:
                    high_tiles += 1
                    high_coords.append([x, y, tile, tile])
                total_tiles += 1
        high_ratio = round(high_tiles / total_tiles, 4) if total_tiles else 0.0
        status = "SUSPICIOUS" if high_ratio > 0.02 else "CLEAN"
        return {
            "name": "Saturation Anomaly Detection",
            "status": status,
            "plain_english": (
                "Real photos have consistent color vibrancy across the image. A color filter or "
                "vivid annotation applied to just one area creates saturation spikes. "
                + (f"{high_ratio * 100:.1f}% of tiles are over-saturated — possible localised color edit."
                   if status == "SUSPICIOUS"
                   else "Saturation is uniform — no localised color edits detected.")
            ),
            "metrics": {
                "mean_saturation": round(mean_s, 3),
                "high_saturation_tile_ratio": high_ratio,
                "threshold_for_suspicious": 0.02,
                "high_saturation_coords_sample": high_coords[:10],
                "interpretation": (
                    "SUSPICIOUS — localised saturation spikes (possible annotation or colour edit)"
                    if status == "SUSPICIOUS"
                    else "NORMAL — saturation distribution uniform across image"
                ),
            },
        }
    except Exception as exc:
        log.warning("Saturation anomaly failed: %s", exc)
        return {"name": "Saturation Anomaly Detection", "status": "ERROR", "error": str(exc), "metrics": {}}


# ── Layer 10: Font Consistency ─────────────────────────────────────────────────

def _font_consistency_analysis(image_path: str) -> Dict[str, Any]:
    """Detect tampered text — e.g. a salary figure typed over the original.

    When someone opens a document image in an editor and types new text (to change
    a number or a name), the new text usually differs from the original in at least
    one of these ways:
    - Stroke width (thickness of the letter strokes)
    - Character height (the new font is a slightly different size)
    - Sharpness (anti-aliasing differs between the original scan and the new text)

    We find all text blobs, measure those three properties, and look for spatially-
    clustered groups of characters that are statistical outliers from the rest.
    A cluster of outlier characters in the same region is strong evidence of
    text replacement.

    Note: this check requires at least 20 detected characters to be meaningful.
    It returns N/A for photos and non-text documents.
    """

    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=15, C=8,
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        char_data: List[Tuple] = []
        for i in range(1, num_labels):
            cx = int(stats[i, cv2.CC_STAT_LEFT])
            cy = int(stats[i, cv2.CC_STAT_TOP])
            cbw = int(stats[i, cv2.CC_STAT_WIDTH])
            cbh = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if cbh < 6 or cbh > h * 0.12 or cbw < 2 or cbw > w * 0.25:
                continue
            if area < 20:
                continue
            aspect = cbw / (cbh + 1e-6)
            if aspect > 6.0 or aspect < 0.05:
                continue
            bx1, by1 = max(0, cx), max(0, cy)
            bx2, by2 = min(w, cx + cbw), min(h, cy + cbh)
            try:
                blob_crop = (labels[by1:by2, bx1:bx2] == i).astype(np.uint8)
                dist = cv2.distanceTransform(blob_crop, cv2.DIST_L2, 3)
                stroke_w = float(np.max(dist)) * 2.0
                if not np.isfinite(stroke_w):
                    stroke_w = 0.0
            except Exception:
                stroke_w = 0.0
            x1 = max(0, cx - 3); y1 = max(0, cy - 3)
            x2 = min(w, cx + cbw + 3); y2 = min(h, cy + cbh + 3)
            roi = gray[y1:y2, x1:x2].astype(np.float64)
            sharpness = float(np.var(cv2.Laplacian(roi, cv2.CV_64F)))
            char_data.append((cx, cy, cbw, cbh, stroke_w, sharpness))

        if len(char_data) < 20:
            return {
                "name": "Font Consistency Analysis",
                "status": "N/A",
                "plain_english": "Insufficient text characters detected — not a text document or very few characters.",
                "metrics": {
                    "char_count": len(char_data),
                    "skipped": True,
                    "reason": "Too few text characters detected",
                    "interpretation": "N/A — insufficient text for font analysis",
                },
            }

        char_arr = np.array(char_data, dtype=np.float32)
        heights = char_arr[:, 3]
        strokes = char_arr[:, 4]
        sharpness = char_arr[:, 5]

        def _safe_cv(arr: "np.ndarray") -> float:
            finite = arr[np.isfinite(arr)]
            if len(finite) < 2:
                return 0.0
            mu = float(np.mean(finite))
            return round(float(np.std(finite)) / (mu + 1e-6), 4) if mu > 1e-6 else 0.0

        def _iqr_outliers(arr: "np.ndarray", factor: float = 1.5) -> "np.ndarray":
            finite = arr[np.isfinite(arr)]
            if len(finite) < 4:
                return np.zeros(len(arr), dtype=bool)
            q1, q3 = np.percentile(finite, 25), np.percentile(finite, 75)
            iqr = q3 - q1
            return (arr < q1 - factor * iqr) | (arr > q3 + factor * iqr)

        strokes_finite = strokes[np.isfinite(strokes) & (strokes > 0.5)]
        height_out = _iqr_outliers(heights)
        stroke_out = np.zeros(len(strokes), dtype=bool)
        if len(strokes_finite) >= 4:
            stroke_finite_mask = np.isfinite(strokes) & (strokes > 0.5)
            stroke_out[stroke_finite_mask] = _iqr_outliers(strokes_finite)
        sharp_out = _iqr_outliers(sharpness, factor=2.0)

        height_cv = _safe_cv(heights)
        stroke_cv = _safe_cv(strokes_finite if len(strokes_finite) >= 2 else strokes)
        sharp_cv = _safe_cv(sharpness)
        height_outlier_ratio = round(float(np.sum(height_out)) / len(heights), 4)
        stroke_outlier_ratio = round(float(np.sum(stroke_out)) / len(strokes), 4)
        sharp_outlier_ratio = round(float(np.sum(sharp_out)) / len(sharpness), 4)

        combined_suspicious = (
            height_out.astype(int) + stroke_out.astype(int) + sharp_out.astype(int)
        ) >= 2

        CELL = 64
        grid_h = h // CELL + 1
        grid_w = w // CELL + 1
        grid = np.zeros((grid_h, grid_w), dtype=np.int32)
        for idx, (cx, cy, cbw, cbh, _sw, _sh) in enumerate(char_data):
            gc = (cx + cbw // 2) // CELL
            gr = (cy + cbh // 2) // CELL
            if combined_suspicious[idx]:
                grid[gr, gc] += 1
        hot_map = (grid >= 2).astype(np.uint8)
        suspicious_regions: List[Dict] = []
        if hot_map.any():
            n_reg, reg_labels, reg_stats, _ = cv2.connectedComponentsWithStats(hot_map, connectivity=8)
            for r in range(1, n_reg):
                rx = int(reg_stats[r, cv2.CC_STAT_LEFT]) * CELL
                ry = int(reg_stats[r, cv2.CC_STAT_TOP]) * CELL
                rw = int(reg_stats[r, cv2.CC_STAT_WIDTH]) * CELL
                rh = int(reg_stats[r, cv2.CC_STAT_HEIGHT]) * CELL
                n_susp = int(np.sum(grid[reg_labels == r]))
                suspicious_regions.append({"bounding_box": [rx, ry, rw, rh], "suspicious_chars": n_susp})
        suspicious_regions.sort(key=lambda r: r["suspicious_chars"], reverse=True)
        n_regions = len(suspicious_regions)
        is_suspicious = (
            (stroke_cv > 0.40 and n_regions >= 1)
            or (n_regions >= 2)
            or (sharp_outlier_ratio > 0.25 and n_regions >= 1)
        )
        status = "SUSPICIOUS" if is_suspicious else "CLEAN"
        return {
            "name": "Font Consistency Analysis",
            "status": status,
            "plain_english": (
                "This checks all the letters in the document for consistent thickness, height, and "
                "sharpness. If some letters look different it suggests someone typed over the original "
                "or inserted text from a different source. "
                + (f"{n_regions} spatially-coherent anomaly region(s) found (stroke_cv={stroke_cv:.3f})."
                   if is_suspicious
                   else f"{len(char_data)} characters analysed — font metrics uniform, no anomalous regions.")
            ),
            "metrics": {
                "char_count": len(char_data),
                "height_cv": height_cv,
                "height_outlier_ratio": height_outlier_ratio,
                "stroke_cv": stroke_cv,
                "stroke_outlier_ratio": stroke_outlier_ratio,
                "sharpness_cv": sharp_cv,
                "sharpness_outlier_ratio": sharp_outlier_ratio,
                "n_suspicious_regions": n_regions,
                "suspicious_regions": suspicious_regions[:5],
                "interpretation": (
                    f"SUSPICIOUS — {n_regions} font anomaly region(s)"
                    if is_suspicious
                    else "CONSISTENT — font metrics uniform across document"
                ),
            },
        }
    except Exception as exc:
        log.warning("Font consistency analysis failed: %s", exc)
        return {"name": "Font Consistency Analysis", "status": "ERROR", "error": str(exc), "metrics": {}}


# ── Layer 11: AI Artifact Detection ───────────────────────────────────────────

def _ai_artifact_analysis(image_path: str) -> Dict[str, Any]:
    """Detect AI-generated images using Fourier Transform frequency analysis.

    AI image generators (Stable Diffusion, Midjourney, DALL-E, etc.) use a
    mathematical structure based on grid convolutions. This leaves a faint
    'checkerboard' or 'frequency spike' pattern in the image's frequency domain
    that is invisible to the human eye but very clear in the FFT spectrum.

    Algorithm:
    1. Convert the image to greyscale.
    2. Apply a 2-D Fast Fourier Transform to get the frequency spectrum.
    3. Shift the zero-frequency component to the centre.
    4. Measure the strength of high-frequency spikes relative to the background.
    5. A spike ratio > 3.0 indicates unnatural periodicity (AI generation).
    """

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"name": "AI Generative Model Detection", "status": "ERROR", "error": "Could not read image", "metrics": {}}
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        mask = dist_from_center > (min(h, w) * 0.3)
        high_freq_data = magnitude_spectrum[mask]
        high_freq_mean = float(np.mean(high_freq_data)) if len(high_freq_data) else 0.0
        high_freq_max = float(np.max(high_freq_data)) if len(high_freq_data) else 0.0
        spike_ratio = round(high_freq_max / (high_freq_mean + 1e-6), 2)
        status = "SUSPICIOUS" if spike_ratio > 3.0 else "CLEAN"
        return {
            "name": "AI Generative Model Detection",
            "status": status,
            "plain_english": (
                "AI image generators (like Midjourney or Stable Diffusion) leave invisible "
                "'checkerboard' grid artifacts detectable via Fourier Transform. "
                + (f"Spike ratio {spike_ratio} > 3.0 — unnatural high-frequency grid artifacts detected."
                   if status == "SUSPICIOUS"
                   else f"Spike ratio {spike_ratio} is below the 3.0 threshold — frequency spectrum looks natural, not AI-generated.")
            ),
            "metrics": {
                "high_freq_mean": round(high_freq_mean, 2),
                "high_freq_max": round(high_freq_max, 2),
                "spike_ratio": spike_ratio,
                "threshold_for_suspicious": 3.0,
                "interpretation": (
                    "SUSPICIOUS — Unnatural high-frequency grid artifacts (possible AI generation)"
                    if status == "SUSPICIOUS"
                    else "NORMAL — Frequency spectrum looks natural"
                ),
            },
        }
    except Exception as exc:
        log.warning("AI artifact analysis failed: %s", exc)
        return {"name": "AI Generative Model Detection", "status": "ERROR", "error": str(exc), "metrics": {}}


# ── Scoring ────────────────────────────────────────────────────────────────────

def _compute_score(layers: Dict[str, Any], file_size_bytes: int) -> Tuple[float, str, List[str]]:
    """Aggregate all 11 layer results into a single 0–100 forgery score.

    Each triggered check adds a fixed number of points. Points are calibrated to
    reflect how reliably each signal indicates actual tampering:
    - High-confidence signals (colour anomaly, DCT comb) add up to 35 pts.
    - Medium-confidence signals (ELA, font, AI artefact) add up to 20–25 pts.
    - Supporting signals (noise, clone, edge, saturation) add up to 8–15 pts.
    - Metadata flags each add 10 pts.

    Final verdict:
      < 15   → ORIGINAL
      15–29  → UNCERTAIN (some signals, could be an innocent re-save)
      30–54  → LIKELY TAMPERED (multiple signals align)
      >=55   → TAMPERED (strong, multi-layer evidence)
    """
    score: float = 0.0
    evidence: List[str] = []

    ela = layers.get("layer_1_ela", {}).get("metrics", {})
    if ela.get("suspicious_block_ratio", 0) > 0.05:
        score += 25
        evidence.append(f"ELA: {ela['suspicious_block_ratio'] * 100:.1f}% of blocks have anomalous re-compression")
    elif ela.get("mean_ela", 0) > 8:
        score += 12
        evidence.append(f"ELA: elevated mean ELA ({ela.get('mean_ela', 0):.1f}) — possible re-editing")

    meta = layers.get("layer_2_metadata", {})
    for flag in meta.get("suspicious_flags", []):
        score += 10
        evidence.append(f"Metadata: {flag}")

    noise = layers.get("layer_4_noise", {}).get("metrics", {})
    if noise.get("hotspot_tile_ratio", 0) > 0.10:
        score += 15
        evidence.append(f"Noise: {noise['hotspot_tile_ratio'] * 100:.1f}% of tiles have anomalous noise")

    dct = layers.get("layer_5_dct", {}).get("metrics", {})
    if not dct.get("skipped") and dct.get("comb_ratio", 0) > 1.3:
        score += 20
        evidence.append(f"DCT: double-compression comb detected (ratio {dct.get('comb_ratio')})")

    clone = layers.get("layer_6_clone", {}).get("metrics", {})
    if clone.get("clone_ratio", 0) > 0.25:
        score += 12
        evidence.append(f"Clone: clone_ratio {clone.get('clone_ratio', 0):.3f} — possible copy-move")

    color = layers.get("layer_7_color", {}).get("metrics", {})
    ratio = color.get("anomaly_ratio", 0)
    blobs = color.get("anomaly_blobs", [])
    largest_blob = blobs[0]["area_px"] if blobs else 0
    if ratio > 0.01 or largest_blob > 2000:
        score += 35
        evidence.append(f"Color anomaly: {ratio * 100:.2f}% of pixels suspicious (largest blob {largest_blob:,} px)")
    elif ratio > 0.003:
        score += 18
        evidence.append(f"Color anomaly: {ratio * 100:.3f}% of pixels suspicious — possible light edit")

    edge = layers.get("layer_8_edge", {}).get("metrics", {})
    if edge.get("high_density_tile_ratio", 0) > 0.06:
        score += 12
        evidence.append(f"Edge: {edge['high_density_tile_ratio'] * 100:.1f}% of tiles have unnaturally high edge density")

    sat = layers.get("layer_9_saturation", {}).get("metrics", {})
    if sat.get("high_saturation_tile_ratio", 0) > 0.02:
        score += 8
        evidence.append(f"Saturation: {sat['high_saturation_tile_ratio'] * 100:.1f}% of tiles are over-saturated")

    font = layers.get("layer_10_font", {}).get("metrics", {})
    if not font.get("skipped"):
        stroke_cv = font.get("stroke_cv", 0.0)
        n_regions = font.get("n_suspicious_regions", 0)
        sharp_out = font.get("sharpness_outlier_ratio", 0.0)
        if stroke_cv > 0.40 and n_regions >= 1:
            score += 20
            evidence.append(f"Font: stroke_cv={stroke_cv:.3f} with {n_regions} anomaly region(s)")
        elif n_regions >= 2:
            score += 15
            evidence.append(f"Font: {n_regions} spatially-coherent character anomaly clusters")
        elif sharp_out > 0.25 and n_regions >= 1:
            score += 15
            evidence.append(f"Font: sharpness outlier ratio {sharp_out * 100:.1f}% in {n_regions} region(s)")

    ai_gen = layers.get("layer_11_ai", {}).get("metrics", {})
    spike_ratio = ai_gen.get("spike_ratio", 0)
    if spike_ratio > 3.5:
        score += 25
        evidence.append(f"AI Generation: severe frequency grid artifacts (spike ratio {spike_ratio})")
    elif spike_ratio > 3.0:
        score += 15
        evidence.append(f"AI Generation: high-frequency anomalies (spike ratio {spike_ratio})")

    score = min(score, 100.0)
    verdict = (
        "TAMPERED" if score >= 55
        else "LIKELY TAMPERED" if score >= 30
        else "UNCERTAIN" if score >= 15
        else "ORIGINAL"
    )
    return round(score, 1), verdict, evidence


# ── Public entry point ─────────────────────────────────────────────────────────

def run_forensics(path: str) -> Dict[str, Any]:
    """Main entry point: run all 11 forensic layers on one image file.

    Call this function with the absolute path to any image file.
    It runs every layer, collects the results, computes the final score,
    and returns a structured dict ready to be stored in the database.

    Returns UNAVAILABLE immediately if the required libraries are not installed.
    All individual layers catch their own exceptions so a single layer failure
    never stops the rest of the analysis.
    """

    """Run the full 11-layer forensic analysis on a single image file.

    Parameters
    ----------
    path : str
        Absolute path to the image file (JPEG, PNG, TIFF, BMP, WebP).
        For PDFs, render page 1 to a temporary PNG before calling this.

    Returns
    -------
    dict with keys:
        scan_summary   -- source_file, format, file_size_bytes, file_entropy_bits,
                          forensic_verdict, forgery_score_0_100, overall_explanation,
                          evidence (list of strings)
        layers         -- dict with layer_1_ela … layer_11_ai, each containing:
                          name, status (CLEAN/SUSPICIOUS/N/A/ERROR),
                          plain_english, metrics
    """
    if not _FORENSICS_AVAILABLE:
        log.warning("run_forensics: forensics libraries not available, returning stub result")
        return {
            "scan_summary": {
                "source_file": os.path.basename(path),
                "forensic_verdict": "UNAVAILABLE",
                "forgery_score_0_100": 0.0,
                "overall_explanation": "Forensics libraries (numpy, opencv, Pillow) are not installed.",
                "evidence": [],
            },
            "layers": {},
        }

    path = str(path)
    fmt = _detect_format(path)         # JPEG, PNG, TIFF, etc.
    file_size = os.path.getsize(path) if os.path.exists(path) else 0

    log.info("run_forensics: starting analysis", extra={"path": path, "format": fmt, "size_bytes": file_size})

    # Run all 11 layers. Each call is independent — a failure in one layer
    # does not prevent the others from running.
    layers = {
        "layer_1_ela":        _ela_analysis(path),           # Error Level Analysis
        "layer_2_metadata":   _metadata_analysis(path, fmt), # EXIF / metadata inspection
        "layer_3_entropy":    _entropy_layer(path),          # Shannon entropy of raw bytes
        "layer_4_noise":      _noise_analysis(path),         # Sensor noise consistency
        "layer_5_dct":        _dct_analysis(path, fmt),      # Double-JPEG compression (JPEG only)
        "layer_6_clone":      _clone_detection(path),        # Copy-move / clone stamp
        "layer_7_color":      _color_anomaly_analysis(path), # Chromatic palette anomalies
        "layer_8_edge":       _edge_analysis(path),          # Unnatural sharp edge density
        "layer_9_saturation": _saturation_anomaly(path),     # Localised over-saturation
        "layer_10_font":      _font_consistency_analysis(path), # Font uniformity (text docs)
        "layer_11_ai":        _ai_artifact_analysis(path),   # AI-generation artefacts (FFT)
    }

    score, verdict, evidence = _compute_score(layers, file_size)

    suspicious_count = sum(
        1 for v in layers.values() if isinstance(v, dict) and v.get("status") == "SUSPICIOUS"
    )
    clean_count = sum(
        1 for v in layers.values() if isinstance(v, dict) and v.get("status") == "CLEAN"
    )

    if verdict == "ORIGINAL":
        explanation = (
            f"All {clean_count} applicable forensic layers came back clean. "
            "No significant tampering signals were detected."
        )
    elif verdict in ("LIKELY TAMPERED", "TAMPERED"):
        explanation = (
            f"{suspicious_count} of 11 forensic layers flagged suspicious signals "
            f"(score {score}/100). Key evidence: {'; '.join(evidence[:3])}."
        )
    else:
        explanation = (
            f"Mixed results — {suspicious_count} suspicious, {clean_count} clean layers. "
            "Manual review is recommended."
        )

    log.info(
        "run_forensics: complete",
        extra={"path": path, "verdict": verdict, "score": score, "evidence_count": len(evidence)},
    )

    return {
        "scan_summary": {
            "source_file": os.path.basename(path),
            "format": fmt,
            "file_size_bytes": file_size,
            "file_entropy_bits": layers["layer_3_entropy"]["metrics"].get("file_entropy_bits", 0.0),
            "forensic_verdict": verdict,
            "forgery_score_0_100": score,
            "overall_explanation": explanation,
            "evidence": evidence,
        },
        "layers": layers,
    }


def run_forensics_on_pdf(pdf_path: str) -> Dict[str, Any]:
    """Run all 11 forensic layers on a PDF document.

    PDFs cannot be analysed directly by pixel-based forensics, so we first
    render the first page to a temporary PNG image at 200 DPI (high enough to
    preserve all text details and compression artefacts), then call
    :func:`run_forensics` on that PNG.

    The temporary PNG is deleted immediately after analysis regardless of
    whether the analysis succeeds or fails.

    Requires PyMuPDF (fitz). Falls back to a stub result if fitz is
    not installed.
    """
    try:
        import fitz  # noqa: PLC0415
    except ImportError:
        log.warning("run_forensics_on_pdf: PyMuPDF (fitz) not installed — skipping forensics for PDF")
        return {
            "scan_summary": {
                "source_file": os.path.basename(pdf_path),
                "forensic_verdict": "UNAVAILABLE",
                "forgery_score_0_100": 0.0,
                "overall_explanation": "PyMuPDF (fitz) is not installed — cannot render PDF page for forensic analysis.",
                "evidence": [],
            },
            "layers": {},
        }

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        doc = fitz.open(pdf_path)
        page = doc[0]
        mat = fitz.Matrix(200 / 72, 200 / 72)  # 200 DPI
        pix = page.get_pixmap(matrix=mat)
        pix.save(tmp_path)
        doc.close()
        log.info("run_forensics_on_pdf: rendered page 1 to temp PNG", extra={"tmp": tmp_path})
        result = run_forensics(tmp_path)
        result["scan_summary"]["source_file"] = os.path.basename(pdf_path)
        result["scan_summary"]["analyzed_as"] = f"{os.path.basename(tmp_path)} (page 1 rendered at 200 DPI)"
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
