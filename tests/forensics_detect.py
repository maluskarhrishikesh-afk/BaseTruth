"""
forensics_detect.py  —  Universal image tampering detector.

Works on JPEG documents (marksheets, IDs, certificates) AND on PNG/JPEG
photos.  No LLM dependency.  The output JSON is structured to be passed
directly to an LLM for reasoning / narrative generation.

Usage:
    python forensics_detect.py <image_path> [<reference_image_path>]

Single-image mode:  absolute signal scoring.
Two-image mode:     peer-relative comparison (higher score = more suspicious).

Techniques (applied to every image; graceful skip where not applicable):
  1  ELA               Error Level Analysis          (JPEG + PNG)
  2  Metadata          EXIF + PNG tEXt chunks        (JPEG + PNG)
  3  File entropy      Shannon entropy of raw bytes  (all)
  4  Noise residual    Gaussian-blur high-freq noise (all)
  5  DCT analysis      Double-compression comb       (JPEG only)
  6  Clone detection   SIFT copy-move                (all)
  7  Color anomaly     HSV chromatic outlier pixels  (all)
  8  Edge density      Canny edge density heatmap    (all)
  9  Saturation        Over-saturated tile map       (all)

Outputs (written alongside the first input image):
    forensics_report.json
    ela_<label>.png
    noise_<label>.png
    color_anomaly_<label>.png
    edge_<label>.png
"""

from __future__ import annotations
import io, json, os, sys, warnings
import numpy as np
import cv2
from PIL import Image, ImageChops
from PIL.ExifTags import TAGS
import exifread

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
ELA_QUALITY = 75     # JPEG quality used as ELA re-compression baseline
ELA_AMPLIFY = 10     # multiply ELA pixel delta for visualisation

_EDIT_SOFTWARE = [
    "photoshop", "gimp", "lightroom", "paint", "snapseed", "affinity",
    "picsart", "pixelmator", "canva", "inkscape", "adobe", "capture one",
    "darktable", "rawtherapee", "medibang", "clip studio", "procreate",
    "preview", "irfanview", "xnview", "imagemagick", "pillow", "opencv",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _out_dir(path: str) -> str:
    return os.path.dirname(os.path.abspath(path)) or "."

def _detect_format(path: str) -> str:
    """Return 'JPEG' or 'PNG' (falls back to extension)."""
    try:
        return Image.open(path).format or (
            "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"
        )
    except Exception:
        return "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"


# ── 1. ELA (Error Level Analysis) ────────────────────────────────────────────────
def ela_analysis(image_path: str, save_path: str) -> dict:
    """
    SIMPLE EXPLANATION:
    When a JPEG image is saved, the quality drops slightly. If someone pastes a new
    piece of high-quality text onto an older image and saves it again, the newly pasted
    text will "stand out" because it hasn't lost as much quality as the rest of the image.
    This function looks for those parts standing out.
    """
    orig = Image.open(image_path).convert("RGB")
    buf  = io.BytesIO()
    orig.save(buf, format="JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    ela_arr = np.array(ImageChops.difference(orig, recompressed), dtype=np.float32)

    ela_vis = np.clip(ela_arr * ELA_AMPLIFY, 0, 255).astype(np.uint8)
    Image.fromarray(ela_vis).save(save_path)

    mean_ela = float(np.mean(ela_arr))
    h, w     = ela_arr.shape[:2]
    block    = 32
    high = total = 0
    for y in range(0, h - block, block):
        for x in range(0, w - block, block):
            if np.mean(ela_arr[y:y+block, x:x+block]) > mean_ela * 2.5:
                high += 1
            total += 1
    suspicious_ratio = round(high / total, 4) if total else 0.0

    return {
        "mean_ela":               round(mean_ela, 3),
        "max_ela":                round(float(np.max(ela_arr)), 3),
        "std_ela":                round(float(np.std(ela_arr)), 3),
        "suspicious_block_ratio": suspicious_ratio,
        "ela_heatmap":            save_path,
        "interpretation": (
            "HIGH — many blocks have anomalous re-compression levels (tampering likely)"
            if suspicious_ratio > 0.05 else
            "LOW — uniform ELA across image (consistent with original)"
        ),
    }


# ── 2. Metadata (Hidden Information) ───────────────────────────────────────────
def metadata_analysis(image_path: str, fmt: str) -> dict:
    """
    SIMPLE EXPLANATION:
    Checks the invisible tags attached to the photo (metadata). 
    It looks for clues that photo-editing software (like Photoshop) was used, or if 
    the original camera's details have been suspiciously deleted.
    """
    tags: dict[str, str]       = {}
    suspicious_flags: list[str] = []

    # JPEG path — rich EXIF via exifread
    if fmt == "JPEG":
        with open(image_path, "rb") as f:
            raw = exifread.process_file(f, details=False, strict=False)
        for k, v in raw.items():
            tags[k] = str(v)

        software      = tags.get("Image Software", "")
        make          = tags.get("Image Make", "")
        model         = tags.get("Image Model", "")
        datetime_orig = tags.get("EXIF DateTimeOriginal", "")
        datetime_img  = tags.get("Image DateTime", "")

        if any(s in software.lower() for s in _EDIT_SOFTWARE):
            suspicious_flags.append(f"Editing software detected: {software!r}")
        if not make and not model:
            suspicious_flags.append("No camera Make/Model in EXIF (re-saved without camera metadata)")
        if not datetime_orig:
            suspicious_flags.append("No DateTimeOriginal (stripped or synthetically created)")
        if datetime_orig and datetime_img and datetime_orig != datetime_img:
            suspicious_flags.append(
                f"DateTimeOriginal ({datetime_orig}) ≠ ImageDateTime ({datetime_img}) — post-edit save"
            )
        if not raw:
            suspicious_flags.append("EXIF completely absent — metadata stripped (common after editing)")

    # All formats — PIL: PNG tEXt/iTXt + embedded EXIF
    pil_img = Image.open(image_path)
    for k, v in (pil_img.info or {}).items():
        if isinstance(v, (str, bytes)):
            str_v = v.decode(errors="replace") if isinstance(v, bytes) else v
            if k not in tags:
                tags[k] = str_v

    exif_raw = pil_img.getexif()
    if exif_raw:
        for tag_id, val in exif_raw.items():
            key = f"EXIF:{TAGS.get(tag_id, tag_id)}"
            if key not in tags:
                tags[key] = str(val)

    # Non-JPEG format checks
    if fmt != "JPEG":
        software = ""
        for k, v in tags.items():
            if isinstance(v, str) and any(
                kw in k.lower() for kw in ("software", "creator", "comment")
            ):
                if any(s in v.lower() for s in _EDIT_SOFTWARE):
                    software = v
                    suspicious_flags.append(f"Editing tool in metadata: {k!r} = {v!r}")

        if not tags:
            suspicious_flags.append(
                "No metadata whatsoever — all metadata stripped (common after editing)"
            )
        elif not any(
            kw in k.lower() for kw in ("make", "model", "camera") for k in tags
        ):
            suspicious_flags.append(
                "No camera Make/Model — image may be screen-captured or re-exported"
            )

    return {
        "format":           fmt,
        "size":             list(pil_img.size),
        "total_tags":       len(tags),
        "software":         (
            tags.get("Image Software") or tags.get("Software") or tags.get("software") or None
        ),
        "suspicious_flags": suspicious_flags,
        "tamper_risk": (
            "HIGH"   if len(suspicious_flags) >= 2 else
            "MEDIUM" if suspicious_flags else
            "LOW"
        ),
    }


# ── 3. File Entropy (Data Randomness) ──────────────────────────────────────────
def file_entropy(path: str) -> float:
    """
    SIMPLE EXPLANATION:
    Checks how 'random' the file's data is. Files that have been heavily edited 
    and re-saved many times lose some of their natural randomness.
    """
    data   = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    counts = np.bincount(data, minlength=256).astype(np.float64)
    counts = counts[counts > 0]
    p      = counts / counts.sum()
    return round(float(-np.sum(p * np.log2(p))), 4)


# ── 4. Noise Residual (Static Background) ──────────────────────────────────────
def noise_analysis(image_path: str, save_path: str) -> dict:
    """
    SIMPLE EXPLANATION:
    Every real photo has a tiny, invisible amount of 'static' (like old TV static) 
    created by the camera sensor. If a picture has been patched together from 
    different images, the static won't match up in the glued areas.
    """
    img      = cv2.imread(image_path)
    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = np.abs(gray - cv2.GaussianBlur(gray, (5, 5), 0))

    mean_n = float(np.mean(residual))
    tile   = 64
    h, w   = residual.shape
    cv_vals = []
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            patch = residual[y:y+tile, x:x+tile]
            cv_vals.append(np.std(patch) / (np.mean(patch) + 1e-6))

    cv_arr    = np.array(cv_vals)
    cv_global = float(np.mean(cv_arr))
    hotspot   = round(float(np.sum(cv_arr > cv_global * 2.0)) / len(cv_arr), 4)

    vis = np.clip(residual * 4, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, cv2.applyColorMap(vis, cv2.COLORMAP_JET))

    return {
        "mean_noise":         round(mean_n, 4),
        "std_noise":          round(float(np.std(residual)), 4),
        "noise_cv_global":    round(cv_global, 4),
        "noise_cv_max":       round(float(np.max(cv_arr)), 4),
        "hotspot_tile_ratio": hotspot,
        "noise_heatmap":      save_path,
        "interpretation": (
            "ANOMALOUS — localised noise spikes detected (possible splice boundary)"
            if hotspot > 0.10 else
            "UNIFORM — noise residual consistent across entire image"
        ),
    }


# ── 5. DCT Analysis (Double-Save Patterns for JPEG) ───────────────────────────
def dct_analysis(image_path: str, fmt: str) -> dict | None:
    """
    SIMPLE EXPLANATION:
    When a JPEG image is edited and saved again, it leaves an invisible repeating 
    pattern behind (like a comb). This specifically hunts for that "double-saved" signature.
    """
    if fmt != "JPEG":
        return None

    from scipy.signal import argrelextrema

    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    gray   = gray[:h8, :w8]

    all_ac: list[float] = []
    for y in range(0, h8, 8):
        for x in range(0, w8, 8):
            dct_block = cv2.dct(gray[y:y+8, x:x+8] - 128.0)
            all_ac.extend(dct_block.flatten()[1:11].tolist())

    ac_arr = np.array(all_ac)
    hist, _ = np.histogram(ac_arr, bins=200, range=(-100, 100))

    local_min = argrelextrema(hist, np.less,    order=2)[0]
    local_max = argrelextrema(hist, np.greater, order=2)[0]
    comb_ratio  = round(len(local_min) / (len(local_max) + 1e-6), 3)
    ac_kurtosis = round(
        float(np.mean((ac_arr - ac_arr.mean()) ** 4) / (ac_arr.std() ** 4 + 1e-10)), 3
    )

    return {
        "dct_ac_mean":             round(float(np.mean(ac_arr)), 4),
        "dct_ac_std":              round(float(np.std(ac_arr)), 4),
        "dct_ac_kurtosis":         ac_kurtosis,
        "histogram_local_minima":  int(len(local_min)),
        "histogram_local_maxima":  int(len(local_max)),
        "comb_ratio":              comb_ratio,
        "interpretation": (
            "DOUBLE-COMPRESSED — comb signature detected (image re-saved after editing)"
            if comb_ratio > 1.3 else
            "SINGLE-COMPRESSED — no double-compression signature found"
        ),
    }


# ── 6. Clone / Copy-Move Detection ──────────────────────────────────────────────
def clone_detection(image_path: str) -> dict:
    """
    SIMPLE EXPLANATION:
    Matches parts of the picture to other parts of the exact same picture. This 
    catches people using a 'clone stamp' tool to hide something by copying the 
    background over it.
    """
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=3000)
    kps, descs = sift.detectAndCompute(gray, None)

    if descs is None or len(descs) < 10:
        return {"keypoints": 0, "clone_matches": 0, "clone_ratio": 0.0,
                "interpretation": "Too few keypoints to analyse"}

    bf      = cv2.BFMatcher(cv2.NORM_L2)
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

    return {
        "keypoints":     len(kps),
        "clone_matches": clone_hits,
        "clone_ratio":   clone_ratio,
        "interpretation": (
            "SUSPICIOUS — high number of spatially-distant self-matches (possible clone stamp)"
            if clone_ratio > 0.25 else
            "CLEAN — no significant copy-move pattern detected"
        ),
    }


# ── 7. Color Anomaly (Strange Colors) ──────────────────────────────────────────
def color_anomaly_analysis(image_path: str, save_path: str) -> dict:
    """
    SIMPLE EXPLANATION:
    Looks for pixels that don't fit in with the rest of the picture's color palette.
    This is great for spotting if someone used a digital paintbrush to cover something 
    up or pasted text with a slightly different ink color.
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    foreground_mask = (V > 30) & (S > 15)

    h_fg = H[foreground_mask].astype(int)
    hist, _ = np.histogram(h_fg, bins=36, range=(0, 180))
    top3_bins = np.argsort(hist)[-3:]

    dominant_mask = np.zeros(36, dtype=bool)
    for b in top3_bins:
        for delta in range(-2, 3):
            dominant_mask[(b + delta) % 36] = True

    h_bin       = (H / 5).astype(int).clip(0, 35)
    is_dominant = dominant_mask[h_bin]
    anomaly_mask = (~is_dominant) & (S > 60) & (V > 40)

    anomaly_pixels  = int(np.sum(anomaly_mask))
    total_fg_pixels = int(np.sum(foreground_mask))
    anomaly_ratio   = round(anomaly_pixels / (total_fg_pixels + 1), 6)

    # Blob analysis
    anomaly_u8     = anomaly_mask.astype(np.uint8) * 255
    kernel         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    anomaly_closed = cv2.morphologyEx(anomaly_u8, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        anomaly_closed, connectivity=8
    )

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
                "area_px":          area,
                "bounding_box":     [bx, by, bw, bh],
                "mean_hue_degrees": round(float(np.mean(blob_px[:, 0])) * 2, 1),
                "mean_saturation":  round(float(np.mean(blob_px[:, 1])), 1),
            })

    blobs.sort(key=lambda b: b["area_px"], reverse=True)

    vis = img_rgb.copy()
    vis[anomaly_mask] = [255, 50, 50]
    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    return {
        "anomaly_pixels": anomaly_pixels,
        "anomaly_ratio":  anomaly_ratio,
        "anomaly_blobs":  blobs[:5],
        "interpretation": (
            "HIGHLY SUSPICIOUS — large chromatically-anomalous region(s) detected "
            "(possible drawn annotation, paint-over, or colour splice)"
            if anomaly_ratio > 0.003 else
            "CLEAN — no significant chromatic anomalies detected"
        ),
    }


# ── 8. Edge Discontinuity (Unnatural Sharpness) ───────────────────────────────
def edge_analysis(image_path: str, save_path: str) -> dict:
    """
    SIMPLE EXPLANATION:
    Finds areas with unnaturally sharp edges or way too many edges compared to 
    the rest of the photo. Digitally typed text or drawn lines often look much 
    sharper than natural objects in a photo.
    """
    img   = cv2.imread(image_path)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    tile = 32
    h, w = edges.shape
    densities = []
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            densities.append(float(np.mean(edges[y:y+tile, x:x+tile])))

    d_arr        = np.array(densities)
    mean_d       = float(np.mean(d_arr))
    high_d_ratio = round(float(np.sum(d_arr > mean_d * 3.0)) / len(d_arr), 4)

    density_map = np.zeros((h, w), dtype=np.float32)
    idx = 0
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            density_map[y:y+tile, x:x+tile] = d_arr[idx]
            idx += 1
    norm = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(save_path, cv2.applyColorMap(norm, cv2.COLORMAP_HOT))

    return {
        "mean_edge_density":       round(mean_d, 4),
        "high_density_tile_ratio": high_d_ratio,
        "edge_heatmap":            save_path,
        "interpretation": (
            "SUSPICIOUS — elevated localised edge density (drawn line / sharp boundary anomaly)"
            if high_d_ratio > 0.06 else
            "NORMAL — edge distribution consistent with natural image"
        ),
    }


# ── 9. Saturation Anomaly (Fake Vibrancy) ──────────────────────────────────────
def saturation_anomaly(image_path: str) -> dict:
    """
    SIMPLE EXPLANATION:
    Detects small areas that are unusually bright or colorful (saturated) compared 
    to the rest of the document, hinting at a digital color edit or filter applied 
    to just one spot.
    """
    img    = cv2.imread(image_path)
    S      = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    mean_s = float(np.mean(S))

    tile = 32
    h, w = S.shape
    high_tiles = total_tiles = 0
    high_coords: list[list[int]] = []
    for y in range(0, h - tile, tile):
        for x in range(0, w - tile, tile):
            if float(np.mean(S[y:y+tile, x:x+tile])) > mean_s * 3.0:
                high_tiles += 1
                high_coords.append([x, y, tile, tile])
            total_tiles += 1

    high_ratio = round(high_tiles / total_tiles, 4) if total_tiles else 0.0

    return {
        "mean_saturation":            round(mean_s, 3),
        "high_saturation_tile_ratio": high_ratio,
        "high_saturation_coords":     high_coords[:10],
        "interpretation": (
            "SUSPICIOUS — localised saturation spikes (possible annotation or colour edit)"
            if high_ratio > 0.02 else
            "NORMAL — saturation distribution uniform across image"
        ),
    }


# ── 10. Font Consistency (Mismatched Text) ────────────────────────────────────
def font_consistency_analysis(image_path: str, save_path: str) -> dict:
    """
    SIMPLE EXPLANATION:
    Looks specifically at all the letters in the document. It checks if some letters 
    have a different thickness, height, or blurriness than the rest. This catches 
    situations where someone printed a document, wrote over it physically, and then
    took a picture of it.
    """
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Adaptive threshold: handles scan-line gradients and uneven phone-camera lighting
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=15, C=8,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    char_data: list[tuple] = []  # (x, y, bw, bh, stroke_width, sharpness)
    for i in range(1, num_labels):
        cx   = int(stats[i, cv2.CC_STAT_LEFT])
        cy   = int(stats[i, cv2.CC_STAT_TOP])
        cbw  = int(stats[i, cv2.CC_STAT_WIDTH])
        cbh  = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        # Filter: reject noise (< 20 px) and non-character objects (borders, logos)
        if cbh < 6 or cbh > h * 0.12 or cbw < 2 or cbw > w * 0.25:
            continue
        if area < 20:
            continue
        aspect = cbw / (cbh + 1e-6)
        if aspect > 6.0 or aspect < 0.05:
            continue

        # Stroke width ≈ 2× the maximum inscribed-circle radius (distance transform).
        # Crop to bounding box — avoids running a full-image distance transform per blob.
        bx1, by1 = max(0, cx), max(0, cy)
        bx2, by2 = min(w, cx + cbw), min(h, cy + cbh)
        try:
            blob_crop = (labels[by1:by2, bx1:bx2] == i).astype(np.uint8)
            dist      = cv2.distanceTransform(blob_crop, cv2.DIST_L2, 3)
            stroke_w  = float(np.max(dist)) * 2.0
            if not np.isfinite(stroke_w):
                stroke_w = 0.0
        except Exception:
            stroke_w = 0.0

        # Local sharpness: Laplacian variance inside the character bounding box (±3 px)
        x1 = max(0, cx - 3);  y1 = max(0, cy - 3)
        x2 = min(w, cx + cbw + 3);  y2 = min(h, cy + cbh + 3)
        roi       = gray[y1:y2, x1:x2].astype(np.float64)
        sharpness = float(np.var(cv2.Laplacian(roi, cv2.CV_64F)))

        char_data.append((cx, cy, cbw, cbh, stroke_w, sharpness))

    if len(char_data) < 20:
        return {
            "char_count": len(char_data),
            "skipped":    True,
            "reason":     "Too few text characters detected — not a text document",
            "interpretation": "N/A — insufficient text for font analysis",
        }

    char_arr  = np.array(char_data, dtype=np.float32)
    heights   = char_arr[:, 3]
    strokes   = char_arr[:, 4]
    sharpness = char_arr[:, 5]

    def _safe_cv(arr: np.ndarray) -> float:
        """Coefficient of variation, ignoring non-finite values."""
        finite = arr[np.isfinite(arr)]
        if len(finite) < 2:
            return 0.0
        mu = float(np.mean(finite))
        return round(float(np.std(finite)) / (mu + 1e-6), 4) if mu > 1e-6 else 0.0

    def _iqr_outliers(arr: np.ndarray, factor: float = 1.5) -> np.ndarray:
        finite = arr[np.isfinite(arr)]
        if len(finite) < 4:
            return np.zeros(len(arr), dtype=bool)
        q1, q3 = np.percentile(finite, 25), np.percentile(finite, 75)
        iqr    = q3 - q1
        return (arr < q1 - factor * iqr) | (arr > q3 + factor * iqr)

    # Use only valid (finite, non-zero) stroke values for stroke statistics
    strokes_finite = strokes[np.isfinite(strokes) & (strokes > 0.5)]

    height_out = _iqr_outliers(heights)
    stroke_out = np.zeros(len(strokes), dtype=bool)
    if len(strokes_finite) >= 4:
        stroke_finite_mask       = np.isfinite(strokes) & (strokes > 0.5)
        stroke_out[stroke_finite_mask] = _iqr_outliers(strokes_finite)
    # Higher IQR factor for sharpness — natural variance is larger
    sharp_out  = _iqr_outliers(sharpness, factor=2.0)

    height_cv = _safe_cv(heights)
    stroke_cv = _safe_cv(strokes_finite if len(strokes_finite) >= 2 else strokes)
    sharp_cv  = _safe_cv(sharpness)

    height_outlier_ratio = round(float(np.sum(height_out)) / len(heights),   4)
    stroke_outlier_ratio = round(float(np.sum(stroke_out)) / len(strokes),   4)
    sharp_outlier_ratio  = round(float(np.sum(sharp_out))  / len(sharpness), 4)

    # A character is suspicious when ≥ 2 independent metrics flag it as an outlier
    combined_suspicious = (
        height_out.astype(int) + stroke_out.astype(int) + sharp_out.astype(int)
    ) >= 2

    # ── Spatial clustering ─────────────────────────────────────────────────────
    CELL   = 64
    grid_h = h // CELL + 1
    grid_w = w // CELL + 1
    grid   = np.zeros((grid_h, grid_w), dtype=np.int32)

    for idx, (cx, cy, cbw, cbh, _sw, _sh) in enumerate(char_data):
        gc = (cx + cbw // 2) // CELL
        gr = (cy + cbh // 2) // CELL
        if combined_suspicious[idx]:
            grid[gr, gc] += 1

    hot_map = (grid >= 2).astype(np.uint8)
    suspicious_regions: list[dict] = []
    if hot_map.any():
        n_reg, reg_labels, reg_stats, _ = cv2.connectedComponentsWithStats(
            hot_map, connectivity=8
        )
        for r in range(1, n_reg):
            rx = int(reg_stats[r, cv2.CC_STAT_LEFT])   * CELL
            ry = int(reg_stats[r, cv2.CC_STAT_TOP])    * CELL
            rw = int(reg_stats[r, cv2.CC_STAT_WIDTH])  * CELL
            rh = int(reg_stats[r, cv2.CC_STAT_HEIGHT]) * CELL
            n_susp = int(np.sum(grid[reg_labels == r]))
            suspicious_regions.append({
                "bounding_box":     [rx, ry, rw, rh],
                "suspicious_chars": n_susp,
            })
    suspicious_regions.sort(key=lambda r: r["suspicious_chars"], reverse=True)

    # ── Visualisation ──────────────────────────────────────────────────────────
    vis = img.copy()
    for idx, (cx, cy, cbw, cbh, _sw, _sh) in enumerate(char_data):
        if combined_suspicious[idx]:
            cv2.rectangle(vis, (cx, cy), (cx + cbw, cy + cbh), (0, 0, 220), 1)
    for reg in suspicious_regions:
        rx, ry, rw, rh = reg["bounding_box"]
        cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 140, 255), 2)
    cv2.imwrite(save_path, vis)

    n_regions    = len(suspicious_regions)
    is_suspicious = (
        (stroke_cv > 0.40 and n_regions >= 1) or
        (n_regions >= 2) or
        (sharp_outlier_ratio > 0.25 and n_regions >= 1)
    )

    return {
        "char_count":              len(char_data),
        "height_cv":               height_cv,
        "height_outlier_ratio":    height_outlier_ratio,
        "stroke_cv":               stroke_cv,
        "stroke_outlier_ratio":    stroke_outlier_ratio,
        "sharpness_cv":            sharp_cv,
        "sharpness_outlier_ratio": sharp_outlier_ratio,
        "n_suspicious_regions":    n_regions,
        "suspicious_regions":      suspicious_regions[:5],
        "font_heatmap":            save_path,
        "interpretation": (
            f"SUSPICIOUS — {n_regions} spatially-coherent font anomaly region(s): "
            f"stroke_cv={stroke_cv:.3f}, height_outlier={height_outlier_ratio*100:.1f}%, "
            f"sharpness_outlier={sharp_outlier_ratio*100:.1f}%"
            if is_suspicious else
            "CONSISTENT — font metrics uniform across document (no anomalous regions)"
        ),
    }


# ── 11. AI Generative Model Detection (Spectral & Structural) ─────────────────
def ai_artifact_analysis(image_path: str, save_path: str) -> dict:
    """
    SIMPLE EXPLANATION:
    Modern AI image generators (like Midjourney or Stable Diffusion) don't use 
    copy-pasting. Instead, they generate pixels from scratch. This process often 
    leaves invisible "checkerboard" grid patterns (upsampling artifacts) and 
    creates text that looks like unreadable, garbled blobs.
    This checks for those weird grids using frequency mathematics (FFT).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"skipped": True, "reason": "Could not read image"}

    # 1. Spectral Analysis (FFT) for grid artifacts
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
    
    # Mask out the low frequencies (center core)
    mask = dist_from_center > (min(h, w) * 0.3)
    high_freq_data = magnitude_spectrum[mask]
    
    if len(high_freq_data) == 0:
        high_freq_mean = 0.0
        high_freq_max = 0.0
    else:
        high_freq_mean = float(np.mean(high_freq_data))
        high_freq_max = float(np.max(high_freq_data))
        
    spike_ratio = round(high_freq_max / (high_freq_mean + 1e-6), 2)
    
    # Save the spectrum heatmap
    vis = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(save_path, cv2.applyColorMap(vis, cv2.COLORMAP_JET))
    
    is_suspicious = spike_ratio > 3.0
    
    return {
        "skipped": False,
        "high_freq_mean":  round(high_freq_mean, 2),
        "high_freq_max":   round(high_freq_max, 2),
        "spike_ratio":     spike_ratio,
        "ai_heatmap":      save_path,
        "interpretation": (
            "SUSPICIOUS — Unnatural high-frequency grid artifacts detected (Possible AI generation)"
            if is_suspicious else
            "NORMAL — Frequency spectrum looks natural"
        )
    }


# ── Scoring ────────────────────────────────────────────────────────────────────
def compute_score(
    result: dict, peer_result: dict | None = None
) -> tuple[float, list[str]]:
    """
    SIMPLE EXPLANATION:
    This is the judge. It takes all the suspicious flags found by the tests above 
    and adds up "guilt points". If the total score hits certain thresholds, 
    it brands the image as TAMPERED.
    """
    score: float       = 0.0
    evidence: list[str] = []

    # 1. ELA
    ela = result.get("ela", {})
    if ela.get("suspicious_block_ratio", 0) > 0.05:
        score += 25
        evidence.append(
            f"ELA: {ela['suspicious_block_ratio']*100:.1f}% of blocks have "
            f"anomalous re-compression (>2.5× mean)"
        )
    elif ela.get("mean_ela", 0) > 8:
        score += 12
        evidence.append(f"ELA: elevated mean ELA ({ela['mean_ela']:.1f}) — possible re-editing")

    # 2. Metadata flags
    meta = result.get("metadata", {})
    for flag in meta.get("suspicious_flags", []):
        score += 10
        evidence.append(f"Metadata: {flag}")
    # Peer: fewer tags than reference → metadata was stripped after editing
    if peer_result:
        own_tags  = meta.get("total_tags", 0)
        peer_tags = peer_result.get("metadata", {}).get("total_tags", 0)
        if peer_tags > 0 and own_tags < peer_tags * 0.6:
            score += 12
            evidence.append(
                f"Metadata: only {own_tags} tags vs peer's {peer_tags} "
                f"— metadata stripped (common in edited re-saves)"
            )

    # 3. File entropy (peer-relative: identical content re-compressed → lower entropy)
    if peer_result:
        own_ent  = result.get("file_entropy_bits", 0.0)
        peer_ent = peer_result.get("file_entropy_bits", 0.0)
        if peer_ent > 0 and own_ent < peer_ent * 0.97:
            score += 8
            evidence.append(
                f"Entropy: {own_ent:.4f} bits/byte vs peer {peer_ent:.4f} "
                f"— lower entropy suggests re-compression loss"
            )

    # 4. Noise residual
    noise = result.get("noise", {})
    if noise.get("hotspot_tile_ratio", 0) > 0.10:
        score += 15
        evidence.append(
            f"Noise: {noise['hotspot_tile_ratio']*100:.1f}% of tiles have "
            f"anomalous noise (possible splice boundary)"
        )

    # 5. DCT double-compression (JPEG only)
    dct = result.get("dct")
    if dct and dct.get("comb_ratio", 0) > 1.3:
        score += 20
        evidence.append(
            f"DCT: double-compression comb detected (ratio {dct['comb_ratio']}) "
            f"— image was re-saved after editing"
        )

    # 6. Clone / copy-move
    clone            = result.get("clone", {})
    own_clone_ratio  = clone.get("clone_ratio", 0)
    peer_clone_ratio = (
        peer_result.get("clone", {}).get("clone_ratio", 0) if peer_result else 0
    )
    if peer_result:
        if own_clone_ratio > peer_clone_ratio * 1.3 and own_clone_ratio > 0.20:
            score += 12
            evidence.append(
                f"Clone: ratio {own_clone_ratio:.3f} >> peer {peer_clone_ratio:.3f} "
                f"— unique copy-move pattern"
            )
    elif own_clone_ratio > 0.25:
        score += 12
        evidence.append(f"Clone: clone_ratio {own_clone_ratio:.3f} — possible copy-move")

    # 7. Color anomaly (strongest single signal for painted/spliced tampering)
    color        = result.get("color_anomaly", {})
    ratio        = color.get("anomaly_ratio", 0)
    blobs        = color.get("anomaly_blobs", [])
    largest_blob = blobs[0]["area_px"] if blobs else 0
    if ratio > 0.01 or largest_blob > 2000:
        score += 35
        evidence.append(
            f"Color anomaly: {ratio*100:.2f}% of pixels have implausible hue "
            f"(largest blob {largest_blob:,} px) — drawn annotation or colour splice"
        )
    elif ratio > 0.003:
        score += 18
        evidence.append(
            f"Color anomaly: {ratio*100:.3f}% of pixels suspicious — possible light edit"
        )

    # 8. Edge discontinuity
    edge = result.get("edge", {})
    if edge.get("high_density_tile_ratio", 0) > 0.06:
        score += 12
        evidence.append(
            f"Edge: {edge['high_density_tile_ratio']*100:.1f}% of tiles have "
            f"unnaturally high edge density (drawn line or sharp boundary)"
        )

    # 9. Saturation anomaly
    sat = result.get("saturation", {})
    if sat.get("high_saturation_tile_ratio", 0) > 0.02:
        score += 8
        evidence.append(
            f"Saturation: {sat['high_saturation_tile_ratio']*100:.1f}% of tiles "
            f"are over-saturated vs image mean"
        )

    # 10. Font consistency
    font = result.get("font", {})
    if not font.get("skipped"):
        n_regions  = font.get("n_suspicious_regions", 0)
        stroke_cv  = font.get("stroke_cv", 0.0)
        sharp_out  = font.get("sharpness_outlier_ratio", 0.0)
        height_out = font.get("height_outlier_ratio", 0.0)
        if stroke_cv > 0.40 and n_regions >= 1:
            score += 20
            evidence.append(
                f"Font: stroke_cv={stroke_cv:.3f} with {n_regions} localised anomaly region(s) "
                f"— multiple stroke widths / font families detected (possible sticker or different printer)"
            )
        elif n_regions >= 2:
            score += 15
            evidence.append(
                f"Font: {n_regions} spatially-coherent character anomaly clusters "
                f"— height_outlier={height_out*100:.1f}%, stroke_outlier={font.get('stroke_outlier_ratio', 0)*100:.1f}%"
            )
        elif sharp_out > 0.25 and n_regions >= 1:
            score += 15
            evidence.append(
                f"Font: sharpness outlier ratio {sharp_out*100:.1f}% in {n_regions} region(s) "
                f"— tampered region at different focal plane (re-photograph of altered printout)"
            )
        elif stroke_cv > 0.55:
            score += 10
            evidence.append(
                f"Font: high stroke-width variation (cv={stroke_cv:.3f}) "
                f"— possible mixed font families or handwriting insertion"
            )

    # 11. File size (peer-relative: re-saved files are typically smaller)
    if peer_result:
        own_size  = result.get("file_size_bytes", 0)
        peer_size = peer_result.get("file_size_bytes", 0)
        if peer_size > 0 and own_size < peer_size * 0.85:
            score += 8
            evidence.append(
                f"File size: {own_size:,} B vs peer {peer_size:,} B — "
                f"{100*(1-own_size/peer_size):.0f}% smaller (re-compression after editing)"
            )

    # 12. AI Generative artifacts
    ai_gen = result.get("ai_artifacts", {})
    if not ai_gen.get("skipped"):
        spike_ratio = ai_gen.get("spike_ratio", 0)
        if spike_ratio > 3.5:
            score += 25
            evidence.append(f"AI Generation: Severe frequency grid artifacts (spike ratio {spike_ratio}) — likely AI generated")
        elif spike_ratio > 3.0:
            score += 15
            evidence.append(f"AI Generation: High-frequency anomalies (spike ratio {spike_ratio}) — possible AI upsampling")

    return min(score, 100.0), evidence


# ── Per-image orchestrator ────────────────────────────────────────────────────
def analyse_image(path: str, out_dir: str) -> dict:
    """
    SIMPLE EXPLANATION:
    This is the central manager for a single picture. It calls all 11 tests 
    one by one, prints out their quick results, and collects all the data 
    into one big report.
    """
    label = _label(path)
    fmt   = _detect_format(path)

    print(f"\n  [1/10] ELA ...")
    ela = ela_analysis(path, os.path.join(out_dir, f"ela_{label}.png"))
    print(f"        mean={ela['mean_ela']}  "
          f"suspicious_blocks={ela['suspicious_block_ratio']*100:.1f}%")

    print(f"  [2/10] Metadata ...")
    meta = metadata_analysis(path, fmt)
    print(f"        format={fmt}  tags={meta['total_tags']}  "
          f"flags={len(meta['suspicious_flags'])}")
    for flag in meta["suspicious_flags"]:
        print(f"        ⚑ {flag}")

    print(f"  [3/10] File entropy ...")
    entropy = file_entropy(path)
    print(f"        {entropy} bits/byte")

    print(f"  [4/10] Noise residual ...")
    noise = noise_analysis(path, os.path.join(out_dir, f"noise_{label}.png"))
    print(f"        hotspot={noise['hotspot_tile_ratio']*100:.1f}%  "
          f"→ {noise['interpretation'][:55]}")

    print(f"  [5/10] DCT double-compression "
          f"{'(skipped — not JPEG)' if fmt != 'JPEG' else ''} ...")
    dct = dct_analysis(path, fmt)
    if dct:
        print(f"        comb_ratio={dct['comb_ratio']}  → {dct['interpretation'][:55]}")
    else:
        print(f"        — N/A for {fmt}")

    print(f"  [6/10] Clone / copy-move ...")
    clone = clone_detection(path)
    print(f"        keypoints={clone['keypoints']}  "
          f"clone_ratio={clone['clone_ratio']}")

    print(f"  [7/10] Color anomaly ...")
    color = color_anomaly_analysis(
        path, os.path.join(out_dir, f"color_anomaly_{label}.png")
    )
    print(f"        anomaly_ratio={color['anomaly_ratio']*100:.3f}%  "
          f"blobs={len(color['anomaly_blobs'])}")
    for b in color["anomaly_blobs"][:3]:
        print(f"        blob: {b['area_px']:,} px  "
              f"hue={b['mean_hue_degrees']}°  sat={b['mean_saturation']:.0f}")

    print(f"  [8/10] Edge discontinuity ...")
    edge = edge_analysis(path, os.path.join(out_dir, f"edge_{label}.png"))
    print(f"        high_density_ratio={edge['high_density_tile_ratio']*100:.1f}%  "
          f"→ {edge['interpretation'][:55]}")

    print(f"  [9/10] Saturation anomaly ...")
    sat = saturation_anomaly(path)
    print(f"        high_sat_ratio={sat['high_saturation_tile_ratio']*100:.1f}%  "
          f"→ {sat['interpretation'][:55]}")

    print(f"  [10/10] Font consistency ...")
    font = font_consistency_analysis(
        path, os.path.join(out_dir, f"font_consistency_{label}.png")
    )
    if font.get("skipped"):
        print(f"        skipped — {font['reason']}")
    else:
        print(f"        chars={font['char_count']}  "
              f"stroke_cv={font['stroke_cv']}  "
              f"suspicious_regions={font['n_suspicious_regions']}")
        print(f"        → {font['interpretation'][:70]}")

    print(f"  [11/11] AI Generative model detection ...")
    ai_art = ai_artifact_analysis(
        path, os.path.join(out_dir, f"ai_artifacts_{label}.png")
    )
    if ai_art.get("skipped"):
        print(f"        skipped — {ai_art.get('reason')}")
    else:
        print(f"        spike_ratio={ai_art['spike_ratio']}  "
              f"→ {ai_art['interpretation'][:60]}")

    return {
        "path":              path,
        "format":            fmt,
        "file_size_bytes":   os.path.getsize(path),
        "file_entropy_bits": entropy,
        "ela":               ela,
        "metadata":          meta,
        "noise":             noise,
        "dct":               dct,
        "clone":             clone,
        "color_anomaly":     color,
        "edge":              edge,
        "saturation":        sat,
        "font":              font,
        "ai_artifacts":      ai_art,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python forensics_detect.py <image> [<reference_image>]")
        sys.exit(1)

    paths  = sys.argv[1:]
    labels = [_label(p) for p in paths]
    out_dir = _out_dir(paths[0])

    for path in paths:
        if not os.path.exists(path):
            print(f"ERROR: file not found — {path}")
            sys.exit(1)

    mode   = "pair" if len(paths) == 2 else "single"
    report: dict = {"mode": mode, "images": {}}

    for path, label in zip(paths, labels):
        print(f"\n{'='*62}")
        print(f"  Analysing: {label}")
        print(f"  File:      {path}  ({os.path.getsize(path):,} bytes)")
        print(f"{'='*62}")
        report["images"][label] = analyse_image(path, out_dir)

    # Score all images (peer-aware in pair mode)
    img_labels = list(report["images"].keys())
    for i, label in enumerate(img_labels):
        result = report["images"][label]
        peer   = report["images"][img_labels[1 - i]] if mode == "pair" else None
        score, evidence = compute_score(result, peer)
        verdict = (
            "TAMPERED"        if score >= 55 else
            "LIKELY TAMPERED" if score >= 30 else
            "UNCERTAIN"       if score >= 15 else
            "ORIGINAL"
        )
        result["forgery_score_0_100"] = round(score, 1)
        result["verdict"]             = verdict
        result["evidence"]            = evidence

        print(f"\n  ► {label}  score={score:.0f}/100  verdict={verdict}")
        for ev in evidence:
            print(f"    • {ev}")

    # Pair summary verdict
    if mode == "pair":
        s0 = report["images"][img_labels[0]]["forgery_score_0_100"]
        s1 = report["images"][img_labels[1]]["forgery_score_0_100"]

        if abs(s0 - s1) < 10:
            original = tampered = "UNCERTAIN"
            conclusion = "Scores too close to determine with confidence — manual review recommended."
        elif s0 > s1:
            original, tampered = img_labels[1], img_labels[0]
            conclusion = (
                f"{img_labels[0]} scores {s0:.0f}/100 (TAMPERED). "
                f"{img_labels[1]} is likely ORIGINAL ({s1:.0f}/100)."
            )
        else:
            original, tampered = img_labels[0], img_labels[1]
            conclusion = (
                f"{img_labels[1]} scores {s1:.0f}/100 (TAMPERED). "
                f"{img_labels[0]} is likely ORIGINAL ({s0:.0f}/100)."
            )

        report["pair_verdict"] = {
            "original":   original,
            "tampered":   tampered,
            "scores":     {img_labels[0]: s0, img_labels[1]: s1},
            "conclusion": conclusion,
        }

        print(f"\n{'='*62}")
        print(f"  PAIR VERDICT")
        print(f"{'='*62}")
        print(f"  {conclusion}")

    # Write report
    out_path = os.path.join(out_dir, "forensics_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Report    → {out_path}")
    print(f"  Heatmaps  → {out_dir}/")
    for label in img_labels:
        print(f"    ela_{label}.png  noise_{label}.png  "
              f"color_anomaly_{label}.png  edge_{label}.png  font_consistency_{label}.png  ai_artifacts_{label}.png")


if __name__ == "__main__":
    main()
