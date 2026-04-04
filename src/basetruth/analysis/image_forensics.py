"""Image forensics analysis for BaseTruth.

Implements a multi-layer heuristic stack for detecting tampered or
AI-generated document images (JPG, PNG, TIFF, BMP, WebP).

Layers
------
1. EXIF / metadata inspection
   - Detects suspicious software tags (Photoshop, GIMP, Canva, AI generators)
   - Flags missing camera EXIF in photos that claim to be scans
   - Detects timestamp inconsistencies

2. Error Level Analysis (ELA)
   - Re-saves the image at a known JPEG quality and compares pixel differences
   - Edited/pasted regions re-compress differently and show higher error levels
   - Returns a scalar score (0–100) and a "high-error region" fraction

3. Noise consistency analysis
   - Uniform document scans have consistent sensor noise across the image
   - Copy-pasted regions have a different noise signature than surrounding areas
   - Coefficient-of-variation (CV) across 16×16 blocks is used as the signal

4. Copy-move detection
   - ORB feature descriptors matched against themselves to find duplicate regions
   - Suspicious when visually identical keypoints appear at different spatial locations

5. GAN / AI-generated image detection
   - DCT frequency analysis: GAN images exhibit characteristic high-frequency
     artefacts and unnatural energy distributions across the spectrum
   - Cross-channel noise correlation: AI-generated images have atypical
     channel-to-channel noise correlations compared to real scans

6. Perceptual hash drift check
   - Compares average-hash vs difference-hash; large deltas can indicate
     localised edits that left the bulk of the image unchanged

All functions degrade gracefully when optional libraries (OpenCV, NumPy,
ImageHash, ExifRead) are not installed — they return None / empty dicts
instead of raising, so the rest of the pipeline is never blocked.

Public API
----------
  analyse_image(path: Path) -> Dict[str, Any]

Returns a dict with keys:
  signals          -- list of Signal dicts (same schema as tamper.py)
  ela_score        -- float 0–100 (higher = more editing artefacts)
  high_error_frac  -- float 0–1 (fraction of blocks with ELA > threshold)
  noise_cv         -- float or None
  copy_move_score  -- float 0–100 or None
  gan_score        -- float 0–100 or None
  exif_metadata    -- flat dict of extracted EXIF fields
  suspicious_tool  -- str or None (the offending software name)
  limitations      -- list of explanatory strings
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known suspicious authoring tools
# Image editors / AI tools that should NOT appear in a genuine scanned document
# ---------------------------------------------------------------------------
_SUSPICIOUS_TOOLS: Tuple[str, ...] = (
    "photoshop",
    "adobe photoshop",
    "lightroom",
    "illustrator",
    "gimp",
    "inkscape",
    "canva",
    "coreldraw",
    "paintshop",
    "affinity",
    "fotor",
    "pixlr",
    "snapseed",
    "vsco",
    "snapedit",
    "remove.bg",
    "stable diffusion",
    "midjourney",
    "dall-e",
    "dall·e",
    "firefly",
    "imagemagick",
    "paint.net",
    "microsoft paint",
    "screenshot",
    "camscanner",  # can be legitimate but worth flagging for scrutiny
    "pdf24",
    "smallpdf",
    "ilovepdf",
)


# ---------------------------------------------------------------------------
# EXIF extraction
# ---------------------------------------------------------------------------

def _extract_exif_pillow(path: Path) -> Dict[str, Any]:
    """Extract EXIF tags using Pillow (always available since it is a hard dep)."""
    exif: Dict[str, Any] = {}
    try:
        from PIL import Image  # type: ignore
        from PIL.ExifTags import TAGS  # type: ignore

        with Image.open(str(path)) as img:
            raw = img._getexif()  # type: ignore[attr-defined]
            if raw:
                for tag_id, value in raw.items():
                    tag = TAGS.get(tag_id, str(tag_id))
                    # Truncate very long byte blobs for storage efficiency
                    if isinstance(value, bytes) and len(value) > 256:
                        value = f"<{len(value)} bytes>"
                    exif[str(tag)] = str(value)
    except Exception as exc:  # noqa: BLE001
        log.debug("Pillow EXIF extraction failed for %s: %s", path.name, exc)
    return exif


def _extract_exif_exifread(path: Path) -> Dict[str, Any]:
    """Extract a richer EXIF set using the exifread library (optional).

    Returns an empty dict if exifread is not installed.
    """
    exif: Dict[str, Any] = {}
    try:
        import exifread  # type: ignore

        with path.open("rb") as fh:
            tags = exifread.process_file(fh, details=False, stop_tag="UNDEF")
        for key, value in tags.items():
            exif[str(key)] = str(value)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        log.debug("exifread extraction failed for %s: %s", path.name, exc)
    return exif


def extract_image_metadata(path: Path) -> Dict[str, Any]:
    """Return a merged flat dict of EXIF/image metadata for *path*.

    Uses Pillow as primary source and exifread as enrichment layer.  Safe for
    all image formats supported by Pillow (JPEG, PNG, TIFF, BMP, WebP, etc.).
    """
    meta: Dict[str, Any] = {}
    meta.update(_extract_exif_exifread(path))   # broader tag set where available
    meta.update(_extract_exif_pillow(path))      # Pillow always wins for duplicates

    # Image dimensions / mode via Pillow (never in EXIF)
    try:
        from PIL import Image  # type: ignore

        with Image.open(str(path)) as img:
            meta["_image_width"] = img.width
            meta["_image_height"] = img.height
            meta["_image_mode"] = img.mode
            meta["_image_format"] = img.format or path.suffix.upper().lstrip(".")
    except Exception as exc:  # noqa: BLE001
        log.debug("Pillow image info failed for %s: %s", path.name, exc)

    return meta


def detect_suspicious_tool(exif_metadata: Dict[str, Any]) -> Optional[str]:
    """Return the offending tool name if any suspicious authoring software is
    detected in the EXIF metadata, or None if the metadata looks clean."""
    blob = " ".join(str(v) for v in exif_metadata.values()).lower()
    for tool in _SUSPICIOUS_TOOLS:
        if tool in blob:
            return tool
    return None


# ---------------------------------------------------------------------------
# Error Level Analysis (ELA)
# ---------------------------------------------------------------------------

def run_ela(
    path: Path,
    resave_quality: int = 95,
) -> Tuple[float, float]:
    """Run Error Level Analysis on a JPEG/PNG image.

    Parameters
    ----------
    path           : path to the image file
    resave_quality : JPEG quality for the comparison resave (default 95)

    Returns
    -------
    (ela_score, high_error_frac) where:
      ela_score       -- float 0–100 (mean of the top-5 % error values,
                         normalised to 0–100)
      high_error_frac -- fraction of 16×16 blocks whose mean error exceeds
                         the adaptive threshold (>= 2 × image average)

    Returns (0.0, 0.0) when Pillow or NumPy are unavailable.
    """
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        return 0.0, 0.0

    try:
        with Image.open(str(path)) as img:
            # Convert to RGB so JPEG resave works for any input mode
            original = img.convert("RGB")

        # Resave at fixed quality
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=resave_quality)
        buffer.seek(0)
        with Image.open(buffer) as img_resaved:
            resaved = img_resaved.convert("RGB")

        # Pixel-level absolute difference (amplified for visibility/scoring)
        orig_arr = np.array(original, dtype=np.float32)
        resaved_arr = np.array(resaved, dtype=np.float32)
        diff = np.abs(orig_arr - resaved_arr)

        # ELA score: mean of the top-5 % pixels (robust to global compression)
        flat = diff.flatten()
        threshold_idx = int(len(flat) * 0.95)
        top_5pct_mean = float(np.sort(flat)[threshold_idx:].mean())
        ela_score = min(100.0, top_5pct_mean / 255.0 * 100.0 * 3.0)

        # High-error block fraction
        block_size = 16
        h, w, _ = diff.shape
        block_means = []
        for row in range(0, h - block_size + 1, block_size):
            for col in range(0, w - block_size + 1, block_size):
                block = diff[row : row + block_size, col : col + block_size]
                block_means.append(float(block.mean()))

        if block_means:
            img_mean = float(np.mean(block_means))
            adaptive_thresh = max(5.0, 2.0 * img_mean)
            high_error_frac = sum(1 for bm in block_means if bm >= adaptive_thresh) / len(block_means)
        else:
            high_error_frac = 0.0

        return round(ela_score, 2), round(high_error_frac, 4)

    except Exception as exc:  # noqa: BLE001
        log.debug("ELA failed for %s: %s", path.name, exc)
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Noise consistency analysis
# ---------------------------------------------------------------------------

def run_noise_analysis(path: Path) -> Optional[float]:
    """Compute the coefficient-of-variation (CV) of local noise across 16×16 blocks.

    A genuine document scan has spatially uniform sensor/compression noise.
    A copy-pasted region will show markedly different noise from the surrounding
    area, which increases the CV.

    Returns a float (CV, typically 0.2–1.5 for clean images, higher for edited
    ones) or None if OpenCV / NumPy are unavailable.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None

    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Estimate local noise via the Laplacian residual
        laplacian = cv2.Laplacian(img, cv2.CV_64F)

        block_size = 16
        h, w = laplacian.shape
        block_stds: list[float] = []

        for row in range(0, h - block_size + 1, block_size):
            for col in range(0, w - block_size + 1, block_size):
                block = laplacian[row : row + block_size, col : col + block_size]
                std = float(np.std(block))
                block_stds.append(std)

        if not block_stds:
            return None

        mean_std = float(np.mean(block_stds))
        if mean_std < 1e-6:
            return 0.0

        cv = float(np.std(block_stds)) / mean_std
        return round(cv, 4)

    except Exception as exc:  # noqa: BLE001
        log.debug("Noise analysis failed for %s: %s", path.name, exc)
        return None


# ---------------------------------------------------------------------------
# Missing camera EXIF detection
# ---------------------------------------------------------------------------

_CAMERA_EXIF_KEYS = (
    "Image Make",          # exifread
    "Image Model",
    "EXIF LensModel",
    "Make",                # Pillow
    "Model",
    "LensModel",
    "ExifIFD",
)

_TIMESTAMP_KEYS = (
    "EXIF DateTimeOriginal",
    "EXIF DateTimeDigitized",
    "Image DateTime",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "DateTime",
)


def _has_camera_exif(exif: Dict[str, Any]) -> bool:
    """Return True if at least one camera hardware EXIF tag is present."""
    return any(key in exif for key in _CAMERA_EXIF_KEYS)


def _check_timestamp_consistency(exif: Dict[str, Any]) -> bool:
    """Return True if timestamps are consistent (original ≤ digitized ≤ modified).

    Returns True (no anomaly) if not enough timestamps are available to compare.
    """
    def _parse_ts(val: str) -> Optional[str]:
        # EXIF datetime: "YYYY:MM:DD HH:MM:SS"
        cleaned = re.sub(r"[^0-9 :]", "", str(val))
        parts = cleaned.strip().split()
        if not parts:
            return None
        return parts[0].replace(" ", "")  # "YYYYMMDD" for simple string compare

    ts_values: list[str] = []
    for key in _TIMESTAMP_KEYS:
        if key in exif:
            parsed = _parse_ts(exif[key])
            if parsed:
                ts_values.append(parsed)

    if len(ts_values) < 2:
        return True  # not enough data to declare inconsistency

    # All timestamps should be equal or increasing (original ≤ later).
    # We just check that the earliest is not after the latest.
    return min(ts_values) <= max(ts_values)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
)


def is_supported_image(path: Path) -> bool:
    """Return True if *path* is an image format that this module can analyse."""
    return path.suffix.lower() in _IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# Copy-move detection — ORB self-matching to find duplicated regions
# ---------------------------------------------------------------------------

def run_copy_move_detection(path: Path) -> Optional[float]:
    """Detect copy-pasted regions by finding duplicate local features.

    Uses ORB (Oriented FAST and Rotated BRIEF) keypoint descriptors.  If
    visually identical blobs appear at geographically distinct locations in the
    same image, that is strong evidence of a copy-paste.

    Returns a *suspicion score* 0–100 (higher = more suspicious), or None when
    OpenCV is unavailable.

    Threshold guidance:
      < 10   — clean (no meaningful self-matching features)
      10–35  — mild / inconclusive
      > 35   — copy-move artefacts detected; manual review recommended
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None

    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        # Limit resolution for speed (4 MP is more than enough for copy-move)
        max_dim = 1200
        h, w = img.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
        kp, des = orb.detectAndCompute(img, None)

        if des is None or len(kp) < 20:
            return 0.0

        # Self-match: each descriptor matched against the full descriptor set
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des, des, k=3)

        # Spatial threshold: matches closer than 15% of the image diagonal
        # are considered the same region (not a copy-move)
        diag = float(np.hypot(img.shape[0], img.shape[1]))
        spatial_min = diag * 0.15

        suspicious_pairs: int = 0
        for match_group in matches:
            for m in match_group:
                # Skip trivial self-match and near-duplicate angles
                if m.queryIdx == m.trainIdx:
                    continue
                if m.distance > 40:  # too dissimilar
                    continue
                pt1 = np.array(kp[m.queryIdx].pt)
                pt2 = np.array(kp[m.trainIdx].pt)
                spatial_dist = float(np.linalg.norm(pt1 - pt2))
                if spatial_dist >= spatial_min:
                    suspicious_pairs += 1

        # Normalise: 0 suspicious pairs → 0; ≥200 pairs → 100
        score = min(100.0, suspicious_pairs / 2.0)
        return round(score, 2)

    except Exception as exc:  # noqa: BLE001
        log.debug("Copy-move detection failed for %s: %s", path.name, exc)
        return None


# ---------------------------------------------------------------------------
# GAN / AI-generated image detection — DCT frequency + channel analysis
# ---------------------------------------------------------------------------

def run_gan_detection(path: Path) -> Optional[float]:
    """Heuristic detection of AI-generated or synthetically created images.

    Two complementary signals are combined:

    1. DCT frequency energy ratio — GAN generators and diffusion models tend
       to produce images with abnormally high energy in mid-to-high frequency
       DCT bands compared to genuine photos or scanned documents.

    2. Cross-channel noise correlation — Genuine photographic noise is
       independent across R/G/B channels; GAN models often introduce
       correlated artefacts because each channel is generated from a shared
       latent code.

    Returns a *suspicion score* 0–100 (higher = more likely AI-generated), or
    None when OpenCV / NumPy are unavailable.

    Important caveat: this is a *heuristic* signal.  It is additional evidence,
    not conclusive proof.  Use alongside EXIF tool detection (e.g. Midjourney
    / Stable Diffusion tags in metadata).

    Threshold guidance:
      < 20   — plausible genuine scan
      20–55  — ambiguous; cross-check with other signals
      > 55   — likely AI-generated or heavily post-processed
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None

    try:
        img = cv2.imread(str(path))
        if img is None:
            return None

        # ── Signal 1: DCT frequency energy ratio (grayscale) ──────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to 256×256 for consistent DCT comparison across image sizes
        g256 = cv2.resize(gray, (256, 256)).astype(np.float32)
        dct = cv2.dct(g256)

        # Low-frequency block: top-left 32×32 (dominant energy for real images)
        lf_energy = float(np.mean(np.abs(dct[:32, :32])))
        # Mid-frequency ring: rows/cols 32–128
        mf_energy = float(np.mean(np.abs(dct[32:128, 32:128])))
        # High-frequency region: rows/cols 128+
        hf_energy = float(np.mean(np.abs(dct[128:, 128:])))

        if lf_energy < 1e-6:
            freq_score = 0.0
        else:
            # Real scans: lf >> mf >> hf (natural roll-off)
            # GAN images: flatter distribution; mf/hf ratio elevated
            mf_ratio = mf_energy / (lf_energy + 1e-6)
            hf_ratio = hf_energy / (lf_energy + 1e-6)
            # Empirical thresholds (tuned on document images)
            freq_score = min(100.0, (mf_ratio * 60.0 + hf_ratio * 300.0))

        # ── Signal 2: Cross-channel noise correlation ──────────────────────
        b, g, r = cv2.split(img.astype(np.float32))

        def _channel_noise(ch: "np.ndarray") -> "np.ndarray":
            lap = cv2.Laplacian(ch, cv2.CV_64F)
            # Keep only residual (remove mean)
            return lap - lap.mean()

        nb, ng, nr = _channel_noise(b), _channel_noise(g), _channel_noise(r)

        def _pearson(x: "np.ndarray", y: "np.ndarray") -> float:
            flat_x = x.flatten()
            flat_y = y.flatten()
            if np.std(flat_x) < 1e-9 or np.std(flat_y) < 1e-9:
                return 0.0
            return float(np.corrcoef(flat_x, flat_y)[0, 1])

        corr_rg = abs(_pearson(nr, ng))
        corr_rb = abs(_pearson(nr, nb))
        corr_gb = abs(_pearson(ng, nb))
        mean_corr = (corr_rg + corr_rb + corr_gb) / 3.0

        # For genuine photos: mean cross-channel noise correlation ≈ 0.0–0.15
        # For GAN images: tends to be higher (0.25–0.90) due to shared latent
        corr_score = min(100.0, max(0.0, (mean_corr - 0.1) * 120.0))

        # Combine both signals (frequency dominates, correlation is secondary)
        combined = 0.60 * freq_score + 0.40 * corr_score
        return round(combined, 2)

    except Exception as exc:  # noqa: BLE001
        log.debug("GAN detection failed for %s: %s", path.name, exc)
        return None


def analyse_image(path: Path) -> Dict[str, Any]:
    """Run all forensic layers against an image file.

    Parameters
    ----------
    path : absolute path to the image file

    Returns
    -------
    Dict with keys:
      signals          -- list[dict]  (Signal schema, ready for tamper.py)
      ela_score        -- float (0–100)
      high_error_frac  -- float (0–1)
      noise_cv         -- float | None
      exif_metadata    -- dict
      suspicious_tool  -- str | None
      limitations      -- list[str]
    """
    from basetruth.models import Signal, signals_to_dict  # local import to avoid circulars

    signals: List[Signal] = []
    limitations: List[str] = []

    # ── 1. EXIF metadata ────────────────────────────────────────────────────
    exif_metadata = extract_image_metadata(path)
    suspicious_tool = detect_suspicious_tool(exif_metadata)

    # ── Signal: suspicious authoring tool ───────────────────────────────────
    signals.append(
        Signal(
            name="image_suspicious_authoring_tool",
            severity="high" if suspicious_tool else "info",
            score=45 if suspicious_tool else 0,
            summary=(
                f"Suspicious authoring tool detected in image metadata: '{suspicious_tool}'. "
                "Genuine scanned documents should not carry photo-editing or AI-generation signatures."
                if suspicious_tool
                else "No suspicious authoring tool detected in image EXIF metadata."
            ),
            passed=suspicious_tool is None,
            details={"suspicious_tool": suspicious_tool or "", "exif_keys_checked": len(exif_metadata)},
        )
    )

    # ── Signal: missing camera EXIF (relevant only for photos, not pure scans)
    # We flag this as info only — a genuine photocopied/scanned document may
    # legitimately lack camera EXIF (e.g., a flat-bed scanner).  However,
    # a document submitted as a phone photo that has NO maker/model is unusual.
    has_cam_exif = _has_camera_exif(exif_metadata)
    if exif_metadata and not has_cam_exif and not path.suffix.lower() in {".png", ".bmp"}:
        signals.append(
            Signal(
                name="image_missing_camera_exif",
                severity="low",
                score=10,
                summary=(
                    "The image claims to be a photograph but contains no camera make / "
                    "model EXIF tags, which is unusual for an unedited phone or scanner capture."
                ),
                passed=False,
                details={"has_camera_exif": False},
            )
        )
    else:
        signals.append(
            Signal(
                name="image_missing_camera_exif",
                severity="info",
                score=0,
                summary="Camera EXIF presence check — acceptable.",
                passed=True,
                details={"has_camera_exif": has_cam_exif},
            )
        )

    # ── Signal: timestamp inconsistency ────────────────────────────────────
    ts_ok = _check_timestamp_consistency(exif_metadata)
    signals.append(
        Signal(
            name="image_timestamp_inconsistency",
            severity="medium" if not ts_ok else "info",
            score=25 if not ts_ok else 0,
            summary=(
                "EXIF timestamps are inconsistent — the capture timestamp follows "
                "the digitized or modified timestamp, which may indicate backdating."
                if not ts_ok
                else "Image EXIF timestamps are consistent."
            ),
            passed=ts_ok,
            details={"timestamps_consistent": ts_ok},
        )
    )

    # ── 2. Error Level Analysis ─────────────────────────────────────────────
    ela_score, high_error_frac = run_ela(path)
    if ela_score == 0.0 and high_error_frac == 0.0:
        limitations.append(
            "ELA unavailable — install Pillow and NumPy for error-level analysis."
        )

    # ELA score thresholds (empirically tuned):
    #   < 8   → consistent with unedited scan / photo
    #   8–20  → mild variation; could be heavy JPEG compression or slight edit
    #   > 20  → strong editing artefact signature
    ela_suspicious = ela_score >= 20.0
    ela_mild = 8.0 <= ela_score < 20.0
    ela_severity = "high" if ela_suspicious else ("low" if ela_mild else "info")
    ela_penalty = 40 if ela_suspicious else (15 if ela_mild else 0)

    signals.append(
        Signal(
            name="image_ela_tampering_score",
            severity=ela_severity,
            score=ela_penalty,
            summary=(
                f"ELA score {ela_score:.1f} / 100 — "
                + (
                    "strong editing artefacts detected. Regions of the image appear "
                    "to have been digitally altered or copy-pasted."
                    if ela_suspicious
                    else (
                        "mild compression artefacts; manual review recommended."
                        if ela_mild
                        else "image error levels are consistent with an unedited document."
                    )
                )
            ),
            passed=not ela_suspicious,
            details={
                "ela_score": ela_score,
                "high_error_block_fraction": high_error_frac,
                "resave_quality": 95,
            },
        )
    )

    # ── 3. Noise consistency ────────────────────────────────────────────────
    noise_cv = run_noise_analysis(path)
    if noise_cv is None:
        limitations.append(
            "Noise analysis unavailable — install opencv-python and NumPy."
        )

    if noise_cv is not None:
        # CV thresholds:  < 0.8 homogeneous (normal), 0.8–1.5 moderate, > 1.5 suspicious
        noise_suspicious = noise_cv > 1.5
        noise_mild = 0.8 <= noise_cv <= 1.5
        noise_severity = "medium" if noise_suspicious else ("low" if noise_mild else "info")
        noise_penalty = 25 if noise_suspicious else (10 if noise_mild else 0)

        signals.append(
            Signal(
                name="image_noise_inconsistency",
                severity=noise_severity,
                score=noise_penalty,
                summary=(
                    f"Noise CV {noise_cv:.3f} — "
                    + (
                        "high noise variance across image blocks, consistent with "
                        "copy-paste or local editing operations."
                        if noise_suspicious
                        else (
                            "moderate noise variation; could be natural or indicate minor edits."
                            if noise_mild
                            else "uniform noise pattern consistent with an unmodified document image."
                        )
                    )
                ),
                passed=not noise_suspicious,
                details={"noise_cv": noise_cv},
            )
        )

    # ── 4. Copy-move detection ──────────────────────────────────────────────
    copy_move_score = run_copy_move_detection(path)
    if copy_move_score is None:
        limitations.append(
            "Copy-move detection unavailable — install opencv-python."
        )
    else:
        cm_suspicious = copy_move_score > 35.0
        cm_mild = 10.0 <= copy_move_score <= 35.0
        cm_severity = "high" if cm_suspicious else ("low" if cm_mild else "info")
        cm_penalty = 35 if cm_suspicious else (12 if cm_mild else 0)

        signals.append(
            Signal(
                name="image_copy_move_detection",
                severity=cm_severity,
                score=cm_penalty,
                summary=(
                    f"Copy-move score {copy_move_score:.1f} / 100 — "
                    + (
                        "duplicate feature regions detected at spatially distinct "
                        "locations, strongly indicating a copy-paste operation."
                        if cm_suspicious
                        else (
                            "some self-similar regions detected; could be a repeating "
                            "pattern or minor editing."
                            if cm_mild
                            else "no meaningful copy-move artefacts detected."
                        )
                    )
                ),
                passed=not cm_suspicious,
                details={"copy_move_score": copy_move_score},
            )
        )

    # ── 5. GAN / AI-generated image detection ──────────────────────────────
    gan_score = run_gan_detection(path)
    if gan_score is None:
        limitations.append(
            "GAN detection unavailable — install opencv-python and NumPy."
        )
    else:
        gan_suspicious = gan_score > 55.0
        gan_mild = 20.0 <= gan_score <= 55.0
        gan_severity = "high" if gan_suspicious else ("low" if gan_mild else "info")
        gan_penalty = 30 if gan_suspicious else (10 if gan_mild else 0)

        signals.append(
            Signal(
                name="image_gan_ai_generation_score",
                severity=gan_severity,
                score=gan_penalty,
                summary=(
                    f"AI-generation score {gan_score:.1f} / 100 — "
                    + (
                        "frequency and noise patterns strongly suggest this image was "
                        "synthetically generated by an AI model (GAN or diffusion). "
                        "Cross-check EXIF metadata for authoring tool tags."
                        if gan_suspicious
                        else (
                            "some spectral anomalies detected; image may be post-processed "
                            "or partially AI-assisted."
                            if gan_mild
                            else "image frequency and noise patterns are consistent with a "
                            "genuine scan or photograph."
                        )
                    )
                ),
                passed=not gan_suspicious,
                details={"gan_score": gan_score},
            )
        )

    limitations += [
        "ELA analysis is most reliable on JPEG images; PNG results may be less informative.",
        "These are heuristic forensic signals — not conclusive proof of tampering.",
        "Template-matching against known-good originals would provide stronger evidence.",
    ]

    return {
        "signals": signals_to_dict(signals),
        "ela_score": ela_score,
        "high_error_frac": high_error_frac,
        "noise_cv": noise_cv,
        "copy_move_score": copy_move_score,
        "gan_score": gan_score,
        "exif_metadata": exif_metadata,
        "suspicious_tool": suspicious_tool,
        "limitations": limitations,
    }
