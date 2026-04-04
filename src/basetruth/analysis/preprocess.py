"""Image preprocessing for BaseTruth OCR pipeline.

Applies deskewing (rotation correction) and perspective correction to raw
document images before text extraction, significantly improving OCR accuracy
for PAN cards, Aadhaar cards, and other identity documents captured with a
phone camera.

Public API
----------
  preprocess_for_ocr(path: Path) -> np.ndarray | None
      Full pipeline: load → deskew → perspective correction → contrast enhance.
      Returns a processed numpy (BGR) image, or None if OpenCV is unavailable.

  preprocess_pil_for_ocr(img: PIL.Image) -> PIL.Image
      Same pipeline, accepts and returns a PIL Image.
      Useful for drop-in replacement of the current pytesseract call sites.

Design notes
------------
- All steps degrade gracefully: if a step cannot detect a correction angle /
  document corners, the image is returned unchanged rather than distorted.
- Edge-case guard: if the detected skew/warp would make the result *worse*
  (very small angle, near-perfect rectangle already), corrections are skipped.
- Every function is pure (no side effects, no file writes).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deskewing — corrects rotation caused by tilted camera / scanner
# ---------------------------------------------------------------------------

def _detect_skew_angle(gray: "np.ndarray") -> float:  # type: ignore[name-defined]
    """Estimate text-line skew angle from a grayscale image.

    Uses the weighted mean of Hough-line angles to robustly handle images that
    have only a few readable lines.  Returns 0.0 if detection fails or the
    detected angle is smaller than 0.3 degrees (no meaningful correction).
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        # Binarise with Otsu so lines are clean
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Dilate horizontally to merge text fragments into line blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours of line blobs
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) < 3:
            # Not enough lines to reliably estimate angle
            return 0.0

        angles: list[float] = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 200:
                continue
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            # cv2.minAreaRect returns angles in [-90, 0); convert to [-45, 45)
            if angle < -45:
                angle += 90
            angles.append(angle)

        if not angles:
            return 0.0

        median_angle = float(np.median(angles))

        # Skip tiny corrections — they can introduce resampling artefacts
        if abs(median_angle) < 0.3:
            return 0.0

        return median_angle

    except Exception as exc:  # noqa: BLE001
        log.debug("Skew detection failed: %s", exc)
        return 0.0


def deskew_image(img: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """Rotate *img* to correct text skew.

    Parameters
    ----------
    img : BGR uint8 array (as returned by cv2.imread)

    Returns
    -------
    Rotated image, or the original image if no correction is needed.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = _detect_skew_angle(gray)

        if angle == 0.0:
            return img

        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Expand canvas so corners are not cropped
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(
            img, M, (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        log.debug("Deskewed by %.2f degrees", angle)
        return rotated

    except Exception as exc:  # noqa: BLE001
        log.debug("Deskew failed, returning original: %s", exc)
        return img


# ---------------------------------------------------------------------------
# Perspective correction — straightens trapezoid-shaped card captures
# ---------------------------------------------------------------------------

def _order_points(pts: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """Order four corner points as [top-left, top-right, bottom-right, bottom-left]."""
    import numpy as np  # type: ignore

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left: smallest sum
    rect[2] = pts[np.argmax(s)]       # bottom-right: largest sum
    rect[1] = pts[np.argmin(diff)]    # top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]    # bottom-left: largest diff
    return rect


def _four_point_transform(
    img: "np.ndarray",  # type: ignore[name-defined]
    pts: "np.ndarray",  # type: ignore[name-defined]
) -> "np.ndarray":  # type: ignore[name-defined]
    """Apply a perspective warp to bring *pts* to a top-down rectangle."""
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_top = math.hypot(float(br[0] - bl[0]), float(br[1] - bl[1]))
    width_bot = math.hypot(float(tr[0] - tl[0]), float(tr[1] - tl[1]))
    max_w = max(int(width_top), int(width_bot))

    height_right = math.hypot(float(tr[0] - br[0]), float(tr[1] - br[1]))
    height_left  = math.hypot(float(tl[0] - bl[0]), float(tl[1] - bl[1]))
    max_h = max(int(height_right), int(height_left))

    if max_w < 50 or max_h < 50:
        return img  # degenerate quad — skip

    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (max_w, max_h))


def correct_perspective(img: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """Detect the document boundary and warp it to a flat rectangle.

    Looks for the largest 4-sided contour in the image.  If no clean document
    boundary is detected (e.g. the card fills the frame edge-to-edge), the
    original image is returned unchanged.

    Parameters
    ----------
    img : BGR uint8 array

    Returns
    -------
    Perspective-corrected image, or the original if correction is not possible.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Equalise contrast so the document edges are sharp even in low-light
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged   = cv2.Canny(blurred, 50, 150)

        # Dilate edges to close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edged  = cv2.dilate(edged, kernel, iterations=1)

        contours, _ = cv2.findContours(
            edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return img

        # Take the largest contour by area
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)

        img_area = img.shape[0] * img.shape[1]

        for cnt in contours_sorted[:5]:
            area = cv2.contourArea(cnt)
            # Must be at least 15% of image area to be the document
            if area < img_area * 0.15:
                break

            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                pts = approx.reshape(4, 2).astype(np.float32)
                warped = _four_point_transform(img, pts)
                log.debug("Perspective correction applied (contour area %.0f)", area)
                return warped

        return img  # No suitable 4-sided contour found

    except Exception as exc:  # noqa: BLE001
        log.debug("Perspective correction failed, returning original: %s", exc)
        return img


# ---------------------------------------------------------------------------
# Contrast / sharpness enhancement
# ---------------------------------------------------------------------------

def enhance_for_ocr(img: "np.ndarray") -> "np.ndarray":  # type: ignore[name-defined]
    """Apply CLAHE contrast enhancement and gentle unsharp-masking.

    These steps improve OCR accuracy on low-contrast or slightly blurry
    photos of identity documents.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        # Convert to LAB for luminance-only enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)

        enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Unsharp mask (gentle): sharpens text edges
        blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2)
        sharpened = cv2.addWeighted(enhanced, 1.4, blur, -0.4, 0)

        return sharpened

    except Exception as exc:  # noqa: BLE001
        log.debug("Enhance failed, returning as-is: %s", exc)
        return img


# ---------------------------------------------------------------------------
# Full pipeline — the single entry point used by the OCR layer
# ---------------------------------------------------------------------------

def preprocess_for_ocr(path: Path) -> Optional["np.ndarray"]:  # type: ignore[name-defined]
    """Load an image and run the full preprocessing pipeline.

    Returns
    -------
    Processed BGR uint8 numpy array, or None if OpenCV / the file is
    unavailable.
    """
    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path))
        if img is None:
            log.debug("Could not load image for preprocessing: %s", path.name)
            return None

        img = correct_perspective(img)
        img = deskew_image(img)
        img = enhance_for_ocr(img)
        return img

    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001
        log.debug("preprocess_for_ocr failed for %s: %s", path.name, exc)
        return None


def preprocess_pil_for_ocr(img: "PIL.Image.Image") -> "PIL.Image.Image":  # type: ignore[name-defined]
    """PIL-native wrapper around the full preprocessing pipeline.

    Converts to BGR numpy internally, runs all corrections, converts back.
    Returns the original image unchanged if OpenCV is unavailable.
    """
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore

        bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        bgr = correct_perspective(bgr)
        bgr = deskew_image(bgr)
        bgr = enhance_for_ocr(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    except Exception as exc:  # noqa: BLE001
        log.debug("preprocess_pil_for_ocr failed: %s", exc)
        return img
