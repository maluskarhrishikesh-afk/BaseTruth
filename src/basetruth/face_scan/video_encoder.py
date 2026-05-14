"""Video encoder for Face Scan live session recordings.

This module converts a list of raw JPEG frame bytes (captured per-frame during a live
Face Scan session) into a single H.264 MP4 file. The MP4 bytes are then uploaded to
MinIO so operators can replay exactly what the server analysed.

Why imageio[ffmpeg]?
  - Wraps FFmpeg with a clean Python API — no system FFmpeg install required.
  - imageio-ffmpeg downloads its own FFmpeg binary on first use.
  - Much lighter dependency than adding opencv-video writers or av (PyAV).
  - Simple writer interface: open writer → append numpy frames → close → get bytes.

Encoding happens synchronously in the FastAPI executor thread (same thread as
build_live_face_scan_result).  The user is already viewing their result screen at this
point, so a 2–5 s encoding delay is invisible.
"""

from __future__ import annotations

import io
import tempfile
from typing import List

import cv2
import numpy as np

from basetruth.logger import get_logger

log = get_logger(__name__)

# Hard cap on the number of frames we will encode.  At 10 FPS this is 2 minutes.
# Frames beyond this cap are silently discarded before encoding begins.
MAX_ENCODABLE_FRAMES: int = 1200


class VideoEncoderError(Exception):
    """Raised when frame encoding fails (empty list, decode error, ffmpeg error, etc.)."""


def encode_frames_to_mp4(frames: List[bytes], fps: int = 10) -> bytes:
    """Encode a list of raw JPEG frame bytes into an H.264 MP4 and return the bytes.

    Each element of *frames* must be the raw bytes of a single JPEG image.  All
    frames should have the same resolution — they are decoded with OpenCV so any
    valid JPEG works, but a mix of resolutions will cause imageio to raise.

    The returned bytes begin with the MP4 ``ftyp`` box and can be uploaded directly
    to MinIO and played back with any modern browser via an HTML5 ``<video>`` tag.

    Args:
        frames: List of raw JPEG bytes, one element per captured frame.
        fps: Playback frame rate.  Should match the capture rate (10 FPS).

    Returns:
        Raw MP4 bytes.

    Raises:
        VideoEncoderError: When *frames* is empty, any frame cannot be decoded, or
            imageio/FFmpeg fails to produce valid output.
    """
    if not frames:
        raise VideoEncoderError("encode_frames_to_mp4: frames list is empty — nothing to encode")

    # Apply hard cap: keep the most recent MAX_ENCODABLE_FRAMES frames so the buffer
    # never produces a file so large it stalls the MinIO upload or the browser.
    if len(frames) > MAX_ENCODABLE_FRAMES:
        log.warning(
            "encode_frames_to_mp4: frame buffer exceeded cap — truncating to %d frames (had %d)",
            MAX_ENCODABLE_FRAMES,
            len(frames),
        )
        frames = frames[-MAX_ENCODABLE_FRAMES:]

    # Decode the first frame to get the resolution so we can validate all others.
    first_arr = _decode_jpeg(frames[0], frame_index=0)
    height, width = first_arr.shape[:2]
    log.debug(
        "encode_frames_to_mp4: encoding %d frames at %dx%d %d FPS",
        len(frames),
        width,
        height,
        fps,
    )

    # imageio writes to a file-like object.  We use a NamedTemporaryFile because
    # imageio-ffmpeg needs a real seekable file path for the mp4 container.
    # We write to a temp file, read the bytes back, then delete it.
    try:
        import imageio  # noqa: PLC0415 — lazy import so the module loads even when imageio absent
    except ImportError as exc:
        raise VideoEncoderError(
            "imageio is not installed — run: pip install 'imageio[ffmpeg]'"
        ) from exc

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # quality=5 is a middle-ground H.264 CRF that produces ~200 kbps at 640×480.
        # macro_block_size=None lets imageio choose the nearest multiple of 16 for the
        # height, avoiding green-bar artefacts on odd-height frames.
        writer = imageio.get_writer(
            tmp_path,
            format="FFMPEG",
            fps=fps,
            codec="libx264",
            quality=5,
            macro_block_size=None,
            ffmpeg_log_level="error",
        )

        for idx, jpeg_bytes in enumerate(frames):
            # Decode each JPEG to an RGB numpy array (imageio expects RGB, not BGR).
            bgr = _decode_jpeg(jpeg_bytes, frame_index=idx)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Skip frames that changed resolution mid-session (e.g. camera reconfig).
            # This avoids a fatal "all frames must have the same shape" error from imageio.
            if rgb.shape[:2] != (height, width):
                log.debug(
                    "encode_frames_to_mp4: skipping frame %d — resolution changed (%dx%d vs %dx%d)",
                    idx,
                    rgb.shape[1],
                    rgb.shape[0],
                    width,
                    height,
                )
                continue

            writer.append_data(rgb)

        writer.close()

        # Read the encoded MP4 bytes back from the temp file.
        with open(tmp_path, "rb") as f:
            mp4_bytes = f.read()

    except VideoEncoderError:
        raise
    except Exception as exc:
        raise VideoEncoderError(f"encode_frames_to_mp4: FFmpeg encoding failed — {exc}") from exc
    finally:
        # Always clean up the temp file even if encoding raised.
        import os  # noqa: PLC0415
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not mp4_bytes:
        raise VideoEncoderError("encode_frames_to_mp4: output MP4 is empty — encoding produced no data")

    log.info(
        "encode_frames_to_mp4: encoded %d frames → %d bytes (%.1f KB)",
        len(frames),
        len(mp4_bytes),
        len(mp4_bytes) / 1024,
    )
    return mp4_bytes


def _decode_jpeg(jpeg_bytes: bytes, frame_index: int = 0) -> np.ndarray:
    """Decode raw JPEG bytes to a BGR numpy array using OpenCV.

    Args:
        jpeg_bytes: Raw JPEG bytes for one frame.
        frame_index: Frame position in the buffer (used only for error messages).

    Returns:
        BGR uint8 numpy array with shape (height, width, 3).

    Raises:
        VideoEncoderError: When OpenCV cannot decode the bytes.
    """
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise VideoEncoderError(
            f"encode_frames_to_mp4: frame {frame_index} could not be decoded as JPEG"
        )
    return img
