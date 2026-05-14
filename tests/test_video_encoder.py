"""Unit tests for basetruth.face_scan.video_encoder.

All tests run without a live database, MinIO, or FFmpeg binary:
- Encoding tests that need real output are guarded with an imageio import check.
- Empty-frames and bad-data tests do not need FFmpeg.
"""
from __future__ import annotations

import io
import pytest
import cv2
import numpy as np


def _make_jpeg_bytes(width: int = 64, height: int = 48, color: tuple = (100, 150, 200)) -> bytes:
    """Return the bytes of a tiny solid-colour JPEG frame."""
    img = np.full((height, width, 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_encode_empty_frames_raises_video_encoder_error() -> None:
    """Encoding an empty frame list must raise VideoEncoderError immediately."""
    from basetruth.face_scan.video_encoder import VideoEncoderError, encode_frames_to_mp4

    with pytest.raises(VideoEncoderError, match="frames list is empty"):
        encode_frames_to_mp4([])


def test_decode_jpeg_valid_frame_returns_bgr_array() -> None:
    """_decode_jpeg should return a BGR uint8 numpy array for a valid JPEG."""
    from basetruth.face_scan.video_encoder import _decode_jpeg

    jpeg = _make_jpeg_bytes(64, 48)
    arr = _decode_jpeg(jpeg, frame_index=0)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.shape == (48, 64, 3)


def test_decode_jpeg_invalid_bytes_raises() -> None:
    """_decode_jpeg must raise VideoEncoderError for non-JPEG bytes."""
    from basetruth.face_scan.video_encoder import VideoEncoderError, _decode_jpeg

    with pytest.raises(VideoEncoderError, match="could not be decoded as JPEG"):
        _decode_jpeg(b"not-a-jpeg", frame_index=0)


def _ffmpeg_available() -> bool:
    """Return True only when the imageio FFMPEG plugin is loadable (requires imageio[ffmpeg])."""
    try:
        import imageio
        import tempfile, os
        # Use a valid path so the plugin is actually resolved; failure = plugin missing.
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp = f.name
        try:
            w = imageio.get_writer(tmp, format="FFMPEG", codec="libx264")
            w.close()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return True
    except Exception:
        return False


_SKIP_NO_FFMPEG = pytest.mark.skipif(
    not _ffmpeg_available(),
    reason="imageio[ffmpeg] / FFmpeg binary not installed — skipping encoding test",
)


@_SKIP_NO_FFMPEG
def test_encode_frames_to_mp4_produces_valid_bytes() -> None:
    """Encoding 10 solid-colour JPEG frames should produce non-empty MP4 bytes.

    This test requires imageio[ffmpeg] to be installed. It is skipped when
    the library is absent so CI without FFmpeg does not fail.
    """
    from basetruth.face_scan.video_encoder import encode_frames_to_mp4

    frames = [_make_jpeg_bytes(64, 48) for _ in range(10)]
    mp4_bytes = encode_frames_to_mp4(frames, fps=10)

    # Must be non-empty and start with the ftyp MP4 box signature at offset 4.
    assert len(mp4_bytes) > 100
    # MP4 ftyp box: bytes 4..8 are "ftyp" (0x66747970)
    assert mp4_bytes[4:8] == b"ftyp", f"Expected MP4 ftyp box, got {mp4_bytes[4:8]!r}"


@_SKIP_NO_FFMPEG
def test_encode_frames_to_mp4_respects_hard_cap(monkeypatch) -> None:
    """When more than MAX_ENCODABLE_FRAMES are provided the encoder must still succeed,
    truncating the input to the most recent MAX_ENCODABLE_FRAMES frames.
    """
    from basetruth.face_scan import video_encoder

    # Lower the cap temporarily so we don't need to create 1200 frames in a test.
    monkeypatch.setattr(video_encoder, "MAX_ENCODABLE_FRAMES", 5)

    # Supply 10 frames — the encoder should keep only the last 5.
    frames = [_make_jpeg_bytes(64, 48, color=(i * 20, 100, 200)) for i in range(10)]
    mp4_bytes = video_encoder.encode_frames_to_mp4(frames, fps=5)
    assert len(mp4_bytes) > 0
