"""Unit tests for video-related store functions.

All tests mock the MinIO S3 client so no live MinIO connection is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, ANY
import pytest


def test_save_face_scan_video_returns_correct_key() -> None:
    """save_face_scan_video should construct the key as 'face-scan-video/{session_id}.mp4'
    and call minio_upload with it.
    """
    from basetruth import store

    with patch.object(store, "minio_upload", return_value=True) as mock_upload:
        key = store.save_face_scan_video("sess-abc123", b"fake-mp4-bytes")

    assert key == "face-scan-video/sess-abc123.mp4"
    mock_upload.assert_called_once_with(
        "face-scan-video/sess-abc123.mp4",
        b"fake-mp4-bytes",
        content_type="video/mp4",
    )


def test_save_face_scan_video_raises_on_upload_failure() -> None:
    """save_face_scan_video must raise RuntimeError when minio_upload returns False."""
    from basetruth import store

    with patch.object(store, "minio_upload", return_value=False):
        with pytest.raises(RuntimeError, match="MinIO upload failed"):
            store.save_face_scan_video("sess-fail", b"bytes")


def test_delete_face_scan_video_delegates_to_minio_delete_object() -> None:
    """delete_face_scan_video should delegate to minio_delete_object and return its value."""
    from basetruth import store

    with patch.object(store, "minio_delete_object", return_value=True) as mock_del:
        result = store.delete_face_scan_video("face-scan-video/sess-abc.mp4")

    assert result is True
    mock_del.assert_called_once_with("face-scan-video/sess-abc.mp4")


def test_get_face_scan_video_presigned_url_returns_none_when_no_client() -> None:
    """get_face_scan_video_presigned_url should return None when the presign client is unavailable."""
    from basetruth import store

    with patch.object(store, "_get_minio_s3_presign_client", return_value=None):
        url = store.get_face_scan_video_presigned_url("face-scan-video/sess.mp4")

    assert url is None


def test_get_face_scan_video_presigned_url_returns_url_from_client() -> None:
    """get_face_scan_video_presigned_url must use the presign client (external endpoint).

    The presign client is configured with MINIO_EXTERNAL_ENDPOINT (localhost:9000)
    so the AWS4 Host signature matches what the browser sends — no post-hoc rewrite
    needed.  The URL returned by boto3 already has the correct hostname.
    """
    from basetruth import store

    fake_client = MagicMock()
    # boto3 presigning with localhost:9000 produces a URL with localhost:9000.
    fake_client.generate_presigned_url.return_value = (
        "http://localhost:9000/basetruth-reports/face-scan-video/sess.mp4"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=3600"
    )

    with patch.object(store, "_get_minio_s3_presign_client", return_value=fake_client):
        url = store.get_face_scan_video_presigned_url("face-scan-video/sess.mp4", expires_seconds=3600)

    assert url is not None
    assert "localhost:9000" in url, f"Expected localhost:9000 in URL, got: {url}"
    assert "face-scan-video" in url
    fake_client.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": ANY, "Key": "face-scan-video/sess.mp4"},
        ExpiresIn=3600,
    )


def test_get_face_scan_video_presigned_url_returns_none_on_exception() -> None:
    """get_face_scan_video_presigned_url must return None (not raise) on boto3 errors."""
    from basetruth import store

    fake_client = MagicMock()
    fake_client.generate_presigned_url.side_effect = RuntimeError("boto3 exploded")

    with patch.object(store, "_get_minio_s3_presign_client", return_value=fake_client):
        url = store.get_face_scan_video_presigned_url("face-scan-video/sess.mp4")

    assert url is None
