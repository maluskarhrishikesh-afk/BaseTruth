# Implementation Plan — Face Scan Live Video Recording

> **Status:** Planning  
> **Scope:** Server-side video capture of completed live Face Scan sessions, MinIO storage, playback in the Face Scan UI, and operator-initiated deletion.

---

## What We Are Building

After every completed live Face Scan session whose verdict is **SUSPICIOUS, DEEPFAKE, or INCONCLUSIVE**, the server assembles the raw JPEG frames it already processed into an H.264 MP4 video and uploads it to MinIO. Operators can watch the video directly in the Face Scan page and delete it when the investigation is closed.

GENUINE sessions are **not** recorded by default. This is the correct balance between investigative value and proportionality under data-protection law (biometric video is Article 9 GDPR special-category data).

---

## Recording Specifications

| Setting | Value | Reason |
|---|---|---|
| Codec | H.264 (via `imageio[ffmpeg]`) | Universal compatibility; small file size; supported by all browsers |
| Container | MP4 | Plays natively in `<video>` tag without transcoding |
| FPS | 10 (matches capture rate) | Exact 1:1 with what the server analysed; no interpolation |
| Resolution | 640 × height (aspect-ratio preserved from capture) | Matches frames already in memory |
| Average file size | 1–2 MB per session | 60 s × ~200 kbps H.264 |
| Audio | Disabled (no audio track) | JPEG frames only; no audio was ever captured |
| Encryption at rest | Server-Side Encryption (SSE-S3) via MinIO config | Required — biometric video is sensitive data |
| MinIO object key | `face-scan-video/{session_id}.mp4` | Consistent prefix makes bucket lifecycle rules easy to configure |
| Retention TTL | Configurable via `FACE_SCAN_VIDEO_RETENTION_DAYS` env var (default 90) | Deployments with stricter data-protection requirements can lower this |

---

## Feature Flag

A single environment variable controls the whole feature:

```
FACE_SCAN_RECORD_VIDEO=true   # enable recording (default: false)
```

When `false` (the default), no frames are accumulated, no MinIO writes happen, and no video column is populated. This lets operators disable recording entirely for deployments that cannot hold biometric video.

---

## Files Changed

| File | Change |
|---|---|
| `src/basetruth/face_scan/live.py` | Accumulate raw JPEG bytes per frame; post-session encoding; feature-flag guard |
| `src/basetruth/face_scan/video_encoder.py` | **New** — `encode_frames_to_mp4(frames: list[bytes], fps: int) -> bytes` using `imageio[ffmpeg]` |
| `src/basetruth/store.py` | `save_face_scan_video(session_id, mp4_bytes) -> str` (upload + return MinIO key); `delete_face_scan_video(key) -> bool`; `get_face_scan_video_url(key) -> str` (presigned URL) |
| `src/basetruth/db.py` | Add `video_key` column to a new `FaceScanLiveResult` table (see DB schema below) |
| `src/basetruth/api.py` | Two new endpoints: `GET /api/v1/face-scan/sessions/{session_id}/video` and `DELETE /api/v1/face-scan/sessions/{session_id}/video` |
| `src/basetruth/ui/pages/face_scan.py` | Video player widget + delete button in the live result panel |
| `src/basetruth/face_scan/live.py` (HTML template) | Add consent notice to the browser page before "Start Live Face Scan" button |
| `requirements.txt` | Add `imageio[ffmpeg]` |
| `docs/FACE_SCAN_WORKING.md` | Document the video recording feature |

---

## Database Schema Change

A new table `face_scan_live_results` is added to persist completed session results and the video object key. The in-memory `FaceScanLiveSession` objects expire after 20 minutes — this table is the durable record.

```sql
CREATE TABLE face_scan_live_results (
    id            SERIAL PRIMARY KEY,
    session_id    VARCHAR(50)  NOT NULL UNIQUE,
    verdict       VARCHAR(20)  NOT NULL,
    risk_score    FLOAT,
    confidence    FLOAT,
    report_json   JSONB,
    best_frame_key VARCHAR(500),   -- MinIO key for the best still frame
    video_key     VARCHAR(500),    -- MinIO key for the MP4 (NULL if not recorded)
    created_at    TIMESTAMP WITH TIME ZONE DEFAULT now(),
    updated_at    TIMESTAMP WITH TIME ZONE DEFAULT now()
);
```

The `video_key` column is NULL when:
- the session verdict was GENUINE (not recorded by default), or
- `FACE_SCAN_RECORD_VIDEO=false`, or
- encoding or MinIO upload failed (soft failure — never blocks the result)

---

## Implementation Steps

### Phase 1 — Frame Accumulation in the Session Object

**File:** `src/basetruth/face_scan/live.py`

Add a new field to `FaceScanLiveSession`:

```python
raw_frame_buffer: List[bytes] = field(default_factory=list)
```

In `process_live_frame_message`, after `raw = base64.b64decode(b64_frame, validate=True)`, add:

```python
if os.environ.get("FACE_SCAN_RECORD_VIDEO", "false").lower() == "true":
    session.raw_frame_buffer.append(raw)
```

**Memory bound:** At 10 FPS × 30 KB/frame × 90 s = ~27 MB per session. With a hard cap of `MAX_FRAME_BUFFER = 1200` frames (2 minutes), the worst-case buffer is ~36 MB. If the cap is hit, the oldest frames are dropped (sliding window) so the buffer never grows unbounded.

The buffer is only populated when the feature flag is on — zero cost when disabled.

---

### Phase 2 — Video Encoder Module

**File:** `src/basetruth/face_scan/video_encoder.py` (new)

```
encode_frames_to_mp4(frames: list[bytes], fps: int = 10) -> bytes
```

- Uses `imageio.get_writer(format="mp4", codec="libx264", fps=fps, quality=5)`
- Decodes each JPEG frame to a numpy array and appends it to the writer
- Returns the final MP4 bytes
- Raises `VideoEncoderError` (custom exception) on failure so the caller can log and skip gracefully

`imageio[ffmpeg]` is the right choice here because:
- It wraps FFmpeg with a clean Python API
- Already used in many data-science environments
- No system-level FFmpeg installation required (downloads its own binary on first use via `imageio-ffmpeg`)
- `opencv-python` would also work but is a much heavier dependency for just encoding

---

### Phase 3 — Post-Session Upload in `build_live_face_scan_result`

**File:** `src/basetruth/face_scan/live.py`

At the end of `build_live_face_scan_result`, after the result dict is assembled:

```python
# Video recording: only for non-GENUINE verdicts, only when flag is on
video_key = None
if (
    os.environ.get("FACE_SCAN_RECORD_VIDEO", "false").lower() == "true"
    and verdict in ("SUSPICIOUS", "DEEPFAKE", "INCONCLUSIVE")
    and session.raw_frame_buffer
):
    try:
        mp4_bytes = encode_frames_to_mp4(session.raw_frame_buffer, fps=10)
        video_key = save_face_scan_video(session.session_id, mp4_bytes)
        log.info("face_scan_live: video saved — session=%s key=%s size=%d bytes",
                 session.session_id, video_key, len(mp4_bytes))
    except Exception as exc:
        log.warning("face_scan_live: video encoding/upload failed — session=%s error=%s",
                    session.session_id, exc)
    finally:
        session.raw_frame_buffer.clear()  # free memory immediately

result["video_key"] = video_key  # None if not recorded / not applicable
```

Encoding happens in the same thread as the result build, which is already running in a FastAPI async handler via `asyncio.run_in_executor`. The 2–5 second encoding delay happens after the challenges are done, while the result JSON is being assembled — the user is already seeing the result screen.

---

### Phase 4 — MinIO Store Functions

**File:** `src/basetruth/store.py`

Three new functions following the same pattern as `minio_upload` / `minio_get_object`:

```python
def save_face_scan_video(session_id: str, mp4_bytes: bytes) -> str:
    """Upload MP4 to MinIO and return the object key. Raises on failure."""
    key = f"face-scan-video/{session_id}.mp4"
    ok = minio_upload(key, mp4_bytes, content_type="video/mp4")
    if not ok:
        raise RuntimeError(f"MinIO upload failed for video key {key}")
    return key

def get_face_scan_video_presigned_url(key: str, expires_seconds: int = 3600) -> Optional[str]:
    """Return a time-limited presigned URL for the video. Returns None if key not found."""
    # Uses client.generate_presigned_url("get_object", ...) 

def delete_face_scan_video(key: str) -> bool:
    """Delete the video object from MinIO. Returns True on success."""
    return minio_delete_object(key)
```

The presigned URL expires in 1 hour by default. This means the video link shown in the Streamlit UI is valid for viewing but not permanently accessible — which is correct for sensitive biometric video.

---

### Phase 5 — API Endpoints

**File:** `src/basetruth/api.py`

Two new endpoints registered in the face-scan router block:

#### `GET /api/v1/face-scan/sessions/{session_id}/video`

Returns a presigned URL for the video. Used by the Streamlit UI to obtain a short-lived playback link.

```
Response 200: { "video_url": "https://...", "expires_in_seconds": 3600 }
Response 404: session not found or no video recorded
Response 403: feature flag is off
```

The endpoint does **not** stream the video directly — it returns a presigned URL so the browser fetches the MP4 directly from MinIO. This avoids routing large binary data through the FastAPI process.

#### `DELETE /api/v1/face-scan/sessions/{session_id}/video`

Deletes the video from MinIO and nulls `video_key` in the DB row.

```
Response 200: { "deleted": true }
Response 404: key not found
```

Both endpoints require the existing API auth. No new auth layer needed.

---

### Phase 6 — Streamlit UI Changes

**File:** `src/basetruth/ui/pages/face_scan.py`

In the live result panel (after the verdict badge, risk score, and evidence bullets), add a new collapsible section **"Session Recording"** that is only rendered when `result.get("video_key")` is not None.

```
▶ Session Recording
  ┌──────────────────────────────────────────────┐
  │  [▶ Watch Recording]                         │
  │                                              │
  │  <video autoplay controls width=640>         │
  │    <source src="{presigned_url}" type=mp4>   │
  │  </video>                                    │
  │                                              │
  │  Recorded: {timestamp}                       │
  │  Retention: auto-deleted after {N} days      │
  │                                              │
  │           [ 🗑 Delete Recording ]            │
  └──────────────────────────────────────────────┘
```

Implementation notes:
- "Watch Recording" button calls `GET /api/v1/face-scan/sessions/{session_id}/video` to fetch the presigned URL, then renders it in an `st.video(url)` component
- "Delete Recording" button calls `DELETE /api/v1/face-scan/sessions/{session_id}/video` with a confirmation step (`st.warning` + checkbox before the call fires)
- After deletion the section shows "Recording has been deleted" and the button disappears
- The `st.video()` component accepts an HTTPS URL directly — no need to download the bytes to the Streamlit process

---

### Phase 7 — Consent Notice on the Browser Page

**File:** `src/basetruth/face_scan/live.py` (HTML template `_FACE_SCAN_LIVE_PAGE_HTML`)

Before the "Start Live Face Scan" button, add:

```html
<p class="consent-notice">
  ⚠️ This session may be recorded for fraud investigation purposes if the
  result is flagged as suspicious. Recordings are stored securely and deleted
  automatically after [RETENTION_DAYS] days. By continuing you consent to
  this recording.
</p>
```

This is rendered as plain HTML in the lightweight browser page. No JavaScript interaction needed — clicking "Start" is the consent action.

`[RETENTION_DAYS]` is injected at render time from the `FACE_SCAN_VIDEO_RETENTION_DAYS` env var (default 90), so the notice is always accurate.

---

## MinIO Bucket Lifecycle Rule

Set a lifecycle rule on the `basetruth-reports` bucket (or a dedicated `face-scan-video` bucket) to automatically expire objects under the `face-scan-video/` prefix after `FACE_SCAN_VIDEO_RETENTION_DAYS` days.

This is the safety net. Even if the operator never manually deletes a recording, it is automatically purged after the retention window. The rule is configured in `docker-compose.yml` via a MinIO init container or the `mc` CLI at startup.

---

## Memory Budget

| Condition | Memory cost |
|---|---|
| Feature flag off | 0 (no buffer allocated) |
| Feature flag on, 1 active session | ~27 MB (90 s × 10 FPS × ~30 KB/frame) |
| Feature flag on, 10 concurrent sessions | ~270 MB |
| Feature flag on, 100 concurrent sessions | ~2.7 GB |

For deployments with more than ~10 concurrent suspicious sessions, consider offloading frame accumulation to a temporary on-disk buffer (`tempfile.SpooledTemporaryFile`) instead of in-memory lists. This is a follow-up optimisation, not needed at current scale.

---

## Testing Plan

Following `docs/TESTING.md`:

| Test | File | What it proves |
|---|---|---|\n| `test_encode_frames_to_mp4_produces_valid_bytes` | `tests/test_video_encoder.py` | Encoding 10 dummy JPEG frames returns non-empty bytes starting with the MP4 ftyp box |
| `test_encode_frames_empty_raises` | `tests/test_video_encoder.py` | Encoding zero frames raises `VideoEncoderError` |
| `test_save_face_scan_video_calls_minio_upload` | `tests/test_store.py` | `save_face_scan_video` constructs the correct key and calls `minio_upload` |
| `test_video_key_in_result_when_suspicious` | `tests/test_face_scan_live.py` | `build_live_face_scan_result` returns `video_key` non-None for a SUSPICIOUS session when feature flag is on |
| `test_video_key_none_when_genuine` | `tests/test_face_scan_live.py` | `video_key` is None for a GENUINE session even with feature flag on |
| `test_video_key_none_when_flag_off` | `tests/test_face_scan_live.py` | `video_key` is None for a SUSPICIOUS session when feature flag is off |
| `test_delete_video_endpoint_returns_200` | `tests/test_face_scan_api.py` | DELETE endpoint calls `delete_face_scan_video` and returns 200 |
| `test_get_video_url_endpoint_404_when_no_key` | `tests/test_face_scan_api.py` | GET endpoint returns 404 when `video_key` is None |

All tests mock MinIO and `encode_frames_to_mp4` to avoid live network or FFmpeg calls.

---

## Rollout Order

1. `video_encoder.py` + unit tests (no side effects)
2. `store.py` new functions + tests
3. `db.py` migration + `face_scan_live_results` table
4. `live.py` frame accumulation + post-session encoding (feature flag off by default)
5. `api.py` two new endpoints
6. `face_scan.py` UI panel + delete button
7. Consent notice in browser HTML
8. MinIO lifecycle rule in `docker-compose.yml`
9. Enable `FACE_SCAN_RECORD_VIDEO=true` in staging, verify, then production

Steps 1–6 can be merged behind the feature flag. Step 9 is a config-only change that activates the feature.

---

## What Operators See After This Is Shipped

For a session that returned SUSPICIOUS:
- The existing verdict badge, risk score, evidence bullets, and per-challenge scores remain unchanged
- A new "Session Recording" expander appears below the evidence
- Inside it: a `<video>` player showing exactly the frames the server analysed, with challenge boundary timings visible from the natural motion in the video
- A "Delete Recording" button that permanently removes the video from MinIO and updates the DB row
- A one-line caption: "Recorded — auto-deleted after 90 days"

For GENUINE sessions: the expander does not appear at all.
