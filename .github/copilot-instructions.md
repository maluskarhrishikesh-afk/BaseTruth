# BaseTruth — GitHub Copilot Instructions

You are working on **BaseTruth**, an AI-powered document fraud detection and identity verification platform.

## Mandatory: Read Before Every Change

Before writing any code, always read these documents first:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System overview, layer descriptions, and technical decisions
- [`docs/FUNCTIONALITY.md`](docs/FUNCTIONALITY.md) — Screen-by-screen behaviour, every button action, and rules that must never be broken
- [`docs/IDENTITY_VERIFICATION.md`](docs/IDENTITY_VERIFICATION.md) — KYC/face-match flow details

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (`src/basetruth/api.py`), port 8000 |
| Frontend UI | Streamlit (`src/basetruth/ui/app.py`), port 8501 |
| Database | PostgreSQL via SQLAlchemy (`src/basetruth/db.py`) |
| Object Storage | MinIO / S3-compatible (`src/basetruth/store.py`) |
| Face Detection | InsightFace (buffalo_l, Python ≤ 3.12) + MediaPipe fallback (Python 3.13+) |
| Deployment | Docker Compose (`docker-compose.yml`) |

## Non-Negotiable Rules

These rules exist because they have caused bugs in the past. **Do not violate them.**

1. **DB availability in the UI render path** — use `_db_available_cached()` and `_minio_available_cached()` (30-second TTL). Never call `db_available()` or `minio_available()` directly from any Streamlit render function — they make live network calls and freeze the UI on every re-render.

2. **Streamlit icon parameter** — `st.info(..., icon=...)`, `st.warning(...)`, `st.error(...)`, `st.success(...)`: the `icon` value must be a real unicode emoji string like `"📧"`. Emoji shortcode strings like `"info"` or `":bell:"` raise `StreamlitAPIException`.

3. **Page title consistency** — every page calls `st.markdown(_page_title(emoji, "Title Text"), unsafe_allow_html=True)`. The emoji and title text must exactly match the corresponding entry in the `_PAGES` dict in `app.py`. Both must be kept in sync when either is changed.

4. **Silent DB failures** — every call to `save_identity_check()`, `reset_db()`, `minio_truncate_bucket()`, or any other write function must either show a success message or a visible error to the user. Never leave a failed write silent.

5. **`init_db()` retry logic** — do not set `st.session_state["db_init_done"] = True` unless `init_db()` returned `True`. The app must keep retrying to create the schema on subsequent renders until the DB comes online.

6. **`_draw_face()` in `vision/face.py`** — this function must always have a proper `def _draw_face(img, face):` declaration before any reference to it. It is called from `compare_faces()` and any typo or missing `def` line causes a `NameError` at runtime on the Identity Verification screen.

7. **Blink liveness (Video KYC)** — blink detection must ALWAYS use Eye Aspect Ratio (EAR) from MediaPipe, even when InsightFace is active. In `api.py _process_kyc_frame`, after InsightFace processes the frame, run `get_mediapipe_faces(img)` and attach `_mp_faces[0].ear` to each InsightFace face object. InsightFace's `det_score` is not a reliable blink indicator.

8. **Database destructive operations** — `TRUNCATE TABLE` must always be inside a `with st.spinner(...)` block so the user sees progress. Never use raw `DELETE FROM` for bulk deletes.

## File Map

| File | Purpose |
|---|---|
| `src/basetruth/ui/app.py` | Streamlit entry point; `_PAGES` dict; sidebar; `main()` router |
| `src/basetruth/ui/components.py` | Shared imports, DB helpers, `_page_title()`, cached availability helpers |
| `src/basetruth/ui/pages/*.py` | One file per screen |
| `src/basetruth/api.py` | FastAPI routes + Video KYC WebSocket |
| `src/basetruth/kyc/liveness.py` | Liveness challenge logic (`analyze_challenge`, `extract_features`) |
| `src/basetruth/vision/face.py` | Face detection + `compare_faces()` + `_draw_face()` |
| `src/basetruth/store.py` | All PostgreSQL + MinIO read/write functions |
| `src/basetruth/db.py` | SQLAlchemy engine, models, `init_db()`, `db_available()` |
| `docs/ARCHITECTURE.md` | Architecture reference — keep updated |
| `docs/FUNCTIONALITY.md` | Screen behaviour reference — keep updated |

## Test Policy

- Run `python -m pytest tests/ -q --tb=short` after every change.
- All tests must pass before committing.
- The test file `tests/test_kyc_ws.py` is excluded (requires a live server) via `pyproject.toml`.

## Commit Policy

- Commit message format: `fix:`, `feat:`, `docs:`, or `refactor:` followed by a short description.
- Always push to `main` after all tests pass.
