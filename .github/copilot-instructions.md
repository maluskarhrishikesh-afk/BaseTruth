# BaseTruth — GitHub Copilot Instructions

You are working on **BaseTruth**, an AI-powered document fraud detection and identity verification platform.

## Mandatory: Read Before Every Change

Before writing any code, always read these documents first in this order:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — System overview, layer descriptions, and technical decisions
- [`docs/FUNCTIONALITY.md`](docs/FUNCTIONALITY.md) — Screen-by-screen behaviour, every button action, and rules that must never be broken
- [`docs/IDENTITY_VERIFICATION.md`](docs/IDENTITY_VERIFICATION.md) — KYC/face-match flow details
- [`docs/TESTING.md`](docs/TESTING.md) — How BaseTruth tests should be written, what must be unit tested, and how to validate changes

These documents are the product contract. If implementation, tests, and docs diverge, bring them back into sync in the same change.

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

9. **Documentation Is Source of Truth** — Always follow the documentation first. If behaviour changes, update the relevant docs in the same change. Do not leave stale or contradictory documentation behind.

10. **File Synchronization** — Always keep all configuration files, AI rule files (`.antigravityrules`, `.cursorrules`, `copilot-instructions.md`), and related technical documentation in sync whenever modifying one.

11. **Git Hygiene** — Respect `.gitignore` at all times. Never add ignored files, generated artifacts, local credentials, logs, models, or runtime outputs to git. Keep changes in a clean, reviewable, push-ready state. If git operations are requested, run tests first, then commit and push.

12. **Testing Is Mandatory** — Every non-trivial code change must add or update unit tests. Prefer narrow, deterministic `pytest` tests that do not require a live DB, MinIO, Ollama, webcam, or network. If a behaviour truly cannot be unit tested, add the cheapest higher-level test possible and document the limitation.

13. **Remove Stale Code, Docs, And Unwanted Files** — When replacing behaviour, remove dead code, obsolete helpers, stale branches, unused imports, outdated comments, stale documentation, and tracked unwanted files such as generated binaries, one-off sample outputs, runtime logs, or scratch artifacts in the same area. BaseTruth should not accumulate superseded logic or stray files.

14. **Design Principles** — Follow coding standards like SOLID design principles uniformly. Ensure separation of concerns, single responsibilities, and well-structured interfaces to keep the application modular.

15. **Temporary Files** — Whenever the "Coding Agent" creates temporary files or scratchpad tests, they MUST be deleted after use to avoid confusion and maintain clean codebases.

16. **Meaningful Loggers** — Every non-trivial function, API endpoint, service method, and background process must include structured log calls using `get_logger(__name__)`. Log at the right level: `log.info` for key lifecycle events (scan started, entity saved, session created), `log.warning` for recoverable issues (fallback used, field missing), `log.error` for failures that need attention, `log.debug` for step-by-step diagnostic detail. Always include relevant context in the message (entity_ref, scan_id, doc_type, etc.) so log entries are self-contained and searchable without needing a debugger.

17. **Code Comments in Simple Language** — Every significant function, algorithm, and non-obvious block of logic must have inline comments written in plain, simple English that any developer can understand on first read. Comments must explain *why* the code does something (the intent and the reason for choosing this approach), not just *what* it does (which the code itself already shows). Single-letter variables or complex maths must always be followed by a comment explaining what they represent. Forensic functions (ELA, DCT, noise, clone, etc.) must each have a plain-English docstring explaining the technique in 3–5 sentences a non-expert can follow.

## File Map

| File | Purpose |
|---|---|
| `src/basetruth/ui/app.py` | Streamlit entry point; `_PAGES` dict; sidebar; `main()` router |
| `src/basetruth/ui/components.py` | Shared imports, DB helpers, `_page_title()`, cached availability helpers |
| `src/basetruth/ui/pages/*.py` | One file per screen |
| `src/basetruth/api.py` | FastAPI routes + Video KYC WebSocket |
| `src/basetruth/kyc/liveness.py` | Liveness challenge logic (`analyze_challenge`, `extract_features`) |
| `src/basetruth/vision/face.py` | Face detection + `compare_faces()` + `_draw_face()` |
| `src/basetruth/store.py` | All PostgreSQL + MinIO read/write functions; `save_scan_to_db` handles both OCR (structured_summary) and Bulk (\_document\_extraction key present) paths |
| `src/basetruth/db.py` | SQLAlchemy engine, models, `init_db()`, `db_available()` |
| `src/basetruth/integrations/document_extract.py` | Gemma4-powered field extraction for bulk scans (payslips, marksheets, offer letters, etc.); called per-document in `bulk.py` after forensics; result stored in `document_extractions` |
| `docs/ARCHITECTURE.md` | Architecture reference — keep updated |
| `docs/FUNCTIONALITY.md` | Screen behaviour reference — keep updated |
| `docs/TESTING.md` | Testing reference — how to write, scope, and run BaseTruth tests |

## Test Policy

- Follow `docs/TESTING.md` for every code change.
- Add or update unit tests for every non-trivial behaviour change.
- Run the narrowest relevant `pytest` command first, then run `python -m pytest tests/ -q --tb=short`.
- All tests must pass before any commit or push.
- The test file `tests/test_kyc_ws.py` is excluded (requires a live server) via `pyproject.toml`.

## Commit Policy

- Commit message format: `fix:`, `feat:`, `docs:`, or `refactor:` followed by a short description.
- Respect `.gitignore`; never commit ignored files, local secrets, runtime outputs, or generated artifacts.
- When git operations are requested, push only after all relevant tests pass.
