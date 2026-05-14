"""collect_live_scan_samples.py — export Face Scan Live sessions to a labeled CSV.

This script reads completed face scan live results from the BaseTruth database
(or from JSON files in your_data/) and appends one row per session to:

    fraud_model/data/training_data_face_scan_live.csv

Each row contains the 20 feature values from ml_scorer_live.FEATURE_NAMES plus
a 'label' column that you must supply:

    0 = genuine (real live person)
    1 = spoof   (replay, screen recording, virtual camera injection, etc.)

Usage
-----
    # Export sessions from the database (default — reads all completed sessions)
    python scripts/collect_live_scan_samples.py

    # Export from JSON files in a directory (for sessions not yet in the DB)
    python scripts/collect_live_scan_samples.py --from-json your_data/

    # Dry run — print rows to stdout without writing the CSV
    python scripts/collect_live_scan_samples.py --dry-run

    # Force a specific label for all exported rows (useful when you know a batch
    # is all genuine or all spoof)
    python scripts/collect_live_scan_samples.py --label 0

Labeling Strategy
-----------------
After running this script, open the CSV in a spreadsheet and set the 'label'
column for each row:
    - Sessions you personally completed on a real webcam          → 0 (genuine)
    - Sessions you ran with OBS Virtual Camera / screen replay    → 1 (spoof)
    - Sessions where you are unsure                               → delete the row

A minimum of 5 rows per class is required before training.
A practical minimum for reliable CV metrics is 20+ rows per class.

The script is idempotent: if a session_id is already present in the CSV it will
not be added again, so you can run it repeatedly as new sessions accumulate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── Resolve repo root and add src/ to sys.path so basetruth imports work ─────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from basetruth.face_scan.ml_scorer_live import FEATURE_NAMES, build_feature_vector  # noqa: E402
from basetruth.logger import get_logger  # noqa: E402

log = get_logger(__name__)

_CSV_PATH = _REPO_ROOT / "fraud_model" / "data" / "training_data_face_scan_live.csv"

# All columns written to the CSV.  The 'label' column is appended last so
# it is easy to find and edit in a spreadsheet.
_CSV_COLUMNS = ["session_id", "timestamp_utc", "verdict"] + FEATURE_NAMES + ["label"]


# ─────────────────────────────────────────────────────────────────────────────
# Extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _row_from_result(result: dict, label: int = -1) -> dict:
    """Convert a completed Face Scan Live result dict to a CSV row dict.

    label defaults to -1 (unlabeled) so you can manually fill it in the CSV.
    Pass 0 (genuine) or 1 (spoof) if you already know the ground truth.
    """
    checks      = result.get("checks", {})
    environment = result.get("environment", {})
    trace       = result.get("trace", {})

    # Build the feature vector using the canonical extractor.
    vec = build_feature_vector(checks, environment)

    row: dict = {
        "session_id":    trace.get("decision_trace_id", result.get("filename", "unknown")),
        "timestamp_utc": trace.get("timestamp_utc", ""),
        "verdict":       result.get("verdict", ""),
    }
    # Add each feature by name — zip with FEATURE_NAMES preserves order.
    for name, value in zip(FEATURE_NAMES, vec):
        # NaN → empty string so pandas reads it as NaN on load.
        row[name] = "" if (value != value) else round(float(value), 6)

    row["label"] = label
    return row


def _load_existing_ids(csv_path: Path) -> set:
    """Return the set of session_ids already present in the CSV."""
    if not csv_path.exists():
        return set()
    try:
        import pandas as pd  # noqa: PLC0415
        df = pd.read_csv(csv_path, usecols=["session_id"])
        return set(df["session_id"].astype(str).tolist())
    except Exception:
        return set()


def _append_rows(rows: list, csv_path: Path) -> int:
    """Append rows to the CSV, creating it with a header if it does not exist.

    Returns the number of rows actually written.
    """
    if not rows:
        return 0

    import pandas as pd  # noqa: PLC0415

    new_df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        new_df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(csv_path, mode="w", header=True, index=False)

    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Source: JSON files
# ─────────────────────────────────────────────────────────────────────────────

def _collect_from_json(directory: str, label: int, existing_ids: set) -> list:
    """Walk a directory tree and extract rows from any *.json files that look
    like Face Scan Live result payloads (must have 'scan_type': 'face_scan' and
    'mode': 'live' at the top level).
    """
    rows = []
    base = Path(directory)
    if not base.exists():
        log.warning("Directory not found", extra={"path": str(base)})
        return rows

    for json_file in sorted(base.rglob("*.json")):
        try:
            with open(json_file, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            log.debug("Skipping unreadable file", extra={"file": str(json_file), "error": str(exc)})
            continue

        # Must be a Face Scan Live result.
        if data.get("scan_type") != "face_scan" or data.get("mode") != "live":
            continue

        session_id = data.get("trace", {}).get("decision_trace_id", str(json_file))
        if session_id in existing_ids:
            log.debug("Skipping already-collected session", extra={"session_id": session_id})
            continue

        row = _row_from_result(data, label=label)
        rows.append(row)
        existing_ids.add(session_id)
        log.info("Collected session from JSON", extra={"session_id": session_id, "file": str(json_file)})

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Source: Database
# ─────────────────────────────────────────────────────────────────────────────

def _collect_from_db(label: int, existing_ids: set) -> list:
    """Query the BaseTruth database for completed face scan live results.

    Reads all rows from the 'identity_checks' table where scan_type = 'face_scan'
    and the structured_summary has mode = 'live'.  Returns a list of row dicts.

    Falls back gracefully if the database is unavailable.
    """
    rows = []
    try:
        from basetruth.db import db_available, SessionLocal  # noqa: PLC0415
        if not db_available():
            log.warning("ml_scorer_live: database not available — skipping DB collection")
            return rows

        from basetruth import db as _db  # noqa: PLC0415
        with SessionLocal() as session:
            # Query rows that look like face scan live results.
            results = (
                session.query(_db.IdentityCheck)
                .filter(_db.IdentityCheck.scan_type == "face_scan")
                .all()
            )
            for rec in results:
                try:
                    summary = rec.structured_summary
                    if isinstance(summary, str):
                        summary = json.loads(summary)
                    if not isinstance(summary, dict):
                        continue
                    if summary.get("mode") != "live":
                        continue

                    session_id = summary.get("trace", {}).get("decision_trace_id", str(rec.id))
                    if session_id in existing_ids:
                        continue

                    row = _row_from_result(summary, label=label)
                    rows.append(row)
                    existing_ids.add(session_id)
                    log.info("Collected session from DB", extra={"session_id": session_id})
                except Exception as rec_exc:
                    log.debug("Skipping DB row", extra={"id": rec.id, "error": str(rec_exc)})
    except Exception as exc:
        log.warning("ml_scorer_live: DB collection failed", extra={"error": str(exc)})

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Face Scan Live sessions to the training CSV."
    )
    parser.add_argument(
        "--from-json",
        metavar="DIR",
        default=None,
        help="Path to a directory to scan for JSON result files (default: read from database).",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Ground-truth label to assign: 0=genuine, 1=spoof, -1=unlabeled (default: -1). "
             "Edit the CSV afterwards to fill in -1 rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print rows to stdout instead of writing to the CSV.",
    )
    args = parser.parse_args()

    existing_ids = _load_existing_ids(_CSV_PATH)
    log.info("Existing sessions in CSV", extra={"count": len(existing_ids)})

    if args.from_json:
        rows = _collect_from_json(args.from_json, args.label, existing_ids)
    else:
        rows = _collect_from_db(args.label, existing_ids)

    if not rows:
        print("No new sessions found to collect.")
        return

    if args.dry_run:
        import json as _json
        for row in rows:
            print(_json.dumps(row, indent=2))
        print(f"\nDry run: {len(rows)} row(s) would have been written to {_CSV_PATH}")
        return

    written = _append_rows(rows, _CSV_PATH)
    print(f"Written {written} new row(s) to {_CSV_PATH}")
    print(f"Total rows in CSV: {len(existing_ids) + written}")

    if args.label == -1:
        print("\nReminder: open the CSV and set 'label' to 0 (genuine) or 1 (spoof) for each row.")
        print("Rows with label=-1 will be filtered out during training.")


if __name__ == "__main__":
    main()
