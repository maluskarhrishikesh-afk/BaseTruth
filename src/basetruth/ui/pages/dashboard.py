"""Dashboard page."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from basetruth.logger import get_logger
from basetruth.service import BaseTruthService
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _page_title,
    db_dashboard_stats,
)

log = get_logger(__name__)


def _format_percent(value: Optional[float]) -> str:
    """Return a human-friendly percentage string for dashboard cards."""
    if value is None:
        return "Not available"
    return f"{value * 100:.1f}%"


def _count_csv_rows(csv_path: Path) -> Optional[int]:
    """Return the number of data rows in a CSV, excluding the header."""
    if not csv_path.exists() or not csv_path.is_file():
        return None
    try:
        with csv_path.open("r", encoding="utf-8") as handle:
            # Subtract the header row when the file is non-empty.
            row_count = sum(1 for _ in handle) - 1
        return max(row_count, 0)
    except OSError as exc:
        log.warning("dashboard: could not count CSV rows", extra={"path": str(csv_path), "error": str(exc)})
        return None


def _trim_feature_frame_for_model(feature_frame: Any, model: Any) -> Any:
    """Trim extra feature columns when an older saved model expects fewer inputs."""
    try:
        expected = getattr(model.named_steps.get("model"), "n_features_in_", None)
    except Exception:  # noqa: BLE001
        expected = None
    if expected and getattr(feature_frame, "shape", (0, 0))[1] > expected:
        return feature_frame.iloc[:, :expected]
    return feature_frame


def _evaluate_image_model_accuracy() -> tuple[Optional[float], Optional[int], int]:
    """Load the saved image model and score it against the current training CSV."""
    try:
        import joblib  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        from sklearn.metrics import accuracy_score  # noqa: PLC0415
        from basetruth.analysis import ml_scorer as image_ml  # noqa: PLC0415

        csv_path = image_ml._REPO_ROOT / "fraud_model" / "data" / "training_data_image.csv"
        rows = _count_csv_rows(csv_path)
        signal_count = len(image_ml.FEATURE_NAMES)
        if not image_ml._MODEL_PATH.exists() or not csv_path.exists():
            return None, rows, signal_count

        model = joblib.load(image_ml._MODEL_PATH)
        frame = pd.read_csv(csv_path)
        if "ela_suspicious_block_ratio" in frame.columns:
            frame = image_ml._remap_raw_csv(frame)
        drop_present = [col for col in image_ml._DROP_COLS if col in frame.columns]
        frame = frame.drop(columns=drop_present)
        for col in image_ml.FEATURE_NAMES:
            if col not in frame.columns:
                frame[col] = 0.0

        features = _trim_feature_frame_for_model(frame[image_ml.FEATURE_NAMES].copy().astype(float), model)
        labels = (frame["label"].astype(int) > 0).astype(int)
        accuracy = float(accuracy_score(labels, model.predict(features)))
        signal_count = getattr(model.named_steps.get("model"), "n_features_in_", signal_count)
        return accuracy, len(frame), int(signal_count)
    except Exception as exc:  # noqa: BLE001
        log.warning("dashboard: image model evaluation failed", extra={"error": str(exc)})
        return None, None, 18


def _evaluate_pdf_model_accuracy() -> tuple[Optional[float], Optional[int], int]:
    """Load the saved PDF model and score it against the current training CSV."""
    try:
        import joblib  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        from sklearn.metrics import accuracy_score  # noqa: PLC0415
        from basetruth.analysis import ml_scorer_pdf as pdf_ml  # noqa: PLC0415

        csv_path = pdf_ml._REPO_ROOT / "fraud_model" / "data" / "training_data_pdf.csv"
        rows = _count_csv_rows(csv_path)
        signal_count = len(pdf_ml.PDF_FEATURE_NAMES)
        if not pdf_ml._MODEL_PATH.exists() or not csv_path.exists():
            return None, rows, signal_count

        model = joblib.load(pdf_ml._MODEL_PATH)
        frame = pd.read_csv(csv_path)
        if "total_hidden_spans" in frame.columns:
            frame = pdf_ml._remap_raw_pdf_csv(frame)
        drop_present = [col for col in pdf_ml._PDF_DROP_COLS if col in frame.columns]
        frame = frame.drop(columns=drop_present)
        for col in pdf_ml.PDF_FEATURE_NAMES:
            if col not in frame.columns:
                frame[col] = 0.0

        features = _trim_feature_frame_for_model(frame[pdf_ml.PDF_FEATURE_NAMES].copy().astype(float), model)
        labels = frame["label"].astype(int)
        accuracy = float(accuracy_score(labels, model.predict(features)))
        signal_count = getattr(model.named_steps.get("model"), "n_features_in_", signal_count)
        return accuracy, len(frame), int(signal_count)
    except Exception as exc:  # noqa: BLE001
        log.warning("dashboard: PDF model evaluation failed", extra={"error": str(exc)})
        return None, None, 17


def _evaluate_face_scan_live_model_accuracy() -> tuple[Optional[float], Optional[int], int]:
    """Load the saved live Face Scan model and score it against the current training CSV."""
    try:
        import joblib  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        from sklearn.metrics import accuracy_score  # noqa: PLC0415
        from basetruth.face_scan import ml_scorer_live as live_ml  # noqa: PLC0415

        rows = _count_csv_rows(live_ml._CSV_PATH)
        signal_count = len(live_ml.FEATURE_NAMES)
        if not live_ml._MODEL_PATH.exists() or not live_ml._CSV_PATH.exists():
            return None, rows, signal_count

        model = joblib.load(live_ml._MODEL_PATH)
        frame = pd.read_csv(live_ml._CSV_PATH)
        drop_present = [col for col in live_ml._DROP_COLS if col in frame.columns]
        frame = frame.drop(columns=drop_present)
        for col in live_ml.FEATURE_NAMES:
            if col not in frame.columns:
                frame[col] = float("nan")

        features = _trim_feature_frame_for_model(frame[live_ml.FEATURE_NAMES].copy().astype(float), model)
        labels = frame["label"].astype(int)
        accuracy = float(accuracy_score(labels, model.predict(features)))
        signal_count = getattr(model.named_steps.get("model"), "n_features_in_", signal_count)
        return accuracy, len(frame), int(signal_count)
    except Exception as exc:  # noqa: BLE001
        log.warning("dashboard: live face model evaluation failed", extra={"error": str(exc)})
        return None, None, 24


def _inspect_image_model_card() -> Dict[str, Any]:
    accuracy, rows, signal_count = _evaluate_image_model_accuracy()
    from basetruth.analysis import ml_scorer as image_ml  # noqa: PLC0415

    return {
        "label": "Document image model",
        "description": "Used for image-based document fraud checks.",
        "trained": image_ml._MODEL_PATH.exists(),
        "signal_count": signal_count,
        "rows": rows,
        "accuracy": accuracy,
    }


def _inspect_pdf_model_card() -> Dict[str, Any]:
    accuracy, rows, signal_count = _evaluate_pdf_model_accuracy()
    from basetruth.analysis import ml_scorer_pdf as pdf_ml  # noqa: PLC0415

    return {
        "label": "PDF fraud model",
        "description": "Used for digital PDF tamper checks.",
        "trained": pdf_ml._MODEL_PATH.exists(),
        "signal_count": signal_count,
        "rows": rows,
        "accuracy": accuracy,
    }


def _inspect_face_scan_live_model_card() -> Dict[str, Any]:
    accuracy, rows, signal_count = _evaluate_face_scan_live_model_accuracy()
    from basetruth.face_scan import ml_scorer_live as live_ml  # noqa: PLC0415

    return {
        "label": "Live face model",
        "description": "Used for live Face Scan and Video KYC spoof checks.",
        "trained": live_ml._MODEL_PATH.exists(),
        "signal_count": signal_count,
        "rows": rows,
        "accuracy": accuracy,
    }


def _collect_dashboard_model_cards(
    inspectors: Optional[List[Callable[[], Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Collect the Dashboard's machine-learning model cards."""
    active = inspectors or [
        _inspect_image_model_card,
        _inspect_pdf_model_card,
        _inspect_face_scan_live_model_card,
    ]
    return [inspector() for inspector in active]


@st.cache_data(ttl=300, show_spinner=False)
def _get_dashboard_model_cards() -> List[Dict[str, Any]]:
    """Return cached model cards so the Dashboard stays responsive."""
    return _collect_dashboard_model_cards()


def _build_dashboard_summary_cards(
    stats: Dict[str, Any],
    model_cards: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build the short, meaningful metrics shown at the top of the Dashboard."""
    trained_models = sum(1 for card in model_cards if card.get("trained"))
    return [
        {
            "label": "Entities",
            "value": str(stats.get("entities", 0)),
            "help": "Every check and every document is linked to an entity.",
        },
        {
            "label": "Saved documents",
            "value": str(stats.get("total_scans", 0)),
            "help": "How many document scan results are already saved.",
        },
        {
            "label": "Needs review",
            "value": str(stats.get("pending_review", 0)),
            "help": "Documents waiting for a human decision in Review Scans.",
        },
        {
            "label": "ML models ready",
            "value": f"{trained_models}/{len(model_cards)}",
            "help": "How many machine learning models are trained and ready to use.",
        },
    ]


def _build_dashboard_workflow_steps() -> List[Dict[str, Any]]:
    """Return the plain-language workflow guide shown on the Dashboard."""
    return [
        {
            "title": "Choose how you want to verify the person.",
            "body": (
                "Use Identity Verification when the person is with you in person. "
                "Use Video KYC when the person is joining remotely."
            ),
            "actions": [
                {"label": "Open Identity Verification", "page": "identity"},
                {"label": "Open Video KYC", "page": "video_kyc"},
            ],
        },
        {
            "title": "Choose the right document screen.",
            "body": (
                "Use Scan Document for one file. Use Bulk Scan for many files such as payslips, "
                "experience letters, offer letters, bank statements, invoices, and similar folders."
            ),
            "actions": [
                {"label": "Open Scan Document", "page": "scan"},
                {"label": "Open Bulk Scan", "page": "bulk"},
            ],
        },
        {
            "title": "Use the specialist screens when you only need one type of check.",
            "body": (
                "Use Forensic Scan for a quick tamper check on one document. "
                "Use Face Scan for a face-only authenticity check."
            ),
            "actions": [
                {"label": "Open Forensic Scan", "page": "forensic_scan"},
                {"label": "Open Face Scan", "page": "face_scan"},
            ],
        },
        {
            "title": "Keep everything under one entity.",
            "body": (
                "Each person or case should live under one entity. "
                "Document Intelligence is the best place to open one entity and see all linked results together."
            ),
            "actions": [
                {"label": "Open Document Intelligence", "page": "document_intelligence"},
            ],
        },
        {
            "title": "Review and read the final outputs.",
            "body": (
                "Use Review Scans when risky documents need human approval. "
                "Use Reports or Review Reports when you want to read saved outputs and audit evidence."
            ),
            "actions": [
                {"label": "Open Review Scans", "page": "scans"},
                {"label": "Open Reports", "page": "reports"},
            ],
        },
    ]


def _render_dashboard_workflow() -> None:
    """Render the plain-language workflow guide."""
    st.subheader("How to use BaseTruth")
    st.caption("Read this once when you are new. It tells you which screen to use and in what order.")

    for idx, step in enumerate(_build_dashboard_workflow_steps(), start=1):
        st.markdown(f"**{idx}. {step['title']}**")
        st.write(step["body"])
        actions = step.get("actions", [])
        if actions:
            cols = st.columns(len(actions))
            for col, action in zip(cols, actions):
                with col:
                    if st.button(action["label"], key=f"dashboard_step_{idx}_{action['page']}", use_container_width=True):
                        st.session_state["page"] = action["page"]
                        st.rerun()
        st.divider()

    st.info(
        "ML Training Pipeline, Log Analyzer, Database Viewer, Swagger, and BaseTruth AI Copilot are support screens. "
        "Use them when you need model maintenance, logs, database checks, API docs, or guided questions.",
        icon="ℹ️",
    )


def _render_dashboard_model_cards(model_cards: List[Dict[str, Any]]) -> None:
    """Render one simple card per machine-learning model."""
    st.subheader("Machine learning models")
    st.caption("These are the models that help BaseTruth score documents and live face sessions.")
    cols = st.columns(len(model_cards))
    for idx, (col, card) in enumerate(zip(cols, model_cards), start=1):
        with col:
            status = "Ready" if card.get("trained") else "Not trained yet"
            st.markdown(f"**{card['label']}**")
            st.write(card["description"])
            st.write(f"Status: **{status}**")
            st.write(f"Signals used: **{card.get('signal_count', '—')}**")
            st.write(f"Training examples: **{card.get('rows') if card.get('rows') is not None else '—'}**")
            st.write(f"Accuracy: **{_format_percent(card.get('accuracy'))}**")
            st.caption("Accuracy is checked against the saved training data on this machine.")
            if idx < len(model_cards):
                st.markdown("&nbsp;", unsafe_allow_html=True)

    if st.button("Open ML Training Pipeline", key="dashboard_open_ml_training", use_container_width=True):
        st.session_state["page"] = "ml_training"
        st.rerun()


def _page_dashboard(service: BaseTruthService) -> None:
    st.markdown(_page_title("🏠", "Dashboard"), unsafe_allow_html=True)

    st.write(
        "This Dashboard is your simple starting point. "
        "It tells you what needs attention, which screen to use next, and whether the machine learning models are ready."
    )

    model_cards = _get_dashboard_model_cards()

    if _DB_IMPORTS_OK and _db_available_cached():
        stats = db_dashboard_stats()
        if not stats:
            st.warning("Could not load dashboard statistics from the database.")
            return

        summary_cards = _build_dashboard_summary_cards(stats, model_cards)
        cols = st.columns(len(summary_cards))
        for col, card in zip(cols, summary_cards):
            with col:
                st.metric(card["label"], card["value"], help=card["help"])

        if stats.get("pending_review", 0) > 0:
            st.warning(
                f"You have {stats['pending_review']} document(s) waiting for review. "
                "Open Review Scans next.",
                icon="⚠️",
            )
        else:
            st.success("Nothing is waiting for manual review right now.", icon="✅")

    else:
        st.info(
            "Database is offline. The simple workflow guide and model summary still work, but saved document counts may be lower than the real database totals.",
            icon="📴",
        )
        reports = service.list_reports()
        ver_reports = [r for r in reports if r.get("kind") == "verification"]
        offline_stats = {
            "entities": 0,
            "total_scans": len(ver_reports),
            "pending_review": 0,
        }
        summary_cards = _build_dashboard_summary_cards(offline_stats, model_cards)
        cols = st.columns(len(summary_cards))
        for col, card in zip(cols, summary_cards):
            with col:
                st.metric(card["label"], card["value"], help=card["help"])

    st.divider()
    _render_dashboard_workflow()

    st.divider()
    _render_dashboard_model_cards(model_cards)
