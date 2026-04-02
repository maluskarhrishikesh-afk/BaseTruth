"""Settings page."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from basetruth.ui.components import _default_artifact_root, _page_title


def _page_settings() -> None:
    st.markdown(_page_title("⚙️", "Settings"), unsafe_allow_html=True)

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
Settings control how BaseTruth stores data and how the service is run.

- **Artifact root** — the local folder where all reports, structured summaries, and case records are written. Change it in the sidebar to point BaseTruth at a different workspace or network share.
- **Product information** — version numbers, runtime info, and API endpoint details.
- **Quick start commands** — copy-paste commands to start/stop the service, run tests, or rebuild the Docker containers.

Most settings are controlled via environment variables in `.env`. See the README for a full reference.
"""
        )

    st.subheader("Artifact root")
    st.markdown(
        "All reports, structured summaries, and case records are stored under this directory. "
        "Change it in the sidebar to point BaseTruth at a different workspace."
    )
    artifact_root = str(st.session_state.get("artifact_root", _default_artifact_root()))
    st.code(artifact_root, language=None)
    if Path(artifact_root).exists():
        items = list(Path(artifact_root).rglob("*"))
        st.metric("Items in artifact root", len(items))

    st.divider()
    st.subheader("Product information")
    st.markdown(
        """
        | Property | Value |
        |---|---|
        | Product | **BaseTruth** |
        | Version | 0.1.0 |
        | Python | `basetruth` package |
        | UI runtime | Streamlit |
        | REST API | `uvicorn basetruth.api:app` |
        """
    )

    st.divider()
    st.subheader("Quick start commands")
    st.code(
        "# CLI scan\npython -m basetruth.cli scan --input /path/to/doc.pdf\n\n"
        "# Compare payslips across months\n"
        "python -m basetruth.cli compare-payslips --input-dir /path/to/payslips\n\n"
        "# Start UI\nstreamlit run src/basetruth/ui/app.py\n\n"
        "# Start REST API\nuvicorn basetruth.api:app --host 0.0.0.0 --port 8502",
        language="bash",
    )
