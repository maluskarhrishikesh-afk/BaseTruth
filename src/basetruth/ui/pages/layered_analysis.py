"""Layered Analysis page — explainable AI audit view across all verification activity."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import streamlit as st

from basetruth.service import BaseTruthService
from basetruth.ui.components import (
    _DB_IMPORTS_OK,
    _db_available_cached,
    _page_title,
    get_all_entities_with_scans,
    get_entity_layered_analysis,
    mark_layered_report_generated,
    minio_delete_object,
    minio_get_object,
    minio_upload,
)

_LAYERED_PDF_KEY = "{entity_ref}/layered_analysis_report.pdf"


def _verdict_badge(verdict: str) -> str:
    normalized = str(verdict or "UNKNOWN").upper()
    if normalized == "PASS":
        color = "#16a34a"
    elif normalized in {"FAIL", "CRITICAL"}:
        color = "#dc2626"
    else:
        color = "#d97706"
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:999px;'
        f'background:{color}1a;color:{color};font-size:12px;font-weight:700;">{normalized}</span>'
    )


def _render_definition_list(items: Iterable[tuple[str, Any]]) -> None:
    rows = [(label, value) for label, value in items if value not in (None, "", [], {})]
    if not rows:
        st.caption("No stored details available.")
        return
    for label, value in rows:
        st.markdown(f"**{label}:** {value}")


def _render_check_block(title: str, payload: Dict[str, Any]) -> None:
    passed = payload.get("passed")
    if passed is True:
        renderer = st.success
        prefix = "PASS"
    elif passed is False:
        renderer = st.error
        prefix = "FAIL"
    else:
        renderer = st.info
        prefix = "INFO"
    renderer(f"**{title}: {prefix}**  \n{payload.get('message', 'No detail available.')}")


def _legacy_visible_pan_layers(pan_layers: Dict[str, Any]) -> List[Dict[str, Any]]:
    allowed_prefixes = (
        "Layer 1",
        "Layer 2",
        "Layer 4",
    )
    return [
        layer
        for layer in (pan_layers.get("layers") or [])
        if str(layer.get("title", "")).startswith(allowed_prefixes)
    ]


def _render_authenticity_checks(payload: Dict[str, Any]) -> None:
    checks = payload.get("checks") or []
    if not checks:
        st.caption("No stored upload authenticity checks found for this section.")
        return
    st.markdown("**Upload Authenticity Checks**")
    for check in checks:
        _render_check_block(check.get("title", "Check"), check)


def _render_pan_layers(pan_layers: Dict[str, Any]) -> None:
    layers = _legacy_visible_pan_layers(pan_layers)
    if not layers:
        st.caption("No stored PAN layered analysis found for this record.")
        return

    for layer in layers:
        st.markdown(
            f"**{layer.get('title', 'Layer')}**  \n"
            f"Status: {layer.get('status', 'UNKNOWN')}  \n"
            f"{layer.get('detail', '')}"
        )
        st.divider()

    if pan_layers.get("overall_score") is not None:
        st.info(
            f"Overall PAN score: {pan_layers['overall_score']}/100 — {pan_layers.get('risk_label', 'UNKNOWN')}"
        )


def _render_identity_section(entries: List[Dict[str, Any]]) -> None:
    sections = {entry.get("section_name", ""): entry for entry in entries}
    st.subheader("Identity Verification")
    if not sections:
        st.caption("No saved Identity Verification analysis entries for this entity.")
        return
    aadhaar = (sections.get("Aadhaar") or {}).get("details_captured_json") or {}
    pan = (sections.get("PAN Card") or {}).get("details_captured_json") or {}
    photo = (sections.get("Photo Upload") or {}).get("details_captured_json") or {}
    verification = (sections.get("Run Verification") or {}).get("details_captured_json") or {}
    cross_checks = verification.get("cross_checks") or {}

    with st.expander("Aadhaar", expanded=False):
        aadhaar_qr = aadhaar.get("aadhaar_qr") or {}
        _render_definition_list(
            [
                ("Document", aadhaar.get("doc_filename")),
                ("Name", aadhaar_qr.get("name")),
                ("DOB/YOB", aadhaar_qr.get("dob") or aadhaar_qr.get("yob")),
                ("Gender", aadhaar_qr.get("gender")),
                ("District", ", ".join(filter(None, [aadhaar_qr.get("dist"), aadhaar_qr.get("state")]))),
                ("UID", aadhaar_qr.get("uid")),
                ("QR type", aadhaar_qr.get("qr_type")),
                ("Captured at", aadhaar.get("captured_at")),
            ]
        )
        _render_authenticity_checks(aadhaar.get("authenticity_checks") or {})
        st.json(aadhaar)

    with st.expander("PAN Card", expanded=False):
        st.markdown(_verdict_badge((pan.get("pan_format") or {}).get("passed") and "PASS" or "INFO"), unsafe_allow_html=True)
        _render_definition_list(
            [
                ("Document", pan.get("doc_filename")),
                ("PAN", (pan.get("pan_extraction") or {}).get("pan_number")),
                ("Entity type", (pan.get("pan_format") or {}).get("entity_type")),
                ("Full name", (pan.get("pan_extraction") or {}).get("full_name")),
                ("Father's name", (pan.get("pan_extraction") or {}).get("father_name")),
                ("Date of birth", (pan.get("pan_extraction") or {}).get("date_of_birth")),
                ("Extraction source", (pan.get("pan_extraction") or {}).get("extraction_source")),
                ("Captured at", pan.get("captured_at")),
            ]
        )
        _render_check_block("PAN Format", pan.get("pan_format", {}))
        st.markdown("**PAN Layered Analysis**")
        _render_pan_layers(pan.get("pan_layers") or {})
        st.json(pan)

    with st.expander("Photo Upload", expanded=False):
        _render_definition_list(
            [
                ("Document filename", photo.get("document_filename")),
                ("Selfie filename", photo.get("selfie_filename")),
                ("Document uploaded", photo.get("document_uploaded")),
                ("Selfie uploaded", photo.get("selfie_uploaded")),
                ("Captured at", photo.get("captured_at")),
            ]
        )
        _render_authenticity_checks(photo.get("authenticity_checks") or {})
        st.json(photo)

    with st.expander("Run Verification", expanded=False):
        st.markdown(_verdict_badge(verification.get("verdict", "UNKNOWN")), unsafe_allow_html=True)
        _render_definition_list(
            [
                ("Status", verification.get("status")),
                ("Verdict", verification.get("verdict")),
                ("Display score", f"{verification.get('display_score', 0) or 0:.1f}%"),
                ("Cosine similarity", f"{verification.get('cosine_similarity', 0) or 0:.4f}"),
                ("Threshold", verification.get("threshold")),
                ("Match", verification.get("is_match")),
                ("Captured at", verification.get("captured_at")),
            ]
        )
        _render_check_block("First Name & Last Name Match", cross_checks.get("first_last_name_match", {}))
        _render_check_block("DOB Match", cross_checks.get("dob_match", {}))
        _render_check_block("PAN Format", cross_checks.get("pan_format", {}))
        _render_check_block("Photo Match", cross_checks.get("photo_match", {}))
        st.json(verification)


def _render_video_kyc_section(entries: List[Dict[str, Any]]) -> None:
    st.subheader("Video KYC")
    if not entries:
        st.caption("No saved Video KYC analysis entries for this entity.")
        return

    for entry in entries:
        details = entry.get("details_captured_json") or {}
        label = f"{entry.get('section_name', 'Session')} — {(entry.get('updated_at') or '')[:19]}"
        with st.expander(label, expanded=False):
            st.markdown(_verdict_badge(details.get("verdict", "UNKNOWN")), unsafe_allow_html=True)
            _render_definition_list(
                [
                    ("Liveness passed", details.get("liveness_passed")),
                    ("Liveness state", details.get("liveness_state")),
                    ("Display score", f"{details.get('display_score', 0) or 0:.1f}%"),
                    ("Cosine similarity", f"{details.get('cosine_similarity', 0) or 0:.4f}"),
                    ("Reference document", details.get("doc_filename")),
                    ("Live capture", details.get("selfie_filename")),
                ]
            )
            if details.get("reference_document_authenticity"):
                _render_authenticity_checks(details.get("reference_document_authenticity") or {})
            if details.get("live_capture_authenticity"):
                _render_authenticity_checks(details.get("live_capture_authenticity") or {})
            with st.expander("Raw Stored Payload", expanded=False):
                st.json(details)


def _render_scan_section(screen_name: str, entries: List[Dict[str, Any]]) -> None:
    st.subheader(screen_name)
    if not entries:
        st.caption(f"No saved {screen_name.lower()} analysis entries for this entity.")
        return

    for entry in entries:
        details = entry.get("details_captured_json") or {}
        structured_summary = details.get("structured_summary") or {}
        key_fields = structured_summary.get("key_fields") or structured_summary.get("named_fields") or {}
        signals = details.get("signals") or []
        label = (
            f"{entry.get('section_name', 'Scan')} — "
            f"{(details.get('generated_at') or entry.get('updated_at') or '')[:19]}"
        )
        with st.expander(label, expanded=False):
            st.markdown(_verdict_badge(details.get("risk_level", details.get("verdict", "UNKNOWN"))), unsafe_allow_html=True)
            _render_definition_list(
                [
                    ("Source file", details.get("source_name")),
                    ("Document type", details.get("document_type")),
                    ("Truth score", details.get("truth_score")),
                    ("Risk level", details.get("risk_level")),
                    ("Parse method", details.get("parse_method")),
                    ("Verdict", details.get("verdict")),
                ]
            )
            _render_authenticity_checks(details.get("authenticity_checks") or {})
            st.markdown("**Key Extracted Fields**")
            if key_fields:
                _render_definition_list(list(key_fields.items())[:10])
            else:
                st.caption("No key fields were stored for this scan.")

            st.markdown("**Top Forensic Signals**")
            if signals:
                for signal in signals[:8]:
                    st.markdown(
                        f"- **{signal.get('type', 'signal')}**: {signal.get('message') or signal.get('reason') or signal.get('description') or 'No detail available.'}"
                    )
            else:
                st.caption("No forensic signals were stored for this scan.")

            with st.expander("Raw Stored Payload", expanded=False):
                st.json(details)


def _page_layered_analysis(service: BaseTruthService) -> None:
    _ = service
    st.markdown(_page_title("🧾", "Layered Analysis"), unsafe_allow_html=True)
    st.caption(
        "Explainable AI audit view for regulators and auditors — extraction outputs, deterministic checks, model evidence, and stored fraud signals."
    )

    with st.expander("ℹ️ How to use this screen", expanded=False):
        st.markdown(
            """
This screen is intentionally detailed and audit-focused.

- It shows the stored evidence trail behind Identity Verification, Video KYC, and Document Scan decisions.
- Use it to review exactly what was extracted, what rules were applied, and what the final verdict was.
- `Generate Final Report` produces a detailed explainable-AI PDF for the selected entity.
"""
        )

    if not _DB_IMPORTS_OK or not _db_available_cached():
        st.info("📴 Database is offline. Connect PostgreSQL to review stored layered analysis.")
        return

    entities = get_all_entities_with_scans()
    if not entities:
        st.info("No saved verification activity found yet.")
        return

    search = st.text_input(
        "🔍 Filter applicants",
        placeholder="Name, PAN, email, or BT-reference...",
        key="layered_analysis_search",
    ).strip().lower()

    filtered = [
        entity
        for entity in entities
        if not search
        or search in (entity.get("name") or "").lower()
        or search in (entity.get("pan_number") or "").lower()
        or search in (entity.get("email") or "").lower()
        or search in (entity.get("entity_ref") or "").lower()
    ]

    st.caption(f"{len(filtered)} applicant(s) shown")
    st.divider()

    for entity in filtered:
        entity_ref = entity["entity_ref"]
        layered = get_entity_layered_analysis(entity_ref)
        screens = layered.get("screens") or {}
        report_state = layered.get("report_state") or {}
        if not screens:
            continue

        title = f"🧾 **{entity.get('name') or entity_ref}** — {entity_ref}"
        subtitle_parts = [part for part in [entity.get("pan_number"), entity.get("email")] if part]
        if subtitle_parts:
            title += f"  ·  {'  ·  '.join(subtitle_parts)}"

        with st.expander(title, expanded=False):
            m1, m2, m3 = st.columns(3)
            m1.metric("Identity Sections", len(screens.get("Identity Verification", [])))
            m2.metric("Video KYC Sections", len(screens.get("Video KYC", [])))
            m3.metric(
                "Scan Sections",
                len(screens.get("Scan Document", [])) + len(screens.get("Bulk Scan", [])),
            )
            st.divider()

            action_col1, action_col2 = st.columns(2)
            with action_col1:
                can_generate = bool(report_state.get("has_entries")) and bool(report_state.get("can_generate"))
                if not report_state.get("has_entries"):
                    st.info("No layered-analysis entries exist yet for this entity.")
                elif report_state.get("generated"):
                    st.success(
                        "Final report already generated for the current evidence set. Add fresh data for this entity to unlock regeneration."
                    )
                if st.button(
                    "📄 Generate Final Report",
                    key=f"layered_generate_{entity_ref}",
                    use_container_width=True,
                    disabled=not can_generate,
                ):
                    with st.spinner("Building layered analysis report..."):
                        try:
                            from basetruth.reporting.pdf import render_layered_analysis_pdf  # noqa: PLC0415

                            pdf_bytes = render_layered_analysis_pdf(
                                entity=entity,
                                layered_analysis=layered,
                            )
                            pdf_key = _LAYERED_PDF_KEY.format(entity_ref=entity_ref)
                            minio_delete_object(pdf_key)
                            upload_ok = minio_upload(pdf_key, pdf_bytes, "application/pdf")
                            mark_ok = mark_layered_report_generated(entity_ref, pdf_key) if upload_ok else False
                            st.session_state[f"layered_pdf_{entity_ref}"] = pdf_bytes
                            if upload_ok and mark_ok:
                                st.success("Final layered-analysis report generated and saved.")
                            else:
                                st.warning("PDF generated, but report state could not be fully updated. Use the download button below.")
                        except Exception as exc:  # noqa: BLE001
                            st.error(f"Failed to build layered-analysis report: {exc}")

                pdf_bytes = st.session_state.get(f"layered_pdf_{entity_ref}")
                if not pdf_bytes:
                    pdf_bytes = minio_get_object(report_state.get("minio_key") or _LAYERED_PDF_KEY.format(entity_ref=entity_ref))
                    if pdf_bytes:
                        st.session_state[f"layered_pdf_{entity_ref}"] = pdf_bytes
                if pdf_bytes:
                    st.download_button(
                        "⬇ Download Final Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"layered_analysis_{entity_ref}.pdf",
                        mime="application/pdf",
                        key=f"layered_download_{entity_ref}",
                        use_container_width=True,
                    )

            with action_col2:
                st.info("This view is audit-facing. Final approval remains with human review.")
                _render_definition_list(
                    [
                        ("Final report generated", "Yes" if report_state.get("generated") else "No"),
                        ("Generated at", report_state.get("generated_at")),
                        ("Last evidence update", report_state.get("updated_at")),
                    ]
                )

            _render_identity_section(screens.get("Identity Verification", []))
            st.divider()
            _render_video_kyc_section(screens.get("Video KYC", []))
            st.divider()
            _render_scan_section("Scan Document", screens.get("Scan Document", []))
            if screens.get("Bulk Scan"):
                st.divider()
                _render_scan_section("Bulk Scan", screens.get("Bulk Scan", []))