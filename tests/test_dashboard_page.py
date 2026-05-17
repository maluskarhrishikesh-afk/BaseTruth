from basetruth.ui.pages.dashboard import (
    _build_dashboard_summary_cards,
    _build_dashboard_workflow_steps,
    _collect_dashboard_model_cards,
    _format_percent,
)


def test_format_percent_handles_missing_values() -> None:
    assert _format_percent(None) == "Not available"
    assert _format_percent(0.9382) == "93.8%"


def test_build_dashboard_summary_cards_counts_ready_models() -> None:
    stats = {"entities": 12, "total_scans": 41, "pending_review": 3}
    model_cards = [
        {"trained": True},
        {"trained": False},
        {"trained": True},
    ]

    cards = _build_dashboard_summary_cards(stats, model_cards)

    assert [card["label"] for card in cards] == [
        "Entities",
        "Saved documents",
        "Needs review",
        "ML models ready",
    ]
    assert [card["value"] for card in cards] == ["12", "41", "3", "2/3"]



def test_build_dashboard_workflow_steps_use_simple_language() -> None:
    steps = _build_dashboard_workflow_steps()

    assert len(steps) == 5
    assert "Identity Verification" in steps[0]["body"]
    assert "Video KYC" in steps[0]["body"]
    assert "Bulk Scan" in steps[1]["body"]
    assert "entity" in steps[3]["body"].lower()
    assert "Review Scans" in steps[4]["body"]



def test_collect_dashboard_model_cards_uses_given_inspectors() -> None:
    cards = _collect_dashboard_model_cards(
        inspectors=[
            lambda: {"label": "One", "trained": True},
            lambda: {"label": "Two", "trained": False},
        ]
    )

    assert cards == [
        {"label": "One", "trained": True},
        {"label": "Two", "trained": False},
    ]
