from basetruth.integrations import db_query


def test_schema_summary_mentions_video_kyc_and_face_scan_tables(monkeypatch) -> None:
    monkeypatch.setattr(db_query, "_load_database_md", lambda: "DATABASE SNAPSHOT")
    monkeypatch.setattr(db_query, "get_qna_db_rules", lambda: "DB RULES")
    monkeypatch.setattr(db_query, "get_qna_training_examples", lambda: "TRAINING EXAMPLES")

    summary = db_query.get_schema_summary()

    assert "video_kyc_checks" in summary
    assert "face_scan_live_results" in summary
    assert "face match + Video KYC results" not in summary
    assert "Live Face Scan session rows live in face_scan_live_results and use session_id" in summary


def test_qna_prompt_assets_define_split_identity_and_face_scan_tables() -> None:
    db_query._load_qna_prompts.cache_clear()

    prompts = db_query._load_qna_prompts()

    assert "VALID TABLES — only these 7 tables exist" in prompts["system_prompt"]
    assert "video_kyc_checks" in prompts["system_prompt"]
    assert "face_scan_live_results" in prompts["system_prompt"]
    assert "TABLE: video_kyc_checks" in prompts["db_query_rules"]
    assert "TABLE: face_scan_live_results" in prompts["db_query_rules"]
    assert '"face scan" / "live face scan" / "spoof check" / "deepfake check" → face_scan_live_results table' in prompts["glossary"]