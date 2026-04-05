from basetruth.reporting.pdf import render_layered_analysis_pdf


def test_render_layered_analysis_pdf_from_persisted_entries() -> None:
    pdf_bytes = render_layered_analysis_pdf(
        entity={
            "entity_ref": "BT-000001",
            "name": "Hrishikesh Maluskar",
            "pan_number": "AQVPM3058P",
            "email": "test@example.com",
        },
        layered_analysis={
            "entries": [
                {
                    "screen_name": "Identity Verification",
                    "section_name": "Aadhaar",
                    "details_captured_json": {
                        "aadhaar_qr": {
                            "name": "HRISHIKESH NAMDEO MALUSKAR",
                            "dob": "20/10/1983",
                            "gender": "M",
                            "dist": "Pune",
                            "state": "Maharashtra",
                        },
                        "authenticity_checks": {
                            "checks": [
                                {
                                    "title": "Layer 1 - Format / Structural Check",
                                    "status": "PASS",
                                    "message": "Aadhaar QR decoded successfully.",
                                },
                                {
                                    "title": "Layer 4 - Image Tampering (ELA)",
                                    "status": "CLEAN (1.0/100)",
                                    "message": "ELA residuals are consistent with an unedited document image.",
                                },
                            ]
                        },
                    },
                }
            ],
            "screens": {
                "Identity Verification": [
                    {
                        "section_name": "Aadhaar",
                        "details_captured_json": {
                            "aadhaar_qr": {
                                "name": "HRISHIKESH NAMDEO MALUSKAR",
                                "dob": "20/10/1983",
                                "gender": "M",
                                "dist": "Pune",
                                "state": "Maharashtra",
                            },
                            "authenticity_checks": {
                                "checks": [
                                    {
                                        "title": "Layer 1 - Format / Structural Check",
                                        "status": "PASS",
                                        "message": "Aadhaar QR decoded successfully.",
                                    },
                                    {
                                        "title": "Layer 4 - Image Tampering (ELA)",
                                        "status": "CLEAN (1.0/100)",
                                        "message": "ELA residuals are consistent with an unedited document image.",
                                    },
                                ]
                            },
                        },
                    },
                    {
                        "section_name": "PAN Card",
                        "details_captured_json": {
                            "pan_extraction": {
                                "pan_number": "AQVPM3058P",
                                "full_name": "HRISHIKESH N MALUSKAR",
                                "date_of_birth": "20/10/1983",
                                "extraction_source": "gemma4",
                            },
                            "pan_format": {"entity_type": "Individual"},
                            "pan_layers": {
                                "layers": [
                                    {
                                        "title": "Layer 1 — PAN Format Check",
                                        "status": "PASS",
                                        "detail": "PAN is syntactically valid.",
                                    }
                                ]
                            },
                        },
                    },
                    {
                        "section_name": "Photo Upload",
                        "details_captured_json": {
                            "document_filename": "aadhaar.jpg",
                            "selfie_filename": "selfie.jpg",
                            "authenticity_checks": {
                                "checks": [
                                    {
                                        "title": "Layer 1 - Format / Structural Check",
                                        "status": "PASS",
                                        "message": "Selfie image decoded successfully.",
                                    }
                                ]
                            },
                        },
                    },
                    {
                        "section_name": "Run Verification",
                        "details_captured_json": {
                            "verdict": "PASS",
                            "display_score": 92.4,
                            "cosine_similarity": 0.8123,
                            "cross_checks": {
                                "first_last_name_match": {"passed": True, "message": "Names match."},
                                "dob_match": {"passed": True, "message": "DOB matches."},
                                "pan_format": {"passed": True, "message": "PAN format valid."},
                                "photo_match": {"passed": True, "message": "Photo match passed."},
                            },
                        },
                    },
                ],
                "Video KYC": [
                    {
                        "section_name": "Remote Session",
                        "details_captured_json": {
                            "verdict": "PASS",
                            "doc_filename": "aadhaar.jpg",
                            "display_score": 88.0,
                            "cosine_similarity": 0.7444,
                            "liveness_passed": True,
                            "liveness_state": "verified",
                            "reference_document_authenticity": {
                                "checks": [
                                    {
                                        "title": "Layer 1 - Format / Structural Check",
                                        "status": "PASS",
                                        "message": "Reference document stored successfully.",
                                    }
                                ]
                            },
                        },
                    }
                ],
                "Scan Document": [
                    {
                        "screen_name": "Scan Document",
                        "section_name": "payslip_mar.pdf",
                        "details_captured_json": {
                            "source_name": "payslip_mar.pdf",
                            "document_type": "payslip",
                            "truth_score": 91,
                            "risk_level": "low",
                            "verdict": "CLEAR",
                            "parse_method": "liteparse",
                            "authenticity_checks": {
                                "checks": [
                                    {
                                        "title": "Layer 1 - Format / Structural Check",
                                        "status": "PASS",
                                        "message": "Payslip structure parsed successfully using liteparse.",
                                    },
                                    {
                                        "title": "Layer 4 - Image Tampering (ELA)",
                                        "status": "N/A",
                                        "message": "No image-specific ELA result was stored for this file type.",
                                    },
                                ]
                            },
                            "structured_summary": {
                                "key_fields": {"employee_name": "Hrishikesh Maluskar"}
                            },
                            "signals": [
                                {"type": "metadata", "message": "No metadata anomalies found."}
                            ],
                        },
                    }
                ],
            },
            "report_state": {
                "generated": False,
                "updated_at": "2026-04-05T10:00:00+00:00",
            },
        },
    )

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_render_layered_analysis_pdf_wraps_long_unbroken_tokens() -> None:
    long_token = "X" * 400

    pdf_bytes = render_layered_analysis_pdf(
        entity={
            "entity_ref": "BT-999999",
            "name": long_token,
            "pan_number": "ABCDE1234F",
            "email": "qa@example.com",
        },
        layered_analysis={
            "entries": [],
            "screens": {
                "Identity Verification": [
                    {
                        "section_name": "Aadhaar",
                        "details_captured_json": {
                            "aadhaar_qr": {"name": long_token, "uid": long_token},
                            "authenticity_checks": {
                                "checks": [
                                    {
                                        "title": "Layer 1 - Format / Structural Check",
                                        "status": "PASS",
                                        "message": long_token,
                                    }
                                ]
                            },
                        },
                    },
                    {
                        "section_name": "PAN Card",
                        "details_captured_json": {
                            "pan_extraction": {"pan_number": "ABCDE1234F", "full_name": long_token},
                            "pan_format": {"passed": True, "message": long_token},
                            "pan_layers": {
                                "layers": [
                                    {"title": "Layer 1 - PAN Format Check", "status": "PASS", "detail": long_token},
                                    {"title": "Layer 4 - Image Tampering (ELA)", "status": "CLEAN", "detail": long_token},
                                ]
                            },
                        },
                    },
                    {
                        "section_name": "Photo Upload",
                        "details_captured_json": {
                            "document_filename": long_token,
                            "selfie_filename": long_token,
                            "authenticity_checks": {
                                "checks": [
                                    {
                                        "title": "Layer 4 - Image Tampering (ELA)",
                                        "status": "CLEAN (1.0/100)",
                                        "message": long_token,
                                    }
                                ]
                            },
                        },
                    },
                    {
                        "section_name": "Run Verification",
                        "details_captured_json": {
                            "verdict": "PASS",
                            "display_score": 77.7,
                            "cosine_similarity": 0.7777,
                            "cross_checks": {
                                "first_last_name_match": {"passed": True, "message": long_token},
                                "dob_match": {"passed": True, "message": long_token},
                                "pan_format": {"passed": True, "message": long_token},
                                "photo_match": {"passed": True, "message": long_token},
                            },
                        },
                    },
                ],
                "Video KYC": [
                    {
                        "section_name": long_token,
                        "details_captured_json": {
                            "verdict": "PASS",
                            "doc_filename": long_token,
                            "selfie_filename": long_token,
                            "display_score": 88.0,
                            "cosine_similarity": 0.7444,
                            "liveness_passed": True,
                            "liveness_state": long_token,
                            "reference_document_authenticity": {
                                "checks": [
                                    {
                                        "title": "Layer 1 - Format / Structural Check",
                                        "status": "PASS",
                                        "message": long_token,
                                    }
                                ]
                            },
                        },
                    }
                ],
                "Scan Document": [
                    {
                        "screen_name": "Scan Document",
                        "section_name": long_token,
                        "details_captured_json": {
                            "source_name": long_token,
                            "document_type": long_token,
                            "truth_score": 91,
                            "risk_level": "low",
                            "verdict": "CLEAR",
                            "parse_method": long_token,
                            "authenticity_checks": {
                                "checks": [
                                    {
                                        "title": "Layer 1 - Format / Structural Check",
                                        "status": "PASS",
                                        "message": long_token,
                                    }
                                ]
                            },
                            "structured_summary": {"key_fields": {"employee_name": long_token}},
                            "signals": [{"type": long_token, "message": long_token}],
                        },
                    }
                ],
            },
            "report_state": {"updated_at": long_token},
        },
    )

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000