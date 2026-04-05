from basetruth.analysis.identity_checks import compare_dob_values, compare_first_last_names


def test_compare_first_last_names_accepts_middle_name_variation() -> None:
    result = compare_first_last_names("Hrishikesh Namdeo Maluskar", "Hrishikesh Maluskar")

    assert result["passed"] is True
    assert result["aadhaar_first_name"] == "HRISHIKESH"
    assert result["pan_last_name"] == "MALUSKAR"


def test_compare_first_last_names_rejects_last_name_mismatch() -> None:
    result = compare_first_last_names("Hrishikesh Maluskar", "Hrishikesh Patil")

    assert result["passed"] is False
    assert "Aadhaar shows 'HRISHIKESH MALUSKAR'" in result["message"]


def test_compare_dob_values_matches_full_date() -> None:
    result = compare_dob_values("12/08/1995", "1995-08-12")

    assert result["passed"] is True
    assert result["comparison_type"] == "exact"
    assert result["aadhaar_dob"] == "12/08/1995"
    assert result["pan_dob"] == "12/08/1995"


def test_compare_dob_values_matches_year_only_when_needed() -> None:
    result = compare_dob_values("1995", "12/08/1995")

    assert result["passed"] is True
    assert result["comparison_type"] == "year_only"


def test_compare_dob_values_rejects_mismatch() -> None:
    result = compare_dob_values("1995", "12/08/1994")

    assert result["passed"] is False
    assert "does not match" in result["message"]