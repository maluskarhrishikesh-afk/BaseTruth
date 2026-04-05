"""Pure helpers for identity-document cross-checks."""
from __future__ import annotations

import re
from typing import Any, Dict


def _name_parts(value: str) -> tuple[str, str]:
    tokens = re.findall(r"[A-Z]+", str(value or "").upper())
    if not tokens:
        return "", ""
    return tokens[0], tokens[-1]


def compare_first_last_names(aadhaar_name: str, pan_name: str) -> Dict[str, Any]:
    """Compare the first and last names on Aadhaar and PAN case-insensitively."""
    aadhaar_first, aadhaar_last = _name_parts(aadhaar_name)
    pan_first, pan_last = _name_parts(pan_name)

    if not aadhaar_first or not aadhaar_last or not pan_first or not pan_last:
        return {
            "passed": None,
            "message": "First/last name comparison could not run because one document is missing a usable full name.",
            "aadhaar_first_name": aadhaar_first,
            "aadhaar_last_name": aadhaar_last,
            "pan_first_name": pan_first,
            "pan_last_name": pan_last,
        }

    passed = aadhaar_first == pan_first and aadhaar_last == pan_last
    if passed:
        message = (
            f"First name '{aadhaar_first}' and last name '{aadhaar_last}' match across Aadhaar and PAN."
        )
    else:
        message = (
            f"Aadhaar shows '{aadhaar_first} {aadhaar_last}' but PAN shows '{pan_first} {pan_last}'."
        )
    return {
        "passed": passed,
        "message": message,
        "aadhaar_first_name": aadhaar_first,
        "aadhaar_last_name": aadhaar_last,
        "pan_first_name": pan_first,
        "pan_last_name": pan_last,
    }


def _normalize_dob(value: str) -> Dict[str, str]:
    text = str(value or "").strip()
    if not text:
        return {"raw": "", "canonical": "", "year": ""}

    parts = [part for part in re.split(r"[^0-9]", text) if part]
    year = ""
    canonical = ""

    if len(parts) == 1 and len(parts[0]) == 4:
        year = parts[0]
    elif len(parts) >= 3:
        first, second, third = parts[0], parts[1], parts[2]
        if len(first) == 4:
            year = first
            canonical = f"{third.zfill(2)}/{second.zfill(2)}/{first}"
        elif len(third) == 4:
            year = third
            canonical = f"{first.zfill(2)}/{second.zfill(2)}/{third}"

    if not year:
        match = re.search(r"(19|20)\d{2}", text)
        if match:
            year = match.group(0)

    return {"raw": text, "canonical": canonical, "year": year}


def compare_dob_values(aadhaar_dob: str, pan_dob: str) -> Dict[str, Any]:
    """Compare Aadhaar and PAN DOB values, using year-only fallback when needed."""
    aadhaar = _normalize_dob(aadhaar_dob)
    pan = _normalize_dob(pan_dob)

    if not aadhaar["raw"] or not pan["raw"]:
        return {
            "passed": None,
            "comparison_type": "missing",
            "message": "DOB comparison could not run because one of the documents does not expose a DOB/YOB value.",
            "aadhaar_dob": aadhaar["raw"],
            "pan_dob": pan["raw"],
        }

    if aadhaar["canonical"] and pan["canonical"]:
        passed = aadhaar["canonical"] == pan["canonical"]
        return {
            "passed": passed,
            "comparison_type": "exact",
            "message": (
                f"DOB matches exactly at {aadhaar['canonical']}."
                if passed
                else f"Aadhaar DOB '{aadhaar['canonical']}' does not match PAN DOB '{pan['canonical']}'."
            ),
            "aadhaar_dob": aadhaar["canonical"],
            "pan_dob": pan["canonical"],
        }

    if aadhaar["year"] and pan["year"]:
        passed = aadhaar["year"] == pan["year"]
        return {
            "passed": passed,
            "comparison_type": "year_only",
            "message": (
                f"Year of birth matches at {aadhaar['year']}."
                if passed
                else f"Aadhaar YOB '{aadhaar['year']}' does not match PAN DOB year '{pan['year']}'."
            ),
            "aadhaar_dob": aadhaar["raw"],
            "pan_dob": pan["raw"],
        }

    return {
        "passed": None,
        "comparison_type": "unparsed",
        "message": "DOB values were present but could not be normalized reliably for comparison.",
        "aadhaar_dob": aadhaar["raw"],
        "pan_dob": pan["raw"],
    }