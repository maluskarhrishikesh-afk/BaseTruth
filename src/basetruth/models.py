from __future__ import annotations

"""
Shared data-model dataclasses for BaseTruth.

Classes
-------
  Signal          -- a single tamper/validation check result (name, passed,
                     severity, detail).  Used by the tamper scorer and validators.
  VerificationReport -- the top-level output object produced by service.scan_document().
                     Serialised to JSON and Markdown artifacts on disk.
  CaseNote        -- a timestamped analyst note attached to a CaseRecord.
  CaseRecord      -- a persistent case that groups one or more documents for
                     triage, investigation, and disposition tracking.

All classes are plain Python dataclasses so they can be trivially serialised
with json.dumps(..., default=lambda v: v.__dict__) without a custom encoder.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Signal:
    name: str
    severity: str
    score: int
    summary: str
    passed: Optional[bool] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactPaths:
    raw_parse_path: str = ""
    structured_summary_path: str = ""
    verification_json_path: str = ""
    verification_markdown_path: str = ""


@dataclass
class CaseNote:
    created_at: str
    author: str
    text: str


@dataclass
class CaseRecord:
    case_key: str
    status: str = "new"
    disposition: str = "open"
    priority: str = "normal"
    assignee: str = ""
    labels: List[str] = field(default_factory=list)
    notes: List[CaseNote] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class VerificationReport:
    schema_version: int
    generated_at: str
    source: Dict[str, Any]
    pdf_metadata: Dict[str, Any]
    structured_summary: Dict[str, Any]
    tamper_assessment: Dict[str, Any]
    artifacts: Dict[str, Any]
    comparison: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def signals_to_dict(signals: List[Signal]) -> List[Dict[str, Any]]:
    return [asdict(signal) for signal in signals]
