from __future__ import annotations

"""
Command-line interface for BaseTruth.

Entry point: `basetruth` (declared in pyproject.toml [project.scripts]).

Commands
--------
  scan   <path>          -- run a full integrity scan on a single document.
  bulk   <directory>     -- scan all supported files in a directory tree.
  compare <directory>    -- cross-month payslip comparison for a folder of
                            previously scanned structured summaries.
  serve  [--host] [--port] -- start the Streamlit web UI.

See `basetruth --help` or `basetruth <command> --help` for full usage.
"""

import argparse
import json
from pathlib import Path

from basetruth.service import BaseTruthService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="basetruth", description="BaseTruth document integrity CLI")
    parser.add_argument("--artifact-root", default="artifacts", help="Directory where BaseTruth should write outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Scan a document or structured summary")
    scan_parser.add_argument("--input", required=True, help="Path to a PDF or a structured/raw JSON file")

    compare_parser = subparsers.add_parser("compare-payslips", help="Compare payslips across months")
    compare_parser.add_argument("--input-dir", required=True, help="Directory containing payslip PDFs or structured summaries")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    service = BaseTruthService(Path(args.artifact_root))

    if args.command == "scan":
        report = service.scan_document(args.input)
        print(json.dumps({
            "truth_score": report["tamper_assessment"]["truth_score"],
            "risk_level": report["tamper_assessment"]["risk_level"],
            "verification_json_path": report["artifacts"]["verification_json_path"],
            "verification_markdown_path": report["artifacts"]["verification_markdown_path"],
        }, indent=2))
        return 0

    if args.command == "compare-payslips":
        comparison = service.compare_payslip_folder(args.input_dir)
        print(json.dumps({
            "summary_count": comparison["summary_count"],
            "anomaly_count": len(comparison.get("anomalies", [])),
            "comparison_json_path": comparison["artifacts"]["comparison_json_path"],
            "comparison_markdown_path": comparison["artifacts"]["comparison_markdown_path"],
        }, indent=2))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
