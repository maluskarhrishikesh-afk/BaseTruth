"""train_ml_scorer_pdf.py — Train and save the XGBoost PDF fraud classifier.

Usage:
    python scripts/train_ml_scorer_pdf.py

Philosophy: Real data only.
  Synthetic datasets were removed because in fraud detection fake patterns
  produce false confidence.  This script trains exclusively on PDFs that
  your forensic engine has actually scanned and labelled.

What this script does:
  1. Loads data/training_data_pdf.csv  (real PDFs collected by you)
       - Columns are automatically remapped to the 18-feature expert schema
  2. Passes it to ml_scorer_pdf.train_pdf() which:
       - Runs 5-fold stratified CV (XGBoost: n_estimators=400, max_depth=6)
       - Saves data/ml_scorer_pdf.pkl ONLY if ROC AUC >= 0.80
  3. Prints a metrics summary with feature importances

Adding more data:
  Scan more PDFs using collect_training_samples_pdf.py, then re-run this
  script.  The model automatically picks up the updated CSV.
"""
from __future__ import annotations

import sys
from pathlib import Path

# fraud_model/scripts/ → fraud_model/ → repo root (3 levels)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from basetruth.analysis.ml_scorer_pdf import train_pdf  # noqa: E402
from basetruth.logger import get_logger                  # noqa: E402

log = get_logger(__name__)

_OWN_CSV    = str(_REPO_ROOT / "fraud_model" / "data" / "training_data_pdf.csv")
_OUTPUT_PKL = str(_REPO_ROOT / "fraud_model" / "models" / "ml_scorer_pdf.pkl")


def main() -> None:
    csv_paths = []

    # Real PDFs only — scanned and labelled from actual documents
    if Path(_OWN_CSV).exists():
        csv_paths.append(_OWN_CSV)
        print(f"[OK] Real PDF CSV loaded: {_OWN_CSV}")
    else:
        print(f"[WARN] Real PDF CSV not found: {_OWN_CSV}", file=sys.stderr)
        print("  Add PDFs to tests/sample/ and run collect_training_samples_pdf.py first.", file=sys.stderr)

    if not csv_paths:
        print("[ERROR] No training CSVs found. Aborting.", file=sys.stderr)
        sys.exit(1)

    print("\nTraining XGBoost PDF fraud scorer …\n")

    try:
        metrics = train_pdf(csv_paths, _OUTPUT_PKL)
    except ValueError as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    print("─────────────────────────────────────────────────")
    print("  PDF ML Scorer — Training Results")
    print("─────────────────────────────────────────────────")
    print(f"  Rows trained  : {metrics['rows_trained']:,}")
    print(f"  Accuracy (CV) : {metrics['accuracy']:.4f}")
    print(f"  F1 Score (CV) : {metrics['f1']:.4f}")
    print(f"  ROC AUC  (CV) : {metrics['roc_auc']:.4f}")

    if metrics.get("hard_case"):
        hc = metrics["hard_case"]
        print(f"\n  Hard Cases    : {hc['n_hard']} rows")
        print(f"  HC Accuracy   : {hc['accuracy']:.4f}")
        print(f"  HC F1 Score   : {hc['f1']:.4f}")

    if metrics.get("feature_importances"):
        print("\n  Feature Importances:")
        sorted_imp = sorted(metrics["feature_importances"].items(), key=lambda x: -x[1])
        for feat, imp in sorted_imp:
            bar = "█" * int(imp * 40)
            print(f"    {feat:<40} {imp:.4f}  {bar}")

    print("\n─────────────────────────────────────────────────")
    print(f"  Model saved → {metrics['model_path']}")
    print("─────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
