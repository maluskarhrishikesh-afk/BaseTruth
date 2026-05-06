"""train_ml_scorer.py — Train and save the XGBoost forensic image fraud classifier.

Usage:
    python scripts/train_ml_scorer.py

Philosophy: Real data only.
  Synthetic datasets were removed because in fraud detection fake patterns
  produce false confidence.  This script trains exclusively on documents that
  your forensic engine has actually scanned and labelled.

What this script does:
  1. Loads data/training_data_image.csv  (real images collected by you)
  2. Passes it to ml_scorer.train() which:
       - Remaps raw-engine columns to the normalised 11-feature schema
       - Runs 5-fold stratified cross-validation (XGBoost or RF fallback)
       - Saves data/ml_scorer_image.pkl ONLY if ROC AUC >= 0.80
  3. Prints the final metrics table

Adding more data:
  Scan more documents using collect_training_samples.py, then re-run this
  script.  The model automatically picks up the updated CSV.

The ROC AUC gate (0.80) prevents saving a model that can't outperform a
naive classifier on the test folds.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from either the project root or the scripts/ directory
# fraud_model/scripts/ → fraud_model/ → repo root (3 levels)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from basetruth.analysis.ml_scorer import train  # noqa: E402
from basetruth.logger import get_logger  # noqa: E402

log = get_logger(__name__)

_OWN_CSV    = str(_REPO_ROOT / "fraud_model" / "data" / "training_data_image.csv")
_OUTPUT_PKL = str(_REPO_ROOT / "fraud_model" / "models" / "ml_scorer_image.pkl")


def main() -> None:
    csv_paths = []

    # Real images only — scanned and labelled from actual documents
    if Path(_OWN_CSV).exists():
        csv_paths.append(_OWN_CSV)
        print(f"[OK] Real image CSV loaded: {_OWN_CSV}")
    else:
        print(f"[WARN] Real image CSV not found: {_OWN_CSV}", file=sys.stderr)
        print("  Add images to tests/sample/ and run collect_training_samples.py first.", file=sys.stderr)

    if not csv_paths:
        print("[ERROR] No training CSVs found. Aborting.", file=sys.stderr)
        sys.exit(1)

    print("\nTraining XGBoost forensic scorer …")
    try:
        metrics = train(csv_paths, _OUTPUT_PKL)
    except ValueError as exc:
        # train() raises ValueError when ROC AUC < 0.80
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        sys.exit(1)

    # Print a clean metrics summary
    print("\n" + "-" * 45)
    print("  ML Scorer -- Training Results")
    print("-" * 45)
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
            bar = "#" * int(imp * 40)
            print(f"    {feat:<30} {imp:.4f}  {bar}")

    print("\n" + "-" * 45)
    print(f"  Model saved -> {metrics['model_path']}")
    print("-" * 45 + "\n")


if __name__ == "__main__":
    main()
