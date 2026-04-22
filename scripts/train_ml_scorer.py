"""train_ml_scorer.py — Train and save the XGBoost forensic fraud classifier.

Usage:
    python scripts/train_ml_scorer.py

What this script does:
  1. Loads data/forensic_training_10000_rows.csv  (expert backbone, 10k rows)
  2. Loads data/training_data_image.csv           (our real-image supplement)
  3. Passes both to ml_scorer.train() which:
       - Remaps raw-engine columns to the normalised expert feature schema
       - Concatenates the two datasets (~10,062 rows)
       - Runs 5-fold stratified cross-validation (XGBoost or RF fallback)
       - Evaluates against hard_case rows in the expert CSV
       - Saves data/ml_scorer_image.pkl ONLY if ROC AUC >= 0.80
  4. Prints the final metrics table

The ROC AUC gate (0.80) prevents saving a model that can't outperform a
naive classifier on the test folds — raise it if you want stricter quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from either the project root or the scripts/ directory
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from basetruth.analysis.ml_scorer import train  # noqa: E402
from basetruth.logger import get_logger  # noqa: E402

log = get_logger(__name__)

_EXPERT_CSV = str(_REPO_ROOT / "data" / "forensic_training_10000_rows.csv")
_OWN_CSV = str(_REPO_ROOT / "data" / "training_data_image.csv")
_OUTPUT_PKL = str(_REPO_ROOT / "data" / "ml_scorer_image.pkl")


def main() -> None:
    csv_paths = []

    # Primary backbone — expert-crafted 10k dataset
    if Path(_EXPERT_CSV).exists():
        csv_paths.append(_EXPERT_CSV)
        print(f"[OK] Expert CSV loaded: {_EXPERT_CSV}")
    else:
        print(f"[WARN] Expert CSV not found, skipping: {_EXPERT_CSV}", file=sys.stderr)

    # Supplementary — our own real-image scans (62+ rows)
    if Path(_OWN_CSV).exists():
        csv_paths.append(_OWN_CSV)
        print(f"[OK] Own CSV loaded: {_OWN_CSV}")
    else:
        print(f"[WARN] Own CSV not found, skipping: {_OWN_CSV}", file=sys.stderr)

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
    print("\n─────────────────────────────────────────────")
    print("  ML Scorer — Training Results")
    print("─────────────────────────────────────────────")
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
            print(f"    {feat:<30} {imp:.4f}  {bar}")

    print("\n─────────────────────────────────────────────")
    print(f"  Model saved → {metrics['model_path']}")
    print("─────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
