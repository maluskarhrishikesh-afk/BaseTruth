"""train_ml_scorer_pdf.py — Train and save the XGBoost PDF fraud classifier.

Usage:
    python scripts/train_ml_scorer_pdf.py

What this script does:
  1. Loads data/pdf_ultimate_dataset_10000_rows.csv  (expert backbone, 10k rows)
     - 18 features, perfectly balanced 5000/5000 label split
     - Includes hard_subtle_case column for difficult-case evaluation
  2. Loads data/training_data_pdf.csv  (our 66-row real-PDF supplement)
     - 33 original (label=0) + 33 synthetically tampered (label=1)
     - Column names are automatically remapped to the expert schema
  3. Passes both to ml_scorer_pdf.train_pdf() which:
       - Remaps our CSV columns to the expert 18-feature schema
       - Concatenates: ~10,066 total rows
       - Runs 5-fold stratified CV (XGBoost: n_estimators=400, max_depth=6)
       - Evaluates against hard_subtle_case rows in the expert CSV
       - Saves data/ml_scorer_pdf.pkl ONLY if ROC AUC >= 0.80
  4. Prints a metrics summary with feature importances

The hyperparameters (n_estimators=400, max_depth=6, learning_rate=0.05,
subsample=0.9, colsample_bytree=0.9) exactly match the expert notebook
(data/pdf_xgboost_shap_notebook.ipynb) to reproduce their intended results.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from basetruth.analysis.ml_scorer_pdf import train_pdf  # noqa: E402
from basetruth.logger import get_logger                  # noqa: E402

log = get_logger(__name__)

_EXPERT_CSV = str(_REPO_ROOT / "data" / "pdf_ultimate_dataset_10000_rows.csv")
_OWN_CSV    = str(_REPO_ROOT / "data" / "training_data_pdf.csv")
_OUTPUT_PKL = str(_REPO_ROOT / "data" / "ml_scorer_pdf.pkl")


def main() -> None:
    csv_paths = []

    # Primary backbone — expert 10k PDF dataset
    if Path(_EXPERT_CSV).exists():
        csv_paths.append(_EXPERT_CSV)
        print(f"[OK] Expert PDF CSV  : {_EXPERT_CSV}")
    else:
        print(f"[WARN] Expert CSV not found: {_EXPERT_CSV}", file=sys.stderr)

    # Supplementary — our own real + synthetically tampered PDFs
    if Path(_OWN_CSV).exists():
        csv_paths.append(_OWN_CSV)
        print(f"[OK] Own PDF CSV     : {_OWN_CSV}")
    else:
        print(f"[WARN] Own CSV not found: {_OWN_CSV}", file=sys.stderr)

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
