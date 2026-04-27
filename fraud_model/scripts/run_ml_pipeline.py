"""run_ml_pipeline.py — Full end-to-end ML training pipeline for BaseTruth.

Philosophy: Train ONLY on real documents.
-----------------------------------------
Synthetic datasets were previously included as a training backbone but have
been intentionally removed.  In fraud detection, models trained on fake data
learn fake patterns — leading to false confidence on real documents.  Every
real document you scan and label is worth far more than thousands of generated
rows.  This pipeline uses only what your forensic engine extracted from actual
files, so the model is always honest about what it has actually seen.

What this does
--------------
  Step 1 — Collect features from ORIGINAL images         (label = 0)
  Step 2 — Collect features from ORIGINAL DERIVED images (label = 1)
  Step 3 — Collect features from TAMPERED images         (label = 2)
  Step 4 — Collect features from TAMPERED DERIVED images (label = 3)
  Step 5 — Collect forensic features from ORIGINAL PDFs  (label = 0)
  Step 6 — Collect forensic features from TAMPERED PDFs  (label = 1)
  Step 7 — Train XGBoost Image Fraud Classifier  → fraud_model/models/ml_scorer_image.pkl
  Step 8 — Train XGBoost PDF Fraud Classifier    → fraud_model/models/ml_scorer_pdf.pkl

Input folders (relative to repo root):
  fraud_model/sample/original_images         — genuine docs, directly phone-clicked
  fraud_model/sample/original_derived_images — save-as copies of originals (still genuine,
                                               but carry re-compression artifacts; teaching
                                               the model that re-compression ≠ tampering)
  fraud_model/sample/tampered_images         — directly manipulated/forged documents
  fraud_model/sample/tampered_derived_images — save-as copies of tampered docs (forensic
                                               "laundering" — ELA/clone signals are softer;
                                               teaching the model to catch laundered fraud)
  fraud_model/sample/original_pdfs           — genuine PDFs
  fraud_model/sample/tampered_pdfs           — tampered PDFs

Why 4 image folders instead of 2?
  When a fraudster saves-as a tampered image, JPEG re-compression washes out ELA
  and clone signals, making it look "cleaner".  If we only train on raw tampered
  images, this laundering trick will fool the model.  Derived folders close that gap.
  Likewise, derived originals prevent the model from mis-flagging everyday re-saves
  of genuine documents as suspicious due to their higher dct_comb_ratio.

Growing the dataset:
  Place more real images in the sample folders above, then re-run
  train_model.exe.  The pipeline picks them up automatically.

Output models:
  fraud_model/models/ml_scorer_image.pkl
  fraud_model/models/ml_scorer_pdf.pkl
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Force UTF-8 output so Unicode box-drawing characters (─, =, etc.) don't
# crash on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Make sure src/ is importable whether we are run directly or via the launcher ──
# fraud_model/scripts/ → fraud_model/ → repo root (3 levels)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ── Sample folder paths — all documents live inside fraud_model/sample/ ───────
# Phone-fresh originals: raw unmodified documents as captured by camera.
_ORIG_IMG_DIR         = _REPO_ROOT / "fraud_model" / "sample" / "original_images"
# Save-as copies of originals: still genuine but carry extra JPEG re-compression
# artifacts.  Including these stops the model from treating re-compression as fraud.
_ORIG_DERIVED_IMG_DIR = _REPO_ROOT / "fraud_model" / "sample" / "original_derived_images"
# Directly manipulated/forged documents: strongest forensic signals.
_TAMP_IMG_DIR         = _REPO_ROOT / "fraud_model" / "sample" / "tampered_images"
# Save-as copies of tampered docs: the ELA/clone signals are softer ("laundered")
# because re-compression partially masks the edit.  Training on these prevents
# the model from missing fraud that has been through a save-as step.
_TAMP_DERIVED_IMG_DIR = _REPO_ROOT / "fraud_model" / "sample" / "tampered_derived_images"
_ORIG_PDF_DIR         = _REPO_ROOT / "fraud_model" / "sample" / "original_pdfs"
_TAMP_PDF_DIR         = _REPO_ROOT / "fraud_model" / "sample" / "tampered_pdfs"

# ── Output CSV paths — training data lives in fraud_model/data/ ───────────────
_IMG_CSV  = str(_REPO_ROOT / "fraud_model" / "data" / "training_data_image.csv")
_PDF_CSV  = str(_REPO_ROOT / "fraud_model" / "data" / "training_data_pdf.csv")

# ── Output model paths — trained models live in fraud_model/models/ ───────────
_IMG_PKL  = str(_REPO_ROOT / "fraud_model" / "models" / "ml_scorer_image.pkl")
_PDF_PKL  = str(_REPO_ROOT / "fraud_model" / "models" / "ml_scorer_pdf.pkl")


# ── Pretty printing helpers ────────────────────────────────────────────────────

def _banner(title: str) -> None:
    """Print a wide banner line to make pipeline steps easy to spot in the console."""
    width = 64
    print("\n" + "=" * width, flush=True)
    print(f"  {title}", flush=True)
    print("=" * width, flush=True)


def _step(n: int, total: int, msg: str) -> None:
    """Print a numbered step header."""
    print(f"\n[Step {n}/{total}]  {msg}", flush=True)
    print("-" * 48, flush=True)


def _ok(msg: str) -> None:
    print(f"  ✔  {msg}", flush=True)


def _warn(msg: str) -> None:
    print(f"  ⚠  {msg}", flush=True)


def _fail(msg: str) -> None:
    print(f"  ✘  {msg}", flush=True)


# ── Step helpers ──────────────────────────────────────────────────────────────

def _collect_images(folder: Path, label: int, output_csv: str, append: bool) -> bool:
    """Run image forensics on every file in *folder* and write rows to *output_csv*.

    Returns True on success, False if the folder is missing or empty.
    The function is a thin wrapper around collect_samples() from
    collect_training_samples.py — imported directly to avoid a subprocess hop.
    """
    if not folder.is_dir():
        _warn(f"Folder not found — skipping: {folder}")
        return False

    # Count supported image files before starting so the user sees a number
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    files = [f for f in folder.iterdir() if f.suffix.lower() in image_exts]
    if not files:
        _warn(f"No image files found in {folder} — skipping")
        return False

    print(f"  Folder  : {folder}", flush=True)
    print(f"  Images  : {len(files)} found", flush=True)
    print(f"  Label   : {label}  ({'original/clean' if label == 0 else 'tampered/forged'})", flush=True)
    print(f"  Output  : {output_csv}  ({'append' if append else 'create new'})", flush=True)

    # Import the collection function from the sibling script
    _scripts_dir = Path(__file__).resolve().parent
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))

    from collect_training_samples import collect_samples  # noqa: PLC0415
    collect_samples(str(folder), label, output_csv, append)
    return True


def _collect_pdfs(folder: Path, label: int, output_csv: str, append: bool) -> bool:
    """Run PDF forensics on every .pdf in *folder* and write rows to *output_csv*.

    Returns True on success, False if the folder is missing or empty.
    """
    if not folder.is_dir():
        _warn(f"Folder not found — skipping: {folder}")
        return False

    files = [f for f in folder.iterdir() if f.suffix.lower() == ".pdf"]
    if not files:
        _warn(f"No PDF files found in {folder} — skipping")
        return False

    print(f"  Folder  : {folder}", flush=True)
    print(f"  PDFs    : {len(files)} found", flush=True)
    print(f"  Label   : {label}  ({'original/clean' if label == 0 else 'tampered/forged'})", flush=True)
    print(f"  Output  : {output_csv}  ({'append' if append else 'create new'})", flush=True)

    _scripts_dir = Path(__file__).resolve().parent
    if str(_scripts_dir) not in sys.path:
        sys.path.insert(0, str(_scripts_dir))

    from collect_training_samples_pdf import collect_samples as collect_pdf_samples  # noqa: PLC0415
    collect_pdf_samples(str(folder), label, output_csv, append)
    return True


def _train_image_model() -> bool:
    """Load the real-image CSV collected in Steps 1-2 and train ml_scorer_image.pkl.

    Only real documents are used — no synthetic data.
    Returns True if the model was saved successfully, False on any failure.
    """
    from basetruth.analysis.ml_scorer import train  # noqa: PLC0415

    csv_paths: list[str] = []

    # Real images only — collected from tests/sample/ in Steps 1 and 2 above
    if Path(_IMG_CSV).exists():
        csv_paths.append(_IMG_CSV)
        _ok(f"Real image CSV    : {Path(_IMG_CSV).name}")
    else:
        _warn(f"Real image CSV not found — {_IMG_CSV}")

    if not csv_paths:
        _fail("No training CSVs available — aborting image model training")
        return False

    print("\n  Training XGBoost image scorer …", flush=True)
    try:
        metrics = train(csv_paths, _IMG_PKL)
    except ValueError as exc:
        # train() raises ValueError when ROC AUC < 0.80
        _fail(f"Training rejected: {exc}")
        return False

    # Print results summary
    print("\n  ─────────────────────────────────────────────", flush=True)
    print("  Image Scorer — Training Results", flush=True)
    print("  ─────────────────────────────────────────────", flush=True)
    print(f"  Rows trained  : {metrics['rows_trained']:,}", flush=True)
    print(f"  Accuracy (CV) : {metrics['accuracy']:.4f}", flush=True)
    print(f"  F1 Score (CV) : {metrics['f1']:.4f}", flush=True)
    print(f"  ROC AUC  (CV) : {metrics['roc_auc']:.4f}", flush=True)

    if metrics.get("hard_case"):
        hc = metrics["hard_case"]
        print(f"\n  Hard Cases    : {hc['n_hard']} rows", flush=True)
        print(f"  HC Accuracy   : {hc['accuracy']:.4f}", flush=True)
        print(f"  HC F1 Score   : {hc['f1']:.4f}", flush=True)

    if metrics.get("feature_importances"):
        print("\n  Feature Importances (top 5):", flush=True)
        sorted_imp = sorted(metrics["feature_importances"].items(), key=lambda x: -x[1])
        for feat, imp in sorted_imp[:5]:
            bar = "█" * int(imp * 40)
            print(f"    {feat:<30} {imp:.4f}  {bar}", flush=True)

    _ok(f"Model saved → {_IMG_PKL}")
    return True


def _train_pdf_model() -> bool:
    """Load the real-PDF CSV collected in Steps 3-4 and train ml_scorer_pdf.pkl.

    Only real documents are used — no synthetic data.
    Returns True if the model was saved successfully, False on any failure.
    """
    from basetruth.analysis.ml_scorer_pdf import train_pdf  # noqa: PLC0415

    csv_paths: list[str] = []

    # Real PDFs only — collected from tests/sample/ in Steps 3 and 4 above
    if Path(_PDF_CSV).exists():
        csv_paths.append(_PDF_CSV)
        _ok(f"Real PDF CSV      : {Path(_PDF_CSV).name}")
    else:
        _warn(f"Real PDF CSV not found — {_PDF_CSV}")

    if not csv_paths:
        _fail("No training CSVs available — aborting PDF model training")
        return False

    print("\n  Training XGBoost PDF scorer …", flush=True)
    try:
        metrics = train_pdf(csv_paths, _PDF_PKL)
    except ValueError as exc:
        _fail(f"Training rejected: {exc}")
        return False

    # Print results summary
    print("\n  ─────────────────────────────────────────────", flush=True)
    print("  PDF Scorer — Training Results", flush=True)
    print("  ─────────────────────────────────────────────", flush=True)
    print(f"  Rows trained  : {metrics['rows_trained']:,}", flush=True)
    print(f"  Accuracy (CV) : {metrics['accuracy']:.4f}", flush=True)
    print(f"  F1 Score (CV) : {metrics['f1']:.4f}", flush=True)
    print(f"  ROC AUC  (CV) : {metrics['roc_auc']:.4f}", flush=True)

    if metrics.get("hard_case"):
        hc = metrics["hard_case"]
        print(f"\n  Hard Cases    : {hc['n_hard']} rows", flush=True)
        print(f"  HC Accuracy   : {hc['accuracy']:.4f}", flush=True)
        print(f"  HC F1 Score   : {hc['f1']:.4f}", flush=True)

    if metrics.get("feature_importances"):
        print("\n  Feature Importances (top 5):", flush=True)
        sorted_imp = sorted(metrics["feature_importances"].items(), key=lambda x: -x[1])
        for feat, imp in sorted_imp[:5]:
            bar = "█" * int(imp * 40)
            print(f"    {feat:<30} {imp:.4f}  {bar}", flush=True)

    _ok(f"Model saved → {_PDF_PKL}")
    return True


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> int:
    """Run all 8 pipeline steps in sequence.

    Returns 0 on full success, 1 if any step failed (so the launcher can
    show the right exit message to the user).
    """
    start_ts = time.time()
    TOTAL_STEPS = 8

    _banner("BaseTruth — ML Training Pipeline")
    print(f"  Repo root : {_REPO_ROOT}", flush=True)
    print(f"  Started   : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    failures: list[str] = []

    # ── Step 1: Original images ────────────────────────────────────────────────
    # Phone-fresh genuine documents — the forensic baseline.
    # append=False means this step CREATES a fresh CSV (wipes any previous run).
    _step(1, TOTAL_STEPS, "Collecting ORIGINAL image features  (label = 0)")
    ok = _collect_images(_ORIG_IMG_DIR, label=0, output_csv=_IMG_CSV, append=False)
    if not ok:
        failures.append("Step 1 — original images")

    # ── Step 2: Original derived images ───────────────────────────────────────
    # Save-as copies of genuine docs — label=1 (ORIGINAL-DERIVED).
    # A second JPEG compression cycle raises dct_comb_ratio and compression_mismatch;
    # these features teach the 4-class model to separate re-saves from raw originals
    # so that a legitimate save-as is never misclassified as tampered.
    _step(2, TOTAL_STEPS, "Collecting ORIGINAL DERIVED image features  (label = 1)")
    ok = _collect_images(_ORIG_DERIVED_IMG_DIR, label=1, output_csv=_IMG_CSV, append=True)
    if not ok:
        failures.append("Step 2 — original derived images")

    # ── Step 3: Tampered images ────────────────────────────────────────────────
    # Directly manipulated/forged documents — label=2 (TAMPERED).
    # Strong ELA, clone, and font signals present here.
    _step(3, TOTAL_STEPS, "Collecting TAMPERED image features  (label = 2)")
    ok = _collect_images(_TAMP_IMG_DIR, label=2, output_csv=_IMG_CSV, append=True)
    if not ok:
        failures.append("Step 3 — tampered images")

    # ── Step 4: Tampered derived images ───────────────────────────────────────
    # Save-as copies of tampered docs — label=3 (TAMPERED-DERIVED).
    # ELA and clone signals are softer because re-compression partially masks edits.
    # Training on these teaches the model to flag laundered fraud that has been
    # through a save-as step after tampering.
    _step(4, TOTAL_STEPS, "Collecting TAMPERED DERIVED image features  (label = 3)")
    ok = _collect_images(_TAMP_DERIVED_IMG_DIR, label=3, output_csv=_IMG_CSV, append=True)
    if not ok:
        failures.append("Step 4 — tampered derived images")

    # ── Step 5: Original PDFs ─────────────────────────────────────────────────
    _step(5, TOTAL_STEPS, "Collecting ORIGINAL PDF features    (label = 0)")
    ok = _collect_pdfs(_ORIG_PDF_DIR, label=0, output_csv=_PDF_CSV, append=False)
    if not ok:
        failures.append("Step 5 — original PDFs")

    # ── Step 6: Tampered PDFs ─────────────────────────────────────────────────
    _step(6, TOTAL_STEPS, "Collecting TAMPERED PDF features    (label = 1)")
    ok = _collect_pdfs(_TAMP_PDF_DIR, label=1, output_csv=_PDF_CSV, append=True)
    if not ok:
        failures.append("Step 6 — tampered PDFs")

    # ── Step 7: Train image model ──────────────────────────────────────────────
    _step(7, TOTAL_STEPS, "Training XGBoost Image Fraud Classifier")
    ok = _train_image_model()
    if not ok:
        failures.append("Step 7 — image model training")

    # ── Step 8: Train PDF model ────────────────────────────────────────────────
    _step(8, TOTAL_STEPS, "Training XGBoost PDF Fraud Classifier")
    ok = _train_pdf_model()
    if not ok:
        failures.append("Step 8 — PDF model training")

    # ── Final summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_ts
    _banner("Pipeline Complete")
    print(f"  Finished  : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  Elapsed   : {elapsed / 60:.1f} minutes", flush=True)

    if failures:
        print("\n  The following steps had warnings or were skipped:", flush=True)
        for f in failures:
            _warn(f)
        print("\n  Models that did train successfully are ready to use.", flush=True)
        return 1

    print("\n  All steps completed successfully.", flush=True)
    print(f"\n  Image model : {_IMG_PKL}", flush=True)
    print(f"  PDF model   : {_PDF_PKL}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
