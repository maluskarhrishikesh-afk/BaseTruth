#!/usr/bin/env python3
"""Download helper for public document corpora useful for document-forensics.

Strategy: use HuggingFace streaming=True to pull only the first --samples rows
from each dataset, saving as JSONL + any embedded images.  This means:
  - DocVQA 1 000 samples  ≈  50–80 MB   ≈  1–2 min
  - DocBank 1 000 samples ≈  20–40 MB   ≈  1–2 min
  - PubLayNet 1 000 samples ≈ 200–400 MB ≈  3–8 min

Full dataset sizes for reference (stream them or plan overnight runs):
  - DocVQA full   : ~4 GB
  - DocBank full  : ~20 GB
  - PubLayNet full: ~10 GB  (PDF tar), ~50 GB expanded

Usage:
  python scripts/download_public_docs.py --datasets all --samples 1000 --token hf_...
  python scripts/download_public_docs.py --datasets docvqa --samples 500
"""
from __future__ import annotations
import argparse
import json
import os
import io
import sys
import traceback
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "datasets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hf_login(token: str | None) -> None:
    if not token:
        token = os.environ.get("HF_TOKEN")
    if token:
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("Logged in to HuggingFace Hub.")
        except Exception as exc:
            print(f"[warn] HF login skipped: {exc}")


def _stream_and_save(
    dataset_id: str,
    split: str,
    dest: Path,
    n_samples: int,
    image_column: str | None = None,
    extra_columns: list[str] | None = None,
    config: str | None = None,
) -> None:
    """Stream up to n_samples rows from a HuggingFace dataset and write them to disk."""
    from datasets import load_dataset

    dest.mkdir(parents=True, exist_ok=True)
    jsonl_path = dest / "samples.jsonl"
    img_dir = dest / "images"

    if jsonl_path.exists():
        print(f"  Already exists ({jsonl_path}), skipping.")
        return

    print(f"  Streaming {n_samples} samples from '{dataset_id}' (split={split}) …")
    load_kwargs: dict = dict(streaming=True)
    if config:
        load_kwargs["name"] = config
    try:
        ds = load_dataset(dataset_id, split=split, **load_kwargs)
    except Exception:
        # fall back without split kwarg (returns DatasetDict; index into it)
        ds_dict = load_dataset(dataset_id, **load_kwargs)
        ds = ds_dict[split] if split in ds_dict else next(iter(ds_dict.values()))

    img_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for i, sample in enumerate(ds):
            if i >= n_samples:
                break
            row: dict = {}
            # Copy scalar / list fields (skip raw PIL images for JSON)
            for k, v in sample.items():
                if k == image_column:
                    continue
                try:
                    json.dumps(v)       # test serializability
                    row[k] = v
                except (TypeError, ValueError):
                    row[k] = str(v)

            # Save image if present
            if image_column and image_column in sample:
                img = sample[image_column]
                img_filename = f"{i:06d}.png"
                img_path = img_dir / img_filename
                try:
                    if hasattr(img, "save"):   # PIL Image
                        img.save(img_path)
                    elif isinstance(img, bytes):
                        img_path.write_bytes(img)
                    row["image_file"] = str(img_path.relative_to(ROOT))
                except Exception as ie:
                    row["image_save_error"] = str(ie)

            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 100 == 0:
                print(f"    … {written} rows written", end="\r", flush=True)

    print(f"\n  Done: {written} rows → {jsonl_path}")


# ---------------------------------------------------------------------------
# Per-dataset functions
# ---------------------------------------------------------------------------

def download_docvqa(dest: Path, n: int) -> None:
    print("\n[DocVQA] ~50–80 MB for 1 000 samples, expected ~1–3 min")
    _stream_and_save(
        dataset_id="lmms-lab/DocVQA",
        split="test",
        dest=dest / "docvqa",
        n_samples=n,
        image_column="image",
        config="DocVQA",
    )


def download_docbank(dest: Path, n: int) -> None:
    """Download form/financial-document datasets relevant to mortgage fraud forensics.

    DocBank and DocLayNet both rely on loading scripts that are no longer supported
    by the current HuggingFace datasets library.  We substitute three Parquet-backed
    datasets that cover the same ground and are directly useful for mortgage documents:

      - FUNSD   : form field detection (addresses, employer names, signatures)
      - CORD-v2 : receipt and financial document layout
      - DeepForm: financial regulatory filings (loan applications, disclosures)
    """
    candidates = [
        dict(dataset_id="nielsr/funsd",           split="train", dest_name="funsd",   image_column="image"),
        dict(dataset_id="naver-clova-ix/cord-v2", split="train", dest_name="cord_v2", image_column="image"),
        dict(dataset_id="jinhybr/OCR-DocVQA-200", split="train", dest_name="ocr_docvqa_small", image_column="image"),
    ]
    for c in candidates:
        print(f"\n[{c['dest_name'].upper()}] ~20–60 MB for {n} samples, expected ~1–3 min")
        try:
            _stream_and_save(
                dataset_id=c["dataset_id"],
                split=c["split"],
                dest=dest / c["dest_name"],
                n_samples=n,
                image_column=c["image_column"],
            )
        except Exception as e:
            print(f"  skipped {c['dest_name']}: {e}")


def download_publaynet(dest: Path, n: int) -> None:
    print("\n[PubLayNet] ~200–400 MB for 1 000 samples, expected ~3–8 min")
    _stream_and_save(
        dataset_id="jordanparker6/publaynet",
        split="train",
        dest=dest / "publaynet",
        n_samples=n,
        image_column="image",
    )


def instruct_rvl_cdip() -> None:
    print("\n[RVL-CDIP] Requires Kaggle API token. Run:")
    print("  pip install kaggle")
    print("  # Place kaggle.json in %USERPROFILE%\\.kaggle\\")
    print("  kaggle datasets download -d patchi/rvl-cdip -p data/datasets/rvl_cdip --unzip")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--datasets", nargs="+",
        choices=["docbank", "publaynet", "docvqa", "rvl-cdip", "all"],
        default=["all"],
    )
    p.add_argument("--samples", type=int, default=1000,
                   help="Number of rows to stream per dataset (default: 1000)")
    p.add_argument("--token", default=None,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    args = p.parse_args()

    _hf_login(args.token)

    choices = set(args.datasets)
    if "all" in choices:
        choices = {"docbank", "publaynet", "docvqa"}

    for ds in sorted(choices):
        try:
            if ds == "docvqa":
                download_docvqa(OUT_DIR, args.samples)
            elif ds == "docbank":
                download_docbank(OUT_DIR, args.samples)
            elif ds == "publaynet":
                download_publaynet(OUT_DIR, args.samples)
            elif ds == "rvl-cdip":
                instruct_rvl_cdip()
        except Exception:
            print(f"[ERROR] {ds} failed:")
            traceback.print_exc()

    print("\nAll requested datasets processed. Files in:", OUT_DIR)


if __name__ == "__main__":
    main()

