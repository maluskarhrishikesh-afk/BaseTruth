#!/usr/bin/env python3
"""Synthetic mortgage document generator.

Generates simple payslip, bank-statement and gift-letter images and a small
metadata JSON for each sample. Intended for quick model prototyping and
privacy-safe testing.

Usage:
  python scripts/generate_synthetic_documents.py --out data/synthetic --n 50
"""
from __future__ import annotations
import argparse
import json
import os
import random
from pathlib import Path

from faker import Faker
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_DEFAULT = ROOT / "data" / "synthetic"
OUT_DIR_DEFAULT.mkdir(parents=True, exist_ok=True)

fake = Faker()


def render_text_image(lines, size=(800, 500), font=None):
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    y = 20
    x = 20
    for line in lines:
        draw.text((x, y), line, fill="black", font=font)
        y += 20
    return img


def make_payslip(i: int, out_dir: Path) -> dict:
    name = fake.name()
    emp = fake.company()
    salary = random.randint(20000, 250000)
    lines = [f"Payslip #{i}", f"Employee: {name}", f"Employer: {emp}", f"Monthly Salary: {salary}", "\nBreakdown:"]
    for m in range(1, 6):
        lines.append(f"Month {m}: {random.randint(1500, 2500)}")
    img = render_text_image(lines, size=(900, 400))
    fname = out_dir / f"payslip_{i}.png"
    img.save(fname)
    meta = {"type": "payslip", "file": str(fname), "name": name, "employer": emp, "salary": salary}
    return meta


def make_bank_statement(i: int, out_dir: Path) -> dict:
    name = fake.name()
    bank = fake.company() + " Bank"
    lines = [f"Bank Statement #{i}", f"Account Holder: {name}", f"Bank: {bank}", "Transactions:"]
    for _ in range(8):
        amt = random.randint(-5000, 50000)
        lines.append(f"{fake.date_between(start_date='-1y', end_date='today')} : {amt}")
    img = render_text_image(lines, size=(900, 500))
    fname = out_dir / f"bank_{i}.png"
    img.save(fname)
    meta = {"type": "bank_statement", "file": str(fname), "name": name, "bank": bank}
    return meta


def make_gift_letter(i: int, out_dir: Path) -> dict:
    donor = fake.name()
    recipient = fake.name()
    lines = [f"Gift Letter #{i}", f"Donor: {donor}", f"Recipient: {recipient}", "Amount: 50000", "This letter confirms gift..."]
    img = render_text_image(lines, size=(800, 300))
    fname = out_dir / f"gift_{i}.png"
    img.save(fname)
    meta = {"type": "gift_letter", "file": str(fname), "donor": donor, "recipient": recipient}
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT_DIR_DEFAULT))
    p.add_argument("--n", type=int, default=50)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_list = []
    for i in range(1, args.n + 1):
        meta_list.append(make_payslip(i, out))
        meta_list.append(make_bank_statement(i, out))
        if i % 5 == 0:
            meta_list.append(make_gift_letter(i // 5, out))
    meta_path = out / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_list, fh, indent=2)
    print(f"Generated {len(meta_list)} synthetic documents in {out}")


if __name__ == "__main__":
    main()
