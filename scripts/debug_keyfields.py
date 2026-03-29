"""Check what key fields are extracted for each document in case_050."""
from basetruth.service import BaseTruthService
from pathlib import Path
import json

svc = BaseTruthService(Path('artifacts/debug'))
case_dir = Path('data/mortgage_docs/cases/case_050')

for pdf in sorted(case_dir.glob('*.pdf')):
    r = svc.scan_document(pdf)
    kf = r['structured_summary'].get('key_fields', {})
    # Filter out transactions list for brevity
    flat_kf = {k: v for k, v in kf.items() if k != 'transactions' and not isinstance(v, (dict, list))}
    print(f"\n=== {pdf.name} ===")
    for k, v in flat_kf.items():
        print(f"  {k}: {v}")
