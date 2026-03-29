from basetruth.service import BaseTruthService
from pathlib import Path

svc = BaseTruthService(Path('artifacts/debug'))
for case_id, fraud_type in [('case_053', 'bank_date_range_fabricated'), ('case_054', 'bank_address_mismatch')]:
    r = svc.scan_document(f'data/mortgage_docs/cases/{case_id}/bank_statement.pdf')
    ta = r['tamper_assessment']
    print(f"{case_id} ({fraud_type}): SCORE={ta['truth_score']} RISK={ta['risk_level']}")
    for s in ta['signals']:
        if not s.get('passed', True):
            print(f"  [FAIL] {s.get('name')} | {str(s.get('summary',''))[:70]}")
    print()
