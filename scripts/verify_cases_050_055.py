"""Verify detection results for cases 050-055 after fixes."""
from basetruth.service import BaseTruthService
from pathlib import Path
import json

svc = BaseTruthService(Path('artifacts/verification_test'))

EXPECTED = {
    'case_050': {'tampered': False, 'tamper_types': []},
    'case_051': {'tampered': True,  'tamper_types': ['ifsc_account_mismatch']},
    'case_052': {'tampered': True,  'tamper_types': ['salary_cross_doc_mismatch']},
    'case_053': {'tampered': True,  'tamper_types': ['bank_date_range_fabricated']},
    'case_054': {'tampered': True,  'tamper_types': ['bank_address_mismatch']},
    'case_055': {'tampered': True,  'tamper_types': ['bank_arithmetic_error']},
}

for case_id in sorted(EXPECTED.keys()):
    case_dir = Path(f'data/mortgage_docs/cases/{case_id}')
    pdfs = sorted(case_dir.glob('*.pdf'))
    reports = [svc.scan_document(p) for p in pdfs]
    
    reconciliation = svc.reconcile_income_documents(reports)
    
    print(f"\n=== {case_id} (expected tampered={EXPECTED[case_id]['tampered']}, "
          f"type={EXPECTED[case_id]['tamper_types']}) ===")
    
    for r in reports:
        fname = r['source']['name']
        score = r['tamper_assessment']['truth_score']
        risk = r['tamper_assessment']['risk_level']
        icon = '🚨' if risk == 'high' else '⚠️' if risk == 'medium' else '✅'
        print(f"  {icon} {fname:<35} Score:{score:>3}/100  Risk:{risk}")
        for s in r['tamper_assessment']['signals']:
            if not s.get('passed', True):
                print(f"     [FAIL] {s.get('name',''):<40} {str(s.get('summary',''))[:50]}")
    
    # Show reconciliation results
    anoms = reconciliation.get('anomalies', [])
    if anoms:
        print(f"  INCOME: {len(anoms)} inconsistency(ies)")
        for a in anoms:
            print(f"    - {a.get('type','')}")
    else:
        print("  INCOME: Consistent")
