"""Check arithmetic errors across several clean cases to see the false positive rate."""
from basetruth.service import BaseTruthService
from pathlib import Path

svc = BaseTruthService(Path('artifacts/debug'))

results = []
for case_num in range(46, 56):  # cases 046-055
    case_id = f'case_{case_num:03d}'
    bs_path = Path(f'data/mortgage_docs/cases/{case_id}/bank_statement.pdf')
    if not bs_path.exists():
        continue
    r = svc.scan_document(bs_path)
    ta = r['tamper_assessment']
    arith = next((s for s in ta['signals'] if 'bank_debit_credit_arithmetic' in s.get('name','')), None)
    if arith:
        errors = len(arith.get('details', {}).get('arithmetic_errors', []))
        total = arith.get('details', {}).get('total_transactions', 0)
        results.append((case_id, ta['truth_score'], ta['risk_level'], errors, total, arith.get('passed', True)))
    else:
        results.append((case_id, ta['truth_score'], ta['risk_level'], 0, 0, True))

print(f"{'Case':<12} {'Score':>5} {'Risk':<8} {'Errors/Total':<15} {'Passed'}")
print("-" * 55)
for case_id, score, risk, errors, total, passed in results:
    print(f"{case_id:<12} {score:>5} {risk:<8} {errors}/{total:<12} {passed}")
