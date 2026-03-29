from basetruth.service import BaseTruthService
from pathlib import Path
from datetime import datetime

svc = BaseTruthService(Path('artifacts/debug'))
r = svc.scan_document('data/mortgage_docs/cases/case_050/bank_statement.pdf')
kf = r['structured_summary'].get('key_fields', {})
print('opening_balance:', kf.get('opening_balance'))
print('closing_balance:', kf.get('closing_balance'))
print('total_credits:', kf.get('total_credits'))
print('total_debits:', kf.get('total_debits'))

def parse_dt(s):
    for fmt in ('%d-%m-%Y','%d-%b-%Y','%d/%m/%Y','%d %b %Y','%d %B %Y'):
        try:
            return datetime.strptime(s.strip(), fmt)
        except Exception:
            pass
    return datetime.min

txns = kf.get('transactions', [])
s = sorted(txns, key=lambda t: parse_dt(str(t.get('date', ''))))

prev = None
errors = 0
for i, t in enumerate(s):
    bal = t.get('balance')
    dr = t.get('debit') or 0
    cr = t.get('credit') or 0
    if prev is not None and bal is not None:
        expected = prev + int(cr) - int(dr)
        ok = abs(expected - int(bal)) <= max(500, int(abs(expected) * 0.01))
        if not ok:
            errors += 1
            print(f'Row {i+1}: {t["date"]} {str(t.get("desc",""))[:30]} dr={dr} cr={cr} bal={bal} expected={expected} diff={int(bal)-expected}')
    if bal is not None:
        prev = int(bal)

print(f'Total errors: {errors} out of {len(s)} transactions')
