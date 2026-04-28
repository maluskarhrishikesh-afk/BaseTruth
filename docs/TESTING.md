# BaseTruth Testing Guide

> **Purpose:** This document defines how BaseTruth should be tested, what belongs in unit tests vs higher-level tests, and what quality bar every code change must meet.
>
> **Policy:** Every non-trivial code change must update tests and documentation together. If the code changes but the test plan or product behaviour changed too, this file or the linked product docs must be updated in the same change.

---

## 1. Testing Principles

BaseTruth handles fraud scoring, KYC, OCR routing, and persistence. That means tests must optimise for **trust**, not just execution count.

The default testing rules are:

- **Follow the product docs first.** `docs/FUNCTIONALITY.md`, `docs/ARCHITECTURE.md`, and `docs/IDENTITY_VERIFICATION.md` define the expected behaviour. Tests should enforce those rules, not invent new behaviour.
- **Write the cheapest test that can fail for the right reason.** Prefer a focused unit test before a broad integration test.
- **Keep unit tests deterministic.** No live DB, MinIO, Ollama, webcam, network, or wall-clock dependence unless the test is explicitly integration/manual.
- **Test behaviour, not implementation noise.** Assert outcomes, persisted payloads, error messages, and decision rules. Avoid brittle assertions on internal call order unless the order is the behaviour.
- **Cover risky paths, not just happy paths.** Fraud logic, identity checks, persistence upserts, fallback behaviour, and user-visible failures all need negative-path coverage.
- **Delete stale tests when behaviour is removed.** Old tests that protect deleted behaviour are as harmful as stale production code.

---

## 2. BaseTruth Test Pyramid

| Layer | Purpose | Examples in BaseTruth | Notes |
|---|---|---|---|
| **Unit tests** | Validate one function/class decision in isolation | `kyc/liveness.py`, validation packs, name/DOB matching, DB upsert helpers with fake sessions | Fast, deterministic, mandatory for most logic changes |
| **Service/integration tests** | Validate module boundaries with fakes or controlled dependencies | `store.py` save flows, report builders, orchestration layers | Use monkeypatch/fakes before using real services |
| **Manual / live tests** | Validate hardware, websocket, UI, DB, and model integrations | `tests/test_kyc_ws.py`, Streamlit flows, live Ollama extraction, camera capture | Not a replacement for unit tests |

Use this rule of thumb:

- If the logic can be tested with plain Python objects, it should be a unit test.
- If the logic needs SQLAlchemy session behaviour, use a fake or monkeypatched session first.
- If the logic needs a real server, webcam, or model runtime, keep it out of the regular unit suite.

---

## 3. What Must Be Unit Tested

The following areas are high-priority and should always gain or keep unit coverage when touched:

1. **KYC decision logic**
   - `src/basetruth/kyc/liveness.py`
   - `src/basetruth/kyc/session.py`
   - challenge pass/fail rules, expiry handling, and face-match edge cases

2. **Deterministic identity checks**
   - name comparison
   - DOB comparison
   - PAN validation and interpretation

3. **Persistence and upsert logic**
   - `save_identity_check()`
   - `save_scan_to_db()`
   - document extraction upsert rules
   - visible failure handling for DB/MinIO writes

4. **Validation packs and scoring rules**
   - arithmetic checks
   - required-field checks
   - fraud signal generation
   - ML/heuristic fallback behaviour

5. **Routing and fallback logic**
   - OCR/document-type routing
   - structured-vs-image branching
   - offline fallback paths when Ollama, DB, or MinIO are unavailable

---

## 4. Unit Test Writing Standard

Unit tests in BaseTruth should usually follow this shape:

1. **Arrange** the smallest possible input.
2. **Act** by calling one function or one method.
3. **Assert** the exact behaviour the product depends on.

Good unit tests in this repo should:

- Have descriptive names: `test_run_face_match_returns_failure_message_below_threshold`
- Cover one behaviour per test whenever possible
- Use plain objects, `SimpleNamespace`, or tiny fakes instead of large fixtures when enough
- Use `monkeypatch` for external boundaries
- Assert user-visible messages when those messages drive operator workflows
- Prefer `pytest.approx(...)` for float comparisons

Avoid these patterns:

- Calling live PostgreSQL, MinIO, Ollama, or camera APIs from unit tests
- Asserting huge JSON blobs when only 2 fields matter
- Depending on file ordering, wall-clock timing, or randomness without controlling them
- Testing private implementation details that can change without changing behaviour

---

## 5. Coverage Policy

**Coverage matters, but only as a guardrail.** It helps find blind spots. It does **not** prove correctness on its own.

BaseTruth should use coverage like this:

- Treat low or missing coverage in risky modules as a warning sign.
- Expect every changed behaviour to gain or update tests.
- Use coverage reports to find untested branches in fraud logic, fallback paths, and persistence code.
- Do **not** chase a vanity percentage by adding shallow tests that assert imports or trivial getters.

The practical rule is:

- **Meaningful coverage matters.**
- **A raw percentage target is secondary.**

If forced to choose between:

- 5 strong tests covering real decision branches, or
- 20 weak tests that only inflate the number,

choose the 5 strong tests.

---

## 6. Commands

Use the narrowest command first, then the full suite.

### Focused test run

```powershell
python -m pytest tests/test_kyc_liveness.py -q --tb=short
```

### Multiple focused files

```powershell
python -m pytest tests/test_kyc_liveness.py tests/test_kyc_session.py -q --tb=short
```

### Full suite

```powershell
python -m pytest tests/ -q --tb=short
```

### Manual live KYC script

```powershell
python tests/test_kyc_ws.py
```

`tests/test_kyc_ws.py` is intentionally excluded from the regular suite because it requires a live API server and browser/device interaction.

---

## 7. Test Strategy By Change Type

| Change Type | Minimum Expected Testing |
|---|---|
| Pure logic change | Add or update focused unit tests for the touched function/module |
| Store / DB persistence change | Add unit tests with fake sessions and validate success + failure paths |
| UI page behaviour change | Add unit tests for underlying helpers plus manual UI verification notes if needed |
| API route change | Add unit tests for the route helper/service logic; add integration coverage only when routing itself is the risk |
| OCR / model fallback change | Add deterministic tests around routing and fallback decisions; do not depend on live models in unit tests |
| Bug fix | Add a regression test that fails before the fix and passes after it |

---

## 8. Release Checklist For Testing

Before closing a change:

1. Add or update unit tests for the touched behaviour.
2. Run the narrowest relevant pytest command first.
3. Run the full suite: `python -m pytest tests/ -q --tb=short`.
4. Check whether docs changed; if yes, update the relevant docs in the same change.
5. Remove stale tests, stale helpers, or obsolete docs created by the old behaviour.

---

## 9. Current Notes

- Video KYC has two test layers today: deterministic unit coverage for `kyc/liveness.py` and `kyc/session.py`, plus the live manual websocket script in `tests/test_kyc_ws.py`.
- Store-layer tests should continue using fake SQLAlchemy-style sessions instead of live DB dependencies wherever possible.
- If a future change introduces a new decision-heavy module without tests, add unit coverage in the same pull request instead of deferring it.