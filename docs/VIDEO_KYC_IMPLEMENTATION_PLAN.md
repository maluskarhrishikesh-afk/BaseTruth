# Video KYC Rework Plan

## Goal

Bring the Video KYC flow closer to the Identity Verification experience, while also changing the `video_kyc_checks` storage contract toward the structure described in `docs/new_features.txt`.

The target user experience is:

1. Upload Aadhaar front, PAN front, and address proof.
2. Extract document details immediately and show them in Session Status.
3. Capture the live selfie during the liveness flow.
4. Show current location, resolved current address, extracted proof address, and distance.
5. Let the operator run the final face match from the Session Status tab.
6. Save one current Video KYC record per entity.

## Short Answer

Yes, the direction is correct.

But there is one design detail that must be decided before implementation:

- `docs/new_features.txt` changes `video_kyc_checks` to use `aadhar_dtls`, `pan_dtls`, `aadhaar_pic`, `pan_pic`, and `signature_pic`, but it no longer includes a dedicated `address_proof_pic` column.
- The new UI requirement still says we must capture and show the uploaded address-proof image.
- So we need to choose one of these approaches before coding:
  - keep `address_proof_pic` in the table even if the note forgot it, or
  - store the address-proof image key inside `address_dtls` / `report_json` instead of a dedicated column.

My recommendation: keep the address-proof image persisted explicitly somewhere stable and documented. Hiding that key only inside `report_json` will make later reporting and database browsing harder.

## Current Gap

Today the Video KYC flow is still mostly a remote-session monitor:

- Session Setup creates a remote session and optional reference embedding.
- Session Status mostly shows live polling and final pass/fail.
- The save payload is thin compared with Identity Verification.
- `video_kyc_checks` currently stores `identity_dtls` and `reference_doc_pic`, not separate Aadhaar and PAN payloads like the feature note asks.
- The operator does not get the same document-by-document review experience that exists on the Identity Verification screen.

## Proposed Implementation Plan

### Phase 1 - Lock the storage contract

Decide and document the final `video_kyc_checks` schema first.

Planned table shape:

- keep existing core result fields: `status`, `cosine_similarity`, `display_score`, `threshold`, `is_match`, `liveness_state`, `liveness_passed`, `address_dtls`, `isAddressMatch`, `kyc_comments`, `current_location_json`, `current_address_text`, `address_distance_meters`, `video_kyc_pic`, `challenge_snapshots_json`, `report_json`, `pdf_report`, timestamps
- replace generic identity fields with identity-style fields:
  - `aadhar_dtls`
  - `pan_dtls`
  - `aadhaar_pic`
  - `pan_pic`
  - `signature_pic`
- decide where the address-proof image key lives:
  - preferred: dedicated `address_proof_pic`
  - fallback: inside `address_dtls`

Work in this phase:

- update SQLAlchemy model in `src/basetruth/db.py`
- update `init_db()` migration logic so old databases are upgraded safely
- update `docs/DATABASE.md`
- update Database Viewer metadata so the new columns display correctly

### Phase 2 - Rework Video KYC page structure

Make the Video KYC page behave more like Identity Verification, but still keep the remote-session capability.

Suggested tab structure:

1. Session Setup
2. Session Status

Inside Session Status, show sections similar to Identity Verification:

- Applicant Details
- Aadhaar Details
- PAN Details
- Address Proof Details
- Live Selfie / Liveness Result
- Current Location and Address Match
- Face Match Result
- Save Result

Key change:

- the Session Status tab should become the main operator review surface, not only a polling dashboard

### Phase 3 - Add document extraction to Video KYC state

When the operator uploads documents, extract data immediately and keep it in session state so the Session Status tab can show it.

Required inputs:

- Aadhaar front
- PAN front
- Address proof image

Required extraction behavior:

- Aadhaar front:
  - run Aadhaar QR extraction using the same logic as Identity Verification
  - show extracted fields in Session Status
- PAN front:
  - run PAN extraction and signature crop using the same combined pipeline as Identity Verification
  - show PAN fields and signature preview in Session Status
- Address proof:
  - extract address text from Aadhaar back or passport address page
  - normalize the extracted address for comparison
  - show raw and cleaned address details in Session Status

Implementation note:

- reuse Identity Verification helpers where possible instead of duplicating extraction logic in Video KYC

### Phase 4 - Add applicant auto-fill and entity resolution

Add an Applicant Details form similar to Identity Verification.

Expected behavior:

- prefill applicant name from Aadhaar first, then PAN if Aadhaar name is missing
- prefill PAN number from PAN extraction
- prefill Aadhaar number from QR data when available
- keep email and phone editable by the operator
- allow linking to an existing entity

Entity save behavior:

- continue using the existing entity find-or-create helper logic
- match by PAN first
- then Aadhaar
- then case-insensitive first name + last name
- update the existing entity row when a match is found
- create a new row only when nothing matches

This is effectively the entity UPSERT behavior the request asks for. The main job is to make the Video KYC page always route through that path cleanly.

### Phase 5 - Capture live selfie and challenge evidence properly

During liveness, keep the best live selfie and one best frame per completed challenge.

Expected behavior:

- `video_kyc_pic` stores the best live selfie captured during the liveness session
- `challenge_snapshots_json` stores one best retained frame per completed challenge
- Session Status shows:
  - live selfie preview
  - liveness result
  - challenge progress

Also make sure the final status payload returned by the API includes enough data for the UI to render the complete review state without guessing.

### Phase 6 - Add address proof and location comparison review

Session Status should show both document-side and live-location-side evidence.

Display requirements:

- uploaded address-proof image preview
- extracted address text or structured address fields
- current GPS coordinates
- reverse-geocoded current address
- final address match result
- distance in meters
- clear comment when comparison is inconclusive or geocoding fails

Rules to preserve:

- address comparison must still degrade gracefully when OCR or geocoding is unavailable
- save path must still show a visible error if persistence fails

### Phase 7 - Move face match to the final operator step

The user wants a dedicated face-match action similar to Identity Verification.

Target behavior:

- document extraction can happen earlier
- liveness captures the live selfie and challenge evidence first
- Session Status shows a `Match Face` action after enough evidence exists
- final save only happens after the match result is available

This likely needs a small backend contract change:

- either the API returns enough material to run face match from Streamlit, or
- the API exposes a dedicated endpoint that performs the final match after the live session is complete

The cleaner option is a dedicated backend step so Streamlit does not have to reimplement session result assembly.

### Phase 8 - Expand persistence layer and reports

Update `save_video_kyc_check()` so it saves the richer payload.

It should persist:

- Aadhaar payload into `aadhar_dtls`
- PAN payload into `pan_dtls`
- PAN signature image key into `signature_pic`
- Aadhaar image key into `aadhaar_pic`
- PAN image key into `pan_pic`
- address proof payload into `address_dtls`
- address proof image key into the chosen final field
- live selfie key into `video_kyc_pic`
- challenge snapshots into `challenge_snapshots_json`
- current location and distance fields
- final PDF report key

It should also keep the one-row-per-entity upsert behavior for `video_kyc_checks`.

### Phase 9 - Update downstream readers

Any screen or report that reads Video KYC data must be updated to the new field names.

Likely touch points:

- Database Viewer
- Reports screen
- final report builder / document intelligence readers
- PDF rendering for Video KYC
- any DB query prompt docs that still describe the older schema

### Phase 10 - Add tests before closing the change

Minimum test coverage for the later implementation:

- schema migration test for old `video_kyc_checks` rows
- store-layer upsert test for one current row per entity
- store-layer test that rich document payloads are written to the correct columns
- save failure test that confirms the UI shows a visible error
- session/status payload test for document details and live selfie metadata
- address comparison fallback test when OCR or geocoding is missing

Run order when implementation starts:

1. narrow tests for touched KYC/store modules
2. full `python -m pytest tests/ -q --tb=short`

## File Areas Likely To Change Later

- `src/basetruth/ui/pages/video_kyc.py`
- `src/basetruth/ui/pages/identity.py` (shared helper extraction reuse only if needed)
- `src/basetruth/api.py`
- `src/basetruth/kyc/session.py`
- `src/basetruth/db.py`
- `src/basetruth/store.py`
- `src/basetruth/reporting/pdf.py`
- `src/basetruth/reporting/final_report_builder.py`
- `src/basetruth/ui/pages/database.py`
- `docs/DATABASE.md`
- `docs/FUNCTIONALITY.md`
- `docs/IDENTITY_VERIFICATION.md`
- tests around KYC session, persistence, and schema migration

## Recommended Implementation Order

1. Finalize schema decision for address-proof image storage.
2. Update DB model, migrations, and docs.
3. Expand session state and API status payloads for Aadhaar, PAN, address proof, and selfie evidence.
4. Rework Session Status UI to mirror the Identity Verification review pattern.
5. Add final face-match action.
6. Update save path and PDF generation.
7. Update Database Viewer and downstream report readers.
8. Add and run tests.

## Main Risks

- schema drift between `docs/new_features.txt`, `docs/DATABASE.md`, ORM model, and Database Viewer
- storing the address-proof image inconsistently if its final column is not decided up front
- duplicating extraction logic between Identity Verification and Video KYC instead of sharing helpers
- saving partial KYC rows without enough evidence if the Session Status flow is not gated carefully
- breaking existing report readers that still expect `identity_dtls` / `reference_doc_pic`

## Practical Recommendation

Implement this as a controlled rework, not as a quick patch.

The best technical path is:

- reuse Identity Verification extraction helpers
- keep Video KYC as a separate save table
- make Session Status the operator review screen
- preserve one-row-per-entity upsert behavior
- update docs and tests in the same change when implementation starts