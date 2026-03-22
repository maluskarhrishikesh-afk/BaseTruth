# Tooling

## What LiteParse Handles Well

LiteParse is well suited for:

- PDF parsing with layout preserved
- OCR-backed text extraction
- tables and spatial text recovery
- generation of structured raw JSON that other detectors can consume

## What LiteParse Does Not Fully Cover

BaseTruth needs more than parsing. The following capabilities require additional tools or modules:

---

## Installed and Active

All packages below are installed in the project venv and importable.

### PDF Extraction and Metadata

| Package | Version | Purpose | pip extra |
|---|---|---|---|
| `pymupdf` (fitz) | 1.27.2 | Best-quality text extraction, page rendering, stream inspection | `pdf` / `ocr` |
| `pypdf` | 6.9.1 | PDF metadata, form fields, cross-reference table, annotations | `pdf` |
| `pdfplumber` | 0.11.9 | Structured table extraction from PDFs — excellent for payslips | `forensics` |
| `pikepdf` | 10.5.1 | qpdf Python wrapper — PDF encryption analysis, signature objects, XRef repair | `forensics` |
| `pypdfium2` | 5.6.0 | Fast PDF rendering (pulled in by ocrmypdf) | — |

### Image Analysis and Hashing

| Package | Version | Purpose | pip extra |
|---|---|---|---|
| `pillow` | 12.1.1 | Image preprocessing, region cropping, format conversion | `ocr` |
| `opencv-python` | 4.13.0 | Image manipulation detection, template alignment, copy-paste region analysis | `forensics` |
| `imagehash` | 4.3.2 | Perceptual hashing — detect near-duplicate or tampered document images | `forensics` |
| `numpy` | 2.4.3 | Array operations underlying opencv and scipy | — |

### OCR

| Package | Version | Purpose | pip extra |
|---|---|---|---|
| `pytesseract` | 0.3.13 | Tesseract OCR for image-only PDFs (Aadhaar, PAN, scanned docs) | `ocr` |
| `ocrmypdf` | 17.4.0 | Scan normalisation — adds OCR text layer to scanned PDFs | `ocr` |
| `pdf2image` | 1.17.0 | PDF-to-image conversion using Poppler | `ocr` |

### Metadata and EXIF

| Package | Version | Purpose | pip extra |
|---|---|---|---|
| `exifread` | 3.5.1 | Pure Python EXIF metadata from images — no binary required | `forensics` |
| `pyexiftool` | 0.5.6 | Rich metadata via ExifTool binary — flags Photoshop/GIMP edit history | `forensics` |

### Cryptography and Signatures

| Package | Version | Purpose | pip extra |
|---|---|---|---|
| `cryptography` | 46.0.5 | PKCS#7 / CMS signature chain parsing, certificate inspection | `forensics` |

### Statistics and ML

| Package | Version | Purpose | pip extra |
|---|---|---|---|
| `scipy` | 1.17.1 | Statistical anomaly detection — field value outlier scoring | `forensics` |
| `scikit-learn` | 1.8.0 | ML fraud pattern detection — isolation forest, clustering | `ml` |
| `faiss-cpu` | 1.13.2 | Vector similarity search — compare documents against known fraud templates | `ml` |

---

## Requires External Binary (manual install)

These Python packages are installed but **need a system binary** to function.  
Install the binary separately and ensure it is on `PATH`.

### Tesseract OCR
- Used by: `pytesseract`, `ocrmypdf`
- Windows installer: https://github.com/UB-Mannheim/tesseract/wiki
- After install, set `pytesseract.pytesseract.tesseract_cmd` or add to PATH

### Poppler
- Used by: `pdf2image`
- Windows binaries: https://github.com/oschwartz10612/poppler-windows/releases
- Add `poppler/Library/bin` to PATH

### ExifTool (Phil Harvey)
- Used by: `pyexiftool`
- Download: https://exiftool.org/
- Place `exiftool.exe` (Windows) on PATH

### Ghostscript
- Used by: `ocrmypdf` (for PDF/A compliance and scan normalisation)
- Download: https://www.ghostscript.com/download.html

### qpdf binary (optional)
- Used standalone for signature extraction workflows
- Download: https://github.com/qpdf/qpdf/releases
- `pikepdf` embeds libqpdf and does not require the binary

---

## Deferred / Not Installed

| Package | Reason | Notes |
|---|---|---|
| `sentence-transformers` | Requires PyTorch (~2–4 GB) | Install manually when semantic document similarity is needed: `pip install sentence-transformers` |
| `torch` | Very large (~2–4 GB) | Required by sentence-transformers; GPU variant recommended |
| `pdfsig` | Part of poppler-utils CLI | Use via subprocess if installed with Poppler |

---

## Installation Reference

```bash
# Core PDF extraction
pip install "basetruth[pdf]"

# OCR pipeline
pip install "basetruth[ocr]"

# Full forensics toolchain (image manipulation, EXIF, signatures, stats)
pip install "basetruth[forensics]"

# ML fraud detection and vector search
pip install "basetruth[ml]"

# Everything at once
pip install "basetruth[pdf,ocr,forensics,ml]"
```

---

## Connector Tooling

| Package | Purpose | pip extra |
|---|---|---|
| `boto3` | S3 datasource sync | `connectors` |
| `google-api-python-client` / `google-auth` | Google Drive datasource sync | `connectors` |
| `requests` | SharePoint / Microsoft Graph datasource sync | `connectors` |

---

## Product-Specific Detector Families

### Banking and payments

- IBAN and account number checksum validation
- merchant statement reconciliation
- invoice and receipt duplicate detection

### Insurance

- claim chronology validation
- signature and stamp region checks (→ `opencv-python`, `imagehash`)
- cross-document claimant identity reconciliation

### Healthcare

- hospital name and provider identifier verification
- inconsistent terminology detection
- treatment date and billing date reconciliation

### Employers and payroll

- payslip and offer-letter template drift analysis (→ `pdfplumber`, `imagehash`)
- compensation anomaly detection (→ `scipy`, `scikit-learn`)
- employee identifier reconciliation across months

