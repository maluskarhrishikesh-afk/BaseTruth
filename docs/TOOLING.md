# Tooling

## Running in Docker (recommended for production / platform-agnostic)

The repo ships a fully self-contained Docker image that bundles **all** external
binaries — no manual PATH setup needed on any OS (Linux, macOS, Windows/WSL2).

### Files

| File | Purpose (Simple English) |
|---|---|
| `Dockerfile` | The master recipe that tells Docker exactly how to build the application and install all required tools automatically |
| `docker-compose.yml` | The blueprint that tells Docker to start both the background worker and the web server together |
| `.dockerignore` | A list of files to ignore during the build so the final application size stays small and fast |
| `requirements.txt` | The exact list of Python packages needed so the app works exactly the same way on every computer |

### Binaries installed into the image (no manual action required)

| Binary | Package | Debian package |
|---|---|---|
| `pdftoppm` / `pdfinfo` | `poppler-utils` | `pdf2image` |
| `exiftool` | `libimage-exiftool-perl` | `pyexiftool` |
| `qpdf` | `qpdf` | standalone signature workflows |
| `node` / `npx` | NodeSource 22.x | `@llamaindex/liteparse` |
| `convert` / `magick` | `imagemagick` | `@llamaindex/liteparse` image-PDF conversion |

### Quick start

```bash
# Build (one time — ~5 min first run, cached after)
docker compose build

# Scan a document
docker compose run --rm basetruth-cli \
    scan --input /app/your_data/your_document.pdf

# Compare payslips in a folder
docker compose run --rm basetruth-cli \
    compare-payslips --input-dir /app/your_data/payslips/

# Start the REST API server (http://localhost:8000)
docker compose up basetruth-api
```

Results are written to `./artifacts/` on your host machine via the bind mount.

### Environment variables (docker-compose or docker run -e)

| Variable | Default | Purpose (Simple English) |
|---|---|---|
| `BASETRUTH_ARTIFACT_ROOT` | `/app/artifacts` | The folder where the system saves its final generated reports and files |
| `EXIFTOOL_PATH` | `/usr/bin/exiftool` | Tells the system where the deep photo checker tool is installed |
| `API_PORT` | `8000` | The web address port where you can talk to the application (e.g. localhost:8000) |

---

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

| Package | Version | Purpose (Simple English) | pip extra |
|---|---|---|---|
| `pymupdf` (fitz) | 1.27.2 | Pulls text directly out of PDF files and helps read them | `pdf` / `ocr` |
| `pypdf` | 6.9.1 | Reads hidden properties inside PDFs like author names, titles, and form data | `pdf` |
| `pdfplumber` | 0.11.9 | Great for extracting data that is organised in tables (like rows and columns on a payslip) | `forensics` |
| `pikepdf` | 10.5.1 | Used to inspect digital signatures and check if a PDF has been secretly modified or encrypted | `forensics` |
| `pypdfium2` | 5.6.0 | Helps process and load PDF pages very quickly | — |

### Image Analysis and Hashing

| Package | Version | Purpose (Simple English) | pip extra |
|---|---|---|---|
| `pillow` | 12.1.1 | A tool to edit, crop, and resize images before we analyse them | `ocr` |
| `opencv-python` | 4.13.0 | Advanced image detective: checks if parts of an image were copied, pasted, or photoshopped | `forensics` |
| `imagehash` | 4.3.2 | Creates a unique digital fingerprint for an image to check if two documents look exactly the same | `forensics` |
| `numpy` | 2.4.3 | Helps other tools do heavy mathematical calculations behind the scenes | — |

### OCR (Reading text from images)

| Package | Version | Purpose (Simple English) | pip extra |
|---|---|---|---|
| `paddleocr` | 3.3.0+ | Reads text from scanned PDFs and uploaded document photos using the same OCR engine across the app | `ocr` |
| `paddlepaddle` | 3.2.0+ | Runtime needed by PaddleOCR so the OCR models can run locally | `ocr` |
| `pdf2image` | 1.17.0 | Converts a PDF file into standard image files (like JPGs or PNGs) | `ocr` |

### Metadata and EXIF

| Package | Version | Purpose (Simple English) | pip extra |
|---|---|---|---|
| `exifread` | 3.5.1 | Reads hidden camera data in photos (like what phone took the picture) | `forensics` |
| `pyexiftool` | 0.5.6 | A deeper hidden data reader — can easily spot if Photoshop or GIMP was used to edit a photo | `forensics` |

### Cryptography and Signatures

| Package | Version | Purpose (Simple English) | pip extra |
|---|---|---|---|
| `cryptography` | 46.0.5 | Checks the digital locks and signatures to make sure a document is genuine and from a trusted source | `forensics` |

### Statistics and ML (Machine Learning)

| Package | Version | Purpose (Simple English) | pip extra |
|---|---|---|---|
| `scipy` | 1.17.1 | Uses maths to spot unusual numbers, amounts, or dates that look out of place | `forensics` |
| `scikit-learn` | 1.8.0 | Uses smart AI methods to look for common patterns that usually mean a document is fake | `ml` |
| `faiss-cpu` | 1.13.2 | A super fast search engine to compare a document against thousands of known fake templates | `ml` |

---

## Requires External Binary (local dev only)

> **Using Docker?** All binaries below are pre-installed in the container — skip this section.

When running outside Docker, these Python packages need a system binary on `PATH`.

### Poppler
- Used by: `pdf2image`
- Windows binaries: https://github.com/oschwartz10612/poppler-windows/releases
- Add `poppler/Library/bin` to PATH

### ExifTool (Phil Harvey)
- Used by: `pyexiftool`
- Download: https://exiftool.org/
- Place `exiftool.exe` (Windows) on PATH

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

| Package | Purpose (Simple English) | pip extra |
|---|---|---|
| `boto3` | Allows the app to talk to and download documents from Amazon S3 storage | `connectors` |
| `google-api-python-client` / `google-auth` | Allows the app to securely log in and download documents from Google Drive | `connectors` |
| `requests` | Used to talk over the internet to download documents from Microsoft SharePoint | `connectors` |

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

