# ─────────────────────────────────────────────────────────────────────────────
# BaseTruth — Platform-agnostic container
#
# Includes ALL external binaries:
#   • Tesseract OCR   (pytesseract, ocrmypdf)
#   • Poppler         (pdf2image)
#   • ExifTool        (pyexiftool)
#   • Ghostscript     (ocrmypdf PDF/A)
#   • qpdf            (optional signature workflows)
#
# Build:
#   docker build -t basetruth:latest .
#
# Run CLI:
#   docker run --rm -v $(pwd)/artifacts:/app/artifacts \
#              -v $(pwd)/your_data:/app/your_data \
#              basetruth:latest scan --input your_data/doc.pdf
#
# Run API:
#   docker run --rm -p 8000:8000 \
#              -v $(pwd)/artifacts:/app/artifacts \
#              basetruth:latest serve
# ─────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Python dependency builder
# Builds wheels so the final image never needs build tools at runtime.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

# Build-time system libraries needed to compile C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
        libssl-dev \
        libjpeg-dev \
        zlib1g-dev \
        libpng-dev \
        libwebp-dev \
        libtiff-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files first — changes here bust the cache; source changes don't
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# Install pinned requirements + dev tools into a prefix we copy to the final image
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt \
 && pip install --no-cache-dir --prefix=/install --no-deps "." \
 && pip install --no-cache-dir --prefix=/install pytest


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime image
# Slim Debian base; all binaries installed from official distro packages.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

LABEL org.opencontainers.image.title="BaseTruth"
LABEL org.opencontainers.image.description="Document integrity and fraud detection pipeline"
LABEL org.opencontainers.image.source="https://github.com/maluskarhrishikesh-afk/BaseTruth"

# ── System binaries ────────────────────────────────────────────────────────────
# tesseract-ocr    → pytesseract + ocrmypdf
# poppler-utils    → pdf2image (pdftoppm, pdfinfo)
# libimage-exiftool-perl → pyexiftool (the `exiftool` binary is in this package)
# ghostscript      → ocrmypdf PDF/A output
# qpdf             → standalone signature extraction (pikepdf bundles libqpdf, this is the CLI)
# libgl1           → opencv headless runtime needs libGL
# libglib2.0-0     → opencv runtime dep
# imagemagick      → @llamaindex/liteparse image-PDF conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-hin \
        poppler-utils \
        libimage-exiftool-perl \
        ghostscript \
        qpdf \
        imagemagick \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Node.js (for @llamaindex/liteparse) ───────────────────────────────────────
# Using NodeSource LTS (22.x) — avoids outdated Debian nodejs package
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/* \
 && npm install -g @llamaindex/liteparse \
 && npm cache clean --force

# ── Python environment from builder ───────────────────────────────────────────
COPY --from=builder /install /usr/local

# ── Application source ────────────────────────────────────────────────────────
WORKDIR /app
COPY src/ ./src/
COPY pyproject.toml ./
COPY README.md ./

# Install the package itself (editable-style, no extra deps — already in /usr/local)
RUN pip install --no-cache-dir --no-deps -e .

# ── Runtime configuration ─────────────────────────────────────────────────────
# Tell pytesseract where Tesseract lives (matches Debian install path)
ENV TESSERACT_CMD=/usr/bin/tesseract
# Tell pyexiftool where exiftool lives (Debian puts it at /usr/bin/exiftool)
ENV EXIFTOOL_PATH=/usr/bin/exiftool
# Poppler pdftoppm is on PATH via poppler-utils; no extra env needed
# Ghostscript binary is `gs` on PATH

# Artifact output directory — mount a volume here to persist results
ENV BASETRUTH_ARTIFACT_ROOT=/app/artifacts
RUN mkdir -p /app/artifacts /app/your_data

# Non-root user for security
RUN groupadd -r basetruth && useradd -r -g basetruth basetruth \
 && chown -R basetruth:basetruth /app
USER basetruth

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import basetruth; print('ok')" || exit 1

# ── Default command ───────────────────────────────────────────────────────────
# Override at `docker run` time:
#   docker run basetruth:latest scan --input /app/your_data/doc.pdf
#   docker run basetruth:latest serve
ENTRYPOINT ["python", "-m", "basetruth.cli"]
CMD ["--help"]
