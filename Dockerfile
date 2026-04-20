# ─────────────────────────────────────────────────────────────────────────────
# BaseTruth — Platform-agnostic container
#
# Includes the external binaries BaseTruth still needs at runtime:
#   • Poppler         (pdf2image)
#   • ExifTool        (pyexiftool)
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
# Docker stays on Python 3.12 because the production OCR/KYC stack still
# depends on wheels that are not published consistently for Python 3.13.
FROM python:3.12-slim AS builder

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

# Copy only dependency manifests first.
# Docker caches each layer independently — as long as requirements.txt and
# pyproject.toml do not change, the expensive pip install below is a cache HIT
# even when source files in src/ change.
COPY requirements.txt pyproject.toml ./

# Install pinned requirements — this layer is reused on every build where
# requirements.txt / pyproject.toml are unchanged (no downloads, ~0s).
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt \
 && pip install --no-cache-dir --prefix=/install pytest

# Copy source AFTER deps are installed.  Changes here only invalidate the fast
# "install package itself" step below, not the multi-GB download layer above.
COPY src/ ./src/
RUN pip install --no-cache-dir --prefix=/install --no-deps "."


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime image
# Slim Debian base; all binaries installed from official distro packages.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="BaseTruth"
LABEL org.opencontainers.image.description="Document integrity and fraud detection pipeline"
LABEL org.opencontainers.image.source="https://github.com/maluskarhrishikesh-afk/BaseTruth"

# ── System binaries ────────────────────────────────────────────────────────────
# poppler-utils    → pdf2image (pdftoppm, pdfinfo)
# libimage-exiftool-perl → pyexiftool (the `exiftool` binary is in this package)
# qpdf             → standalone signature extraction (pikepdf bundles libqpdf, this is the CLI)
# libgl1           → opencv headless runtime needs libGL
# libglib2.0-0     → opencv runtime dep
# imagemagick      → @llamaindex/liteparse image-PDF conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libimage-exiftool-perl \
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
# Copy the docs/ directory so DATABASE.md is available inside the container.
# _load_database_md() also tries the MinIO docs bucket as a fallback, but having
# the file on the filesystem is the most reliable and zero-latency path.
COPY docs/ ./docs/

# Install the package itself (editable-style, no extra deps — already in /usr/local)
RUN pip install --no-cache-dir --no-deps -e .

# ── Runtime configuration ─────────────────────────────────────────────────────
# Tell pyexiftool where exiftool lives (Debian puts it at /usr/bin/exiftool)
ENV EXIFTOOL_PATH=/usr/bin/exiftool
ENV HOME=/home/basetruth
ENV XDG_CACHE_HOME=/home/basetruth/.cache
# Poppler pdftoppm is on PATH via poppler-utils; no extra env needed
# Ghostscript binary is `gs` on PATH

# Artifact output directory — mount a volume here to persist results
ENV BASETRUTH_ARTIFACT_ROOT=/app/artifacts
RUN mkdir -p /app/artifacts /app/your_data

# Non-root user for security. PaddleX writes model/cache state under HOME,
# so the runtime user must have a real writable home directory.
RUN groupadd -r basetruth && useradd -r -m -d /home/basetruth -g basetruth basetruth \
 && mkdir -p /home/basetruth/.cache /home/basetruth/.paddlex \
 && chown -R basetruth:basetruth /app /home/basetruth
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
