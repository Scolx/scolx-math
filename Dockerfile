# syntax=docker/dockerfile:1

# Stage 1: Builder - build dependencies and install packages
FROM python:3.14-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build dependencies for scientific packages like symengine
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy project metadata and source code for installation
COPY pyproject.toml README.md ./
COPY scolx_math ./scolx_math

# Upgrade pip and install the project including dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir .

# Stage 2: Runtime - clean image without build dependencies
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only the installed packages and app code from builder
COPY --from=builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=builder /app/scolx_math ./scolx_math
COPY --from=builder /app/pyproject.toml ./pyproject.toml
COPY --from=builder /app/README.md ./README.md

EXPOSE 8000

CMD ["uvicorn", "scolx_math.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
