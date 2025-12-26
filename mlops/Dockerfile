# =============================================================================
# Pikkit ML API - Production Dockerfile
# Multi-stage build for optimized container size and security
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies and prepare wheel packages
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

# Build arguments
ARG PYTHON_VERSION=3.11
ARG POETRY_VERSION=1.7.1

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production Runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS production

# Labels for container metadata
LABEL maintainer="mike@cornerstonemanagement.io" \
    app.name="pikkit-ml-api" \
    app.description="Pikkit ML Prediction API" \
    app.version="1.0.0"

# Build arguments
ARG APP_VERSION=1.0.0
ARG BUILD_DATE
ARG GIT_COMMIT

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    APP_VERSION=${APP_VERSION} \
    BUILD_DATE=${BUILD_DATE} \
    GIT_COMMIT=${GIT_COMMIT} \
    # Application defaults
    MODEL_DIR=/app/models \
    ENVIRONMENT=production \
    PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 pikkit && \
    useradd --uid 1000 --gid pikkit --shell /bin/bash --create-home pikkit

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/models /app/logs && \
    chown -R pikkit:pikkit /app

# Copy application code
COPY --chown=pikkit:pikkit app/ /app/

# Switch to non-root user
USER pikkit

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]

# -----------------------------------------------------------------------------
# Stage 3: Development (optional - for local development)
# -----------------------------------------------------------------------------
FROM production AS development

USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    less \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    httpx \
    black \
    isort \
    mypy

USER pikkit

# Enable hot reload for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# -----------------------------------------------------------------------------
# Stage 4: Test Runner
# -----------------------------------------------------------------------------
FROM development AS test

COPY --chown=pikkit:pikkit tests/ /app/tests/

CMD ["pytest", "-v", "--cov=/app", "--cov-report=term-missing", "tests/"]
