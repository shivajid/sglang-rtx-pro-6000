# Stage 1: Build native Python bindings using Rust toolchain
FROM rust:1.85-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    clang \
    libclang-dev \
    libopencv-dev \
    patchelf \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY deepseek-recipe /build/deepseek-recipe

WORKDIR /build/deepseek-recipe
RUN python3 -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install 'maturin[patchelf]>=1.9,<2' patchelf && \
    /venv/bin/maturin build --release --manifest-path deepseek-recipe-python/Cargo.toml --locked --out /dist

# Stage 2: Minimal runtime image
FROM python:3.11-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl \
    "fastapi>=0.115,<1" \
    "uvicorn>=0.34,<1" \
    "httpx>=0.27" \
    "anyio>=4.0" \
    && rm -f /tmp/*.whl

COPY main.py /app/main.py

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
