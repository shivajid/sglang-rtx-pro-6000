FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/hf \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip

ARG VLLM_VERSION=0.20.1
RUN pip install "vllm==${VLLM_VERSION}" \
 && pip install "datasets<4.0" "soundfile" "librosa" "numpy"

WORKDIR /bench
COPY run_bench.sh /bench/run_bench.sh
RUN chmod +x /bench/run_bench.sh

ENTRYPOINT ["/bench/run_bench.sh"]
