# Whisper-large-v3 on GCP G4

Optimized configurations and benchmarks for [OpenAI's Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) on GCP G4 instances using SGLang.

## Model Overview
Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.

## Serving Configuration
The model is served using a single-node SGLang setup with Data Parallelism:
- **Tensor Parallelism**: 1
- **Data Parallelism**: 8
- **Precision**: float16 (base model)
- **Serving Image**: `docker.io/lmsysorg/sglang:latest`

The server was launched with the following parameters:
```bash
sglang serve --model-path openai/whisper-large-v3 \
    --tp-size 1 \
    --dp-size 8 \
    --trust-remote-code \
    --mem-fraction-static 0.90 \
    --max-running-requests 1024 \
    --chunked-prefill-size 16384 \
    --cuda-graph-max-bs 128 \
    --attention-backend flashinfer
```

## Benchmark Results
The following benchmarks were conducted on a single `g4-standard-384` instance (8x RTX PRO 6000 Blackwell GPUs) using 512 examples from the `D4nt3/esb-datasets-earnings22-validation-tiny-filtered` dataset with a concurrency of 256.

We tested three different API endpoints supported by the benchmark script:

| Metric | Chat Completions | Audio Transcriptions | Streaming Transcription |
| :--- | :---: | :---: | :---: |
| **Throughput** | 48.21 req/s | **55.63 req/s** | 28.15 req/s |
| **Token Throughput** | 898.04 tok/s | **1048.84 tok/s** | 530.64 tok/s |
| **Average Latency** | 3.2707 s | **0.3220 s** | 0.9063 s |
| **WER** | **12.7570** | 13.1418 | 13.1538 |

*Note: The Audio Transcriptions API yielded the highest throughput and lowest latency.*

## Usage
To deploy the benchmark job, apply the `sglang-whisper-job.yaml` manifest:
```bash
kubectl apply -f sglang-whisper-job.yaml
```
