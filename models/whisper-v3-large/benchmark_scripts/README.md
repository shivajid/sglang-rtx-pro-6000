# Whisper-large-v3 Benchmark

Throughput and latency benchmark for `openai/whisper-large-v3` running behind any
OpenAI-compatible `/v1/audio/transcriptions` endpoint — vLLM, SGLang, TGI, or
anything else that speaks the protocol.

The benchmark runs a sweep over multiple request rates, measures end-to-end
latency, and reports throughput in both requests/sec and **RTFx** (the
real-time factor — seconds of audio processed per wall-clock second).

## What's in here

| File | Purpose |
|------|---------|
| `Dockerfile` | Builds the benchmark client image |
| `run_bench.sh` | Container entrypoint — sweeps over request rates and writes one JSON per rate |

The image bundles `vllm bench serve` (the unified vLLM benchmark CLI), which
works against any OpenAI-compatible audio endpoint regardless of which engine
is on the other end.

## Prerequisites

- Docker
- A running inference server exposing `/v1/audio/transcriptions` with
  `whisper-large-v3` loaded. See [Server setup](#server-setup) below.
- ~5 GB free disk for the HuggingFace dataset cache

## Quick start

```bash
# 1. Build the image
docker build -t whisper-bench .

# 2. Run the benchmark against a server on localhost:30000 (SGLang default)
docker run --rm \
  --network host \
  -v "$PWD/results:/results" \
  -v "$HOME/.cache/huggingface:/cache/hf" \
  -e HOST=localhost \
  -e PORT=30000 \
  whisper-bench
```

Results land in `./results/whisper_rate_<N>.json`, one file per request rate.

## Configuration

All knobs are environment variables passed via `-e`. Defaults in parentheses.

| Variable | Default | Notes |
|----------|---------|-------|
| `HOST` | `host.docker.internal` | Server hostname. Use `localhost` with `--network host` on Linux. |
| `PORT` | `8000` | vLLM default is 8000, SGLang default is 30000. |
| `MODEL` | `openai/whisper-large-v3` | Sent in request body's `model` field. |
| `SERVED_NAME` | `whisper-large-v3` | Must match server's `--served-model-name`. |
| `NUM_PROMPTS` | `1000` | Requests per rate phase. |
| `RATES` | `8 16 32 64 128 256 inf` | Space-separated list. `inf` = fire all at once. |
| `DATASET_PATH` | `edinburghcstr/ami` | Any HF audio dataset. |
| `HF_SUBSET` | `ihm` | |
| `HF_SPLIT` | `test` | |
| `OUTDIR` | `/results` | Inside the container. Mount it to get files on the host. |

### Common variations

A quick sanity check (~1 minute on a fast GPU):

```bash
docker run --rm --network host \
  -v "$PWD/results:/results" \
  -v "$HOME/.cache/huggingface:/cache/hf" \
  -e HOST=localhost -e PORT=30000 \
  -e RATES="32 inf" \
  -e NUM_PROMPTS=200 \
  whisper-bench
```

LibriSpeech instead of AMI (cleaner audio, faster decode):

```bash
docker run --rm --network host \
  -v "$PWD/results:/results" \
  -v "$HOME/.cache/huggingface:/cache/hf" \
  -e HOST=localhost -e PORT=30000 \
  -e DATASET_PATH=librispeech_asr \
  -e HF_SUBSET=clean \
  whisper-bench
```

Hit a remote server:

```bash
docker run --rm \
  -v "$PWD/results:/results" \
  -v "$HOME/.cache/huggingface:/cache/hf" \
  -e HOST=10.0.0.42 -e PORT=8000 \
  whisper-bench
```

## Server setup

The benchmark is engine-agnostic. Either of the following works as the target:

### vLLM

```bash
vllm serve openai/whisper-large-v3 \
  --task transcription \
  --served-model-name whisper-large-v3 \
  --max-model-len 448 \
  --max-num-seqs 400 \
  --gpu-memory-utilization 0.95 \
  --port 8000
```

For maximum throughput on Hopper/Ada GPUs, swap the model for the FP8 variant:
`RedHatAI/whisper-large-v3-FP8-dynamic`.

### SGLang (single GPU)

```bash
python -m sglang.launch_server \
  --model-path openai/whisper-large-v3 \
  --served-model-name whisper-large-v3 \
  --max-running-requests 400 \
  --mem-fraction-static 0.92 \
  --port 30000
```

### SGLang (multi-GPU, Docker, data parallel)

For maximum throughput on a multi-GPU host, run SGLang in Docker with one
replica per GPU using `--dp-size`. Whisper-large-v3 is small enough (~3 GB
in bf16) that data parallelism scales near-linearly — 8× GPUs give roughly
8× throughput.

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN must be set}"

sudo docker run \
  --gpus all --network host --ipc host \
  --rm -d --name sglang-whisper \
  -v /path/to/huggingface_cache:/root/.cache/huggingface \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e OMP_NUM_THREADS=24 \
  -e SGLANG_SET_CPU_AFFINITY=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  lmsysorg/sglang:latest \
  bash -c "pip install torchaudio datasets && \
    python3 -m sglang.launch_server \
      --model-path openai/whisper-large-v3 \
      --served-model-name whisper-large-v3 \
      --tp-size 1 \
      --dp-size 8 \
      --trust-remote-code \
      --mem-fraction-static 0.90 \
      --max-running-requests 1024 \
      --chunked-prefill-size 16384 \
      --cuda-graph-max-bs 128 \
      --attention-backend flashinfer \
      --host 0.0.0.0 \
      --port 30000"
```

Adjust `--dp-size` to your GPU count. Tail logs with
`docker logs -f sglang-whisper` and verify all GPUs are active with
`nvidia-smi -l 1` during the benchmark.

Notes:

- `HF_TOKEN` must be exported in the shell that runs this script — never
  hardcode tokens. Revoke and rotate any token that lands in a script,
  commit, or chat log.
- `--ipc host` is required for SGLang's inter-process communication
  between DP workers.
- `--attention-backend flashinfer` is the fastest option on Hopper/Ada.
  On Ampere, omit it to fall back to the default backend.
- `--served-model-name` must match the `SERVED_NAME` env var passed to
  the benchmark container, otherwise the server returns 404.

## Reading the results

Each `whisper_rate_<N>.json` contains throughput and latency stats for that
phase. Pull the headline numbers with:

```bash
cd results

python3 -c "
import json, glob
rows = []
for f in sorted(glob.glob('whisper_rate_*.json'),
                key=lambda x: float('inf') if 'inf' in x else int(x.split('_')[2].split('.')[0])):
    d = json.load(open(f))
    rows.append((f.replace('whisper_rate_','').replace('.json',''),
                 d['completed'], d['failed'], d['request_throughput'],
                 d.get('rtfx', 0), d['output_throughput'],
                 d['median_ttft_ms'], d['p99_ttft_ms']))
print(f'{\"rate\":>6}  {\"reqs\":>5}  {\"fail\":>4}  {\"rps\":>7}  {\"RTFx\":>8}  {\"tok/s\":>8}  {\"p50_ttft\":>10}  {\"p99_ttft\":>10}')
print('-' * 80)
for r in rows:
    print(f'{r[0]:>6}  {r[1]:>5}  {r[2]:>4}  {r[3]:>7.2f}  {r[4]:>8.1f}  {r[5]:>8.1f}  {r[6]:>9.0f}ms  {r[7]:>9.0f}ms')
"
```

### What to look for

- **`completed` should equal `num_prompts`** and `failed` should be 0. Anything
  else means the server couldn't keep up at that rate.
- **`request_throughput` (rps) plateaus at saturation.** As you walk up the
  rates, rps tracks the offered rate linearly until the server is full. The
  rate where rps stops climbing is your **server capacity ceiling**.
- **`rtfx` is the throughput number to quote** for transcription workloads.
  RTFx = audio seconds processed per wall-clock second. 250 RTFx means one
  hour of audio processes in ~14 seconds.
- **`p99_ttft_ms` tells you when queue pressure hits.** At low rates this is
  pure model latency. When it starts climbing into seconds, requests are
  queueing — the rate just below that is your safe operating point.

### Plot the throughput curve

```bash
pip install matplotlib

python3 -c "
import json, glob, matplotlib.pyplot as plt
data = []
for f in glob.glob('whisper_rate_*.json'):
    d = json.load(open(f))
    rate = f.replace('whisper_rate_','').replace('.json','')
    data.append((rate, d['request_throughput'], d['median_ttft_ms'], d['p99_ttft_ms']))
data.sort(key=lambda x: float('inf') if x[0]=='inf' else int(x[0]))

rps = [x[1] for x in data]
plt.figure(figsize=(8, 5))
plt.plot(rps, [x[2] for x in data], 'o-', label='p50 TTFT')
plt.plot(rps, [x[3] for x in data], 's-', label='p99 TTFT')
plt.xlabel('Achieved throughput (req/s)')
plt.ylabel('Latency (ms)')
plt.title('Whisper-large-v3 throughput vs latency')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('throughput_curve.png', dpi=120)
"
```

## How long does a run take?

For 1000 prompts × 7 rates against a single GPU:

| GPU | Approx total |
|-----|--------------|
| H100 + FP8 | 8–10 min |
| L40S + FP8 | 12–15 min |
| A100 + bf16 | 10–12 min |
| L4 + bf16 | 20–25 min |

Add 5–15 min the first time for the AMI dataset download. With the HF cache
volume mounted, subsequent runs reuse the cached audio.

## Troubleshooting

**`404 model not found`** — `SERVED_NAME` doesn't match the server's
`--served-model-name`. Either align them or pass the full model path as
`SERVED_NAME` so it matches the server's default.

**`AttributeError: 'str' object has no attribute 'decode'`** — A vLLM
benchmark harness bug in the streaming response handler. The Dockerfile
patches it during build; if you've stripped that patch, restore it.

**`ImportError: please install 'torchcodec'`** — `datasets` 4.0+ requires
torchcodec, which has fragile FFmpeg/torch ABI dependencies. The Dockerfile
pins `datasets<4.0` to avoid this entirely; the soundfile-based decoder it
uses produces identical results.

**Server logs show `gen throughput: 0.00`** — Misleading display artifact in
SGLang's per-step decode log. The benchmark JSON is the source of truth.
Whisper outputs are short (50–100 tokens) so the per-interval average often
rounds to zero between bursts even when the server is fully loaded.

**Container can't reach the server** — On Linux, `--network host` with
`HOST=localhost` is simplest. On Docker Desktop (Mac/Windows), use
`HOST=host.docker.internal` and drop `--network host`.

## Notes on accuracy benchmarking

This image only measures throughput and latency. For Word Error Rate
(WER) numbers, evaluate transcription outputs against ground truth using
`jiwer` and OpenAI's English text normalizer — see the
[Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
methodology.

## License

MIT.

