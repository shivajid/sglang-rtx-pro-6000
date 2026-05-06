# Whisper-large-v3 Benchmark

**Note**. This ia an alternative benchmark tool, based on vLLM going against the sglang server.

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

The benchmark client is engine-agnostic — anything exposing an
OpenAI-compatible `/v1/audio/transcriptions` endpoint works. Below is the
SGLang setup we use.

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

## How the sweep works

A single benchmark run executes one **phase per request rate**. Each phase is
an independent measurement: the harness sends `NUM_PROMPTS` audio
transcription requests at the target rate, waits for them all to complete,
records latency and throughput, and writes a JSON file. Then it moves to the
next rate. Phases don't share state — a slow rate doesn't pollute the next
one's measurements.

### Arrival pattern

Requests arrive as an **open-loop Poisson process** at the requested rate
(in requests per second). At rate=32, the harness submits a new request every
~31ms on average, regardless of whether the previous one has completed. This
matches how production traffic actually behaves — clients don't wait for the
server.

The special value `inf` means "fire all `NUM_PROMPTS` at once with no
spacing." This is a stress test, not realistic load — it produces useful
peak-throughput numbers but TTFT will be dominated by queue time.

### Why a sweep instead of one number

A single throughput number doesn't tell you how the server behaves under
real load. The sweep traces a **throughput-vs-latency curve**:

- **Low rates** (well below capacity) → latency is pure model time, throughput
  equals offered rate. The server is idle between requests.
- **Mid rates** (approaching capacity) → latency starts to climb as requests
  begin queueing. Throughput still tracks offered rate.
- **At capacity** → throughput plateaus, latency knees up sharply. This is
  the **saturation point**.
- **Above capacity** → throughput stays flat (server can't go faster),
  latency grows unbounded as the queue backs up. Eventually requests time
  out.

The shape of this curve is what you actually want to know. Two servers might
both peak at 100 req/s, but the one that holds <500ms p99 up to 80 req/s is
much more useful in production than the one that's already at 2s p99 by 50
req/s.

### What each phase measures

For every rate in `RATES`, the harness records:

| Metric | Meaning |
|--------|---------|
| `completed` / `failed` | How many requests succeeded vs errored |
| `duration` | Wall time for the phase, in seconds |
| `request_throughput` | Achieved requests/sec (may differ from offered rate at saturation) |
| `output_throughput` | Generated tokens/sec, summed across all requests |
| `total_token_throughput` | Input + output tokens/sec (input includes audio frames) |
| `rtfx` | Audio seconds processed per wall-clock second — **the headline number for ASR** |
| `mean_ttft_ms` / `median_ttft_ms` / `p99_ttft_ms` | Time to first transcript token. Captures queue + encoder time. |
| `mean_tpot_ms` / `median_tpot_ms` / `p99_tpot_ms` | Time per output token, *after* first token |
| `mean_itl_ms` / `p99_itl_ms` | Inter-token latency (similar to TPOT, slightly different aggregation) |

For Whisper specifically, **TTFT dominates user-perceived latency** because
outputs are short (50–100 tokens) and TPOT is sub-millisecond. A request that
finishes in 800ms typically spent 750ms in queue+encoder and 50ms in
decoding.

### Choosing the right rate set

The default `8 16 32 64 128 256 inf` is geometric, doubling each step. This
is good for "I don't know the server's capacity yet" — it covers two orders
of magnitude and makes the saturation point visible regardless of where it
falls.

Once you know roughly where saturation is, narrow the sweep around that
range for higher resolution:

```bash
# Saturation looks like it's around 60-100 req/s
-e RATES="40 60 80 100 120 inf"
```

Or skip the sweep entirely for a single-point measurement:

```bash
# Just measure peak throughput
-e RATES="inf" -e NUM_PROMPTS=500
```

### Per-phase duration

Each phase takes `max(num_prompts / rate, num_prompts / server_capacity)`
seconds. At rate=8 with 1000 prompts, that's at least 125 seconds regardless
of how fast the server is — the arrival schedule itself takes that long. At
rate=inf, the phase runs as fast as the server can drain the queue.

This means **low-rate phases are usually the longest** in a sweep, even
though they exercise the server least. If you're iterating on server config
and don't need the full curve, skip the bottom rates.

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

