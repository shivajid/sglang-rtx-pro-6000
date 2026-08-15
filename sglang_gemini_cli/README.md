# Gemini CLI as an agentic harness for SGLang

Point the [Gemini CLI](https://github.com/google-gemini/gemini-cli) at a model you are serving with
SGLang, and you get a real coding agent — file edits, shell execution, multi-turn tool calls — with
every token generated on your own GPUs. No external API, no per-token cost.

Validated against **Kimi-K3** and **DeepSeek-V4-Flash-0731**. Any model on this repo's recipes that
handles tool calling well should work; they all expose the same OpenAI-compatible API surface.

> 📖 Also documented on the site: **[Gemini CLI harness](https://shivajid.github.io/sglang-rtx-pro-6000/#geminicli)**

## Why this matters

SGLang gives you tokens; it does not give you an agent. The harness supplies the loop — tool
definitions, file edits, shell execution, multi-turn state — that turns a served checkpoint into
something that can actually build software. The [ViBench](../models/kimik3/) results in this repo were
produced by this class of setup, and the throughput numbers elsewhere are what determine whether an
agent session feels responsive or sluggish.

## What this actually is

The Gemini CLI normally talks to Gemini models. The build used here is a fork, republished to npm as
[`@shivajidnpm2026/gemini-cli`](https://www.npmjs.com/package/@shivajidnpm2026/gemini-cli), patched so
its model backend can be redirected at any **OpenAI-compatible** `/v1` endpoint — which is what SGLang
serves.

Nothing about your SGLang deployment needs to change. This is purely a client-side redirect driven by
three environment variables.

---

## 1. Install Node 20+

The CLI declares `engines.node >= 20` and will refuse to start on older runtimes.

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential python3

# Node.js 20.x LTS via NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

node -v   # v20.x or higher
npm -v    # 10.x or higher
```

## 2. Get a network path to the server

Two options. Pick one — **the port you end up talking to is the port that goes in `SGLANG_BASE_URL`**,
and confusing the two is the most common setup mistake.

| Option | When to use it | Endpoint |
|--------|----------------|----------|
| **A · Direct service/pod IP** | Client VM is on the same VPC as the cluster | `http://<pod-ip>:30000/v1` |
| **B · `kubectl port-forward`** | Working from a laptop or outside the VPC | `http://localhost:<local-port>/v1` |

For option B:

```bash
gcloud container clusters get-credentials <cluster_name> \
  --zone <zone> --project <project_id> --dns-endpoint

# syntax is LOCAL:REMOTE — 30000 locally maps to 30100 in the pod
kubectl port-forward pod/<pod_name> 30000:30100
```

> **Check which port your server actually listens on.** The recipes in this repo bind `30000` or
> `30100` depending on the deployment. With `port-forward 30000:30100` the remote side is `30100` and
> you connect to `localhost:30000`. If `/v1/models` hangs or refuses the connection, this mapping is
> almost always why.

## 3. Verify the endpoint before touching the CLI

Confirm the server answers, and note the exact model string it reports — you need it verbatim in the
next step.

```bash
# list served models — copy the "id" field from the response
curl http://<host>:30000/v1/models

# one round-trip through chat completions
curl http://<host>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V4-Flash-0731",
    "messages": [
      {"role": "user", "content": "Hello! What model are you?"}
    ],
    "max_tokens": 128,
    "temperature": 0.6
  }'
```

## 4. Set the three environment variables

| Variable | Value | Notes |
|----------|-------|-------|
| `SGLANG_BASE_URL` | `http://<host>:30000/v1` | Must include the `/v1` suffix |
| `GEMINI_MODEL` | e.g. `deepseek-ai/DeepSeek-V4-Flash-0731` | Exactly the `id` from `/v1/models` — no aliasing |
| `GEMINI_DEFAULT_AUTH_TYPE` | `sglang` | Any non-empty string; it only bypasses the Google auth flow |

```bash
export SGLANG_BASE_URL=http://<ip>:30000/v1
export GEMINI_MODEL="deepseek-ai/DeepSeek-V4-Flash-0731"
export GEMINI_DEFAULT_AUTH_TYPE="sglang"
```

## 5. Run it

```bash
npx @shivajidnpm2026/gemini-cli
```

The CLI starts in your current working directory and treats it as the project root — `cd` into the
repo you want the agent to work on first. From there it behaves like the upstream Gemini CLI.

---

## Troubleshooting

| Symptom | Likely cause |
|---------|--------------|
| Connection refused / request hangs | Port mismatch between `port-forward` and `SGLANG_BASE_URL`, or the pod listens on a different port than assumed |
| 404 on requests | `/v1` missing from `SGLANG_BASE_URL` |
| Model-not-found errors | `GEMINI_MODEL` doesn't match the `id` in `/v1/models` character for character |
| CLI exits immediately on start | Node older than 20 — check `node -v` |
| Agent stalls on long multi-step tasks | Context window exhaustion. Reasoning-heavy models are verbose; on Kimi-K3 see the `reasoning_effort` discussion in the [ViBench report](../models/kimik3/rev2_prgrs_vibench_kimik3_final_report.md) |
| Sluggish turns under load | TPOT at your concurrency, not the harness — see the [benchmark table](../README.md) for what each recipe sustains |

## Notes

The published build tracks upstream nightly `0.55.0-nightly.20260729` (Apache-2.0, fork of
[google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli)). It is a personal republish
for this workflow, not an official Google distribution — pin the version if you need reproducibility.
