# Cloudflare Clef on GCP G4 (RTX PRO 6000)

Deploy, test and benchmark [Cloudflare/clef](https://huggingface.co/Cloudflare/clef) on GKE with G4 VMs (`g4-standard-48`, one NVIDIA RTX PRO 6000 Blackwell GPU each).

**Highlights** (FP8, one replica per GPU, benchmarked October 7, 2026):
- **10.6K input tokens/s** per GPU, 1.9–2.2× the model's reference implementation
- **147 ms** for a 1.5K-token request, and **7.4 req/s** per replica on mixed traffic
- **Same accuracy as BF16** on BANKING77, at about **$0.15 per 1M input tokens**

## What is Clef?

Clef is a **decision model**. You send a **state** (text or JSON) and a **schema** of typed questions. Clef returns a probability for every allowed answer, from a single forward pass. It never generates text, so there is no output to parse, and cost depends only on input length.

| Question type | You send | You get back |
|---|---|---|
| `noul` | a yes/no question | P(true) |
| `choice` | named options | the most likely option, and every option's probability |
| `score` | ordered levels, lowest first | the expected level, and each level's probability |

Use it wherever a workflow or an agent needs a fast, consistent decision: routing a ticket, flagging a message, choosing the next step.

### How Clef relates to Jev

- **Jev came first.** Typesafe AI's *Jev System One* model introduced this kind of decision model. Jev's API is `POST /v1/systemone`, which is why this server, and Clef's reference code, use that endpoint.
- **Clef is Cloudflare's Jev-compatible model.** It takes the same request body and returns the same response shape, so a request written for Jev works against this server unchanged.
- **Clef is open.** Cloudflare released Clef (Qwen3.8-27B backbone) and the smaller Clef-flash (Qwen3.5-9B backbone) under Apache-2.0, and also hosts both on Workers AI. This folder self-hosts Clef on GKE.

How they compare, according to Cloudflare's [announcement](https://blog.cloudflare.com/clef-decision-models) and [model card](https://huggingface.co/Cloudflare/clef):

| | Jev | Clef |
|---|---|---|
| API | `POST /v1/systemone` | Same request and response format |
| Inputs | Text | Text, images, video |
| Context window | 32K tokens | 64K tokens (this deployment caps inputs at 16K) |
| Decision Index benchmarks won, head to head | 15 of 41 | 26 of 41 |
| Typesafe workflow evals won | 1 of 4 | 3 of 4 |
| Median latency on the Decision Index | 524 ms | 209 ms |

- The Decision Index (Cloudflare also calls it the Jev Decision Index) is a benchmark suite for decision models.
- Clef's leads include classification, for example BANKING77 macro-F1 of 94.2 vs 79.7.
- Jev's biggest leads are on GPQA Diamond, BBH and MMLU-Pro.
- All of these comparisons are Cloudflare's. This repo serves Clef only and did not benchmark Jev.

## Deploy

**You need** a GKE cluster with a node pool of `g4-standard-48` VMs that use local SSD for ephemeral storage, one VM per replica. The manifest targets node pool `clef-g4-pool`; change the `nodeSelector` in [`k8s/clef-server.yaml`](./k8s/clef-server.yaml) to match yours.

1. Package the server code as a ConfigMap:
   ```bash
   cd models/Clef
   kubectl create configmap clef-server-code \
     --from-file=server/app.py --from-file=server/clef_engine.py --from-file=server/fast_patches.py \
     --from-file=server/fp8_linear.py --from-file=server/nvfp4_linear.py \
     --dry-run=client -o yaml | kubectl apply -f -
   ```
2. Deploy the server: 2 replicas, one per node, behind the `clef-server` Service on port 8000. With a single node, set `replicas: 1` first.
   ```bash
   kubectl apply -f k8s/clef-server.yaml
   ```
3. Wait until the pods are ready:
   ```bash
   kubectl rollout status deployment/clef-server
   ```
   The first start on a new node installs the Python packages and downloads the 52 GB of weights. After that, a pod is ready about 25 s after it starts.

> [!CAUTION]
> The Service is internal (ClusterIP) on purpose. The server's `/admin/config` endpoint has no authentication, so never expose the server publicly.

Precision, batch size and other options are listed under [Server settings](./TECHNICAL_DETAILS.md#server-settings).

## Test

### Send a request

```bash
kubectl port-forward svc/clef-server 8000:8000 &

curl -s localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d @- <<'EOF' | python3 -m json.tool
{
  "model": "clef",
  "state": "Hi, I was charged twice for my March invoice, and when I try to update my card the checkout page just times out so I can't pay. Please sort this out today.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this message?",
      "criteria": {
        "billing": "Payments, invoices, refunds or charges",
        "technical": "Bugs, errors or outages",
        "account": "Login, password or profile changes",
        "sales": "Pricing, plans or upgrades"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How soon does the customer need a response?",
      "criteria": ["Can wait", "This week", "Today"]
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Is the customer asking for money back?"
    }
  }
}
EOF
```

Clef's answers to this request:

| Question | Answer |
|---|---|
| `department` | `billing`, probability 0.88 (`technical` 0.11) |
| `urgency` | score 1.96 on the 0–2 scale: "Today", probability 0.96 |
| `refund_requested` | P(true) = 0.05 |

The request used 386 input tokens and 0 output tokens. Wording matters: asking "Would resolving this require refunding money to the customer?" instead raises `refund_requested` to 0.69.

From inside the cluster, use `http://clef-server:8000/v1/systemone`. The full request and response format is in the [API reference](./TECHNICAL_DETAILS.md#api-reference).

### Check decisions against labeled cases

[`run_decisions.py`](./tests/decisions/run_decisions.py) sends each case in a suite file and checks Clef's answers against the expected ones. Write a suite with your production schema and labeled examples; [`example_support_triage.yaml`](./tests/decisions/suites/example_support_triage.yaml) shows the format.

```bash
python3 tests/decisions/run_decisions.py tests/decisions/suites/example_support_triage.yaml
```

It prints accuracy per question and any failing cases (add `-v` to list every case). It exits non-zero when the pass rate is below the suite's `min_pass_rate`, so it can gate a rollout.

## Benchmark

### Results

Measured through the Kubernetes Service with FP8, from a load generator on a separate CPU node pool. Throughput is input tokens per second, because Clef generates no output tokens.

**Latency**, one request at a time:

| Workload | Input tokens | Reference code | This server |
|---|---|---|---|
| short: chat moderation, 3 questions | ~530 | 120 ms | 73 ms |
| medium: ticket triage, 5 questions | ~1.5K | 266 ms | 147 ms |
| banking77: 77-way intent | ~1.8K | 330 ms | 175 ms |
| long: contract review, 8 questions | ~4K | 707 ms | 389 ms |
| xlong: incident report, 4 questions | ~15.8K | 2,993 ms | 1,644 ms |

**Peak throughput:**

| Workload | 1 replica | 2 replicas |
|---|---|---|
| medium | 10.7K tok/s · 7.0 req/s | 21.1K tok/s · 13.8 req/s |
| mixed: 50% short, 35% medium, 15% long | 10.6K tok/s · 7.5 req/s | 20.9K tok/s · 14.5 req/s |
| banking77 | 10.7K tok/s · 5.8 req/s | 21.1K tok/s · 11.5 req/s |
| long | 10.6K tok/s · 2.7 req/s | 20.4K tok/s · 5.1 req/s |

- **One GPU tops out at about 10.6K tokens/s.** That equals SGLang running the backbone alone in FP8, without the Clef head.
- **Plan for about 80% of peak.** At 6 mixed req/s per replica, p99 latency stays under 2 s.
- **FP8 costs no accuracy.** BANKING77 accuracy is 0.953, the same as BF16.
- **NVFP4 is an opt-in speed-up.** It gives +31% throughput for −0.1 to −0.6 points of accuracy.

### Run the benchmark

1. Generate the request workloads once with [`bench/make_workloads.py`](./bench/make_workloads.py). It needs the Clef tokenizer, so run it in a dev pod; see [Reproducing the experiments](./TECHNICAL_DETAILS.md#10-reproducing-the-experiments). Put the generated files in a local `workloads/` folder.
2. Start the load generator on a CPU node pool, and copy in the scripts and workloads. Change the `nodeSelector` in [`k8s/bench-client.yaml`](./k8s/bench-client.yaml) first.
   ```bash
   kubectl apply -f k8s/bench-client.yaml
   kubectl wait --for=condition=Ready pod/clef-bench-client --timeout=300s
   kubectl exec clef-bench-client -- bash -c "pip install -q aiohttp && mkdir -p /work/results"
   kubectl cp bench/loadgen.py clef-bench-client:/work/loadgen.py
   kubectl cp bench/final_bench.sh clef-bench-client:/work/final_bench.sh
   kubectl cp workloads clef-bench-client:/work/workloads
   ```
3. Run the concurrency sweeps over all six workloads:
   ```bash
   kubectl exec clef-bench-client -- bash /work/final_bench.sh http://clef-server:8000 /work/results/run.jsonl my-run
   ```
   To also measure latency at fixed request rates, add the rates as a fourth argument, for example `5,9,12,14,17`.
4. Fetch the results and print the tables:
   ```bash
   kubectl exec clef-bench-client -- cat /work/results/run.jsonl > results/my-run.jsonl
   python3 bench/report_tables.py final results/my-run.jsonl
   ```

To compare replica counts, run the sweep again with a different tag after `kubectl scale deployment clef-server --replicas=1`.

## Learn more

- [Technical details](./TECHNICAL_DETAILS.md): covers
  - how the server works and why it isn't built on SGLang or vLLM
  - each optimization step and its gain, and two serving bugs found and fixed
  - accuracy, the NVFP4 option, capacity and cost, recommendations, and all raw results
- [Clef model card](https://huggingface.co/Cloudflare/clef) and [Cloudflare's announcement](https://blog.cloudflare.com/clef-decision-models)

| Folder | Contents |
|---|---|
| [`server/`](./server/) | The inference server and its custom kernels |
| [`k8s/`](./k8s/) | Kubernetes manifests: server, load generator, dev pods, SGLang baseline |
| [`tests/`](./tests/) | Decision tests and correctness checks |
| [`bench/`](./bench/) | Workload generator, load generator and benchmark scripts |
| [`results/`](./results/) | Raw benchmark results |
