# K3 console

A small chat front end for an SGLang server running Kimi-K3, plus optional
image generation from a second SGLang Diffusion server.

The whole app is one FastAPI process that streams bytes between the browser and
SGLang. No GPU, no database, no build step, no npm. It runs happily in 192Mi on
a GKE CPU node pool.

```
browser ──► k3-console (this app, CPU pool) ──┬──► SGLang · Kimi-K3      /v1/chat/completions
                                              └──► SGLang Diffusion     /v1/images/generations
```

## What it does

- Streams replies token by token, including K3's `reasoning_content` in a
  collapsible trace (K3 reasons on every turn, so this is on by default).
- Image input: attach, paste, or drag images into the composer. They are resized
  in the browser to a 1568px longest edge, then sent as OpenAI `image_url`
  content blocks — the format SGLang's K3 vision path expects.
- Image output: the composer's **Generate image** mode posts to a diffusion
  server and renders the result. Generated images can be handed straight back to
  K3 with **Ask K3 about this**.
- Per-turn telemetry (tokens, tok/s, time to first token) in the left rail.
- History and settings live in the browser's local storage. The server keeps
  nothing, so pods stay stateless and can scale or restart freely.

## Run it locally

```bash
pip install -r requirements.txt
cp .env.example .env            # point SGLANG_BASE_URL at your server
set -a && source .env && set +a
uvicorn app.main:app --reload --port 8080
```

Open http://localhost:8080. The dot in the header turns green once `/readyz`
can reach SGLang.

If your SGLang runs in the cluster, tunnel to it first:

```bash
kubectl port-forward -n serving svc/sglang-kimi-k3 30000:30000
```

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `SGLANG_BASE_URL` | `http://sglang-kimi-k3:30000/v1` | Chat/vision upstream. Include `/v1`. |
| `CHAT_MODEL` | `moonshotai/Kimi-K3` | Must match SGLang's `--served-model-name`. |
| `SGLANG_API_KEY` | – | Only if you launched SGLang with `--api-key`. |
| `IMAGE_BASE_URL` | – | Diffusion upstream. Blank hides the image button. |
| `IMAGE_MODEL` | `Qwen/Qwen-Image` | Whatever that server serves. |
| `MAX_REQUEST_MB` | `24` | Rejects oversized conversations before they hit SGLang. |
| `MAX_IMAGE_MB` | `8` | Per-attachment ceiling after browser-side resizing. |
| `READ_TIMEOUT_SECONDS` | `900` | Long, because 1M-context prefills are slow. |
| `DEFAULT_TEMPERATURE` | `0.6` | Starting value for the slider. |
| `APP_TOKEN` | – | Optional shared secret on `/api/*`. Prefer IAP for real auth. |

## Deploy to GKE

Build for the node pool's architecture — `--platform linux/amd64` unless your
CPU pool is Arm (T2A/C4A):

```bash
PROJECT_ID=your-project
REGION=us-central1
IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/apps/k3-console:0.1.0

gcloud artifacts repositories create apps --repository-format=docker --location=$REGION
docker build --platform linux/amd64 -t $IMAGE .
docker push $IMAGE
```

Edit `k8s/configmap.yaml` (upstream URLs) and the image line in
`k8s/deployment.yaml`, then:

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml     # or skip and use port-forward
kubectl -n chat rollout status deploy/k3-console
```

### Things that bite on GKE

- **Node pool pinning.** `nodeSelector: cloud.google.com/gke-nodepool: cpu-pool-4`
  — rename it to your pool. There is also a node affinity rule that refuses any
  node carrying `cloud.google.com/gke-accelerator`, so a misconfigured selector
  can't quietly park this pod on a GPU node.
- **Streaming and the load balancer.** GCLB backends default to a 30s timeout,
  which truncates long generations. `service.yaml` ships a `BackendConfig` with
  `timeoutSec: 1800`; keep it if you swap in your own Service.
- **Readiness follows the model server.** `/readyz` probes SGLang's `/models`,
  so pods leave the load balancer when the GPU side is down or reloading.
- **Request size.** Base64 images make requests large. Raise `MAX_REQUEST_MB`
  and any proxy body limits together, or attachments will fail at the edge.
- **Image generation is a separate server.** K3 reads images; it does not draw
  them. Point `IMAGE_BASE_URL` at an SGLang Diffusion deployment, for example
  `sglang serve --model-path Qwen/Qwen-Image --port 30010`. Leave it unset and
  the app hides the feature cleanly.

## API

| Route | Purpose |
| --- | --- |
| `POST /api/chat` | Server-sent events: `reasoning`, `delta`, `notice`, `error`, `done`. |
| `POST /api/images` | `{prompt, size, n, seed}` → base64 data URLs. |
| `GET /api/config` | What the UI should show. |
| `GET /healthz` | Liveness. Always cheap. |
| `GET /readyz` | Readiness. Checks the SGLang upstream, cached 10s. |

## Layout

```
app/main.py           gateway: SSE relay, image proxy, health
app/static/           index.html, styles.css, app.js — no framework
k8s/                  namespace, config, deployment, service, ingress, hpa
Dockerfile            python:3.12-slim, non-root, read-only root filesystem
```
