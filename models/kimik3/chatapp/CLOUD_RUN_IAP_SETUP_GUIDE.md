# K3 Console: Cloud Run & Identity-Aware Proxy (IAP) Deployment Guide

This guide documents the complete end-to-end architecture and deployment procedure for hosting **K3 Console** on Google Cloud Run with **Identity-Aware Proxy (IAP)** authentication and **Direct VPC Egress** to a distributed **Moonshot AI Kimi-K3** SGLang server running on Google Kubernetes Engine (GKE).

---

## 1. Architecture Overview

```
                        Google Corporate Network / Browser
                                       │
                                       ▼ (HTTPS)
      ┌─────────────────────────────────────────────────────────────────┐
      │                    Cloud Run Service: k3-console                │
      │  • Security: Identity-Aware Proxy (IAP) + IAM                   │
      │  • Domain Access: domain:google.com (roles/run.invoker)         │
      │  • Networking: Direct VPC Egress (pm-g4-vpc-useast5)            │
      │  • Environment: SGLANG_BASE_URL=http://10.100.0.58:30000/v1     │
      └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼ (Private Andromeda Fabric)
      ┌─────────────────────────────────────────────────────────────────┐
      │             GKE Internal Load Balancer (ILB)                    │
      │  • Service: sglang-kimi-k3-ilb                                  │
      │  • Static Private VIP: 10.100.0.58:30000                        │
      └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
      ┌─────────────────────────────────────────────────────────────────┐
      │             GKE SGLang Serving Pods (4x G4 Nodes)               │
      │  • Master HTTP Pod (PP0): 10.100.0.45:30000                     │
      │  • 32x NVIDIA RTX PRO 6000 Ada (1.53 TB VRAM)                   │
      │  • Hyperdisk ML Mount: 1.5 TB Safetensor weights (/data/model)  │
      └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Prerequisites & Resource Mapping

| Resource | Value |
|---|---|
| **GCP Project** | `northam-ce-mlai-tpu` |
| **Region** | `us-east5` (Columbus) |
| **GKE Cluster** | `pm-g4-1m-cluster` |
| **VPC Network** | `pm-g4-vpc-useast5` |
| **Subnetwork** | `pm-subnet-1-useast5` |
| **Internal ILB VIP** | `10.100.0.58:30000` |
| **Cloud Run URL** | `https://k3-console-q2fm2rxu6a-ul.a.run.app` |

---

## 3. Step-by-Step Deployment Guide

### Step 1: Create GKE Internal Load Balancer (ILB)

To ensure Cloud Run has a resilient, static private IP that does not change upon pod restarts:

1. Create manifest `sglang-kimi-k3-ilb.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: sglang-kimi-k3-ilb
  namespace: default
  annotations:
    networking.gke.io/load-balancer-type: "Internal"
spec:
  type: LoadBalancer
  selector:
    app: sglang-kimi-k3-g4
    apps.kubernetes.io/pod-index: "0"
  ports:
  - name: http
    port: 30000
    targetPort: 30000
```

2. Apply to GKE cluster:
```bash
kubectl apply -f sglang-kimi-k3-ilb.yaml
```

3. Note the assigned internal IP (`10.100.0.58`):
```bash
kubectl get svc sglang-kimi-k3-ilb
```

---

### Step 2: Build & Package Container Image

Build the lightweight FastAPI image using Google Cloud Build directly to Artifact Registry:

```bash
gcloud builds submit KimkK3/chatapp \
  --tag us-east5-docker.pkg.dev/northam-ce-mlai-tpu/debug-repo/k3-console:latest \
  --project northam-ce-mlai-tpu \
  --region us-east5
```

---

### Step 3: Deploy to Cloud Run with Direct VPC Egress

Deploy the service attached to the GKE cluster's VPC subnet:

```bash
gcloud run deploy k3-console \
  --image us-east5-docker.pkg.dev/northam-ce-mlai-tpu/debug-repo/k3-console:latest \
  --region us-east5 \
  --project northam-ce-mlai-tpu \
  --network pm-g4-vpc-useast5 \
  --subnet pm-subnet-1-useast5 \
  --vpc-egress all-traffic \
  --set-env-vars SGLANG_BASE_URL="http://10.100.0.58:30000/v1",CHAT_MODEL="moonshotai/Kimi-K3",READ_TIMEOUT_SECONDS="900" \
  --port 8080 \
  --cpu 1 \
  --memory 512Mi
```

---

### Step 4: Configure Domain Access & IAP Security

1. Grant Cloud Run Invoker role to all `@google.com` accounts:
```bash
gcloud run services add-iam-policy-binding k3-console \
  --region us-east5 \
  --project northam-ce-mlai-tpu \
  --member="domain:google.com" \
  --role="roles/run.invoker"
```

2. Grant IAP Web App User role to all `@google.com` accounts:
```bash
gcloud iap web add-iam-policy-binding \
  --project northam-ce-mlai-tpu \
  --member="domain:google.com" \
  --role="roles/iap.httpsResourceAccessor"
```

3. Enable IAP in Cloud Run Console:
   * Open **Cloud Run** $\rightarrow$ Click `k3-console` $\rightarrow$ Click **Edit & Deploy New Revision**.
   * Under **Security** $\rightarrow$ **Authentication**: Check **Identity Aware Proxy (IAP)**.
   * Click **Deploy**.

---

## 4. How Users Access the Service

* **Web Browser URL**:
  👉 **[https://k3-console-q2fm2rxu6a-ul.a.run.app](https://k3-console-q2fm2rxu6a-ul.a.run.app)**

* **Authentication Flow**:
  1. User opens the link in Chrome.
  2. Google IAP prompts user to sign in with their `@google.com` account.
  3. Upon sign-in, IAP validates membership and establishes a session.
  4. Real-time token streaming with collapsible reasoning traces starts immediately.

* **Optional Local Proxy Command**:
```bash
gcloud run services proxy k3-console --region us-east5 --project northam-ce-mlai-tpu
```
