# Benchmarking Guide: GKE & SGLang on Blackwell G4

This guide provides step-by-step instructions on how to deploy LLM serving pods and execute performance benchmarks using the provided YAML manifests.

## Prerequisites

- A GKE cluster provisioned using the templates in `gkecluster/`.
- `kubectl` configured to access your cluster.
- A Hugging Face token (if required by the model) stored as a Kubernetes secret:
  ```bash
  kubectl create secret generic hf-secret --from-literal=HF_TOKEN=your_token_here
  ```

---

## Step 1: Deploy the Serving Framework

The repository contains model-specific manifests in the `models/` directory.

### For Single-Node Deployment (e.g., DeepSeek NVFP4)
```bash
kubectl apply -f models/DeepSeekv3-2/nvp4/sglang-ds32-job-v2.yaml
```

### For Distributed Multi-Node Deployment (e.g., GLM-5.1 or Kimi-K2.5)
These use a `StatefulSet` and a `Service` for internal communication.
```bash
kubectl apply -f models/GLM5.1/sglang-glm51-v1-04232026.yaml
```

**Verification**:
Ensure the pods are running and the SGLang server is initialized:
```bash
kubectl logs -f <pod-name>
```
Look for `The server is fired up and ready!` in the logs.

---

## Step 2: Run the Benchmark Client

Once the server is ready, deploy the benchmark client pod located in the `benchmarks/` directory.

### Example: Benchmarking DeepSeek-V3.2
1. Update the `API_URL` in `benchmarks/benchmark-dsv2.yaml` if necessary (it should point to the serving service name).
2. Deploy the client:
   ```bash
   kubectl apply -f benchmarks/benchmark-dsv2.yaml
   ```

### Monitoring the Benchmark
Track the progress of the load generator:
```bash
kubectl logs -f sglang-benchmark-client
```

---

## Step 3: Extracting Results

The benchmark client is configured to `sleep infinity` after completion, allowing you to copy the result files:

```bash
# Copy the results from the pod to your local machine
kubectl cp sglang-benchmark-client:/workspace/results_dsv2_3.json ./results.json
```

## Troubleshooting

- **OOM Errors**: If the pod fails with Out-Of-Memory, check the `mem-fraction-static` flag in the YAML. Ensure the model fits within the allocated memory.
- **Networking**: Multi-node setups require the VPC to be configured with high MTU (8896). Ensure you used the `createCluster_template.sh` for infrastructure setup.
