# Qwen3.8-Flash-Next Weight Disk Setup

This directory provides the workflow to ingest `Qwen/Qwen3.8-Flash-Next` weights onto a GCP Hyperdisk volume (`qwen38-flash-hyperdisk-ml`) and serve via SGLang with zero boot disk pressure.

## Step 1: Provision the Disk

The 500GB `hyperdisk-ml` disk has been provisioned:
```bash
gcloud compute disks create qwen38-flash-hyperdisk-ml \
    --project=northam-ce-mlai-tpu \
    --zone=europe-west4-b \
    --type=hyperdisk-ml \
    --size=500GB
```

## Step 2: Ingest Model Weights (Writer Mode)

1. Delete any failing/pending serving StatefulSet to clear the node:
   ```bash
   kubectl delete statefulset sglang-qwen38-flash-next-latency --ignore-not-found
   ```

2. Apply the ReadWriteOnce PV and PVC:
   ```bash
   kubectl apply -f models/Qwen3.8-Flash-Next/model_weight_disk/qwen38-flash-hdml-writer.yaml
   ```

3. Run the downloader job:
   ```bash
   kubectl apply -f models/Qwen3.8-Flash-Next/model_weight_disk/qwen38-flash-downloader-job.yaml
   ```

4. Monitor the download:
   ```bash
   kubectl logs -f job/qwen38-flash-hdml-downloader
   ```

5. Once the job succeeds, release the writer lock:
   ```bash
   kubectl delete job qwen38-flash-hdml-downloader
   kubectl delete pvc qwen38-flash-hdml-writer-pvc
   kubectl delete pv qwen38-flash-hdml-pv
   ```

## Step 3: Mount & Serve

1. Apply the serving PV and PVC:
   ```bash
   kubectl apply -f models/Qwen3.8-Flash-Next/model_weight_disk/qwen38-flash-hdml-ro.yaml
   ```

2. Deploy the SGLang single-node serving workload (TP8 + MTP NEXTN speculative decoding):
   ```bash
   kubectl apply -f models/Qwen3.8-Flash-Next/sglang-qwen38-flash-next-latency-hdml.yaml
   ```
