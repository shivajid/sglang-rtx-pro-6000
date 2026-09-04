# DeepSeek-V4-Flash Weight Disk Setup

This directory provides the workflow to ingest `deepseek-ai/DeepSeek-V4-Flash-0731` weights onto a GCP Hyperdisk volume (`dsv4-flash-hyperdisk-balanced`) and serve via SGLang with zero boot disk pressure.

## Step 1: Provision the Disk

The 500GB `hyperdisk-balanced` disk has been provisioned:
```bash
gcloud compute disks create dsv4-flash-hyperdisk-balanced \
    --project=northam-ce-mlai-tpu \
    --zone=us-east5-a \
    --type=hyperdisk-balanced \
    --size=500GB
```

## Step 2: Ingest Model Weights (Writer Mode)

1. Delete any failing/pending serving StatefulSet to clear the node:
   ```bash
   kubectl delete statefulset sglang-dsv4-flash-1node --ignore-not-found
   ```

2. Apply the ReadWriteOnce PV and PVC:
   ```bash
   kubectl apply -f models/DeepSeekV4-Flash-0731/model_weight_disk/dsv4-flash-hdml-writer.yaml
   ```

3. Run the downloader job:
   ```bash
   kubectl apply -f models/DeepSeekV4-Flash-0731/model_weight_disk/dsv4-flash-downloader-job.yaml
   ```

4. Monitor the download:
   ```bash
   kubectl logs -f job/dsv4-flash-hdml-downloader
   ```

5. Once the job succeeds, release the writer lock:
   ```bash
   kubectl delete job dsv4-flash-hdml-downloader
   kubectl delete pvc dsv4-flash-hdml-writer-pvc
   kubectl delete pv dsv4-flash-hdml-pv
   ```

## Step 3: Mount in ReadOnlyMany Mode & Serve

1. Apply the ReadOnlyMany PV and PVC:
   ```bash
   kubectl apply -f models/DeepSeekV4-Flash-0731/model_weight_disk/dsv4-flash-hdml-ro.yaml
   ```

2. Deploy the SGLang single-node serving workload:
   ```bash
   kubectl apply -f models/DeepSeekV4-Flash-0731/sglang-dsv4-flash-1node-hdml.yaml
   ```
