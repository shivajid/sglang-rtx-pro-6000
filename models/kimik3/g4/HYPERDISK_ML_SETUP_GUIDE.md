# Google Cloud Hyperdisk ML Setup & Multi-Node Weight Ingestion Guide

This document outlines the end-to-end architecture and step-by-step procedures used to provision **Google Cloud Hyperdisk ML**, ingest 1.5 TB of **Moonshot AI Kimi-K3** weights from Google Cloud Storage, and attach the volume simultaneously in `ReadOnlyMany` mode across a multi-node SGLang serving cluster.

---

## 1. Architecture Overview

**Google Cloud Hyperdisk ML** provides high-throughput, read-only multi-attach block storage specifically engineered for large language model inference clusters. It allows tens or hundreds of G4 / GPU compute instances to attach to the same shared disk concurrently, achieving multi-gigabyte/second sustained read throughput per node without network file system bottlenecks.

```
+---------------------------------------------------------------------------------------------------+
|                                 PHASE 1: WRITE & POPULATE (1-to-1)                                |
|                                                                                                   |
|  [ GCS Bucket: gs://<YOUR_CKPT_BUCKET>/ ]                                                         |
|                              | (gcloud storage rsync)                                             |
|                              v                                                                    |
|  [ Downloader Job Pod: kimik3-hdml-downloader ] (<YOUR_G4_NODEPOOL> Node)                         |
|                              | (ReadWriteOnce)                                                    |
|                              v                                                                    |
|  [ Hyperdisk ML Disk: kimik3-hyperdisk-ml (2,000 GB in <YOUR_ZONE>) ]                              |
+---------------------------------------------------------------------------------------------------+
                                               |
                                               v (Unmount Writer Claim)
+---------------------------------------------------------------------------------------------------+
|                               PHASE 2: READ-ONLY MULTI-ATTACH (1-to-Many)                         |
|                                                                                                   |
|                    [ Hyperdisk ML Disk: kimik3-hyperdisk-ml (2,000 GB) ]                          |
|                                               |                                                   |
|                        +----------------------+----------------------+                            |
|                        | (ReadOnlyMany - Shared Block Device)        |                            |
|                        v                                             v                            |
|  [ SGLang Pod 0 (G4 Node 0) ]                                [ SGLang Pod 1 (G4 Node 1) ]         |
|  Mount: /data/model (ro)                                     Mount: /data/model (ro)              |
|                                                                                                   |
|  [ SGLang Pod 2 (G4 Node 2) ]                                [ SGLang Pod 3 (G4 Node 3) ]         |
|  Mount: /data/model (ro)                                     Mount: /data/model (ro)              |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Prerequisites & Infrastructure Setup

* **Google Cloud Project:** `<YOUR_PROJECT_ID>`
* **Region / Zone:** `<YOUR_REGION>` / `<YOUR_ZONE>` (e.g., `us-east5` / `us-east5-a`)
* **GKE Cluster:** `<YOUR_CLUSTER_NAME>`
* **GPU Node Pool:** `<YOUR_G4_NODEPOOL>` (`g4-standard-384` instances, SM120)
* **GCS Checkpoint Bucket:** `gs://<YOUR_CKPT_BUCKET>/` containing:
  * 96 Safetensor model shards (`model-00001-of-00096.safetensors` to `model-00096-of-00096.safetensors`, ~1.5 TB)
  * Tokenizer files (`tiktoken.model`, `tokenizer_config.json`, `config.json`)

> [!IMPORTANT]
> Google Cloud restricts Hyperdisk ML in `ReadOnlyMany` mode to accelerator/optimized machine families (such as `g4-standard-384`, `a3`, `c3`/`c4`). Standard `n2-standard-*` CPU nodes cannot attach Hyperdisk ML in `ReadOnlyMany` mode.

---

## 3. Step-by-Step Implementation

### Step 1: Provision the GCP Hyperdisk ML Disk

Create the 2 TB Hyperdisk ML disk in the same zone as the G4 GPU nodes:

```bash
gcloud compute disks create kimik3-hyperdisk-ml \
    --project=<YOUR_PROJECT_ID> \
    --zone=<YOUR_ZONE> \
    --type=hyperdisk-ml \
    --size=2000GB \
    --description="Shared model weight storage for Kimi-K3 SGLang Serving"
```

---

### Step 2: Create PersistentVolume & Claim in `ReadWriteOnce` Mode

Apply `model_weight_disk/kimik3-hdml-writer.yaml` to bind the Hyperdisk ML disk in write mode:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: kimik3-hdml-pv
spec:
  storageClassName: ""
  capacity:
    storage: 2000Gi
  accessModes:
    - ReadWriteOnce
  claimRef:
    namespace: default
    name: kimik3-hdml-writer-pvc
  csi:
    driver: pd.csi.storage.gke.io
    volumeHandle: projects/<YOUR_PROJECT_ID>/zones/<YOUR_ZONE>/disks/kimik3-hyperdisk-ml
    fsType: ext4
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: kimik3-hdml-writer-pvc
  namespace: default
spec:
  storageClassName: ""
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 2000Gi
  volumeName: kimik3-hdml-pv
```

```bash
kubectl apply -f model_weight_disk/kimik3-hdml-writer.yaml
```

---

### Step 3: Run the Downloader Job from GCS to Hyperdisk ML

Deploy `model_weight_disk/kimik3-downloader-job.yaml` targeting a node in your GPU pool:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: kimik3-hdml-downloader
  namespace: default
spec:
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"
    spec:
      restartPolicy: OnFailure
      serviceAccountName: sglang-sa
      nodeSelector:
        cloud.google.com/gke-nodepool: <YOUR_G4_NODEPOOL>
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: downloader
        image: google/cloud-sdk:slim
        command:
        - /bin/bash
        - -c
        - |
          set -e
          echo "=== Starting Kimi-K3 sync to Hyperdisk ML ==="
          mkdir -p /data/model
          echo "Syncing from gs://<YOUR_CKPT_BUCKET>/ to /data/model/ ..."
          gcloud storage rsync -r gs://<YOUR_CKPT_BUCKET>/ /data/model/
          echo "Sync complete! Checking disk contents:"
          ls -lh /data/model
          echo "Disk usage:"
          df -h /data/model
          echo "=== Finished successfully ==="
        volumeMounts:
        - name: model-disk
          mountPath: /data/model
      volumes:
      - name: model-disk
        persistentVolumeClaim:
          claimName: kimik3-hdml-writer-pvc
```

```bash
kubectl apply -f model_weight_disk/kimik3-downloader-job.yaml
kubectl logs -f job/kimik3-hdml-downloader
```

---

### Step 4: Release Writer Mode & Recreate PV/PVC in `ReadOnlyMany` Mode

Once the download job completes successfully, delete the write resources so the disk can transition to shared read-only mode:

```bash
# 1. Delete Downloader Job
kubectl delete job kimik3-hdml-downloader

# 2. Delete Writer PV & PVC
kubectl delete pvc kimik3-hdml-writer-pvc
kubectl delete pv kimik3-hdml-pv
```

Apply `model_weight_disk/kimik3-hdml-ro.yaml` with `accessModes: [ReadOnlyMany]` and `readOnly: true`:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: kimik3-hdml-ro-pv
spec:
  storageClassName: ""
  capacity:
    storage: 2000Gi
  accessModes:
    - ReadOnlyMany
  claimRef:
    namespace: default
    name: kimik3-hdml-pvc
  csi:
    driver: pd.csi.storage.gke.io
    volumeHandle: projects/<YOUR_PROJECT_ID>/zones/<YOUR_ZONE>/disks/kimik3-hyperdisk-ml
    fsType: ext4
    readOnly: true
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: kimik3-hdml-pvc
  namespace: default
spec:
  storageClassName: ""
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 2000Gi
  volumeName: kimik3-hdml-ro-pv
```

```bash
kubectl apply -f model_weight_disk/kimik3-hdml-ro.yaml
```

---

### Step 5: Mount `kimik3-hdml-pvc` across SGLang Server StatefulSet Pods

In `g4_4node_kimik3.yaml`, attach `kimik3-hdml-pvc` to each pod:

```yaml
        volumeMounts:
        - name: model-weights
          mountPath: /data/model
          readOnly: true
      volumes:
      - name: model-weights
        persistentVolumeClaim:
          claimName: kimik3-hdml-pvc
          readOnly: true
```

Launch the 4-node SGLang server StatefulSet:
```bash
kubectl apply -f g4_4node_kimik3.yaml
```

---

## 4. Verification & Performance Results

### Weight Loading Speed:
* **Total Shards:** 96 Safetensor shards (1.53 TB total).
* **Load Time:** **42.8 seconds total** across all 32 GPUs simultaneously.
* **Loading Rate:** **2.97 shards/second** (~35.7 GB/s aggregate read bandwidth).
* **Warm Cache Initialization:** Subsequent runs and JIT warming completed in <1.2 seconds.

### Disk Verification Command:
```bash
kubectl exec sglang-kimi-k3-g4-0 -- ls -lh /data/model
```
```text
-rw-r--r-- 1 root root  16G Aug 15 20:00 model-00001-of-00096.safetensors
...
-rw-r--r-- 1 root root  16G Aug 15 20:00 model-00096-of-00096.safetensors
-rw-r--r-- 1 root root  24M Aug 15 20:00 tiktoken.model
-rw-r--r-- 1 root root 2.1K Aug 15 20:00 tokenizer_config.json
-rw-r--r-- 1 root root 8.5K Aug 15 20:00 config.json
```
