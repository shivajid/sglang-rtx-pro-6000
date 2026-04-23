# GKE Cluster Creation Template for SGLang

This README explains how to use the `createCluster_template.sh` script to provision a GKE cluster optimized for SGLang multi-node inference.

## Overview

The `createCluster_template.sh` script automates the creation of a complete infrastructure on Google Cloud Platform (GCP) for running high-performance SGLang serving and benchmarking.

It provisions:
- A custom VPC network with high MTU (8896).
- Two subnets with secondary ranges for Pods and Services.
- Firewall rules for internal communication, SSH, and master-to-node traffic.
- A Cloud Router and NAT gateway for private node internet access.
- An Artifact Registry repository for Docker images.
- A GKE cluster with private nodes and multi-networking enabled.
- A GPU node pool with RTX 6000 Ada GPUs (default 2 nodes, scale to 3).
- A Benchmark Client node pool for running load tests.

## Prerequisites

- **Google Cloud SDK (`gcloud`)**: Must be installed and authenticated.
- **IAM Permissions**: You need permissions to create VPC networks, firewall rules, routers, Artifact Registry repos, and GKE clusters.
- **Project ID**: You must have a valid GCP project ID.

## Usage

1. **Set Required Environment Variables**:
   The script requires `PROJECT_NAME` to be set in your environment. It does not have a default to prevent accidental creation in the wrong project.

   ```bash
   export PROJECT_NAME="your-gcp-project-id"
   ```

2. **Override Optional Variables (Optional)**:
   You can customize the deployment by setting other environment variables before running the script. For example:

   ```bash
   export REGION="us-central1"
   export ZONE="us-central1-a"
   export CLUSTER_NAME="my-custom-sglang-cluster"
   ```

3. **Review CIDR Ranges (CRITICAL)**:
   Open the script and review the default CIDR ranges in section 1. Ensure they do not conflict with any existing networks in your project.

4. **Run the Script**:
   ```bash
   chmod +x createCluster_template.sh
   ./createCluster_template.sh
   ```

## Key Configuration Variables

The following variables can be set in your environment to override the defaults in the script:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `PROJECT_NAME` | **Required** GCP Project ID | *None (Fails if not set)* |
| `REGION` | GCP Region | `europe-west4` |
| `ZONE` | GCP Zone | `europe-west4-b` |
| `NETWORK_NAME` | VPC Network Name | `$USER-g4-vpc-1` |
| `CLUSTER_NAME` | GKE Cluster Name | `$USER-gke-cluster-v2` |
| `GPU_POOL_NAME` | GPU Node Pool Name | `$USER-8x-rtx6000-2nic-pool-v2` |
| `GPU_MACHINE_TYPE` | Machine type for GPU nodes | `g4-standard-384` |
| `GPU_POOL_MIN_NODES` | Minimum number of GPU nodes | `2` |
| `GPU_POOL_MAX_NODES` | Maximum number of GPU nodes | `3` |

*Note: `$USER` expands to the username of the person running the script.*
