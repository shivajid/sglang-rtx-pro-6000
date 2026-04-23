# Technical Specifications: GCP G4 (NVIDIA RTX PRO 6000 Blackwell)

Technical reference for Google Cloud G4 instances featuring NVIDIA Blackwell SM120 GPUs and the Titanium offload architecture.

## NVIDIA RTX PRO 6000 Blackwell (SM120)

| Hardware Component | Specification |
|-------------------|---------------|
| **Architecture** | Blackwell |
| **Compute Capability** | 12.0 (SM120) |
| **CUDA Cores** | 24,064 |
| **Tensor Cores** | 752 (5th Generation) |
| **RT Cores** | 188 (4th Generation) |
| **GPU Memory** | 96 GB GDDR7 with ECC |
| **Memory Bandwidth** | 1,792 GB/s |
| **Memory Interface** | 384-bit |
| **Interconnect** | PCIe Gen 5.0 x16 |
| **TDP** | 300W |

### Precision & Tensor Core Throughput
The SM120 architecture introduces native hardware support for 4-bit floating point (NVFP4) operations.

- **NVFP4 (E2M1)**: 1 sign bit, 2 exponent bits, 1 mantissa bit.
- **Micro-Tensor Scaling**: Uses two-level scaling (16-element blocks with FP8 scale factors + global FP32 scale) to manage dynamic range.
- **Supported Formats**: FP4, FP6, FP8, INT8, FP16, BF16, TF32, FP32.
- **Peak Performance**: ~4,000 TOPS (FP4 with sparsity).

## Titanium Infrastructure Architecture

G4 instances utilize the Titanium system to offload infrastructure management tasks to dedicated silicon.

### Networking
- **Titanium Offload Processors (TOPs)**: Dedicated chips for VPC networking and packet processing.
- **Throughput**: Up to 400 Gbps network bandwidth.
- **Protocol Support**: Offloads Andromeda virtual networking and encapsulation.
- **Security**: Line-rate encryption and hardware-based root of trust.

### Storage
- **Titanium Local SSD**: Up to 12 TiB capacity.
- **I/O Path**: Hardware-accelerated data path to Local SSD, bypassing host CPU for storage operations.

## Instance Configurations

| Instance Type | GPUs | vCPUs (AMD Turin) | System Memory | Network Bandwidth |
|---------------|------|-------------------|---------------|-------------------|
| `g4-standard-48` | 1 | 48 | 192 GB | 50 Gbps |
| `g4-standard-96` | 2 | 96 | 384 GB | 100 Gbps |
| `g4-standard-192`| 4 | 192 | 768 GB | 200 Gbps |
| `g4-standard-384`| 8 | 384 | 1.5 TB | 400 Gbps |

## SGLang Implementation Notes

- **VRAM Utilization**: 96GB per GPU allows for high-density KV cache allocation.
- **NVFP4 Serving**: SGLang leverages Blackwell's 5th Gen Tensor Cores for 4-bit inference.
- **Distributed Topology**: Uses PCIe Gen 5 for GPU-to-GPU communication during Tensor Parallel (TP) and Pipeline Parallel (PP) operations.
