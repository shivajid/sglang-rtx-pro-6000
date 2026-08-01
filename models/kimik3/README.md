# KimiK3

#
The model was released on July 27th, 2026

- 4 Node setup with Benchmark sweeps
- 8 Node setup with Vibench Agentic Benchmark. 

Challenges

KimiK3 has really long reasoning, ensure to 


On 8 Node add the following timeouts, to avoid NCCL Errors

          - name: TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC
            value: "3600"
          - name: NCCL_HEARTBEAT_TIMEOUT_SEC
            value: "3600"

GKE Add Block Selector to be on the same NVLINK Block
      spec:
        nodeSelector:
         cloud.google.com/gce-topology-block: "1acd074d42cd3be9e4486b524db2e9ab"

Other experiments
 - DSpark Configs 
 - Hi Cache
```
python3 -m sglang.launch_server \
            --model-path /data/model \
            --served-model-name moonshotai/Kimi-K3 \
            --tp-size 16 \
            --nnodes ${SIZE} \
            --node-rank ${RANK} \
            --dist-init-addr ${LEADER_HOST}:20000 \
            --host 0.0.0.0 \
            --port 30100 \
            --trust-remote-code \
            --reasoning-parser kimi_k3 \
            --tool-call-parser kimi_k3 \
            --mamba-full-memory-ratio 7.21 \
            --dcp-size 16 \
            --mem-fraction-static 0.80 \
            --enable-hierarchical-cache \
            --page-size 64 \
            --hicache-ratio 2.0 \
            --hicache-io-backend direct \
            --hicache-mem-layout page_first_direct \
            --hicache-write-policy write_through \
            --hicache-storage-prefetch-policy=timeout \
            --enable-cache-report \
            --enable-metrics \
            --watchdog-timeout 3600 \
            2>&1 | tee /gcs-cache/shivajid-sglang_server_${RANK}.log
```

Above hangs and crashes. See the change from `--hicache-io-backend direct` from Kernel used in G4s.

Some config calculators:
mamba-full-memory-ratio values

Avg request length (ISL+OSL)	--mamba-full-memory-ratio
11,264	7.21
32,768	2.48
65,536	1.24
131,072	0.62
262,144	0.31
524,288	0.16
1,048,576	0.078


