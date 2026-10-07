#!/bin/bash
# NVFP4 mixed-precision accuracy variants on BANKING77 (run inside a dev pod).
source /cache/env.sh
cd /cache/code
export TRITON_CACHE_DIR=/cache/triton_probe_on
BASE=/cache/results/probs_main_bf16.npy
LOG=/cache/results/accuracy_nvfp4_variants.log
NVFP4_PARTS=gate_up python bench/accuracy.py --apply fused --steps nvfp4_mlp --baseline $BASE --tag nvfp4_gu >> $LOG 2>&1
NVFP4_LAYERS='^layers\.([89]|[1-4][0-9]|5[0-5])\.mlp$' python bench/accuracy.py --apply fused --steps nvfp4_mlp --baseline $BASE --tag nvfp4_mid >> $LOG 2>&1
echo VARIANTS_DONE >> $LOG
