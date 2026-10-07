#!/bin/bash
# One-time environment setup inside a clef-dev pod (idempotent).
# - extra python packages installed with --user into the node-local cache
#   (PYTHONUSERBASE=/cache/pyuser), reusing the image's torch
# - Clef weights downloaded to /cache/models/clef
# - backbone-only symlink view (no joint head files) for SGLang/vLLM
set -euo pipefail
mkdir -p /cache/models /cache/hf /cache/results /cache/pyuser
cat > /cache/env.sh <<'EOF'
export PYTHONUSERBASE=/cache/pyuser
export PATH=/cache/pyuser/bin:$PATH
export HF_HOME=/cache/hf
export MODEL_PATH=/cache/models/clef
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF
source /cache/env.sh
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit --format=csv
which python; python --version
PKGS='transformers==5.10.2 flash-linear-attention fastapi uvicorn[standard] orjson pillow torchao accelerate aiohttp numpy huggingface_hub[hf_xet] scikit-learn pandas pyarrow'
python -m pip install --user -q $PKGS || python -m pip install --user --break-system-packages -q $PKGS
python - <<'EOF'
import torch, transformers, importlib
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("transformers", transformers.__version__)
for m in ("fla", "torchao", "triton", "causal_conv1d", "flash_attn"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__, str(e)[:120])
EOF
if [ ! -f /cache/models/clef/.download_complete ]; then
  python - <<'EOF'
import time
from huggingface_hub import snapshot_download
t = time.time()
snapshot_download("Cloudflare/clef", local_dir="/cache/models/clef", max_workers=16)
print(f"download took {time.time()-t:.1f}s")
EOF
  touch /cache/models/clef/.download_complete
fi
mkdir -p /cache/models/clef-backbone
for f in /cache/models/clef/*; do
  b=$(basename "$f")
  case "$b" in joint_head*|joint_schema_model.py) ;; *) ln -sfn "$f" "/cache/models/clef-backbone/$b" ;; esac
done
du -sh /cache/models/clef
echo SETUP_DONE
