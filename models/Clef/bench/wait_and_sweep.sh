#!/bin/bash
# Wait for a Clef server's /health, then run the HTTP ablation ladder (sweep_clef.sh).
# Usage: bash wait_and_sweep.sh <url> <outfile> <logfile> [configs...]
URL=$1; OUT=$2; LOG=$3; shift 3
cd /work
python - "$URL" <<'EOF' || { echo "server not healthy" > "$LOG"; exit 1; }
import sys, time, urllib.request
url = sys.argv[1] + "/health"
for _ in range(400):
    try:
        if urllib.request.urlopen(url, timeout=5).status == 200:
            sys.exit(0)
    except Exception:
        pass
    time.sleep(3)
sys.exit(1)
EOF
bash sweep_clef.sh "$URL" "$OUT" "$@" > "$LOG" 2>&1
