#!/bin/bash
# Show the latest trial of every running job and the finished jobs.
cd "$(dirname "$0")/.."
for f in results/log_mlp100_*.txt; do [ "$f" = results/log_mlp100_status.txt ] && continue; echo "$f: $(grep -E 'trial|E_Pi|E2' "$f" | tail -1)"; done
echo "--- finished:"; cat results/log_mlp100_status.txt 2>/dev/null
