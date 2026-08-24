#!/usr/bin/env bash
# W7.4: launch llama-server with documented defaults.
# Physical-core detection: nproc reports logical CPUs (wrong under SMT/QEMU);
# prefer lscpu / sysctl, fall back to nproc-1 with an override env var.
set -euo pipefail

MODEL="${TAROT_MODEL:-/tmp/smoke-f16.gguf}"
PORT="${TAROT_PORT:-8079}"
CTX="${TAROT_CTX:-4096}"
HOST="${TAROT_HOST:-127.0.0.1}"

detect_physical_cores() {
  if command -v lscpu >/dev/null 2>&1; then
    local cores
    cores=$(lscpu -b -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
    if [ "$cores" -gt 0 ] 2>/dev/null; then echo "$cores"; return; fi
  fi
  if command -v sysctl >/dev/null 2>&1 && sysctl -n hw.physicalcpu >/dev/null 2>&1; then
    sysctl -n hw.physicalcpu; return
  fi
  echo "$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))"
}

THREADS="${TAROT_THREADS:-$(detect_physical_cores)}"
THREADS_PREFILL="${TAROT_THREADS_PREFILL:-$THREADS}"

echo "[serve.sh] model=$MODEL host=$HOST port=$PORT ctx=$CTX threads=$THREADS prefill=$THREADS_PREFILL"

exec /home/ubuntu/llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  -c "$CTX" \
  -t "$THREADS" \
  -tb "$THREADS_PREFILL" \
  --mmap
