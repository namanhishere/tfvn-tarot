#!/usr/bin/env bash
# W0.1 — Install cmake + build llama.cpp toolchain rootless (SSE4.2 scalar fallback).
#
# Target box: Ubuntu 24.04, CPU WITHOUT AVX2 (SSE4.2 only), no sudo required.
# Strategy: cmake + ninja come from pip wheels into the project venv; llama.cpp is
# built in user space with AVX/AVX2/FMA/F16C explicitly disabled so the binary
# runs on the scalar/SSE4.2 fallback path — deterministic on any host.
#
# Produces: ~/llama.cpp/build/bin/{llama-server,llama-cli,llama-quantize,llama-imatrix}
# Smoke:    llama-cli --version (commit hash), llama-server single completion on a
#           tiny GGUF, llama-quantize quantises that GGUF.
#
# Idempotent: re-running skips completed stages unless --rebuild is passed.
set -euo pipefail

# ------------------------------------------------------------------ config ---
LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_TAG="${LLAMA_CPP_TAG:-b10257}"            # pinned for reproducibility
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
BUILD_DIR="$LLAMA_CPP_DIR/build"
CMAKE_VER="4.4.2"
NINJA_VER="1.13.0"
# Tiny public GGUF for the smoke tests (~90 MB Q4_K_M).
SMOKE_MODEL_URL="${SMOKE_MODEL_URL:-https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf}"
SMOKE_MODEL="${SMOKE_MODEL:-$HOME/.cache/tfvn/smokelm2-135m-q4.gguf}"
SMOKE_OUT="${SMOKE_OUT:-/tmp/tfvn/smoke-q4_0.gguf}"
JOBS="${JOBS:-$(nproc)}"
REBUILD="${REBUILD:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="$REPO_ROOT/.venv/bin/python"
BUILD_INFO="$REPO_ROOT/artifacts/llamacpp_build_info.txt"

# venv provides the cmake launcher + ninja binary; expose them to cmake/ninja
# subprocess lookups without requiring an activated venv.
if [ -d "$REPO_ROOT/.venv/bin" ]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

log() { printf '[W0.1] %s\n' "$*"; }
die() { printf '[W0.1] FATAL: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------- cmake via pip ----
if ! command -v cmake >/dev/null 2>&1; then
  if [ ! -x "$VENV_PY" ]; then
    die "no venv at $VENV_PY — create it first: python3 -m venv .venv"
  fi
  log "installing cmake==$CMAKE_VER + ninja==$NINJA_VER into venv (rootless)"
  "$VENV_PY" -m pip install --quiet "cmake==$CMAKE_VER" "ninja==$NINJA_VER"
fi
CMAKE_BIN="$(command -v cmake || true)"
[ -z "$CMAKE_BIN" ] && [ -x "$REPO_ROOT/.venv/bin/cmake" ] && CMAKE_BIN="$REPO_ROOT/.venv/bin/cmake"
[ -n "$CMAKE_BIN" ] || die "cmake not on PATH; pip install 'cmake' into .venv provides .venv/bin/cmake"
log "cmake: $("$CMAKE_BIN" --version | head -1)"

# --------------------------------------------------------- clone + build ----
if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  log "cloning llama.cpp @ $LLAMA_CPP_TAG"
  git clone --depth 1 --branch "$LLAMA_CPP_TAG" "$LLAMA_CPP_REPO" "$LLAMA_CPP_DIR"
else
  log "llama.cpp already cloned at $LLAMA_CPP_DIR"
fi

if [ "$REBUILD" = "1" ] || [ ! -f "$BUILD_DIR/bin/llama-cli" ]; then
  log "configuring scalar/SSE4.2 build (AVX/AVX2/FMA/F16C disabled)"
  cmake -S "$LLAMA_CPP_DIR" -B "$BUILD_DIR" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF \
    -DGGML_CUDA=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_SERVER=ON
  log "building with $JOBS jobs (this is the slow step on a scalar CPU)"
  cmake --build "$BUILD_DIR" --config Release -j "$JOBS" \
    --target llama-server llama-cli llama-quantize llama-imatrix
else
  log "build present — skipping (pass REBUILD=1 to force)"
fi

BIN="$BUILD_DIR/bin"
for b in llama-server llama-cli llama-quantize llama-imatrix; do
  [ -x "$BIN/$b" ] || die "missing binary: $BIN/$b"
done
log "all four binaries present in $BIN"

# ------------------------------------------------------------- acceptance ---
log "llama-cli --version:"
"$BIN/llama-cli" --version

mkdir -p "$(dirname "$SMOKE_MODEL")" /tmp/tfvn
if [ ! -f "$SMOKE_MODEL" ]; then
  log "downloading smoke model (Q4_K_M, ~90 MB)"
  curl -sSL --fail "$SMOKE_MODEL_URL" -o "$SMOKE_MODEL"
fi

log "smoke: llama-quantize q4_0 on the tiny GGUF"
"$BIN/llama-quantize" "$SMOKE_MODEL" "$SMOKE_OUT" q4_0 >/dev/null
[ -s "$SMOKE_OUT" ] || die "quantised output empty"

log "smoke: llama-server single completion"
PORT=18080
"$BIN/llama-server" -m "$SMOKE_MODEL" -c 512 --port "$PORT" --host 127.0.0.1 >/tmp/tfvn/server.log 2>&1 &
SRV_PID=$!
cleanup() { kill "$SRV_PID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 60); do
  curl -s --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
RESP="$(curl -s --max-time 60 "http://127.0.0.1:$PORT/completion" \
  -d '{"prompt":"The meaning of the Fool card is","n_predict":12,"temperature":0}' || true)"
kill "$SRV_PID" 2>/dev/null || true
trap - EXIT
[ -n "$RESP" ] && [ "$(printf '%s' "$RESP" | python3 -c 'import json,sys;print(bool(json.load(sys.stdin).get("content")))' 2>/dev/null)" = "True" ] \
  || die "llama-server returned no completion — see /tmp/tfvn/server.log"
log "smoke completion OK"

# ------------------------------------------------------------- record -------
mkdir -p "$(dirname "$BUILD_INFO")"
{
  echo "llama.cpp tag: $LLAMA_CPP_TAG"
  echo "repo: $LLAMA_CPP_REPO"
  echo "build dir: $BUILD_DIR"
  "$BIN/llama-cli" --version 2>&1 | head -1
  echo "cmake: $("$CMAKE_BIN" --version | head -1)"
  echo "configure flags: -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "smoke: llama-server single completion OK; llama-quantize q4_0 OK"
} > "$BUILD_INFO"
log "W0.1 complete — toolchain ready; build info in $BUILD_INFO"
