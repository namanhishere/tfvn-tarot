#!/usr/bin/env python3
"""W5.1: merge LoRA adapter into base weights and export an F16 GGUF master.

Plan rules:
  - merge in bf16 on CC>=8 GPUs; NEVER merge on a quantised base
  - retain the F16 GGUF as the master for imatrix vs no-imatrix comparison
  - record checksum

Works CPU-only (float32 merge -> f16 GGUF) so it is verifiable locally with the
0.6B smoke model.

Usage:
  python3 scripts/export_gguf.py \
      --base Qwen/Qwen3-1.7B --adapter artifacts/sft_r32/best \
      --out artifacts/vn-tarot-f16.gguf
Smoke:
  python3 scripts/export_gguf.py --base Qwen/Qwen3-0.6B \
      --adapter /tmp/smoke_run/smoke_final --out /tmp/smoke-f16.gguf --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_DIR = Path.home() / "llama.cpp"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merged-dir", default=None,
                    help="where to keep the merged HF weights (default alongside out)")
    ap.add_argument("--llama-dir", default=str(LLAMA_DIR))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    merged_dir = Path(args.merged_dir or out.with_suffix("").with_suffix("") or "artifacts/merged")
    merged_dir = Path(args.merged_dir or (out.parent / "merged-hf"))
    merged_dir.mkdir(parents=True, exist_ok=True)

    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    dtype = torch.bfloat16 if major >= 8 else torch.float32
    print(f"merge dtype: {dtype} (CC {major} gate)")

    print(f"loading base {args.base} ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    print(f"loading adapter {args.adapter} ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging ...")
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    convert = Path(args.llama_dir) / "convert_hf_to_gguf.py"
    assert convert.exists(), f"missing {convert}"
    ftype = "f16" if dtype != torch.float32 else "f32"
    cmd = [sys.executable, str(convert), str(merged_dir),
           "--outfile", str(out), "--outtype", ftype]
    print("running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    digest = sha256_file(out)
    meta = {
        "gguf": str(out), "sha256": digest,
        "base": args.base, "adapter": args.adapter,
        "merge_dtype": str(dtype), "ftype": ftype,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True, cwd=ROOT).stdout.strip(),
    }
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    # load check via llama-cli is done by the caller (W5 acceptance);
    # here we only assert non-trivial size
    size_mb = out.stat().st_size / 1e6
    assert size_mb > 50 or args.smoke, f"GGUF suspiciously small: {size_mb:.1f} MB"
    return 0


if __name__ == "__main__":
    sys.exit(main())
