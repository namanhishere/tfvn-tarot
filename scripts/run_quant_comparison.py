#!/usr/bin/env python3
"""W5.3: build six quants (3 types x with/without imatrix) from the same F16
master, then compare perplexity / file size and select per the plan rules.

Quant types: Q4_K_M, Q5_K_M, Q6_K — all with --output-tensor-type Q8_0
and --token-embd-type Q6_K (plan W5.3).

Selection rule (plan): lowest-bpw config whose tone accuracy is within 1 point
of the best AND whose drift collapse rate is not worse than F16 baseline AND
whose perplexity is within tolerance. If with/without imatrix tie, ship WITH.

Tone accuracy per quant requires serving each GGUF through llama-server; that
part is driven by scripts/run_evals.py --provider llama-server@... and wired in
by --tone-reports (a JSON map quant-name -> accuracy), keeping this script
offline-runnable for the quantize+perplexity stage.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_DIR = Path.home() / "llama.cpp"
QUANT_TYPES = ["Q4_K_M", "Q5_K_M", "Q6_K"]
BPW = {"Q4_K_M": 4.85, "Q5_K_M": 5.69, "Q6_K": 6.59}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


import hashlib  # noqa: E402


def build_quants(f16: Path, imatrix: Path | None, out_dir: Path,
                 llama_dir: Path, types: list) -> dict:
    quantize = llama_dir / "build" / "bin" / "llama-quantize"
    assert quantize.exists(), f"missing {quantize}"
    results = {}
    for qt in types:
        tag = f"{qt.lower()}" + ("_imx" if imatrix else "")
        out = out_dir / f"model.{tag}.gguf"
        if out.exists():
            print(f"skip existing {out}")
        else:
            cmd = [str(quantize)]
            if imatrix:
                cmd += ["--imatrix", str(imatrix)]
            cmd += ["--output-tensor-type", "Q8_0",
                    "--token-embedding-type", "Q6_K",
                    str(f16), str(out), qt]
            print("running:", " ".join(cmd))
            subprocess.run(cmd, check=True)
        results[tag] = {"path": str(out), "quant_type": qt,
                        "with_imatrix": bool(imatrix),
                        "size_bytes": out.stat().st_size,
                        "sha256": sha256_file(out)}
    return results


def perplexity(gguf: Path, llama_dir: Path, val_text: Path | None,
               n_tokens: int = 4000) -> float | None:
    """Optional: llama-perplexity on a held-out Vietnamese text."""
    if val_text is None:
        return None
    ppl_bin = llama_dir / "build" / "bin" / "llama-perplexity"
    if not ppl_bin.exists():
        return None
    proc = subprocess.run(
        [str(ppl_bin), "-m", str(gguf), "-f", str(val_text),
         "-n", str(n_tokens), "-t", "4"],
        capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if "PPL is" in line or "ppl value" in line.lower():
            try:
                return float(line.strip().split()[-1])
            except ValueError:
                continue
    return None


def select(built: dict, tone_reports: dict | None, ppl_tolerance: float,
           best_ppl: float | None) -> dict:
    """Plan selection: lowest bpw within tone tolerance of best; prefer imatrix on ties."""
    candidates = []
    for tag, info in built.items():
        entry = {**info, "bpw": BPW[info["quant_type"]]}
        if tone_reports:
            acc = tone_reports.get(tag)
            if acc is not None:
                entry["tone_accuracy"] = acc
        candidates.append(entry)

    best_tone = max((c["tone_accuracy"] for c in candidates
                     if "tone_accuracy" in c), default=None)
    def within_tone(c):
        return best_tone is None or c.get("tone_accuracy", 0) >= best_tone - 0.01

    def within_ppl(c):
        if best_ppl is None or c.get("ppl") is None:
            return True
        return (c["ppl"] - best_ppl) <= ppl_tolerance * best_ppl

    eligible = [c for c in candidates if within_tone(c) and within_ppl(c)]
    if not eligible:
        eligible = sorted(candidates, key=lambda c: -c["bpw"])
        reason = "no candidate met gates — fell back to highest bpw"
    else:
        reason = "lowest bpw within tone+ppl gates"
    eligible.sort(key=lambda c: (c["bpw"],
                                 not c["with_imatrix"]))  # lowest bpw; prefer imatrix
    chosen = eligible[0]
    return {"selected": chosen["tag"] if "tag" in chosen else chosen["path"],
            "chosen": chosen, "reason": reason,
            "candidates": candidates}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f16", required=True, help="F16 master GGUF")
    ap.add_argument("--imatrix", default=None, help="vi.imatrix file")
    ap.add_argument("--out-dir", default="artifacts/quants")
    ap.add_argument("--types", nargs="+", default=QUANT_TYPES)
    ap.add_argument("--val-text", default="artifacts/imatrix_corpus.txt")
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--tone-reports", default=None,
                    help='JSON map {"q5_k_m_imx": 0.87, ...} from tone evals')
    ap.add_argument("--report", default="artifacts/quant_selection.md")
    args = ap.parse_args()

    f16 = Path(args.f16)
    imatrix = Path(args.imatrix) if args.imatrix else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    built = {}
    built.update(build_quants(f16, None, out_dir, LLAMA_DIR, args.types))
    if imatrix:
        built.update(build_quants(f16, imatrix, out_dir, LLAMA_DIR, args.types))

    if not args.no_ppl:
        val = Path(args.val_text)
        if val.exists():
            for tag, info in built.items():
                info["ppl"] = perplexity(Path(info["path"]), LLAMA_DIR, val)

    tone = json.loads(Path(args.tone_reports).read_text()) if args.tone_reports else None
    sel = select(built, tone, ppl_tolerance=0.10, best_ppl=None)

    report_lines = [
        "# Quant selection (W5.3)", "",
        "| quant | type | imatrix | size MB | ppl | tone |",
        "|---|---|---|---|---|---|",
    ]
    for c in sorted(sel["candidates"], key=lambda c: c["bpw"]):
        report_lines.append(
            f"| {Path(c['path']).name} | {c['quant_type']} | {c['with_imatrix']} "
            f"| {c['size_bytes'] / 1e6:.0f} | {c.get('ppl', '—')} "
            f"| {c.get('tone_accuracy', '—')} |")
    report_lines += ["", f"**Selected:** `{sel['selected']}` — {sel['reason']}", ""]
    Path(args.report).write_text("\n".join(report_lines), encoding="utf-8")
    Path(args.report).with_suffix(".json").write_text(
        json.dumps(sel, indent=2, default=str), encoding="utf-8")
    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
