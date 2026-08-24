"""Language-drift metric (plan W3e.3): windowed VI:EN token ratio over
generations, excluding whitelisted card-name spans.

Metric is the DELTA from the base-model baseline (change in mean VI fraction
and change in collapse rate), not absolute ratios — legitimate English tarot
terminology appears in both baseline and fine-tuned output.

Collapse definition: a 100-token window starting after token 300 whose
VI fraction drops below ``collapse_threshold``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WINDOW = 100          # tokens per window (plan)
COLLAPSE_AFTER = 300  # drift zone start (plan: ~token 300)


def load_whitelist_surfaces(path: Path) -> List[str]:
    wl = json.loads(Path(path).read_text(encoding="utf-8"))
    surfaces = set(wl.get("canonical_names") or [])
    for a in wl.get("aliases") or []:
        surfaces.add(a.get("canonical") or "")
        surfaces.add(a.get("alias") or a.get("surface") or "")
    return sorted({s for s in surfaces if s}, key=len, reverse=True)


def whitelist_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_card_names(text: str, surfaces: Sequence[str]) -> str:
    """Replace every whitelist surface form with a neutral placeholder so
    legitimate English tarot terminology never counts as drift."""
    out = text
    for s in surfaces:
        out = re.sub(re.escape(s), "CARD", out, flags=re.IGNORECASE)
    return out


_EN_TOKEN = re.compile(r"^[A-Za-z]+$")


def classify_tokens(text: str) -> List[str]:
    """Token classes: 'vi', 'en', or ignored (numbers, punctuation)."""
    classes = []
    for tok in re.findall(r"[^\s]+", text):
        core = tok.strip(".,;:!?()[]{}\"'“”…—–")
        if not core or not re.search(r"[A-Za-zÀ-ỹ]", core):
            classes.append("ign")
        elif _EN_TOKEN.match(core):
            # pure-ASCII word: English unless it's a Vietnamese syllable spelled
            # without diacritics is ambiguous — treat as en (conservative for drift)
            classes.append("en")
        elif re.fullmatch(r"[a-zà-ỹđ]+", core.lower()):
            classes.append("vi")
        else:
            classes.append("ign")
    return classes


def window_vi_fractions(classes: Sequence[str], window: int = WINDOW) -> List[float]:
    fracs = []
    for i in range(0, len(classes) - window + 1, window):
        w = classes[i:i + window]
        vi = sum(1 for c in w if c == "vi")
        en = sum(1 for c in w if c == "en")
        fracs.append(vi / (vi + en) if (vi + en) else 0.0)
    return fracs


def analyse_generation(text: str, surfaces: Sequence[str],
                       collapse_threshold: float,
                       window: int = WINDOW) -> dict:
    masked = mask_card_names(text, surfaces)
    classes = classify_tokens(masked)
    fracs = window_vi_fractions(classes, window)
    tail = [f for k, f in enumerate(fracs) if (k * window) >= COLLAPSE_AFTER]
    collapses = sum(1 for f in tail if f < collapse_threshold)
    return {
        "n_tokens": len([c for c in classes if c != "ign"]),
        "mean_vi_frac": (sum(fracs) / len(fracs)) if fracs else 0.0,
        "windows": fracs,
        "collapses": collapses,
    }


def compare(baseline_report: dict, candidate_report: dict) -> dict:
    b_mean = baseline_report["aggregate"]["mean_vi_frac"]
    c_mean = candidate_report["aggregate"]["mean_vi_frac"]
    b_col = baseline_report["aggregate"]["collapse_rate"]
    c_col = candidate_report["aggregate"]["collapse_rate"]
    return {
        "delta_mean_vi_frac": c_mean - b_mean,
        "delta_collapse_rate": c_col - b_col,
        "regressed": bool((c_mean - b_mean) < -0.05 or (c_col - b_col) > 0.02),
        "whitelist_match": (
            baseline_report.get("whitelist_sha256") ==
            candidate_report.get("whitelist_sha256")),
    }


def run(provider, n_gens: int, max_tokens: int, surfaces: Sequence[str],
        collapse_threshold: float, prompts: Optional[Sequence[str]] = None,
        seed: int = 42) -> dict:
    import random

    rng = random.Random(seed)
    default_prompts = [
        "Hãy viết một lời đọc bài tarot ngắn về tình yêu.",
        "Hãy viết một lời đọc bài tarot ngắn về công việc.",
        "Hãy viết một lời đọc bài tarot ngắn về tài chính.",
        "Hãy viết một lời đọc bài tarot ngắn về sức khỏe.",
    ]
    if prompts is None:
        prompts = default_prompts
    else:
        prompts = list(prompts)

    analyses = []
    for i in range(n_gens):
        p = prompts[i % len(prompts)]
        out = provider.generate(p, temperature=0.8, max_tokens=max_tokens)
        analyses.append(analyse_generation(out, surfaces, collapse_threshold))

    n = max(1, len(analyses))
    total_windows = sum(len(a["windows"]) for a in analyses) or 1
    agg = {
        "mean_vi_frac": sum(a["mean_vi_frac"] for a in analyses) / n,
        "collapse_rate": sum(a["collapses"] for a in analyses) / n,
        "mean_tokens": sum(a["n_tokens"] for a in analyses) / n,
    }
    return {
        "provider": provider.name,
        "n_generations": n_gens,
        "max_tokens": max_tokens,
        "collapse_threshold": collapse_threshold,
        "whitelist_sha256": None,  # filled by caller
        "aggregate": agg,
        "per_generation": [
            {k: v for k, v in a.items() if k != "windows"} | {"windows": a["windows"]}
            for a in analyses
        ],
    }
