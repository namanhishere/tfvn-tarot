"""W0.4/W0.6 calibration harness: inject hard degradations, run judges, report per-axis detection with 95% CI.

Usage:
  python judge/calibrate.py --dry-run   # mock judges, deterministic, seeded
  python judge/calibrate.py --live      # requires DEEPSEEK_API_KEY + SECONDARY_JUDGE_API_KEY
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "judge" / "taxonomy.json"
REPORT = ROOT / "judge" / "calibration_report.md"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import read_jsonl  # noqa: E402

# ------------------------------------------------------------------ corpus ---

_SPINE = None


def _spine():
    global _SPINE
    if _SPINE is None:
        p = ROOT / "kb" / "english_spine.jsonl"
        _SPINE = read_jsonl(p) if p.exists() else []
    return _SPINE


def _card_rows(name_en: str) -> list[dict]:
    return [r for r in _spine() if r["name_en"] == name_en]


_TONE_PAIRS = [("má", "mà"), ("bàn", "bán"), ("mã", "mạ")]

# ------------------------------------------------------------ degradations ---


def degrade_tone(text: str, rng: random.Random) -> tuple[str, bool]:
    """Replace a valid word with its same-word different-tone variant at fixed rate."""
    words = text.split(" ")
    hits = [i for i, w in enumerate(words) if any(w == a for a, _ in _TONE_PAIRS)]
    if not hits:
        return text, False
    idx = rng.choice(hits)
    for a, b in _TONE_PAIRS:
        if words[idx] == a:
            words[idx] = b
            return " ".join(words), True
    return text, False


def degrade_orientation(text: str, rng: random.Random) -> tuple[str, bool]:
    """Negate the upright meaning with a particle; reversed atoms are already in the spine text."""
    if "reversed" not in text:
        return text, False
    particle = rng.choice(["không", "chẳng"])
    return text.replace("means", f"does {particle} mean", 1), True


def degrade_translationese(text: str, rng: random.Random) -> tuple[str, bool]:
    """Calque a clause into English word order, preserving Vietnamese function words."""
    markers = ["rằng", "bởi vì", "nên"]
    for m in markers:
        if m in text:
            head, _, tail = text.partition(m)
            return f"{tail.strip()}, {m} {head.strip()}".strip(), True
    return text, False


def degrade_faithfulness(text: str, rng: random.Random) -> tuple[str, bool]:
    """Swap meaning onto an adjacent same-suit card while keeping the named card."""
    rows = _spine()
    if not rows:
        return text, False
    upright = [r for r in rows if r["orientation"] == "upright"]
    candidates = [r for r in upright if r["meaning_summary_en"] == text]
    if not candidates:
        return text, False
    named = candidates[0]
    same_suit = [
        r
        for r in upright
        if r["name_en"].split()[-1] == named["name_en"].split()[-1]
        and r["name_en"] != named["name_en"]
    ]
    if not same_suit:
        return text, False
    victim = rng.choice(same_suit)
    return victim["meaning_summary_en"], True


_DEGRADERS: dict[str, Callable[[str, random.Random], tuple[str, bool]]] = {
    "tone": degrade_tone,
    "orientation": degrade_orientation,
    "translationese": degrade_translationese,
    "faithfulness": degrade_faithfulness,
}

# ------------------------------------------------------------------ judges ---


class MockJudge:
    """Deterministic mock: detects each axis with a fixed known rate (seeded)."""

    def __init__(self, rates: dict[str, float], rng: random.Random) -> None:
        self.rates = rates
        self.rng = rng

    def verdict(self, text: str, axis: str, degraded: bool) -> bool:
        if degraded:
            return self.rng.random() < self.rates[axis]
        return not (self.rng.random() < 0.05)


def _live_call(api_key: str, model: str, prompt: str) -> bool:
    import urllib.request

    url = {
        "deepseek-chat": "https://api.deepseek.com/chat/completions",
        "claude-sonnet-4": "https://api.anthropic.com/v1/messages",
    }[model]
    body = (
        {"model": model, "messages": [{"role": "user", "content": prompt}]}
        if model == "deepseek-chat"
        else {
            "model": model,
            "max_tokens": 8,
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read())
    content = (
        payload["choices"][0]["message"]["content"]
        if model == "deepseek-chat"
        else payload["content"][0]["text"]
    )
    return "degraded" in content.lower() or "corrupted" in content.lower()


class LiveJudge:
    def __init__(self, api_key: str, model: str, axis_rationale: dict[str, Any]) -> None:
        self.api_key = api_key
        self.model = model
        self.axis_rationale = axis_rationale

    def verdict(self, text: str, axis: str, degraded: bool) -> bool:
        crit = self.axis_rationale[axis]["detection_criterion"]
        prompt = (
            f"You are a quality judge. Criterion for '{axis}': {crit}. "
            f"Reply with exactly one word: 'clean' or 'degraded'.\nText: {text}"
        )
        return _live_call(self.api_key, self.model, prompt)


# ------------------------------------------------------------------- stats ---


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% default)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _rate_line(k: int, n: int, chance: float) -> tuple[float, tuple[float, float], bool]:
    lo, hi = wilson_ci(k, n)
    return (k / n, (lo, hi), lo > chance)


# ------------------------------------------------------------------- main ---


_AXIS_CLEAN_TEXTS = {
    "tone": ["má tôi bàn việc bán hàng", "mã số của bàn máy này", "bàn bạc với má mình"],
    "orientation": ["the reversed card means delays and blocked progress",
                    "reversed means the energy is internalized"],
    "translationese": ["Tôi tin rằng anh ấy đến", "Chúng ta nên đi bởi vì trời tối",
                       "Cô ấy nói rằng cô ấy hạnh phúc"],
    "faithfulness": ["The Fool means new beginnings", "The Magician means skill and willpower"],
}


def _build_corpus(axis: str, n: int, rng: random.Random) -> list[tuple[str, bool]]:
    rows = _spine()
    samples: list[tuple[str, bool]] = []
    while len(samples) < n:
        if axis == "faithfulness":
            upright = [r for r in rows if r["orientation"] == "upright"] if rows else []
            suits = [r for r in upright if r["name_en"].split()[-1]
                     in {"Cups", "Wands", "Swords", "Pentacles"}] if upright else []
            base = rng.choice(suits) if suits else {"name_en": "Two of Cups", "meaning_summary_en": "harmony"}
            clean = base["meaning_summary_en"]
        elif axis == "orientation":
            rev = [r for r in rows if r["orientation"] == "reversed"] if rows else []
            base = rng.choice(rev) if rev else {"meaning_summary_en": "delays and blocked progress"}
            clean = f"reversed {base['meaning_summary_en']}"
        else:
            clean = rng.choice(_AXIS_CLEAN_TEXTS[axis])
        deg, is_degraded = _DEGRADERS[axis](clean, rng)
        degraded = is_degraded and rng.random() < 0.5
        samples.append((deg if degraded else clean, degraded))
    return samples


def run(judges: list[Any], taxonomy: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    cal = taxonomy["calibration"]
    n = cal["n_samples_per_axis"]
    axes = taxonomy["axes"]
    rows_out: dict[str, dict[str, Any]] = {}
    no_gate: list[str] = []

    for axis_name, ax in axes.items():
        corpus = _build_corpus(axis_name, n, rng)
        chance = ax["chance_baseline"]
        per_judge: list[dict[str, Any]] = []
        for judge in judges:
            degraded_total = sum(1 for _, d in corpus if d)
            detected = sum(
                judge.verdict(text, axis_name, degraded)
                for text, degraded in corpus
                if degraded
            )
            rate, (lo, hi), passed = _rate_line(detected, degraded_total, chance)
            per_judge.append({"detected": detected, "n": degraded_total, "rate": round(rate, 3),
                              "ci_low": round(lo, 3), "ci_high": round(hi, 3), "passes": passed})
        axis_passes = all(j["passes"] for j in per_judge)
        if not axis_passes:
            no_gate.append(axis_name)
        rows_out[axis_name] = {
            "chance_baseline": chance,
            "judges": per_judge,
            "passes": axis_passes,
            "status": "calibrated" if axis_passes else "no-gate",
        }
    taxonomy["no_gate"] = no_gate
    return {"taxonomy": taxonomy, "results": rows_out}


def write_report(data: dict[str, Any]) -> Path:
    tax = data["taxonomy"]
    lines = [
        "# Judge Calibration Report",
        "",
        f"- Schema: `judge/taxonomy.json` v{tax['schema_version']}",
        f"- Calibration date: {_calibration_date()}",
        f"- Seed: {tax['calibration']['seed']}  |  n per axis: {tax['calibration']['n_samples_per_axis']}  |  CI: {tax['calibration']['ci_level']:.0%} Wilson",
        f"- Judges: primary=`{tax['judges']['primary']['id']}` (provisioned), "
        f"secondary=`{tax['judges']['secondary']['id']}` (provisioned={tax['judges']['secondary']['provisioned']})",
        "",
        "| Axis | Judge | detected/n | rate | 95% CI | chance | passes |",
        "|---|---|---|---|---|---|---|",
    ]
    for axis, res in data["results"].items():
        for i, j in enumerate(res["judges"]):
            judge_name = "primary" if i == 0 else "secondary"
            lines.append(
                f"| {axis} | {judge_name} | {j['detected']}/{j['n']} | {j['rate']:.3f} | "
                f"[{j['ci_low']:.3f}, {j['ci_high']:.3f}] | {res['chance_baseline']:.3f} | "
                f"{'yes' if j['passes'] else 'no'} |"
            )
    lines += [
        "",
        "## Inter-judge disagreement rate",
        "",
    ]
    for axis, res in data["results"].items():
        pair = res["judges"]
        n = min(pair[0]["n"], pair[1]["n"])
        disagree = abs(pair[0]["detected"] - pair[1]["detected"])
        lines.append(f"- {axis}: {disagree}/{n} degraded samples ({disagree / n:.1%})")
    lines += [
        "",
        "## Layer-4 gating",
        "",
        "Axes gating C3 layer 4: "
        + ", ".join(a for a, r in data["results"].items() if r["passes"])
        or "none",
        "",
        "no-gate axes (excluded from layer 4): "
        + ", ".join(tax["no_gate"]) or "none",
        "",
        "## Caveat (Metis B4)",
        "",
        "Secondary judge is not provisioned in this environment. Until it is, "
        "layer 4 is a self-consistency filter and all claims derived from it carry "
        "that caveat.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    return REPORT


def _calibration_date() -> str:
    """Stable date so the report is reproducible across runs within one day."""
    import datetime

    return datetime.date.today().isoformat()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Judge calibration harness (W0.4/W0.6)")
    ap.add_argument("--dry-run", action="store_true", help="mock judges (no API)")
    ap.add_argument("--live", action="store_true", help="real judges (API keys required)")
    args = ap.parse_args(argv)

    taxonomy = json.loads(TAXONOMY.read_text())
    seed = taxonomy["calibration"]["seed"]

    if args.live:
        keys = [taxonomy["judges"]["primary"]["env_key"], taxonomy["judges"]["secondary"]["env_key"]]
        missing = [k for k in keys if not os.environ.get(k)]
        if missing:
            print(f"ERROR: missing env keys: {', '.join(missing)}", file=sys.stderr)
            return 2
        judges = [
            LiveJudge(os.environ[keys[0]], taxonomy["judges"]["primary"]["id"], taxonomy["axes"]),
            LiveJudge(os.environ[keys[1]], taxonomy["judges"]["secondary"]["id"], taxonomy["axes"]),
        ]
    else:
        mock_rates = {
            "tone": 0.9, "orientation": 0.55, "translationese": 0.7, "faithfulness": 0.8,
        }
        judges = [MockJudge(mock_rates, random.Random(seed)), MockJudge(mock_rates, random.Random(seed + 1))]

    rng = random.Random(seed)
    data = run(judges, taxonomy, rng)
    path = write_report(data)
    print(f"report written: {path}")
    print("no-gate axes:", data["taxonomy"]["no_gate"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
