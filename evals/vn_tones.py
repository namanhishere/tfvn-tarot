"""Vietnamese tone minimal pairs — quantisation gate (plan W5/M4, built here W3e.2).

Mechanically derived from authentic KB prose: take a real sentence, blank one
syllable, and ask which tone-spelling is correct. The correct option is the
original token (gold label is mechanical, no frontier involvement). Distractors
are the same syllable under different tones, each verified against a corpus
vocabulary so every option is a *real* Vietnamese syllable (plan requirement:
"dictionary lookup used to validate perturbed syllables").

Tone engine works on NFD: tone marks (grave 0300, acute 0301, hook 0309,
tilde 0303, dot-below 0323) are stripped/re-inserted independently of vowel
marks (circumflex 0302, breve 0306, horn 031B), then NFC-recomposed.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# combining tone marks, mapped to our canonical tone ids
_TONE_MARKS = {"\u0300": "huyen", "\u0301": "sac", "\u0309": "hoi",
               "\u0303": "nga", "\u0323": "nang"}
_VOWEL_GLYPHS = set("aăâeêioôơuưyAĂÂEÊIOÔƠUƯY")


def decompose(syllable: str) -> tuple:
    """Return (base_without_tone_marks, tone_id|None, tone_mark_index|None).

    ``base`` keeps vowel marks (â ê ô...) but no tone diacritic, NFC-form.
    """
    nfd = unicodedata.normalize("NFD", syllable)
    out: List[str] = []
    tone: Optional[str] = None
    tone_idx: Optional[int] = None
    for ch in nfd:
        t = _TONE_MARKS.get(ch)
        if t:
            tone = t
            tone_idx = None  # resolved later relative to stripped string
        else:
            out.append(ch)
    base = unicodedata.normalize("NFC", "".join(out))
    return base, tone, tone_idx


def apply_tone(base: str, tone: Optional[str]) -> str:
    """Attach ``tone`` to ``base`` (no tone marks). ngang (None) -> unchanged."""
    if tone is None:
        return base
    nfd = unicodedata.normalize("NFD", base)
    # insertion point: first vowel glyph (onset consonants are at most 3: đ/ngh/kh…)
    idx = None
    seen_consonants = 0
    for i, ch in enumerate(nfd):
        if ch in _VOWEL_GLYPHS:
            idx = i
            break
        seen_consonants += 1
        if seen_consonants > 3:
            break
    if idx is None:
        return base
    mark = {"sac": "\u0301", "huyen": "\u0300", "hoi": "\u0309",
            "nga": "\u0303", "nang": "\u0323"}[tone]
    marked = nfd[:idx + 1] + mark + nfd[idx + 1:]
    return unicodedata.normalize("NFC", marked)


def tone_variants(syllable: str) -> List[str]:
    """All five other tone spellings of this syllable (may contain nonsense)."""
    base, _, _ = decompose(syllable)
    out = []
    for t in ("sac", "huyen", "hoi", "nga", "nang"):
        v = apply_tone(base, t)
        if v != syllable:
            out.append(v)
    return out


def build_vocab(texts: Sequence[str]) -> Set[str]:
    """Corpus syllable vocabulary for real-word validation."""
    vocab: Set[str] = set()
    for t in texts:
        for tok in re.findall(r"[^\s\d]+", t):
            tok = tok.strip(".,;:!?()\"'“”…—").lower()
            if tok and re.fullmatch(r"[a-zà-ỹđ]+", tok):
                vocab.add(tok)
    return vocab


def build_items(kb_path: Path, n_items: int = 300, seed: int = 42,
                max_sentence_words: int = 40) -> List[dict]:
    """Build n_items minimal-pair items from authentic KB domain prose."""
    import random

    from tfvn.serialise import read_jsonl

    rows = read_jsonl(kb_path)
    rng = random.Random(seed)

    sentences: List[str] = []
    for r in rows:
        dom = r.get("domain_vi") or {}
        for text in dom.values():
            if isinstance(text, str):
                sentences += [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                              if 6 <= len(s.split()) <= max_sentence_words]
    rng.shuffle(sentences)

    vocab = build_vocab(s for s in sentences)
    items: List[dict] = []
    used_spans: Set[str] = set()
    for sent in sentences:
        if len(items) >= n_items:
            break
        toks = sent.split()
        cand_idx = list(range(len(toks)))
        rng.shuffle(cand_idx)
        placed = False
        for ti in cand_idx:
            raw = toks[ti].strip(".,;:!?()\"'“”…—").lower()
            if not re.fullmatch(r"[a-zà-ỹđ]+", raw):
                continue
            variants = []
            for v in tone_variants(raw):
                if v in vocab and v not in variants:
                    variants.append(v)
            if len(variants) < 3:
                continue
            rng.shuffle(variants)
            key = f"{raw}|{sent[:24]}"
            options = [raw] + variants[:3]
            order = list(range(4))
            rng.shuffle(order)
            opts = [options[j] for j in order]
            answer_idx = opts.index(raw)
            display = " …" + " ".join(
                "_____" if j == ti else w for j, w in enumerate(toks)) + "… "
            items.append({
                "item_id": f"tone_{len(items):04d}",
                "sentence_vi": sent,
                "blanked": display,
                "options": opts,
                "answer_word": raw,
                "answer_idx": answer_idx,
            })
            used_spans.add(key)
            placed = True
            break
        if placed and len(items) >= n_items:
            break
    return items


_LETTER_RE = re.compile(r"\b([ABCD])\b")
_LETTER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}


def format_prompt(item: dict) -> str:
    opts = "\n".join(f"{'ABCD'[i]}. {o}" for i, o in enumerate(item["options"]))
    return (
        "Câu tiếng Việt sau bị thiếu một âm tiết (tại chỗ _____). "
        "Chọn cách viết ĐÚNG chính tả có dấu thanh để hoàn thiện câu.\n\n"
        f"{item['blanked']}\n\n{opts}\n\n"
        "Trả lời bằng đúng một chữ cái (A, B, C hoặc D)."
    )


def score_items(provider, items: Sequence[dict]) -> dict:
    used_ll = bool(getattr(provider, "supports_loglikelihood", lambda: False)())
    correct = 0
    details = []
    for item in items:
        prompt = format_prompt(item)
        if used_ll:
            scores = [provider.loglikelihood(prompt + "\nĐáp án:", f" {'ABCD'[i]}")
                      for i in range(4)]
            pred = max(range(4), key=lambda i: scores[i])
        else:
            out = provider.generate(prompt, temperature=0.0, max_tokens=8)
            m = _LETTER_RE.search(out.upper())
            pred = _LETTER_MAP.get(m.group(1), -1)
        ok = pred == item["answer_idx"]
        correct += ok
        details.append({"item_id": item["item_id"], "pred": pred, "correct": ok})
    return {
        "provider": provider.name,
        "scoring": "loglikelihood" if used_ll else "letter-parse",
        "n_items": len(items),
        "accuracy": correct / len(items) if items else 0.0,
        "details": details,
    }
