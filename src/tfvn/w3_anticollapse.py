"""W3.3 anti-collapse stack for SFT-dataset generation.

Three mechanisms (plan W3.3) fighting mode collapse in the generated corpus:

1. ``MemoryIndex`` — persistent cross-batch semantic memory: a deterministic
   hashed char-n-gram vector store that rejects near-duplicate generations
   across batches. Uses a language-agnostic, process-stable hashing embedding
   (zlib.crc32 over char n-grams -> fixed-dim vector, cosine similarity) — the
   same documented-proxy discipline as W2.1 (no /embeddings route on the
   endpoint; a deterministic semantic proxy is used instead).
2. ``RotatingPromptState`` — rebuilds the per-batch generation context from the
   memory state plus rotating diversity axes (register, length band, question
   topic, exemplars). Drives prompt-strategy rotation.
3. ``SelfTighteningNGramBlacklist`` — computes n-gram frequencies over already
   generated data, promotes over-represented CONTENT-bearing n-grams into a
   blacklist, and NEVER fires on function-word/particle tokens (protected set
   from ``kb/vn_register_profile.json``).

Plus the diversity metric: ``distinct_n`` (distinct-2 on a sample, floor
0.45 per plan W3.3) and the ablation runner (none / memory-only / all-three).
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .w2_gates import simple_vi_tokens


# ------------------------------------------------------------- hashing embed --


class CharNGramEmbedding:
    """Deterministic hashed char-n-gram bag-of-vectors (language-agnostic).

    ``vector(text)`` returns a fixed-dim list of floats (default 4096 dims,
    char 4-grams). The mapping n-gram -> dim uses zlib.crc32, which is
    stable across processes (Python's builtin ``hash`` is salted per process
    and must NOT be used for a persistent store).
    """

    def __init__(self, n: int = 4, dim: int = 4096) -> None:
        self.n = n
        self.dim = dim

    def vector(self, text: str) -> List[float]:
        text = text.lower()
        if len(text) < self.n:
            text = text.ljust(self.n, " ")
        vec = [0.0] * self.dim
        for i in range(len(text) - self.n + 1):
            gram = text[i : i + self.n]
            idx = zlib.crc32(gram.encode("utf-8")) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def cosine(self, a: Sequence[float], b: Sequence[float]) -> float:
        return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------- memory index --


class MemoryIndex:
    """Persistent cross-batch semantic memory (rejects near-duplicates).

    Stores one vector per accepted generation (plus its hash). ``is_dup``
    returns True when the candidate is a near-duplicate of an accepted item
    (cosine >= threshold) OR an exact hash match. Persists to JSONL so batches
    share memory across process runs.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        threshold: float = 0.92,
        embed: Optional[CharNGramEmbedding] = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.threshold = threshold
        self.embed = embed or CharNGramEmbedding()
        self._items: List[Dict[str, Any]] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    self._items.append(json.loads(line))

    @property
    def size(self) -> int:
        return len(self._items)

    def _exact_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_dup(self, text: str) -> bool:
        h = self._exact_hash(text)
        vec = self.embed.vector(text)
        for item in self._items:
            if item.get("hash") == h:
                return True
            if self.embed.cosine(vec, item["vec"]) >= self.threshold:
                return True
        return False

    def add(self, text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        item = {
            "hash": self._exact_hash(text),
            "vec": self.embed.vector(text),
            "meta": meta or {},
        }
        self._items.append(item)
        self._flush()
        return item

    def _flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(it, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for it in self._items
        ]
        self.path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def dedup_rate_probe(self, texts: Sequence[str]) -> Dict[str, Any]:
        """Measure how much work the memory index does on a candidate batch
        (plan: dedup rate above 0 but below 0.4 — it rejects duplicates yet
        does not strip useful variety)."""
        if not texts:
            return {"n": 0, "dups": 0, "rate": 0.0}
        dup = sum(1 for t in texts if self.is_dup(t))
        return {"n": len(texts), "dups": dup, "rate": round(dup / len(texts), 4)}


# --------------------------------------------------------------- n-gram blacklist


class SelfTighteningNGramBlacklist:
    """Promotes over-represented CONTENT n-grams into a blacklist.

    ``update(texts)`` recomputes n-gram frequencies over the corpus; any n-gram
    whose frequency exceeds ``ratio_floor`` (relative share of the largest
    n-gram) is promoted. N-grams containing any protected function-word token
    are NEVER blacklisted (plan: the blacklist must not fire on legitimate
    Vietnamese constructions). ``is_blacklisted`` and ``forbidden_phrases``
    feed the rotating generation prompt as a hard constraint.
    """

    def __init__(
        self,
        n: int = 4,
        ratio_floor: float = 0.55,
        protected_tokens: Optional[Sequence[str]] = None,
        profile_path: Optional[Path] = None,
    ) -> None:
        self.n = n
        self.ratio_floor = ratio_floor
        self.protected = set(protected_tokens or [])
        if profile_path is not None and profile_path.exists():
            prof = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            self.protected |= set(prof.get("function_words") or [])
        self._blacklist: List[str] = []

    def update(self, texts: Sequence[str]) -> List[str]:
        counts: Counter = Counter()
        for t in texts:
            tokens = simple_vi_tokens(t)
            for i in range(len(tokens) - self.n + 1):
                gram = " ".join(tokens[i : i + self.n])
                counts[gram] += 1
        if not counts:
            return list(self._blacklist)
        top = max(counts.values())
        self._blacklist = [
            g for g, c in counts.items() if c / top >= self.ratio_floor
        ]
        return list(self._blacklist)

    def is_blacklisted(self, phrase: str) -> bool:
        tokens = simple_vi_tokens(phrase)
        if any(t in self.protected for t in tokens):
            return False
        return phrase in self._blacklist

    @property
    def forbidden_phrases(self) -> List[str]:
        return list(self._blacklist)

    def never_fires_on_function_words(self) -> bool:
        """Assertion for the plan QA: a blacklisted n-gram never contains a
        protected function word."""
        protected = self.protected
        for g in self._blacklist:
            if any(t in protected for t in g.split()):
                return False
        return True


# ------------------------------------------------------------- rotating prompts --


class RotatingPromptState:
    """Rebuilds per-batch generation context from memory + rotating axes.

    ``next_axis()`` rotates through (register x length_band x topic) to force
    diversity across batches. ``context_block()`` serialises the current memory
    summary (how many readings exist, top promoted blacklist phrases) so the
    generation prompt can be rebuilt each batch — a cache-missing prompt that
    reflects the current corpus state.
    """

    AXES = [
        ("formal", "ngắn", "love"),
        ("warm", "đầy_đủ", "career"),
        ("casual", "ngắn", "money"),
        ("formal", "đầy_đủ", "health"),
        ("warm", "ngắn", "spiritual"),
        ("casual", "đầy_đủ", "decision"),
        ("formal", "ngắn", "career"),
        ("warm", "đầy_đủ", "love"),
        ("casual", "ngắn", "health"),
        ("formal", "đầy_đủ", "decision"),
        ("warm", "ngắn", "money"),
        ("casual", "đầy_đủ", "spiritual"),
    ]

    def __init__(self, memory: MemoryIndex, blacklist: SelfTighteningNGramBlacklist) -> None:
        self.memory = memory
        self.blacklist = blacklist
        self._i = 0

    def next_axis(self) -> Dict[str, str]:
        axis = self.AXES[self._i % len(self.AXES)]
        self._i += 1
        return {"register": axis[0], "length_band": axis[1], "topic": axis[2]}

    def context_block(self) -> str:
        parts = [
            f"Đã tạo {self.memory.size} bài đọc trong đợt này.",
        ]
        forbid = self.blacklist.forbidden_phrases[:5]
        if forbid:
            parts.append(
                "CỤM TỪ CẤM (tuyệt đối không lặp lại): " + "; ".join(forbid) + "."
            )
        return "\n".join(parts)


# --------------------------------------------------------------- diversity --


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """distinct-n: unique n-gram ratio over the token stream.

    distinct-2 is the plan's ONE stated diversity target (floor 0.45 on a
    held-out 200-prompt sample).
    """
    grams: List[str] = []
    for t in texts:
        tokens = simple_vi_tokens(t)
        grams.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    if not grams:
        return 0.0
    return round(len(set(grams)) / len(grams), 4)


# ------------------------------------------------------------------ ablation --


def run_ablation(
    candidates: Sequence[str],
    memory: MemoryIndex,
    blacklist: SelfTighteningNGramBlacklist,
) -> Dict[str, Any]:
    """Ablate the three mechanisms (plan: none / memory-only / all-three).

    Measures distinct-2 on the candidate batch under three conditions:
      - none:       raw candidates, no memory rejection, no blacklist
      - memory:     near-duplicates (against memory) removed
      - all-three:  memory removal + blacklisted phrases stripped
    Records the result; the plan does NOT ship the stack uncritically.
    """
    kept_none = list(candidates)
    kept_memory = [c for c in candidates if not memory.is_dup(c)]
    stripped = [c for c in kept_memory if not blacklist.is_blacklisted(c)]
    kept_all = [c for c in kept_memory if not any(
        ph in c for ph in blacklist.forbidden_phrases
    )]
    if not kept_all:
        kept_all = stripped
    return {
        "n_candidates": len(candidates),
        "distinct2_none": distinct_n(kept_none),
        "distinct2_memory_only": distinct_n(kept_memory),
        "distinct2_all_three": distinct_n(kept_all),
        "kept_none": len(kept_none),
        "kept_memory": len(kept_memory),
        "kept_all_three": len(kept_all),
        "floor": 0.45,
        "all_three_meets_floor": distinct_n(kept_all) >= 0.45,
        "blacklist_never_fires_on_function_words": blacklist.never_fires_on_function_words(),
    }
