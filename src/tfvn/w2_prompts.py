"""Prompt construction for W2.2 Vietnamese reversed-meaning synthesis.

Cardinal rule (plan W2.2): generate native Vietnamese from the English
semantic spine as *constraints*; NEVER show the model English prose to
translate. Translationese correlated with the reversed label is the specific
failure this design avoids. The card's English meaning text is therefore used
only as keyword/constraint context, never quoted as the thing to render.

Two prompt families:

1. ``generation``  — synthesises one Vietnamese reversed prose variant per call,
   returning JSON ``{"keywords_vi": [...], "prose": "..."}``. The declared
   ``keywords_vi`` feed the mechanical containment back-check (G3).
2. ``rubric``      — judge call: scores how many of the card's English keyword
   atoms are semantically covered by a candidate, plus the inline-English-
   card-name rule. Returns JSON with ``atoms_covered`` / ``atoms_total`` and a
   pass/fail ``decision`` (plan: keyword back-check recall >= 70%).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

SYSTEM_GENERATION = (
    "Bạn là chuyên gia viết ý nghĩa bài Tarot bằng tiếng Việt tự nhiên, "
    "đúng văn phong sách huyền học xuất bản. QUY TẮC BẮT BUỘC:\n"
    "1. Viết tiếng Việt thuần, có dấu đầy đủ, tự nhiên như người bản ngữ.\n"
    "2. Tuyệt đối không dịch máy từ tiếng Anh; hãy VIẾT MỚI ý nghĩa bằng tiếng Việt.\n"
    "3. Tên lá bài bằng tiếng Anh phải xuất hiện nguyên văn trong câu văn "
    "(ví dụ: The Fool, Two of Cups) — không viết tên tiếng Việt.\n"
    "4. Dùng từ nối, hư từ (thì, là, mà, của, sẽ, rằng, và, nhưng, khi, để, cho, "
    "với, trong, ngoài, đã, đang, bị, được, các, những, một, này, đó, vì, do, "
    "bởi, nên, cũng, rất, nhiều, ít, không, chưa, chỉ, còn, hay, hoặc, về, như, theo) "
    "đúng tần suất văn xuôi tiếng Việt tự nhiên.\n"
    "5. Viết 3–5 câu, khoảng 120–200 từ, mạch lạc.\n"
    "6. KHÔNG khẳng định chuyện chết chóc, chẩn đoán bệnh, hay tư vấn pháp lý.\n"
    "7. Chỉ trả về JSON hợp lệ, không kèm chú thích ngoài JSON."
)

SYSTEM_RUBRIC = (
    "Bạn là giám khảo kiểm tra độ bao phủ từ khóa của một đoạn ý nghĩa Tarot "
    "tiếng Việt. Nhiệm vụ: đối chiếu danh sách TỪ KHÓA TIẾNG ANH (keyword atoms) "
    "của lá bài với nội dung tiếng Việt được sinh ra, rồi trả về JSON:\n"
    '{"name_inline": true/false, "atoms_covered": <số từ khóa được thể hiện>, '
    '"atoms_total": <tổng số từ khóa>, "decision": "pass"/"fail", "reason": "<ngắn gọn>"}.\n'
    "Cách chấm: MỘT từ khóa được tính là covered nếu ý nghĩa của nó hiện diện rõ "
    "trong câu tiếng Việt (từ đồng nghĩa, cách diễn đạt tương đương). "
    "name_inline=true nếu tên lá bài tiếng Anh xuất hiện nguyên văn trong văn bản. "
    "decision='pass' khi atoms_covered/atoms_total >= 0.7 VÀ name_inline=true; "
    "ngược lại 'fail'. Chỉ trả về JSON."
)


def _exemplar_block(exemplars: Sequence[str]) -> str:
    if not exemplars:
        return "  (không có ví dụ mẫu)"
    return "\n".join(f"  MẪU {i + 1}: {ex}" for i, ex in enumerate(exemplars[:5]))


def build_generation_messages(
    spine_row: Dict[str, Any],
    exemplars: Sequence[str],
    variant: int = 1,
    attempt: int = 1,
) -> List[Dict[str, str]]:
    """Messages for one reversed-prose variant.

    ``spine_row`` must be the REVERSED row of ``kb/english_spine.jsonl``:
    carries card_id, name_en, keyword_atoms_en, polarity_axis, meaning_summary_en.
    ``attempt`` marks the retry round — it changes the prompt text so a retry
    misses the prompt-hash cache and draws fresh content.
    """
    atoms = ", ".join(spine_row.get("keyword_atoms_en") or [])
    polarity = spine_row.get("polarity_axis") or "inverted"
    user = (
        f"LÁ BÀI: {spine_row['name_en']} (card_id {spine_row['card_id']}, "
        f"vị trí NGƯỢC / reversed).\n"
        f"TỪ KHÓA (tiếng Anh, làm ràng buộc ý nghĩa): {atoms}.\n"
        f"TRỤC CỰC (polarity): {polarity} — nghĩa đảo ngược, cản trở, hoặc nội hóa.\n"
        f"VĂN PHONG MẪU (tham khảo giọng văn, KHÔNG chép nội dung):\n"
        f"{_exemplar_block(exemplars)}\n"
        f"NHIỆM VỤ: Viết MỚI bằng tiếng Việt ý nghĩa lá {spine_row['name_en']} "
        f"khi xuất hiện NGƯỢC. Dùng các từ khóa trên làm ý nghĩa cốt lõi, diễn "
        f"đạt tự nhiên theo trục cực {polarity}. Tên tiếng Anh "
        f"{spine_row['name_en']} phải xuất hiện nguyên văn trong bài viết. "
        f"Biến thể {variant}/2, lần thử {attempt}.\n"
        f'Trả về JSON: {{"keywords_vi": ["...", ...] (3-5 từ khóa tiếng Việt '
        f"đại diện cho ý nghĩa này, sẽ dùng để kiểm tra), \"prose\": \"<toàn bộ "
        f"bài viết tiếng Việt>\"}}."
    )
    return [{"role": "system", "content": SYSTEM_GENERATION}, {"role": "user", "content": user}]


def build_rubric_messages(
    spine_row: Dict[str, Any], prose: str, keywords_vi: Sequence[str]
) -> List[Dict[str, str]]:
    """Judge messages: keyword back-check for one candidate (G3)."""
    atoms = spine_row.get("keyword_atoms_en") or []
    user = (
        f"LÁ BÀI: {spine_row['name_en']} (reversed).\n"
        f"TỪ KHÓA TIẾNG ANH: {json.dumps(atoms, ensure_ascii=False)}.\n"
        f"ỨNG VIÊN TIẾNG VIỆT:\n"
        f"  keywords_vi = {json.dumps(list(keywords_vi), ensure_ascii=False)}\n"
        f"  prose = \"{prose}\"\n"
        "Hãy chấm theo rubric ở trên."
    )
    return [{"role": "system", "content": SYSTEM_RUBRIC}, {"role": "user", "content": user}]


def build_attribution_messages(vn_text: str) -> List[Dict[str, str]]:
    """Optional LLM cross-check for W2.1 attribution (off by default; the
    deterministic polarity-lexicon proxy is the default). Returns messages for
    a JSON verdict ``{"verdict": "upright"|"agnostic"|"reversed_skew", "reason": "..."}``."""
    user = (
        "Đoạn văn dưới đây là phần ý nghĩa lá bài Tarot tiếng Việt (bản sao "
        "giống hệt cho cả xuôi và ngược trong nguồn dữ liệu). Hỏi: đoạn văn này "
        "thiên về giọng ý nghĩa XUÔI (thuận lợi, tích cực), giọng NGƯỢC (tiêu cực, "
        "trở ngại), hay TRUNG TÍNH (không nghiêng về hướng nào)?\n\n"
        f"VĂN BẢN: \"{vn_text[:800]}\"\n\n"
        'Trả về JSON: {"verdict": "upright"|"agnostic"|"reversed_skew", "reason": "..."}'
    )
    return [{"role": "user", "content": user}]
