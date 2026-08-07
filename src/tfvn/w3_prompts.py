"""Prompt construction for W3.2 SFT-dataset generation.

Cardinal rule (plan W3.2): generate native Vietnamese. The English semantic
spine appears only as *constraints* (keyword atoms, polarity axis, card names,
position glosses) — NEVER as prose to translate. Translationese correlated with
the reversed label or the task type is the specific failure this design avoids.

Prompt families (each returns a cached prompt-hash key):

1. ``question_draw``    — stage 1 of the Magpie split: draws a natural
   Vietnamese question from querent context x register x length band. High
   temperature (1.0). JSON: ``{"question_vi": "..."}``.
2. ``reading_draw``     — stage 2: the reading given the drawn question + card
   block + position glosses. Low temperature (0.7). JSON:
   ``{"reading_vi": "..."}``.
3. ``critique``         — Sonnet judge: critiques a reading against the card
   block (faithfulness, orientation, naturalness, question-answering,
   forbidden claims). JSON verdict + issues list.
4. ``revise``           — Haiku: produces the revised reading from the original
   + critique. JSON ``{"reading_vi": "..."}``.
5. ``rubric``           — L4 judge rubric (tone / translationese / faithfulness).
6. ``safety_reading``   — reading generation with the safety policy category
   template as a hard constraint (matched-pair members).
7. ``grounding_reading``— reading generation with a grounding defect injected
   (missing card / incomplete context / empty gloss).
8. ``correction_reading``— counter-sycophancy: the querent asserts a wrong
   meaning; the model must correct politely.

Every prompt embeds a ``slot`` counter so retries and re-draws miss the
prompt-hash cache and produce fresh content (same discipline as W2.2).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

# ------------------------------------------------------------------ register --

SYSTEM_READING = (
    "Bạn là chuyên gia đọc bài Tarot bằng tiếng Việt tự nhiên, đúng văn phong "
    "sách huyền học xuất bản và giọng trò chuyện ấm áp của thầy bói giàu kinh nghiệm. "
    "QUY TẮC BẮT BUỘC:\n"
    "1. Viết tiếng Việt thuần, có dấu đầy đủ, tự nhiên như người bản ngữ.\n"
    "2. Tuyệt đối không dịch máy từ tiếng Anh; hãy VIẾT MỚI bằng tiếng Việt.\n"
    "3. Tên lá bài tiếng Anh phải xuất hiện nguyên văn trong câu văn (ví dụ: "
    "The Fool, Two of Cups) — không viết tên tiếng Việt.\n"
    "4. Dùng từ nối, hư từ (thì, là, mà, của, sẽ, rằng, và, nhưng, khi, để, cho, "
    "với, trong, ngoài, đã, đang, bị, được, các, những, một, này, đó, vì, do, "
    "bởi, nên, cũng, rất, nhiều, ít, không, chưa, chỉ, còn, hay, hoặc, về, như, theo) "
    "đúng tần suất văn xuôi tiếng Việt tự nhiên.\n"
    "5. Trả lời ĐÚNG câu hỏi của người hỏi, phục vụ câu hỏi làm trung tâm, không "
    "chỉ liệt kê nghĩa từng lá.\n"
    "6. Phân biệt rõ nghĩa XUÔI và NGƯỢC của từng lá; không gộp chung.\n"
    "7. KHÔNG khẳng định chuyện chết chóc, chẩn đoán bệnh, hay tư vấn pháp lý.\n"
    "8. Chỉ trả về JSON hợp lệ, không kèm chú thích ngoài JSON."
)

SYSTEM_CRITIQUE = (
    "Bạn là giám khảo khắt khe đánh giá bài đọc Tarot tiếng Việt. Đối chiếu bài "
    "đọc với THẺ BÀI và CÂU HỎI đã cho, rồi trả về JSON:\n"
    '{"verdict": "pass"/"fix", "issues": ["...", "..."], "orientation_ok": true/false, '
    '"forbidden_claims": ["..."], "faithful": true/false, "answers_question": true/false, '
    '"vietnamese_natural": true/false}\n'
    "Chấm điểm: faithful=false nếu bài đọc nói về lá không có trong bộ bài, hoặc "
    "bỏ sót vị trí, hoặc bịa nghĩa. orientation_ok=false nếu không phân biệt xuôi/ngược "
    "khi thẻ yêu cầu. forbidden_claims chứa các khẳng định chết chóc/chẩn đoán/tư vấn "
    "pháp lý nếu có. verdict='pass' chỉ khi faithful=true VÀ orientation_ok=true VÀ "
    "answers_question=true VÀ không có forbidden_claims. Chỉ trả về JSON."
)

SYSTEM_RUBRIC = (
    "Bạn là giám khảo chấm bài đọc Tarot tiếng Việt theo 3 trục. Trả về JSON:\n"
    '{"tone": {"pass": true/false, "reason": "..."}, '
    '"translationese": {"pass": true/false, "reason": "..."}, '
    '"faithfulness": {"pass": true/false, "reason": "..."}, '
    '"decision": "pass"/"fail"}\n'
    "tone: tiếng Việt tự nhiên, có dấu đầy đủ, không lai tiếng Anh. "
    "translationese: không mang mùi dịch máy từ tiếng Anh. "
    "faithfulness: mọi lá được nhắc đều thuộc bộ bài đã cho, nghĩa đúng thẻ, "
    "phân biệt xuôi/ngược. decision='pass' khi cả ba trục đều pass. Chỉ trả về JSON."
)

SYSTEM_REFUSAL = (
    "Bạn là trợ lý Tarot tiếng Việt lịch sự, tuân thủ tuyệt đối chính sách an toàn. "
    "Chỉ trả về JSON. Khi cần từ chối, dùng đúng văn mẫu quy định cho hạng mục được "
    "chỉ định, không thêm bịa."
)

REGISTER_GUIDE = {
    "formal": "Giọng trang trọng, lịch sự, dùng 'anh/chị' và kính ngữ.",
    "warm": "Giọng ấm áp, gần gũi, dùng 'bạn', nhẹ nhàng khích lệ.",
    "casual": "Giọng thân mật, tự nhiên như trò chuyện bạn bè.",
}

LENGTH_GUIDE = {
    "ngắn": "Ngắn gọn: 2–3 câu, khoảng 60–90 từ.",
    "đầy_đủ": "Đầy đủ: 4–6 câu, khoảng 120–180 từ.",
}


def _exemplar_block(exemplars: Sequence[str]) -> str:
    if not exemplars:
        return "  (không có ví dụ mẫu)"
    return "\n".join(f"  MẪU {i + 1}: {ex}" for i, ex in enumerate(exemplars[:4]))


def card_block(rows: Sequence[Dict[str, Any]], with_meaning_vi: bool = True) -> str:
    """Render the drawn cards as constraint lines (not prose to translate)."""
    lines = []
    for i, r in enumerate(rows):
        atoms = ", ".join(r.get("keywords_en") or [])
        part = (
            f"[Vị trí {i + 1}] {r['name_en']} — {r['orientation'].upper()} — "
            f"trục cực: {r.get('polarity_axis') or 'xuôi'} — "
            f"keyword: {atoms}"
        )
        if with_meaning_vi and r.get("meaning_vi"):
            part += f" — nghĩa VN (ngữ liệu): {r['meaning_vi'][:220]}"
        lines.append(part)
    return "\n".join(lines)


# ------------------------------------------------------------ stage 1: question


def build_question_draw_messages(
    context: str,
    register: str,
    length_band: str,
    interaction: str,
    seeds: Sequence[str],
    slot: int,
    topic_label_vi: str,
) -> List[Dict[str, str]]:
    seed_ex = "\n".join(f"  MẪU CÂU HỎI {i + 1}: {s}" for i, s in enumerate(seeds[:3]))
    user = (
        f"BỐI CẢNH NGƯỜI HỎI: {topic_label_vi} ({context}).\n"
        f"GIỌNG: {REGISTER_GUIDE.get(register, REGISTER_GUIDE['warm'])}\n"
        f"ĐỘ DÀI CÂU TRẢ LỜI MONG MUỐN: {LENGTH_GUIDE.get(length_band, LENGTH_GUIDE['đầy_đủ'])}\n"
        f"KIỂU TƯƠNG TÁC: {interaction}\n"
        f"MẪU CÂU HỎI TIẾNG VIỆT (tham khảo giọng, KHÔNG chép nguyên văn):\n{seed_ex}\n"
        f"NHIỆM VỤ: Viết MỚI bằng tiếng Việt một câu hỏi tự nhiên mà một người thật "
        f"sẽ hỏi thầy Tarot trong bối cảnh trên (slot {slot}). Câu hỏi nên cụ thể, "
        f"có chi tiết đời thường, KHÔNG nhắc tên lá bài.\n"
        f'Trả về JSON: {{"question_vi": "<câu hỏi tiếng Việt>"}}.'
    )
    return [{"role": "system", "content": SYSTEM_READING}, {"role": "user", "content": user}]


# ------------------------------------------------------------- stage 2: reading


def build_reading_draw_messages(
    question_vi: str,
    rows: Sequence[Dict[str, Any]],
    spread_name_vi: str,
    position_glosses: Sequence[str],
    register: str,
    length_band: str,
    slot: int,
    exemplars: Sequence[str],
) -> List[Dict[str, str]]:
    glosses = "\n".join(
        f"  Vị trí {i + 1}: {g}" for i, g in enumerate(position_glosses)
    )
    user = (
        f"CÂU HỎI CỦA NGƯỜI HỎI: \"{question_vi}\"\n"
        f"TRẢI BÀI: {spread_name_vi}\n"
        f"Ý NGHĨA CÁC VỊ TRÍ:\n{glosses}\n"
        f"THẺ ĐÃ RÚT (ràng buộc — dùng đúng thẻ, không thêm lá khác):\n"
        f"{card_block(rows)}\n"
        f"GIỌNG: {REGISTER_GUIDE.get(register, REGISTER_GUIDE['warm'])}\n"
        f"ĐỘ DÀI: {LENGTH_GUIDE.get(length_band, LENGTH_GUIDE['đầy_đủ'])}\n"
        f"VĂN PHONG MẪU (tham khảo giọng văn, KHÔNG chép nội dung):\n"
        f"{_exemplar_block(exemplars)}\n"
        f"NHIỆM VỤ: Viết MỚI bằng tiếng Việt một bài đọc hoàn chỉnh trả lời câu hỏi "
        f"trên, dùng ĐÚNG các lá đã rút, phân biệt rõ xuôi/ngược, mỗi vị trí được "
        f"giải thích, tên lá tiếng Anh xuất hiện nguyên văn, slot {slot}.\n"
        f'Trả về JSON: {{"reading_vi": "<toàn bộ bài đọc tiếng Việt>"}}.'
    )
    return [{"role": "system", "content": SYSTEM_READING}, {"role": "user", "content": user}]


# ------------------------------------------------------- critique and revise ----
# Critique runs on SONNET, revise on HAIKU (approved W3 mix).


def build_critique_messages(
    question_vi: str,
    rows: Sequence[Dict[str, Any]],
    reading_vi: str,
) -> List[Dict[str, str]]:
    user = (
        f"CÂU HỎI: \"{question_vi}\"\n"
        f"THẺ ĐÃ RÚT:\n{card_block(rows, with_meaning_vi=False)}\n"
        f"BÀI ĐỌC CẦN CHẤM:\n\"{reading_vi}\"\n"
        "Hãy chấm theo rubric ở trên. Trả về JSON."
    )
    return [{"role": "system", "content": SYSTEM_CRITIQUE}, {"role": "user", "content": user}]


def build_revise_messages(
    question_vi: str,
    rows: Sequence[Dict[str, Any]],
    reading_vi: str,
    critique: Dict[str, Any],
    slot: int,
) -> List[Dict[str, str]]:
    issues = "\n".join(f"  - {i}" for i in (critique.get("issues") or [])[:5])
    user = (
        f"CÂU HỎI: \"{question_vi}\"\n"
        f"THẺ ĐÃ RÚT:\n{card_block(rows, with_meaning_vi=False)}\n"
        f"BÀI ĐỌC GỐC:\n\"{reading_vi}\"\n"
        f"NHẬN XÉT CỦA GIÁM KHẢO:\n{issues}\n"
        f"NHIỆM VỤ: Viết LẠI bằng tiếng Việt bài đọc, khắc phục mọi nhận xét trên, "
        f"giữ nguyên thẻ đã rút và tên lá tiếng Anh nguyên văn, slot {slot}.\n"
        f'Trả về JSON: {{"reading_vi": "<bài đọc sửa lại>"}}.'
    )
    return [{"role": "system", "content": SYSTEM_READING}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------- rubric


def build_rubric_messages(
    question_vi: str,
    rows: Sequence[Dict[str, Any]],
    reading_vi: str,
) -> List[Dict[str, str]]:
    user = (
        f"CÂU HỎI: \"{question_vi}\"\n"
        f"THẺ ĐÃ RÚT:\n{card_block(rows, with_meaning_vi=False)}\n"
        f"BÀI ĐỌC CẦN CHẤM:\n\"{reading_vi}\"\n"
        "Hãy chấm theo rubric 3 trục ở trên. Trả về JSON."
    )
    return [{"role": "system", "content": SYSTEM_RUBRIC}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------- safety


def build_safety_reading_messages(
    question_vi: str,
    rows: Sequence[Dict[str, Any]],
    spread_name_vi: str,
    position_glosses: Sequence[str],
    category: str,
    policy_template: str,
    slot: int,
) -> List[Dict[str, str]]:
    glosses = "\n".join(
        f"  Vị trí {i + 1}: {g}" for i, g in enumerate(position_glosses)
    )
    user = (
        f"CÂU HỎI: \"{question_vi}\"\n"
        f"TRẢI BÀI: {spread_name_vi}\nÝ NGHĨA CÁC VỊ TRÍ:\n{glosses}\n"
        f"THẺ ĐÃ RÚT:\n{card_block(rows)}\n"
        f"HẠNG MỤC CHÍNH SÁCH: {category}\n"
        f"VĂN MẪU BẮT BUỘC (phải dùng đúng tinh thần, có thể nối vào bài đọc):\n"
        f"  \"{policy_template}\"\n"
        f"NHIỆM VỤ: Viết bài đọc tiếng Việt trả lời câu hỏi, vẫn giải thích đúng thẻ, "
        f"nhưng TUÂN THỦ tuyệt đối văn mẫu chính sách cho hạng mục {category} — dùng "
        f"văn mẫu nguyên văn hoặc sát nghĩa trong câu trả lời, slot {slot}.\n"
        f"QUY TẮC BẮT BUỘC (không thể bỏ): kể cả khi áp dụng văn mẫu từ chối theo "
        f"chính sách, câu trả lời vẫn phải nêu nguyên văn tên tiếng Anh của từng lá "
        f"trong THẺ ĐÃ RÚT và nói rõ hướng xuôi/ngược của lá đó (ví dụ: \"Ten of "
        f"Wands\" ngược) — không viết lời từ chối trần trụi mà không nhắc lá bài.\n"
        f'Trả về JSON: {{"reading_vi": "<bài đọc tuân thủ chính sách>"}}.'
    )
    return [{"role": "system", "content": SYSTEM_READING}, {"role": "user", "content": user}]


# ------------------------------------------------------------------- grounding


def build_grounding_reading_messages(
    question_vi: str,
    rows: Sequence[Dict[str, Any]],
    spread_name_vi: str,
    position_glosses: Sequence[str],
    defect: Dict[str, Any],
    slot: int,
) -> List[Dict[str, str]]:
    glosses = "\n".join(
        f"  Vị trí {i + 1}: {g}" for i, g in enumerate(position_glosses)
    )
    defect_line = {
        "missing_card": "LƯU Ý: một lá đã rút KHÔNG có trong ngữ liệu thẻ bên dưới.",
        "incomplete_context": "LƯU Ý: ngữ liệu bị cắt — một số thẻ chỉ có tên, không có nghĩa.",
        "empty_gloss": "LƯU Ý: một vị trí trải bài KHÔNG có mô tả ý nghĩa.",
    }.get(defect.get("kind", ""), "")
    user = (
        f"CÂU HỎI: \"{question_vi}\"\n"
        f"TRẢI BÀI: {spread_name_vi}\nÝ NGHĨA CÁC VỊ TRÍ:\n{glosses}\n"
        f"{defect_line}\n"
        f"THẺ ĐÃ RÚT:\n{card_block(rows)}\n"
        f"NHIỆM VỤ: Viết bài đọc tiếng Việt trả lời câu hỏi. QUY TẮC: không bịa lá "
        f"không có trong ngữ liệu, không bịa nghĩa cho thẻ/ vị trí thiếu thông tin — "
        f"nếu thiếu, hãy nói rõ rằng thông tin chưa đủ và diễn giải phần có được, "
        f"slot {slot}.\n"
        f'Trả về JSON: {{"reading_vi": "<bài đọc trung thực với ngữ liệu>"}}.'
    )
    return [{"role": "system", "content": SYSTEM_READING}, {"role": "user", "content": user}]


# ------------------------------------------------------------- counter-sycophancy


def build_correction_reading_messages(
    question_vi: str,
    wrong_claim: str,
    rows: Sequence[Dict[str, Any]],
    spread_name_vi: str,
    position_glosses: Sequence[str],
    slot: int,
) -> List[Dict[str, str]]:
    glosses = "\n".join(
        f"  Vị trí {i + 1}: {g}" for i, g in enumerate(position_glosses)
    )
    user = (
        f"CÂU HỎI: \"{question_vi}\"\n"
        f"NGƯỜI HỎI KHẲNG ĐỊNH SAI: \"{wrong_claim}\"\n"
        f"TRẢI BÀI: {spread_name_vi}\nÝ NGHĨA CÁC VỊ TRÍ:\n{glosses}\n"
        f"THẺ ĐÃ RÚT:\n{card_block(rows)}\n"
        f"NHIỆM VỤ: Viết bài đọc tiếng Việt trả lời câu hỏi, nhưng phải LỊCH SỰ "
        f"ĐÍNH CHÍNH khẳng định sai của người hỏi dựa trên nghĩa đúng của lá bài, "
        f"không xu nịnh, slot {slot}.\n"
        f'Trả về JSON: {{"reading_vi": "<bài đọc có đính chính>"}}.'
    )
    return [{"role": "system", "content": SYSTEM_READING}, {"role": "user", "content": user}]


# --------------------------------------------------------------------- refusal


def build_refusal_messages(
    question_vi: str,
    category: str,
    policy_template: str,
    slot: int,
) -> List[Dict[str, str]]:
    user = (
        f"CÂU HỎI: \"{question_vi}\"\n"
        f"HẠNG MỤC: {category}\n"
        f"VĂN MẪU BẮT BUỘC:\n  \"{policy_template}\"\n"
        f"NHIỆM VỤ: Trả lời bằng tiếng Việt theo đúng văn mẫu trên (slot {slot}), "
        f"có thể thêm một câu dẫn tự nhiên nhưng KHÔNG đi chệch chính sách.\n"
        f'Trả về JSON: {{"reply_vi": "<câu trả lời tuân thủ chính sách>"}}.'
    )
    return [{"role": "system", "content": SYSTEM_REFUSAL}, {"role": "user", "content": user}]
