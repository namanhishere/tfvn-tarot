#!/usr/bin/env python3
"""Platform-agnostic SFT entrypoint (plan W4.2 config, reused by all three
cloud notebooks and by local CPU smoke tests).

Committed configuration (design doc §9.1):
  - LoRA r=32 alpha=32 dropout=0.0 target_modules=all-linear
  - LR 1e-4 cosine -> 1% of peak, warmup STEPS = 3% of total (v5 removed
    warmup_ratio)
  - effective batch 32, seq 2048, NO packing, NO group_by_length (H15)
  - completions-only loss (prompt tokens masked -100)
  - epochs 2 as stopping rule, eval + checkpoint every 1/4 epoch
  - fixed SEED everywhere
  - bf16 gated on get_device_capability()[0] >= 8 (never is_bf16_supported)

W6 safety continuation reuses this script with --lora-r 8 --lora-alpha 16
--safety-slice.

--smoke runs the identical code path on a tiny subset with 4 optimizer steps —
this is how the script is verified without a GPU.

Examples:
  # cloud (notebooks call exactly this):
  python3 scripts/train_sft.py --data datasets/filtered_core.jsonl \
      --out artifacts/sft_r32 --epochs 2
  # ablation:
  python3 scripts/train_sft.py --data datasets/filtered_bulk.jsonl \
      --out artifacts/sft_r64 --lora-r 64
  # safety slice (W6):
  python3 scripts/train_sft.py --data datasets/safety_slice.jsonl \
      --resume-from artifacts/sft_r32/best --lora-r 8 --lora-alpha 16 \
      --out artifacts/safety_continued --epochs 1
  # local verification (no GPU needed):
  python3 scripts/train_sft.py --smoke --out /tmp/smoke_run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tfvn.serialise import read_jsonl  # noqa: E402

SEED = 42


# ------------------------------------------------------------------ data ----

def format_example(row: dict, system_reading: str, system_refusal: str) -> list:
    """Rebuild chat messages from a filtered_core/bulk row's ingredients."""
    task = row.get("task_type") or "reading"
    system = system_refusal if task == "safety" else system_reading

    cards = row.get("cards_used") or []
    draw_lines = []
    for i, c in enumerate(cards):
        pol = f" | polarity: {c['polarity_axis']}" if c.get("polarity_axis") else ""
        draw_lines.append(f"- Vị trí {i + 1}: {c['name_en']} ({c['orientation']}{pol})")
    positions = ", ".join(row.get("position_glosses") or [])
    spread = row.get("spread_name_vi") or "Single-Card Draw"

    user = (
        f"TRẢ BÀI {spread}.\n"
        f"BÀI ĐÃ RÚT:\n" + "\n".join(draw_lines) + "\n"
        + (f"CÁC VỊ TRÍ: {positions}\n" if positions else "")
        + f"CÂU HỎI: {row.get('question_vi', '')}"
    )
    if task == "correction" and row.get("wrong_claim"):
        user += f"\nNGƯỜI HỎI KHẲNG ĐỊNH SAI: {row['wrong_claim']}"

    assistant = row.get("reading_vi") or row.get("target_vi") or ""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def load_rows(paths: list, max_examples=None, seed=SEED):
    rows = []
    for p in paths:
        rows.extend(read_jsonl(Path(p)))
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_examples:
        rows = rows[:max_examples]
    return rows


# ----------------------------------------------------------------- model ----

def pick_dtype() -> str:
    """bf16 only on CC>=8 GPUs (Pascal emulated bf16 is a trap)."""
    import torch

    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        return "bf16" if major >= 8 else "fp16"
    return "no"  # CPU: fp32



def build_tokenized(rows, tokenizer, seq_len):
    """Tokenize with completions-only loss: prompt span masked to -100.
    Returns one {'input_ids': [...], 'labels': [...]} dict per input row."""
    out = []
    for msgs in rows:
        prefix = tokenizer.apply_chat_template(
            msgs[:2], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=False)
        p_ids = tokenizer(prefix, add_special_tokens=False).input_ids
        f_ids = tokenizer(full, add_special_tokens=False).input_ids
        n_mask = min(len(p_ids), len(f_ids))
        labels = [-100] * n_mask + list(f_ids[n_mask:])
        # keep the TAIL when over seq_len so completions survive truncation
        if len(f_ids) > seq_len:
            ids = f_ids[-seq_len:]
            labels = labels[-seq_len:]
        else:
            ids = f_ids
        pad = seq_len - len(ids)
        ids = ids + [tokenizer.pad_token_id or 0] * pad
        labels = labels + [-100] * pad
        out.append({"input_ids": ids, "labels": labels})
    return out


class OrientationTripwire:
    """Plan W4.3: at end of epoch 1, generate upright/reversed for sampled cards;
    HALT training if >= tripwire_rate of pairs exceed the Jaccard threshold."""

    def __init__(self, model, tokenizer, kb_path, threshold=0.24,
                 max_rate=0.05, n_cards=12, device=None):
        import torch

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_rate = max_rate
        self.device = device
        self.report = []
        rows = read_jsonl(kb_path)
        by_id = {}
        for r in rows:
            by_id.setdefault(r["card_id"], {})[r["orientation"]] = r
        self.cards = [(cid, o) for cid, o in sorted(by_id.items())
                      if "upright" in o and "reversed" in o][:n_cards * 4:2][:n_cards]

    @staticmethod
    def _jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _gen(self, question: str, name: str, orientation: str) -> str:
        msgs = [{"role": "user", "content":
                 f"Lá {name} ở trạng thái {'xuôi' if orientation == 'upright' else 'đảo'} "
                 f"nghĩa là gì về tình yêu?"}]
        text = self.tokenizer.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)
        ids = self.tokenizer(text, return_tensors="pt").to(self.device or "cuda")
        with self.torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=180, do_sample=False,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                     skip_special_tokens=True)

    def check(self) -> dict:
        flagged = 0
        for cid, orientations in self.cards:
            name = orientations["upright"]["name_en"]
            up = set(self._gen("tình yêu", name, "upright").lower().split())
            rev = set(self._gen("tình yêu", name, "reversed").lower().split())
            j = self._jaccard(up, rev)
            over = j > self.threshold
            flagged += over
            self.report.append({"card_id": cid, "name_en": name, "jaccard": round(j, 3),
                                "over_threshold": over})
        rate = flagged / max(1, len(self.cards))
        return {"flagged": flagged, "n_cards": len(self.cards),
                "rate": rate, "halt": rate >= self.max_rate, "pairs": self.report}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--kb", default=str(ROOT / "kb/cards.jsonl"))
    ap.add_argument("--data", nargs="+",
                    default=[str(ROOT / "datasets/filtered_core.jsonl")])
    ap.add_argument("--val", default=str(ROOT / "datasets/filtered_core.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=0,
                    help="0 = auto so micro*accum == effective batch 32")
    ap.add_argument("--effective-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tripwire-threshold", type=float, default=0.24)
    ap.add_argument("--no-tripwire", action="store_true")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--resume-from", default=None,
                    help="existing adapter dir to continue from (W6)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run exercising the identical code path")
    args = ap.parse_args()

    import torch
    import transformers
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from tfvn.w3_prompts import SYSTEM_READING, SYSTEM_REFUSAL

    random.seed(args.seed)

    smoke = args.smoke
    if smoke:
        if args.model == "Qwen/Qwen3-1.7B":
            args.model = "Qwen/Qwen3-0.6B"  # fast CPU smoke; same code path
        args.epochs = 1
        args.seq = 256
        args.micro_batch = 1
        args.effective_batch = 2
        args.max_examples = 8
        args.no_tripwire = True

    grad_accum = (args.grad_accum or max(1, args.effective_batch // args.micro_batch))
    dtype = pick_dtype()

    tok_kwargs = {}
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"torch_dtype": {"bf16": torch.bfloat16, "fp16": torch.float16}.get(dtype)}
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if dtype == "no":
        model = model.float()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    peft_kwargs = dict(r=args.lora_r, lora_alpha=args.lora_alpha,
                       lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    try:
        lora_cfg = LoraConfig(target_modules="all-linear", **peft_kwargs)
    except Exception:
        lora_cfg = LoraConfig(**peft_kwargs)  # older peft: infer modules

    if args.resume_from:
        model = PeftModel.from_pretrained(model, args.resume_from, is_trainable=True)
        print(f"resumed adapter from {args.resume_from}")
    else:
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_rows = load_rows(args.data, args.max_examples, args.seed)
    val_rows = load_rows([args.val], max(1, args.max_examples // 4) if args.max_examples else 200,
                         args.seed + 1)
    fmt = lambda r: format_example(r, SYSTEM_READING, SYSTEM_REFUSAL)
    train_ds = Dataset.from_list([build_tokenized([fmt(r)], tokenizer, args.seq)[0]
                                  for r in train_rows])
    val_ds = Dataset.from_list([build_tokenized([fmt(r)], tokenizer, args.seq)[0]
                                for r in val_rows])

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.micro_batch * grad_accum)))
    eval_steps = max(1, int(steps_per_epoch / 4))

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=max(1, args.micro_batch),
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.01},
        warmup_steps=max(1, int(0.03 * steps_per_epoch * args.epochs)),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=6,
        bf16=dtype == "bf16",
        fp16=dtype == "fp16",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    tripwire = None
    if not args.no_tripwire and args.epochs > 1:
        tripwire = OrientationTripwire(model, tokenizer, Path(args.kb),
                                       threshold=args.tripwire_threshold, device=device)

    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=val_ds, processing_class=tokenizer)

    # epoch-1 tripwire hook
    if tripwire is not None:
        from transformers import TrainerCallback

        class TripwireCallback(TrainerCallback):
            def __init__(self):
                self.fired = False

            def on_epoch_end(self, targs_, state, control, **kw):
                if int(round(state.epoch)) == 1 and not self.fired:
                    self.fired = True
                    was_training = model.training
                    model.eval()
                    result = tripwire.check()
                    if was_training:
                        model.train()
                    print(f"[TRIPWIRE] rate={result['rate']:.2%} halt={result['halt']}")
                    Path(args.out, "orientation_tripwire_epoch1.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2))
                    if result["halt"]:
                        raise RuntimeError(
                            "ORIENTATION TRIPWIRE: >=5% pairs exceed Jaccard "
                            "threshold — halting per plan W4.3")

        trainer.add_callback(TripwireCallback())

    trainer.train()

    out = Path(args.out)
    final_dir = out / ("smoke_final" if smoke else "best")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    loss = next((h.get("eval_loss") or h.get("loss")
                 for h in reversed(trainer.state.log_history)
                 if h.get("eval_loss") or h.get("loss") is not None), None)
    meta = {
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                   text=True, cwd=ROOT).stdout.strip(),
        "args": vars(args),
        "dtype": dtype,
        "device": device,
        "transformers_version": transformers.__version__,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "final_loss": loss,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"done -> {final_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
