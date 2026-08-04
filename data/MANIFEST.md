# Vietnamese Tarot LLM — Dataset Manifest

Downloaded `2026-08-01`. Total **490 MB**. Every item below is license-clean for a
commercial product, or is public domain. Nothing scraped, nothing NC/SA-encumbered.

Field counts, row counts and reversed-meaning coverage were verified by opening the
files, not read from READMEs.

---

## Layer 1 — English factual backbone (RAG knowledge base)

| Path | Source | License | Verified contents |
|---|---|---|---|
| `github/tarotoo_cards.json` | Tarotoo-com/tarotoo-tarot-dataset | **MIT** | 78 cards × 22 fields. **7 `_reversed` fields** (`meaning`, `keywords`, `love`, `career`, `mood`, `spiritual`, `yes_no`) |
| `hf/Tarotoo__tarotoo-tarot-card-meanings/` | HF mirror of above | **MIT** | same, parquet/csv |
| `hf/StarTarotOnline__tarot-rws-historical-meanings/` | HF | **CC BY 4.0** | 78 rows. `divinatory_meanings{upright, reversed, upright_section, reversed_section, source_id, editorial_note}` — Waite 1910 verbatim **with per-field citations** |
| `hf/Blacik__deckaura-tarot-card-meanings/` | HF | **MIT** | 78 rows, terse keyword style — use as third-opinion cross-check |
| `github/lindseyb_tarot.json` | LindseyB/tarot-api | **MIT** | 78 cards, upright/reversed keyword arrays + astrology |
| `github/smallcat419_index.json` | smallcat419 | CC0 *(README only — no LICENSE file)* | 78 cards, reversed × 4 domains |
| `github/plateau_cards.json` | geraldfingburke | **Unlicense** | 78 cards, **no reversed** |
| `github/tarotapi_dev_cards.json` | tarotapi.dev bulk export | repo unlicensed; text = PD Waite 1910 | 78 cards |

**Start with `tarotoo_cards.json`.** Every field has a `_reversed` twin, so the Vietnamese
layer is a mechanical 1:1 key mapping → ~1,700 translation units.

## Layer 2 — Spreads & extensible decks

| Path | License | Verified contents |
|---|---|---|
| `github/tarotschema_spreads.json` | **MIT** + CC BY 4.0 | **21 spreads** with `card_positions` prose meanings, `cards_drawn`, `difficulty`, `reading_variants`, elemental-dignity rules |
| `github/tarotschema_decks.json` | same | **7 decks**, incl. bifrost (80 cards) and Marseille (86) — directly covers your "add more cards" requirement |
| `hf/tarotsmith__TarotSchema/` | same | HF mirror |
| `hf/build-small-hackathon__arcana-deliverables/` | **MIT** | Agent traces of an LLM *inventing* a custom themed 22-card deck, incl. the JSON schemas that constrain generation |
| `hf/AskRunes__elder-futhark-rune-meanings/` | **CC BY 4.0** | Not tarot — included for its schema: `reversible` + `visually_symmetrical` booleans formalize *when a reversal is even meaningful*. Copy this for non-standard cards |

Only 3 real spread sources exist; most "spread" repos just shuffle without positional
meaning. TarotSchema is the backbone.

## Layer 3 — Vietnamese

| Path | License | Verified contents |
|---|---|---|
| `vietnamese/Tarot-Vietnamese-API/data.txt` | **MIT** | **156 JSONL records = 78 upright + 78 `ngược`** (verified split). Fields: `title_main`, `title_secondary`, `title_love`, `title_work`, `title_money`, `title_heath` |
| `vietnamese/Tarot-Vietnamese-API/forward/` | MIT | 78 upright JPEGs |
| `vietnamese/Tarot-Vietnamese-API/reverse/` | MIT | 78 **pre-rotated** reversed JPEGs |
| `hf/jakeveo05__chinese-traditional-knowledge/` | **CC BY 4.0** | 69 MB. 54 Vietnamese TCM books + 11 I-Ching + 7 feng-shui. Largest clean-licensed **Vietnamese esoteric** corpus — for DAPT to fix Vietnamese mystical vocabulary |
| `hf/tokushukaijp__astrology-vn/` | none declared ⚠️ | 1,553 rows. Only Vietnamese divination instruct set. Grounding is broken — **mine for register + `<thinking>` format, do not train on the reasoning** |
| `hf/phat1231425__tuvi/` | none declared ⚠️ | Vietnamese tử vi Q&A — authentic VN divination terminology |

`title_heath` is misspelled in the source. Normalize on ingest.

## Layer 4 — Reading transcripts (teaches *reading*, not reciting)

| Path | License | Verified contents |
|---|---|---|
| `hf/Dendory__tarot/` | **MIT** | 5,770 rows `Card1,Card2,Card3,Reading`. Canonical English set; ChatGPT-generated. Load with explicit `data_files=` — no viewer config |
| `hf/tellang__yeji-processed/` | **MIT** | 177 MB, 27,735 rows `instruction/input/output/domain/source`. Filter `domain=="tarot"`. Best license:volume ratio for Asian-language work |
| `hf/sunkencity__tarot-oracle-instruct/` | none declared ⚠️ | 115 rows ChatML. **Only set that narrates the shuffle** (`*[The Oracle shuffles the deck... Draws: ...]*`) — the pattern your deck-fold feature needs. Few-shot seeds, not bulk SFT |

`Dendory/tarot` is upstream of several other "independent" sets — dedupe before combining.

## Layer 5 — Public domain texts (legal backbone, redistributable)

All pre-1931 → US public domain. Cutoff is **Jan 1, 1931** as of 2026, not 1929.

| Path | Year | Words | Per-card + reversed? |
|---|---|---|---|
| `pd-texts/waite_pictorial_key_1911.txt` | 1911 | 35,731 | **YES, all 78.** 57 `Divinatory Meanings` + 128 `Reversed` markers (verified). Human-proofread Wikisource, not OCR |
| `pd-texts/mathers_1888_03.html` | 1888 | 2,085 | **YES, all 78.** 106 `R.` reversal markers. Densest + cleanest parse in the corpus |
| `pd-texts/delaurence_illustrated_key_1918.txt` | 1918 | 39,420 | **YES, all 78** — Waite's text reprinted near-verbatim |
| `pd-texts/platt_card_fortune_telling_1920.txt` | c.1920 | 43,159 | YES (52-card) |
| `pd-texts/foli_fortune_telling_by_cards_1900.txt` | 1900 | 32,835 | YES (52-card) |
| `pd-texts/grand_etteilla_1900_fr.txt` | 1900 | 32,212 | YES (78-card Etteilla, French) |
| `pd-texts/mohammed_ali_telling_fortunes.txt` | early 1900s | 32,399 | YES |
| `pd-texts/cielo_fortunes_and_dreams_1918.txt` | 1918 | 39,890 | YES |

`mathers_1888_0{0,2,4}.html` are the surrounding chapters. sacred-texts.com now returns
402/403 to all automated clients — these came from pinned Wayback snapshots.

**Parsing:** Waite/de Laurence split on the `Divinatory Meanings:` / `Reversed:` delimiter
pair (56 minors direct, 22 trumps in §3). Mathers splits on `N. Name.-- upright; R. reversed`
and yields all 78 in one 2,200-word pass — cheapest high-quality seed for the Vietnamese
translation table.

**Etteilla uses its own deck names/numbering** — do not map to RWS card IDs without a
deliberate correspondence table.

## Layer 6 — Images

| Path | License | Verified |
|---|---|---|
| `hf/multimodalart__1920-raider-waite-tarot-public-domain-cleaned/train/` | **MIT** | **78 PNGs** + `metadata.csv`. Pamela Colman Smith 1909/1911 → PD in US (pre-1931) and life+70 (d. 1951) since 2022 |

Higher-res fallback if needed: `https://upload.wikimedia.org/wikipedia/commons/9/99/The_Pictorial_Key_to_the_Tarot.pdf` (76 MB, full 1922 scan).

---

## Deliberately NOT downloaded

**Copyright / license blockers:**
- `sangde/tarot` — richest Vietnamese structure found, but an unlicensed scrape of tarot.vn
  which is itself a translation of Joan Bunning's *Learning the Tarot*. **Double exposure.**
- `mixvlad/TarotCards` — 630 images, but **CC BY-NC 4.0** forbids commercial use. The
  underlying 1909 scans are PD, so re-source from Wikimedia instead.
- `searge/tarot` — **CC BY-SA 4.0** would force your derived Vietnamese dataset share-alike.
- `dariusk/corpora` tarot file — declares CC0 but credits Mark McElroy's copyrighted text.
  A repo owner cannot CC0 a third party's work. Also its `light`/`shadow` axis is **not**
  upright/reversed — mapping shadow→reversed would be a factual error in your KB.
- Mystic House Vietnamese translations (*78 Độ Minh Triết Tarot* et al.) — dual copyright
  (author + translator), active commercial licensee in your exact market. Buy them and
  hand-build the term-base; terminology is factual and not copyrightable, paragraphs are.
- `dilib.vn`, `thuviensach.vn` — distribute pirated copies of the licensed Pollack translation.
- **Crowley *Book of Thoth* (1944)** — in US copyright until 2040. PD in Vietnam (life+50,
  since 1998) but a redistributable corpus is reachable from the US.
- **Case *A Key to the Wisdom of the Ages*** — **1947, not 1927.** In copyright to ~2043.
- **Regardie *The Golden Dawn*** (1937–40) — in copyright to ~2033–36. Book T's status is
  genuinely uncertain; Waite + Mathers already give complete 78-card coverage, so skip it.
- `tarotguideonline.com` — serves `Content-Signal: ai-train=no`, an express Art. 4
  EU-DSM reservation. Excluded. Worth adding a Content-Signal check to any future crawler.
- **Bactrian-X (vi)** — CC BY-NC; contaminates commercial models and is a component of
  several aggregate Vietnamese sets. Check transitively.

**Quality rejects:**
- `fzlzjerry/tarot-mcp` — MIT and complete, but prose is template-generated ("...the
  constructive expression of the archetype's growth lesson" on every card). Keywords only.
- `11-47/The_Prophet_Tarot_Zodiac_50K` — largest but machine-templated; dedupe cuts >80%.
- `Brhiza/mingyu` (241★) — `tarotCards` is names-only, no meanings, and no LICENSE file.

---

## Known gaps (must be built, not downloaded)

1. **No Vietnamese tarot dataset exists anywhere** — not on HF, Kaggle, or GitHub beyond
   the 156-record phatjkk set. This corpus is the actual deliverable.
2. **No safety-stop data exists.** Nothing encodes "ask the cards whether this question
   should be answered." Synthesize from the ethics codes below.
3. **Reversed meanings are the most common gap** in tarot datasets. Verified per-file above.

### Safety-stop grounding (sources, not files)

- **TABI** https://tabi.org.uk/ethics/ — third-party questions are *"re-phrased or declined"*
  (note: rephrase **before** refuse); refer to qualified professionals; under-18 needs
  guardian consent; *"withdraw tactfully"*; *"readings aim to help the client take charge
  of their own life"* (anti-fatalism, use against death/doom questions).
- **ATA** (site dark; Wayback 2003) — *"legal, financial, medical, or psychological"* is a
  ready-made four-category refusal taxonomy. Earlier version: *"Tarot cards are **not**
  fortune telling cards"*, *"You alone are responsible for your own behavior."*
- Adaptable refusal datasets: `LibrAI/do-not-answer`, `PKU-Alignment/BeaverTails`,
  `allenai/wildguardmix`. Catalogue: https://safetyprompts.com
- **Also test over-refusal** (`allenai/or-bench`, XSTest). A tarot app that refuses
  "will I find love?" as a relationship prediction is broken. This failure mode matters
  as much as under-refusing.
- **Vietnam crisis resources — do not port US "call 988" scaffolding.** Only 5 services
  exist for ~100M people. Verified at https://findahelpline.com/countries/vn:

  | Service | Number | Hours |
  |---|---|---|
  | Ngày Mai (depression/suicide) | 096 306 1414 | **Wed–Sun 13:00–20:30 only** |
  | HOPE (suicide prevention) | 0865 044 400 | see site |
  | National Child Protection | **111** | 24/7 |
  | CSAGA (gender violence) | 024 3333 5599 | see site |
  | Ngôi nhà Bình yên (shelter) | 1900 969 680 | 24/7 |
  | Ambulance / Police | 115 / 113 | 24/7 |

  **Vietnam's main youth depression line is closed most of the week.** A 3 a.m. Tuesday
  crisis has no answer there. The crisis path must be **time-aware** and never present a
  closed line as the only option.

---

## Vietnamese output-style rule (highest-leverage single decision)

**Vietnamese readers write card names in English.** Verified across every credible source:

> "khi **The Lovers** xuất hiện ở trạng thái ngược"
> "**Queen of Cups** xuôi là người phụ nữ nhân hậu... thì ngược là người phụ nữ nhỏ nhen"

Vietnamese names (`Kẻ Khờ`, `Nữ Tư Tế`) appear **only as parenthetical glosses in lookup
tables**, never as the working name in a reading. If the model outputs "Kẻ Khờ ngược cho
thấy…" it reads as machine-translated. **Configure the term-base to leave card names
untranslated by default.**

Core terms: Major/Minor Arcana = **Ẩn Chính / Ẩn Phụ** (never "Arcana Lớn/Nhỏ");
suits = **Gậy / Cốc (or Ly) / Kiếm / Tiền (or Xu)**; court = **Tiểu Đồng / Hiệp Sĩ (Kị Sĩ)
/ Hoàng Hậu / Hoàng Đế**; upright/reversed = **xuôi / ngược**; spread = **trải bài**;
*querent* and *reader* **stay English**; pick-a-card = **chọn tụ** (modern YouTube term).

Encode this distinction — Vietnamese separates card *orientation* from *meaning*:
**bài xuôi/ngược** (the physical card) vs **nghĩa xuôi/ngược** (the interpretive register).
English has no clean equivalent; getting it right is a strong fluency signal.

## Base model note

Don't reflexively pick PhoGPT for being "most Vietnamese." Its 20K Vietnamese-tuned BPE
vocabulary tokenizes **untranslated English card names** badly — and per the rule above,
those appear constantly. A bilingual base (`Viet-Mistral/Vistral-7B-Chat`,
`SeaLLMs/SeaLLMs-v3-7B-Chat`, `sail/Sailor2-8B-Chat`) is likely the better fit. Run a
tokenizer comparison on real mixed VN/English reading text before committing.

## Cross-source disagreement

Waite's Star reversed is *"Arrogance, haughtiness, impotence"*; Mathers' differs. Both are
PD. Waite himself notes *"some of the readings cannot be harmonized."* **Store `source` +
`year` per chunk** rather than collapsing to one canonical meaning.
