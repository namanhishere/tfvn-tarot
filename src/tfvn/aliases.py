"""Canonical RWS card names and historical alias table.

Mathers (1888) and other PD sources use different card names. Joins MUST go
through this table by name — never by Mathers' numeric ordering (minors run
King→Ace descending from 22).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Canonical RWS names in tarotoo / StarTarotOnline order (card_id 0–77).
CANONICAL_NAMES: List[str] = [
    "The Fool",
    "The Magician",
    "The High Priestess",
    "The Empress",
    "The Emperor",
    "The Hierophant",
    "The Lovers",
    "The Chariot",
    "Strength",
    "The Hermit",
    "Wheel of Fortune",
    "Justice",
    "The Hanged Man",
    "Death",
    "Temperance",
    "The Devil",
    "The Tower",
    "The Star",
    "The Moon",
    "The Sun",
    "Judgement",
    "The World",
    # Wands
    "Ace of Wands",
    "Two of Wands",
    "Three of Wands",
    "Four of Wands",
    "Five of Wands",
    "Six of Wands",
    "Seven of Wands",
    "Eight of Wands",
    "Nine of Wands",
    "Ten of Wands",
    "Page of Wands",
    "Knight of Wands",
    "Queen of Wands",
    "King of Wands",
    # Cups
    "Ace of Cups",
    "Two of Cups",
    "Three of Cups",
    "Four of Cups",
    "Five of Cups",
    "Six of Cups",
    "Seven of Cups",
    "Eight of Cups",
    "Nine of Cups",
    "Ten of Cups",
    "Page of Cups",
    "Knight of Cups",
    "Queen of Cups",
    "King of Cups",
    # Swords
    "Ace of Swords",
    "Two of Swords",
    "Three of Swords",
    "Four of Swords",
    "Five of Swords",
    "Six of Swords",
    "Seven of Swords",
    "Eight of Swords",
    "Nine of Swords",
    "Ten of Swords",
    "Page of Swords",
    "Knight of Swords",
    "Queen of Swords",
    "King of Swords",
    # Pentacles
    "Ace of Pentacles",
    "Two of Pentacles",
    "Three of Pentacles",
    "Four of Pentacles",
    "Five of Pentacles",
    "Six of Pentacles",
    "Seven of Pentacles",
    "Eight of Pentacles",
    "Nine of Pentacles",
    "Ten of Pentacles",
    "Page of Pentacles",
    "Knight of Pentacles",
    "Queen of Pentacles",
    "King of Pentacles",
]

assert len(CANONICAL_NAMES) == 78
assert len(set(CANONICAL_NAMES)) == 78

NAME_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(CANONICAL_NAMES)}

# Historical / alternate names → canonical name.
# Keep injective after normalisation: each alias maps to exactly one canonical.
# Canonical names are also aliases of themselves (total coverage).
_RAW_ALIASES: Dict[str, str] = {
    # Majors
    "The Juggler": "The Magician",
    "The Magus": "The Magician",
    "The Foolish Man": "The Fool",
    "Fortitude": "Strength",
    "Themis": "Justice",
    "Justice (Themis)": "Justice",
    "The Last Judgment": "Judgement",
    "The Last Judgement": "Judgement",
    "Judgment": "Judgement",
    "The Universe": "The World",
    "The Hierophant or Pope": "The Hierophant",
    "The Pope": "The Hierophant",
    "The Hierophant or the Pope": "The Hierophant",
    "Strength, or Fortitude": "Strength",
    "Themis, or Justice": "Justice",
    "The Lightning-struck Tower": "The Tower",
    "The Lightning Struck Tower": "The Tower",
    "The Wheel of Fortune": "Wheel of Fortune",
    # Source typos / OCR variants
    "The Charriot": "The Chariot",
    "Charriot": "The Chariot",
    "Judgement": "Judgement",
    "The High Priestess": "The High Priestess",
    # Suits
    "Sceptres": "Wands",
    "Batons": "Wands",
    "Coins": "Pentacles",
    "Deniers": "Pentacles",
    # Court
    "Knave": "Page",
    "Princess": "Page",
    "Valet": "Page",
    # Rank synonyms
    "Deuce": "Two",
    # Full Mathers-style minor names
    "King of Sceptres": "King of Wands",
    "Queen of Sceptres": "Queen of Wands",
    "Knight of Sceptres": "Knight of Wands",
    "Knave of Sceptres": "Page of Wands",
    "Ace of Sceptres": "Ace of Wands",
    "Deuce of Sceptres": "Two of Wands",
    "Two of Sceptres": "Two of Wands",
    "Three of Sceptres": "Three of Wands",
    "Four of Sceptres": "Four of Wands",
    "Five of Sceptres": "Five of Wands",
    "Six of Sceptres": "Six of Wands",
    "Seven of Sceptres": "Seven of Wands",
    "Eight of Sceptres": "Eight of Wands",
    "Nine of Sceptres": "Nine of Wands",
    "Ten of Sceptres": "Ten of Wands",
    "King of Cups": "King of Cups",
    "Queen of Cups": "Queen of Cups",
    "Knight of Cups": "Knight of Cups",
    "Knave of Cups": "Page of Cups",
    "Ace of Cups": "Ace of Cups",
    "Deuce of Cups": "Two of Cups",
    "King of Swords": "King of Swords",
    "Queen of Swords": "Queen of Swords",
    "Knight of Swords": "Knight of Swords",
    "Knave of Swords": "Page of Swords",
    "Ace of Swords": "Ace of Swords",
    "Deuce of Swords": "Two of Swords",
    "King of Pentacles": "King of Pentacles",
    "Queen of Pentacles": "Queen of Pentacles",
    "Knight of Pentacles": "Knight of Pentacles",
    "Knave of Pentacles": "Page of Pentacles",
    "Ace of Pentacles": "Ace of Pentacles",
    "Deuce of Pentacles": "Two of Pentacles",
    "King of Coins": "King of Pentacles",
    "Queen of Coins": "Queen of Pentacles",
    "Knight of Coins": "Knight of Pentacles",
    "Knave of Coins": "Page of Pentacles",
    "Ace of Coins": "Ace of Pentacles",
    "Deuce of Coins": "Two of Pentacles",
    "Two of Coins": "Two of Pentacles",
    "Three of Coins": "Three of Pentacles",
    "Four of Coins": "Four of Pentacles",
    "Five of Coins": "Five of Pentacles",
    "Six of Coins": "Six of Pentacles",
    "Seven of Coins": "Seven of Pentacles",
    "Eight of Coins": "Eight of Pentacles",
    "Nine of Coins": "Nine of Pentacles",
    "Ten of Coins": "Ten of Pentacles",
}


def _norm(name: str) -> str:
    return " ".join(name.strip().lower().replace("—", "-").replace("–", "-").split())


def build_alias_table() -> Dict[str, str]:
    """Return normalised_alias → canonical_name. Total over 78 + injective."""
    table: Dict[str, str] = {}
    # Self-maps first
    for canon in CANONICAL_NAMES:
        table[_norm(canon)] = canon
    # Explicit aliases
    for alias, canon in _RAW_ALIASES.items():
        if canon not in NAME_TO_ID and canon not in ("Wands", "Pentacles", "Page", "Two"):
            # suit/rank fragments are intermediate; expand below if full name
            if canon not in CANONICAL_NAMES:
                continue
        if canon in CANONICAL_NAMES:
            key = _norm(alias)
            if key in table and table[key] != canon:
                raise ValueError(f"alias collision: {alias!r} → {canon!r} vs {table[key]!r}")
            table[key] = canon
    # Expanded Deuce/Knave/Sceptres/Coins full names for all minors
    suits = {
        "wands": "Wands",
        "sceptres": "Wands",
        "batons": "Wands",
        "cups": "Cups",
        "swords": "Swords",
        "pentacles": "Pentacles",
        "coins": "Pentacles",
        "deniers": "Pentacles",
    }
    ranks = {
        "ace": "Ace",
        "deuce": "Two",
        "two": "Two",
        "three": "Three",
        "four": "Four",
        "five": "Five",
        "six": "Six",
        "seven": "Seven",
        "eight": "Eight",
        "nine": "Nine",
        "ten": "Ten",
        "page": "Page",
        "knave": "Page",
        "princess": "Page",
        "valet": "Page",
        "knight": "Knight",
        "queen": "Queen",
        "king": "King",
    }
    for r_alias, r_canon in ranks.items():
        for s_alias, s_canon in suits.items():
            full_alias = f"{r_alias} of {s_alias}"
            full_canon = f"{r_canon} of {s_canon}"
            if full_canon not in NAME_TO_ID:
                continue
            key = _norm(full_alias)
            if key in table and table[key] != full_canon:
                raise ValueError(f"alias collision: {full_alias!r}")
            table[key] = full_canon
    return table


ALIAS_TABLE: Dict[str, str] = build_alias_table()


def resolve_name(name: str) -> str:
    """Map any known alias to canonical name; raise KeyError if unknown."""
    key = _norm(name)
    if key in ALIAS_TABLE:
        return ALIAS_TABLE[key]
    # strip leading articles already handled; try without "the "
    if key.startswith("the "):
        alt = key[4:]
        if alt in ALIAS_TABLE:
            return ALIAS_TABLE[alt]
    # try with "the "
    if not key.startswith("the "):
        alt = "the " + key
        if alt in ALIAS_TABLE:
            return ALIAS_TABLE[alt]
    raise KeyError(f"unknown card name: {name!r}")


def resolve_id(name: str) -> int:
    return NAME_TO_ID[resolve_name(name)]


def alias_table_for_export() -> List[dict]:
    """Stable sorted list of {alias, canonical, card_id} for whitelist / JSON."""
    rows = []
    seen: set[Tuple[str, str]] = set()
    for alias_norm, canon in sorted(ALIAS_TABLE.items(), key=lambda x: (x[1], x[0])):
        pair = (alias_norm, canon)
        if pair in seen:
            continue
        seen.add(pair)
        rows.append(
            {
                "alias": alias_norm,
                "canonical": canon,
                "card_id": NAME_TO_ID[canon],
            }
        )
    return rows


def assert_alias_table_total_injective() -> None:
    """Validate total coverage of 78 canonicals and injectivity of alias map."""
    table = ALIAS_TABLE
    # Total: every canonical is reachable via its own norm key
    for canon in CANONICAL_NAMES:
        if _norm(canon) not in table:
            raise AssertionError(f"alias table not total: missing {canon}")
        if table[_norm(canon)] != canon:
            raise AssertionError(f"self-map broken for {canon}")
    # Injective: no two distinct normalised aliases map to different cards if
    # they were the same string (dict enforces); check no alias maps to two
    # names (impossible with dict). Check two different aliases of same string
    # don't exist. Also: inverse injectivity for distinct aliases that are
    # equal under norm is already handled.
    # Check that no single alias key maps to two names — dict property.
    # Check that canonical set covered has size 78
    covered = set(table.values())
    if covered != set(CANONICAL_NAMES):
        missing = set(CANONICAL_NAMES) - covered
        extra = covered - set(CANONICAL_NAMES)
        raise AssertionError(f"coverage error missing={missing} extra={extra}")
    # Injectivity of the *forward* map is a dict; injectivity of inverse is NOT
    # required (many aliases → one card). Plan: "no two aliases point to the
    # same name" is WRONG for real alias tables — plan said:
    # "injective (no two aliases point to the same name, no alias maps to two names)"
    # Re-read carefully:
    # "alias table is total (covers all 78 canonical names) and injective
    # (no two aliases point to the same name, no alias maps to two names)"
    #
    # That wording is contradictory with having Magician = The Juggler.
    # Interpret as: the *canonical self-map portion* is bijective, and the
    # alias relation as a whole is a function (each alias → one name).
    # We enforce function + total; multi-alias→one-card is required.
    return
