"""Ingredient text cleaning and parsing (Section 5 of the plan).

Pipeline: (optional) OCR fixes -> ingredient-section extraction -> normalisation ->
bracket-aware comma splitting -> list of Ingredient objects.
Everything here is pure Python so it can be unit-tested without OCR or ML libraries.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# words that only describe a function; dropped when they have bracketed sub-ingredients
FUNCTION_WORDS = {
    "colour", "colours", "color", "colors", "preservative", "preservatives", "antioxidant",
    "antioxidants", "emulsifier", "emulsifiers", "stabiliser", "stabilisers", "stabilizer",
    "stabilizers", "acidity regulator", "acidity regulators", "raising agent", "raising agents",
    "flavour enhancer", "flavor enhancer", "sweetener", "sweeteners", "thickener", "gelling agent",
    "anti-caking agent", "firming agent", "humectant", "acid", "acids",
}

START_MARKERS = re.compile(r"\b(?:ingr[eé]dient[s]?|zutaten|ingredientes)\s*[:\-]?", re.I)
STOP_MARKERS = re.compile(
    r"\b(?:nutrition(?:al)?\s+(?:information|facts|values?)|allergen(?:s)?\s*(?:advice|information)?\s*[:\-]|"
    r"storage|store\s+in|best\s+before|use\s+by|manufactured|distributed|directions|net\s+w(?:t|eight)|"
    r"for\s+allergens|typical\s+values|per\s+100\s*g)\b",
    re.I,
)
TAIL_MARKERS = re.compile(
    r"\b(?:may\s+contain|allergy\s+advice|allergen\s+information|produced\s+in\s+a\s+factory|"
    r"packed\s+in\s+a\s+factory|manufactured\s+in|traces?\s+of)\b.*$",
    re.I | re.S,
)
CONTAINS_COLON = re.compile(r"\bcontains\s*:.*$", re.I | re.S)
# US label variants: "contains 2% or less of:", "contains less than 2% of:", "2% or less of:"
LESS_THAN = re.compile(
    r"(?:contains\s+)?(?:less\s+than\s+)?\d+(?:[.,]\d+)?\s*%\s*(?:or\s+less\s+)?of\s*:?"
    r"|contains\s+\d+(?:[.,]\d+)?\s*%?\s*or\s+less\s+of\s*:?",
    re.I,
)
# USDA-only noise: footnote asterisks and embedded curing disclaimers
FOOTNOTE = re.compile(r"\*+")
CURING_NOTE = re.compile(r"[\s\-]*no\s+nitrates\s+or\s+nitrites\s+added[^,;)]*", re.I)
PERCENT = re.compile(r"\(?\s*\d+(?:[.,]\d+)?\s*%\s*\)?")
E_NUMBER = re.compile(r"\be[\s\-]?(\d{3,4}[a-z]?)\b", re.I)


@dataclass
class Ingredient:
    name: str
    position: int      # index among TOP-LEVEL ingredients (order ~ quantity)
    depth: int = 0     # 0 = top level, 1+ = inside brackets
    parent: str | None = None


def fix_ocr_confusions(text: str) -> str:
    """Repair common character confusions only when surrounded by letters (0->o, 1->l, |->l)."""
    text = re.sub(r"(?<=[a-z])0(?=[a-z])", "o", text, flags=re.I)
    text = re.sub(r"(?<=[a-z])1(?=[a-z])", "l", text, flags=re.I)
    text = re.sub(r"(?<=[a-z])\|(?=[a-z])", "l", text, flags=re.I)
    return text


def spell_correct(token: str, vocab: list[str], cutoff: int = 90) -> str:
    """Optional fuzzy correction against an ingredient vocabulary (needs rapidfuzz)."""
    try:
        from rapidfuzz import process
    except ImportError:
        return token
    if not vocab or len(token) < 5:
        return token
    hit = process.extractOne(token, vocab, score_cutoff=cutoff)
    return hit[0] if hit else token


def extract_ingredient_section(raw: str) -> str:
    """Pull the ingredient list out of full-label OCR text."""
    m = START_MARKERS.search(raw)
    text = raw[m.end():] if m else raw
    s = STOP_MARKERS.search(text)
    if s:
        text = text[: s.start()]
    return text.strip()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.lower().replace("\n", " ").replace(";", ",")
    text = re.sub(r"^\s*(?:ingr[eé]dient[s]?|zutaten|ingredientes)\s*[:\-]?\s*", "", text)
    text = FOOTNOTE.sub("", text)
    text = CURING_NOTE.sub("", text)
    text = LESS_THAN.sub(",", text)
    text = TAIL_MARKERS.sub("", text)
    text = CONTAINS_COLON.sub("", text)
    text = PERCENT.sub(" ", text)
    text = E_NUMBER.sub(lambda m: f"e{m.group(1).lower()}", text)
    text = re.sub(r"[\[\{]", "(", text)
    text = re.sub(r"[\]\}]", ")", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:-")
    return text


def split_top_level(text: str) -> list[str]:
    """Split on commas that are NOT inside brackets."""
    parts, depth, buf = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return [p.strip(" .:-") for p in parts if p.strip(" .:-")]


_FUNCTION_PREFIX = re.compile(
    r"^(" + "|".join(sorted(map(re.escape, FUNCTION_WORDS), key=len, reverse=True)) + r")"
    r"(?:\s*:\s*(?P<named>.+)|\s+(?P<enum>e\d{3,4}[a-z]?))$"
)


def _strip_function_word(name: str) -> str:
    """'emulsifier: soy lecithin' -> 'soy lecithin', 'colour e102' -> 'e102' (the function word adds no ingredient)."""
    m = _FUNCTION_PREFIX.match(name)
    return (m.group("named") or m.group("enum")).strip(" .:-") if m else name


def _expand(item: str, position: int, depth: int, parent: str | None, out: list[Ingredient]) -> None:
    m = re.match(r"^(.*?)\((.*)\)\s*(.*)$", item)
    if not m:
        name = _strip_function_word(item.strip(" .:-"))
        if name:
            out.append(Ingredient(name, position, depth, parent))
        return
    head, inner, tail = (m.group(1).strip(" .:-"), m.group(2), m.group(3).strip(" .:-"))
    subs = split_top_level(inner)
    if head and not (head in FUNCTION_WORDS and subs):
        out.append(Ingredient(head, position, depth, parent))
    for sub in subs:
        _expand(sub, position, depth + 1, head or parent, out)
    if tail:
        out.append(Ingredient(tail, position, depth, parent))


def parse_ingredients(text: str, vocab: list[str] | None = None) -> list[Ingredient]:
    """Raw label text -> ordered Ingredient objects (top-level first, sub-ingredients after their parent)."""
    cleaned = normalize_text(fix_ocr_confusions(text or ""))
    out: list[Ingredient] = []
    for pos, item in enumerate(split_top_level(cleaned)):
        _expand(item, pos, 0, None, out)
    if vocab:
        for ing in out:
            ing.name = spell_correct(ing.name, vocab)
    return out


def ingredient_names(text: str, vocab: list[str] | None = None) -> list[str]:
    return [i.name for i in parse_ingredients(text, vocab)]


def normalized_string(text: str) -> str:
    """Canonical string used for deduplication and as model input."""
    return ", ".join(ingredient_names(text))
