"""Rule-based expert system (Section 7): knowledge base + rules + hybrid decision.

Design: knowledge lives in YAML (knowledge_base/), logic lives here. Every output carries the
rule id, message and source so the UI can show "which rules fired".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import config
from .parser import Ingredient

ALLERGEN_FALSE_POSITIVES = ["cocoa butter", "shea butter", "nutmeg", "coconut milk", "butternut", "peanut butter"]
E_RE = re.compile(r"\be\d{3,4}[a-z]?\b")


@dataclass
class Match:
    entry: dict
    ingredient: str
    position: int


@dataclass
class Finding:
    kind: str            # "rule" | "additive" | "allergen"
    id: str
    message: str
    severity: int
    source: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class ExpertResult:
    matches: list[Match]
    findings: list[Finding]
    allergens: dict[str, list[str]]
    nova4_evidence: int
    nova4_markers: list[str]


class ExpertSystem:
    def __init__(self, kb_dir: Path | str = config.KB_DIR):
        kb_dir = Path(kb_dir)
        kb = yaml.safe_load((kb_dir / "additives.yaml").read_text(encoding="utf-8"))
        rules = yaml.safe_load((kb_dir / "rules.yaml").read_text(encoding="utf-8"))
        self.kb_version = kb.get("version", "?")
        self.rules_version = rules.get("version", "?")
        self.entries: dict[str, dict] = {e["id"]: e for e in kb["entries"]}
        self.rules: list[dict] = rules["rules"]
        self.allergen_words: dict[str, list[str]] = rules["allergens"]
        self.preferences: dict[str, dict] = rules["preferences"]
        # alias -> entry id ; E-numbers are aliases too (lowercase)
        self.alias_map: dict[str, str] = {}
        for e in self.entries.values():
            for a in [e["name"], *e.get("aliases", []), *[n.lower() for n in e.get("e_numbers", [])]]:
                self.alias_map[a.lower()] = e["id"]
        self._aliases_by_len = sorted(self.alias_map, key=len, reverse=True)
        # One alternation (longest-first) instead of a regex per alias: same result, far fewer searches.
        self._alias_re = (
            re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(a) for a in self._aliases_by_len) + r")(?![a-z0-9])")
            if self._aliases_by_len else None
        )
        self._long_aliases = [a for a in self.alias_map if len(a) >= 6]  # cached for the fuzzy fallback

    # ---------- matching ----------
    def vocabulary(self) -> list[str]:
        return list(self.alias_map)

    def _match_one(self, name: str) -> str | None:
        name = name.lower()
        if name in self.alias_map:
            return self.alias_map[name]
        if self._alias_re is not None:            # longest alias first
            m = self._alias_re.search(name)
            if m:
                return self.alias_map[m.group(1)]
        if self._long_aliases:                    # optional fuzzy fallback
            try:
                from rapidfuzz import process
                hit = process.extractOne(name, self._long_aliases, score_cutoff=92)
                if hit:
                    return self.alias_map[hit[0]]
            except ImportError:
                pass
        return None

    def match_ingredients(self, ingredients: list[Ingredient]) -> list[Match]:
        seen, out = set(), []
        for ing in ingredients:
            eid = self._match_one(ing.name)
            if eid and eid not in seen:
                seen.add(eid)
                out.append(Match(self.entries[eid], ing.name, ing.position))
        return out

    # ---------- allergens (always rule-based) ----------
    def detect_allergens(self, ingredients: list[Ingredient]) -> dict[str, list[str]]:
        text = " , ".join(i.name for i in ingredients).lower()
        for fp in ALLERGEN_FALSE_POSITIVES:
            text = text.replace(fp, " ")
        found: dict[str, list[str]] = {}
        for group, words in self.allergen_words.items():
            hits = [w for w in words if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", text)]
            if hits:
                found[group] = hits
        return found

    # ---------- rules ----------
    def _eval(self, when: dict, matches: list[Match], ingredients: list[Ingredient] = ()) -> list[str] | None:
        """Return evidence list if the condition holds, else None."""
        t = when["type"]
        cats = {m.entry["category"] for m in matches}
        if t == "all_categories_present":
            if set(when["categories"]) <= cats:
                return [m.ingredient for m in matches if m.entry["category"] in when["categories"]]
        elif t == "ingredient_present":
            ev = [m.ingredient for m in matches if m.entry["id"] in when["ids"]]
            return ev or None
        elif t == "category_in_first_n":
            ev = [m.ingredient for m in matches
                  if m.entry["category"] == when["category"] and m.position < when["n"]]
            return ev or None
        elif t == "category_count_at_least":     # distinct ingredient NAMES (e.g. sugar split across several names)
            names = sorted({i.name for i in ingredients
                            if (eid := self._match_one(i.name)) and self.entries[eid]["category"] == when["category"]})
            return names if len(names) >= when["count"] else None
        elif t == "nova4_markers_at_least":
            ev = [m.ingredient for m in matches if m.entry.get("nova4_marker")]
            return ev if len(ev) >= when["count"] else None
        else:
            raise ValueError(f"Unknown rule type: {t}")
        return None

    def analyze(self, ingredients: list[Ingredient]) -> ExpertResult:
        matches = self.match_ingredients(ingredients)
        findings: list[Finding] = []
        evidence_total = 0
        for rule in self.rules:
            ev = self._eval(rule["when"], matches, ingredients)
            if ev:
                findings.append(Finding("rule", rule["id"], rule["message"], rule["severity"], rule["source"], ev))
                evidence_total += rule.get("effect", {}).get("nova4_evidence", 0)
        for m in matches:
            e = m.entry
            if e["rating"] in ("caution", "avoid") and e["severity"] >= 1:
                findings.append(Finding("additive", e["id"], f'{e["name"]}: {e["reason"]}',
                                        e["severity"], e["source"], [m.ingredient]))
        allergens = self.detect_allergens(ingredients)
        for g, words in allergens.items():
            findings.append(Finding("allergen", f"ALLERGEN_{g}", f"Contains declared allergen source: {g.replace('_', ' ')}.",
                                    2, "EU Regulation 1169/2011 Annex II (keyword match)", words))
        markers = sorted({m.entry["id"] for m in matches if m.entry.get("nova4_marker")})
        findings.sort(key=lambda f: -f.severity)
        return ExpertResult(matches, findings, allergens, evidence_total, markers)

    # ---------- hybrid decision ----------
    def hybrid_nova(self, ml_pred: int, result: ExpertResult, threshold: int = 2) -> tuple[int, list[str]]:
        """Rules may only RAISE the ML prediction to 4, and must state why."""
        if result.nova4_evidence >= threshold and ml_pred < 4:
            fired = [f.id for f in result.findings if f.kind == "rule"]
            return 4, [f"Rule evidence raised NOVA {ml_pred} -> 4 (rules fired: {', '.join(fired)})."]
        return ml_pred, []

    # ---------- rules-only NOVA (ablation baseline: the expert system without any ML) ----------
    CULINARY = {"sugar", "salt", "sea salt", "honey", "maple syrup", "butter", "lard", "vinegar", "olive oil",
                "sunflower oil", "rapeseed oil", "vegetable oil", "coconut oil", "starch", "corn starch"}
    PROCESSING_ADDS = ("salt", "sugar", "oil", "vinegar", "syrup", "yeast", "brine", "cheese")

    def rules_only_nova(self, ingredients: list[Ingredient], result: ExpertResult | None = None,
                        threshold: int = 2) -> int:
        """Hand-written NOVA rules (Monteiro et al. definitions), used only for the ML/rules/hybrid ablation:
        4 if rule evidence or any NOVA-4 marker; 2 if the product is a single culinary ingredient; 3 if a food has
        added salt/sugar/oil-type ingredients; otherwise 1."""
        result = result or self.analyze(ingredients)
        if result.nova4_evidence >= threshold or result.nova4_markers or any(E_RE.search(i.name) for i in ingredients):
            return 4
        top = [i.name for i in ingredients if i.depth == 0]
        if len(top) == 1 and top[0] in self.CULINARY:
            return 2
        if len(top) > 1 and any(any(w in n for w in self.PROCESSING_ADDS) for n in top):
            return 3
        return 1

    # ---------- preferences ----------
    def violates_preference(self, ingredients: list[Ingredient], prefs: list[str]) -> list[str]:
        """Return the list of violated preference names."""
        matches = self.match_ingredients(ingredients)
        ids = {m.entry["id"] for m in matches}
        cats = {m.entry["category"] for m in matches}
        text = " , ".join(i.name for i in ingredients).lower()
        bad = []
        for p in prefs:
            rule = self.preferences.get(p)
            if not rule:
                continue
            if (set(rule.get("ingredient_ids", [])) & ids or set(rule.get("categories", [])) & cats or
                    any(re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", text) for k in rule.get("keywords", []))):
                bad.append(p)
        return bad
