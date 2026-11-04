"""Keep the two targets' feature sets straight (validation report 2.2).

- NOVA model: OFF computes NOVA from the ingredient list, so OFF-derived ingredient columns must not be features.
- Health-tier model: OFF computes Nutri-Score from the nutrition table, so nutriment columns must not be features.
Ingredient TEXT is the legitimate input for both. OFF-derived tags are still fine for FILTERING alternatives.
"""
from __future__ import annotations

OFF_INGREDIENT_DERIVED = {
    "nova_group", "additives_n", "additives_tags", "ingredients_n", "ingredients_analysis_tags", "analysis_tags",
    "ingredients_from_palm_oil_n", "allergens_tags", "traces_tags",
}
NUTRIMENT_NAMES = {"energy-kcal", "energy", "sugars", "fat", "saturated-fat", "salt", "sodium", "proteins", "fiber",
                   "carbohydrates", "nutriments", "nutriscore_score", "nutriscore_grade", "nutrition_score"}


def forbidden_features(columns, target: str) -> list[str]:
    cols = [str(c) for c in columns]
    if target == "nova_group":
        return [c for c in cols if c in OFF_INGREDIENT_DERIVED]
    if target == "health_tier":
        return [c for c in cols if c.endswith("_100g") or c in NUTRIMENT_NAMES]
    return []


def check_features(columns, target: str) -> None:
    bad = forbidden_features(columns, target)
    if bad:
        raise ValueError(f"Leaky features for target '{target}': {bad}")
