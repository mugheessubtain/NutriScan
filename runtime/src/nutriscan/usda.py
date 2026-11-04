"""Plan v2 Section 4.1: make USDA Branded Foods a labelled dataset of its own (not only a barcode join).

1. Category (T3): map USDA `branded_food_category` to the same coarse OFF food groups with ordered keyword rules.
2. Health tier (T2): compute a Nutri-Score grade from USDA nutrients (per 100 g/ml for Branded Foods) with the 2017
   algorithm, fruit/vegetable/nut share set to 0 because USDA does not report it (stated as a limitation).
These labels are independent of OFF, so USDA products absent from OFF can be used for training/testing T2 and T3.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Ordered (first match wins) keyword rules on the lower-cased USDA category -> OFF food group (level 1)
CATEGORY_RULES: list[tuple[str, str]] = [
    (r"alcohol|beer|wine|liquor|spirit|cocktail mix(?!er)", "alcoholic-beverages"),
    (r"baby|infant|toddler", "baby-foods-and-milks"),
    (r"nut (?:& seed )?butter|peanut butter", "fats-and-sauces"),
    (r"breakfast sandwich", "composite-foods"),
    (r"ice cream|frozen yogurt|yogurt|milk(?! chocolate)|cheese|cream(?!er)|butter(?! ?milk)|dairy|kefir|cottage", "milk-and-dairy-products"),
    (r"candy|chocolate|confection|cookie|biscuit|cake|pastr|pie|donut|doughnut|dessert|sweet|jam|jelly|syrup|honey|"
     r"frosting|pudding|gelatin|granola bar|cereal bar|snack bar|popsicle|breakfast pastr", "sugary-snacks"),
    (r"chip|crisp|pretzel|popcorn|puffed|cracker|snack mix|nut|seed|trail mix|pork rind", "salty-snacks"),
    (r"soda|juice|drink|beverage|water|tea|coffee|smoothie|lemonade|cocoa mix|energy|sport", "beverages"),
    (r"sauce|dressing|ketchup|mustard|mayonnaise|oil|vinegar|condiment|dip|salsa|spread|gravy|marinade|seasoning|spice|"
     r"shortening|margarine|pickle", "fats-and-sauces"),
    (r"meat|beef|pork|chicken|turkey|poultry|sausage|hot ?dog|bacon|ham|jerky|fish|seafood|tuna|salmon|shrimp|egg|deli", "fish-meat-eggs"),
    (r"bread|bun|roll|bagel|tortilla|cereal|oat|grain|rice|pasta|noodle|flour|potato|baking|muffin|waffle|pancake|"
     r"stuffing|croutons", "cereals-and-potatoes"),
    (r"fruit|vegetable|bean|legume|lentil|tomato|produce|salad(?! dressing)|olives|mushroom|soup", "fruits-and-vegetables"),
    (r"frozen dinner|entree|meal|pizza|sandwich|prepared|deli salad|burrito|lasagna|dumpling|appetizer|side dish", "composite-foods"),
]
_COMPILED = [(re.compile(p), g) for p, g in CATEGORY_RULES]


def map_category(usda_category) -> str | None:
    if not isinstance(usda_category, str) or not usda_category.strip():
        return None
    c = usda_category.lower()
    for rx, group in _COMPILED:
        if rx.search(c):
            return group
    return None


def mapping_table(usda: pd.DataFrame) -> pd.DataFrame:
    """Documented USDA-category -> OFF-group mapping with product counts (goes in the report appendix)."""
    vc = usda["usda_category"].fillna("(missing)").value_counts()
    return pd.DataFrame({"usda_category": vc.index, "products": vc.values,
                         "off_food_group": [map_category(c) or "(unmapped)" for c in vc.index]})


# ---------------------------------------------------------------- Nutri-Score 2017 (general foods + beverages)
NUTRIENT_IDS = {1008: "energy_kcal_100g", 2000: "sugars_100g", 1258: "saturated_fat_100g", 1093: "sodium_mg_100g",
                1079: "fiber_100g", 1003: "proteins_100g"}
_ENERGY_KJ = [335 * i for i in range(1, 11)]
_SUGARS = [4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45]
_SATFAT = list(range(1, 11))
_SODIUM = [90 * i for i in range(1, 11)]
_FIBRE = [0.9, 1.9, 2.8, 3.7, 4.7]
_PROTEIN = [1.6, 3.2, 4.8, 6.4, 8.0]
_BEV_ENERGY_KJ = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]
_BEV_SUGARS = [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5]


def _pts(v, thresholds) -> int:
    return int(np.sum(np.asarray(thresholds) < v)) if pd.notna(v) else 0


def nutriscore_2017(kcal, sugars, satfat, sodium_mg, fibre, protein, beverage: bool = False) -> str | None:
    """Grade a-e from per-100 g values; None when energy, sugars, saturated fat or sodium is missing."""
    if any(pd.isna(v) for v in (kcal, sugars, satfat, sodium_mg)):
        return None
    kj = kcal * 4.184
    if beverage:
        n = _pts(kj, _BEV_ENERGY_KJ) + _pts(sugars, _BEV_SUGARS) + _pts(satfat, _SATFAT) + _pts(sodium_mg, _SODIUM)
    else:
        n = _pts(kj, _ENERGY_KJ) + _pts(sugars, _SUGARS) + _pts(satfat, _SATFAT) + _pts(sodium_mg, _SODIUM)
    fibre_p = _pts(fibre, _FIBRE)
    protein_p = _pts(protein, _PROTEIN) if n < 11 else 0     # fruit/veg points are 0, so protein only counts when N < 11
    score = n - fibre_p - protein_p
    if beverage:
        return "b" if score <= 1 else "c" if score <= 5 else "d" if score <= 9 else "e"
    return "a" if score <= -1 else "b" if score <= 2 else "c" if score <= 10 else "d" if score <= 18 else "e"


def read_usda_nutrients(path: str | Path, chunksize: int = 2_000_000) -> pd.DataFrame:
    """food_nutrient.csv (plain or inside the FDC zip) -> one row per fdc_id with the six Nutri-Score inputs."""
    path = Path(path)
    keep = set(NUTRIENT_IDS)
    kwargs = dict(usecols=["fdc_id", "nutrient_id", "amount"], dtype={"fdc_id": "int64", "nutrient_id": "int32",
                                                                     "amount": "float64"}, chunksize=chunksize)

    def chunks(f):
        for ch in pd.read_csv(f, **kwargs):
            yield ch[ch["nutrient_id"].isin(keep)]

    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            member = next((n for n in z.namelist() if n.endswith("/food_nutrient.csv") or n == "food_nutrient.csv"), None)
            if member is None:
                return pd.DataFrame(columns=["fdc_id", *NUTRIENT_IDS.values()])
            with z.open(member) as f:
                long = pd.concat(chunks(f), ignore_index=True)
    else:
        long = pd.concat(chunks(path), ignore_index=True)
    wide = long.pivot_table(index="fdc_id", columns="nutrient_id", values="amount", aggfunc="last")
    return wide.rename(columns=NUTRIENT_IDS).reset_index()


def label_usda(usda: pd.DataFrame, nutrients: pd.DataFrame | None) -> pd.DataFrame:
    """Add `category` (mapped OFF group), nutrient columns, `nutriscore_grade_usda` and `health_tier` (from USDA nutrients)."""
    out = usda.copy()
    out["category"] = out["usda_category"].map(map_category)
    if nutrients is not None and len(nutrients) and "fdc_id" in out:
        out["fdc_id"] = pd.to_numeric(out["fdc_id"], errors="coerce").astype("Int64")
        nutrients = nutrients.astype({"fdc_id": "Int64"})
        out = out.merge(nutrients, on="fdc_id", how="left")
        out["salt_100g"] = out["sodium_mg_100g"] * 2.5 / 1000
        bev = out["category"].eq("beverages")
        out["nutriscore_grade_usda"] = [
            nutriscore_2017(r.energy_kcal_100g, r.sugars_100g, r.saturated_fat_100g, r.sodium_mg_100g,
                            r.fiber_100g, r.proteins_100g, beverage=b)
            for r, b in zip(out.itertuples(index=False), bev)]
        out["health_tier"] = out["nutriscore_grade_usda"].map(lambda g: config.health_tier(g) if g else np.nan)
    return out
