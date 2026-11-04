"""Generate a SYNTHETIC OFF-like table and a USDA-like zip so every script runs without the real data.

  python training/make_sample_data.py [--n 3000] [--seed 42]

Writes
  data/sample/products_sample.csv          OFF-like rows (same columns prepare_off.py produces, incl. *_100g nutrients)
  data/sample/usda_branded_sample.zip      branded_food.csv + food_nutrient.csv in the FoodData Central layout

The numbers these files produce are meaningless. Never report them; they only prove the code runs end to end.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runtime" / "src"))
from nutriscan import config  # noqa: E402

# category -> (base ingredients, typical nutrients per 100 g: kcal, sugars, sat fat, salt, protein, fibre)
CATEGORIES = {
    "sugary-snacks": (["sugar", "wheat flour", "cocoa butter", "milk powder", "glucose syrup", "hazelnuts", "cocoa mass"],
                      (480, 35, 12, 0.3, 6, 3)),
    "beverages": (["water", "sugar", "orange juice", "lemon juice", "apple juice", "carbon dioxide"], (40, 9, 0, 0.02, 0.2, 0.2)),
    "cereals-and-potatoes": (["wheat flour", "whole grain oats", "potatoes", "rice", "corn", "barley malt", "salt"],
                             (360, 8, 1.5, 0.8, 10, 7)),
    "milk-and-dairy-products": (["milk", "cream", "cultures", "salt", "skimmed milk powder", "rennet"], (150, 5, 6, 0.6, 8, 0)),
    "fats-and-sauces": (["sunflower oil", "tomatoes", "vinegar", "salt", "mustard seeds", "egg yolk", "spices"],
                        (300, 6, 3, 1.8, 2, 1)),
    "fish-meat-eggs": (["pork", "chicken", "beef", "salt", "spices", "eggs", "tuna"], (220, 1, 6, 1.6, 18, 0)),
    "fruits-and-vegetables": (["tomatoes", "carrots", "peas", "apples", "onions", "spinach", "lemon juice"], (45, 5, 0.1, 0.2, 1.5, 3)),
    "salty-snacks": (["potatoes", "sunflower oil", "salt", "corn", "rice flour", "paprika"], (520, 2, 4, 1.5, 6, 4)),
    "composite-foods": (["water", "wheat flour", "cheese", "tomato puree", "chicken", "onions", "rapeseed oil"],
                        (210, 3, 4, 1.1, 9, 2)),
}
NOVA2 = ["sugar", "salt", "olive oil", "butter", "honey", "maple syrup", "sunflower oil"]
NOVA3_ADD = ["salt", "sugar", "vinegar", "vegetable oil", "yeast"]
NOVA4_ADD = ["glucose-fructose syrup", "flavouring", "emulsifier (soy lecithin)", "modified maize starch", "maltodextrin",
             "palm oil", "colour (e150d)", "e102", "aspartame", "acesulfame k", "sodium benzoate", "monosodium glutamate",
             "dextrose", "hydrogenated vegetable oil", "e471", "natural flavourings", "carrageenan", "xanthan gum",
             "invert sugar syrup", "whey powder", "e330", "e202"]
BRANDS = [f"brand_{i:03d}" for i in range(120)]


def nutriscore_grade(kcal, sugars, satfat, salt, protein, fibre) -> str:
    bad = min(kcal * 4.184 / 335, 10) + min(sugars / 4.5, 10) + min(satfat, 10) + min(salt * 400 / 90, 10)
    good = min(protein / 1.6, 5) + min(fibre / 0.9, 5)
    score = bad - good
    return "a" if score < 0 else "b" if score < 3 else "c" if score < 11 else "d" if score < 19 else "e"


def make_off(n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    cats = list(CATEGORIES)
    for i in range(n):
        cat = cats[rng.integers(len(cats))]
        base, nut = CATEGORIES[cat]
        nova = int(rng.choice([1, 2, 3, 4], p=[0.12, 0.03, 0.2, 0.65]))
        if nova == 1:
            ings = list(rng.choice(base, size=rng.integers(1, 3), replace=False))
        elif nova == 2:
            ings = [str(rng.choice(NOVA2))]
        elif nova == 3:
            ings = list(rng.choice(base, size=rng.integers(2, 4), replace=False)) + list(
                rng.choice(NOVA3_ADD, size=rng.integers(1, 3), replace=False))
        else:
            ings = list(rng.choice(base, size=rng.integers(2, 5), replace=False)) + list(
                rng.choice(NOVA4_ADD, size=rng.integers(2, 7), replace=False))
        if nova >= 3 and rng.random() < 0.3:
            ings.insert(1, f"{rng.choice(base)} ({rng.integers(5, 60)}%)")
        text = ", ".join(ings)
        if rng.random() < 0.15:
            text = "Ingredients: " + text.capitalize() + ". May contain traces of nuts."
        jitter = rng.lognormal(0, 0.35, size=6)
        kcal, sug, sat, salt, prot, fib = (np.array(nut) * jitter).round(2)
        if nova == 4:
            sug, sat = sug * 1.3, sat * 1.2
        missing_nut = rng.random() < 0.25
        rows.append({
            "code": f"{rng.integers(10**11, 10**12):012d}" if rng.random() < 0.5 else f"{rng.integers(10**12, 10**13):013d}",
            "product_name": f"{cat.split('-')[0].title()} product {i}",
            "ingredients_text": text,
            "nova_group": nova if rng.random() > 0.08 else np.nan,
            "nutriscore_grade": (np.nan if missing_nut or rng.random() < 0.1
                                 else nutriscore_grade(kcal, sug, sat, salt, prot, fib)),
            "category": cat if rng.random() > 0.1 else np.nan,
            "brand": str(rng.choice(BRANDS)),
            "lang": str(rng.choice(["en", "fr", "de"], p=[0.8, 0.15, 0.05])),
            "countries": str(rng.choice(["en:united-states", "en:france", "en:united-kingdom", "en:germany", "en:canada"],
                                        p=[0.4, 0.25, 0.15, 0.1, 0.1])),
            "analysis_tags": ",".join(["en:palm-oil" if "palm oil" in text else "en:palm-oil-free",
                                       "en:non-vegan" if any(w in text for w in ("milk", "egg", "pork", "chicken", "beef", "whey"))
                                       else "en:vegan"]),
            "allergens": "",
            "unique_scans_n": int(rng.integers(1, 500)),
            **({} if missing_nut else {"energy_kcal_100g": kcal, "sugars_100g": sug, "saturated_fat_100g": sat,
                                       "salt_100g": salt, "proteins_100g": prot, "fiber_100g": fib}),
        })
    df = pd.DataFrame(rows)
    dup = df.sample(frac=0.03, random_state=1)            # some exact duplicates to exercise the dedup rules
    return pd.concat([df, dup], ignore_index=True)


def make_usda(off: pd.DataFrame, n_extra: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    usda_cat = {"sugary-snacks": "Candy", "beverages": "Soda", "cereals-and-potatoes": "Cereal",
                "milk-and-dairy-products": "Cheese", "fats-and-sauces": "Ketchup, Mustard, BBQ & Cheese Sauce",
                "fish-meat-eggs": "Sausages, Hotdogs & Brats", "fruits-and-vegetables": "Canned Vegetables",
                "salty-snacks": "Chips, Pretzels & Snacks", "composite-foods": "Frozen Dinners & Entrees"}
    shared = off.dropna(subset=["category"]).drop_duplicates("code").sample(frac=0.4, random_state=2)
    rows = []
    for _, r in shared.iterrows():
        ings = r["ingredients_text"].replace("Ingredients: ", "").rstrip(".").upper()
        if rng.random() < 0.4:
            ings = ings.replace(", ", ", CONTAINS 2% OR LESS OF: ", 1)
        rows.append((r["code"].zfill(14), "INGREDIENTS: " + ings, usda_cat[r["category"]]))
    for _ in range(n_extra):
        cat = list(usda_cat)[rng.integers(len(usda_cat))]
        base, _ = CATEGORIES[cat]
        ings = ", ".join(rng.choice(base + NOVA4_ADD, size=rng.integers(3, 8), replace=False)).upper()
        rows.append((f"{rng.integers(10**12, 10**13):014d}", ings, usda_cat[cat]))
    branded = pd.DataFrame(rows, columns=["gtin_upc", "ingredients", "branded_food_category"])
    branded.insert(0, "fdc_id", np.arange(100000, 100000 + len(branded)))
    branded["brand_owner"] = [str(rng.choice(BRANDS)).upper() for _ in range(len(branded))]
    branded["brand_name"] = branded["brand_owner"]
    branded["modified_date"] = "2026-01-01"
    branded["market_country"] = "United States"
    branded["discontinued_date"] = ""
    nutrient_ids = {1008: (100, 500), 2000: (0, 40), 1258: (0, 15), 1093: (5, 1200), 1003: (0, 25), 1079: (0, 8)}
    fn = [(fid, nid, round(float(rng.uniform(lo, hi)), 2)) for fid in branded["fdc_id"] for nid, (lo, hi) in nutrient_ids.items()
          if rng.random() > 0.05]
    food_nutrient = pd.DataFrame(fn, columns=["fdc_id", "nutrient_id", "amount"])
    food_nutrient.insert(0, "id", np.arange(len(food_nutrient)))
    return branded, food_nutrient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    off = make_off(a.n, rng)
    config.SAMPLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    off.to_csv(config.SAMPLE_CSV, index=False)
    branded, food_nutrient = make_usda(off, a.n // 5, rng)
    with zipfile.ZipFile(config.USDA_SAMPLE_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for name, df in (("branded_food.csv", branded), ("food_nutrient.csv", food_nutrient)):
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            z.writestr(f"FoodData_Central_branded_food_csv_sample/{name}", buf.getvalue())
    print(f"Wrote {config.SAMPLE_CSV} ({len(off):,} rows, SYNTHETIC)")
    print(f"Wrote {config.USDA_SAMPLE_ZIP} ({len(branded):,} branded rows, SYNTHETIC)")


if __name__ == "__main__":
    main()
