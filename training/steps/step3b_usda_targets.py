"""Step 3b - give USDA its own labels (plan v2 Section 4.1), independent of Open Food Facts.

USAGE
  python training/steps/step3b_usda_targets.py --usda-zip data/raw/FoodData_Central_branded_food_csv_<date>.zip

Reads data/processed/usda_branded.parquet (step3) and food_nutrient.csv from the same FDC zip, then adds
  * category       USDA branded_food_category mapped to the OFF food groups (usda.CATEGORY_RULES)
  * health_tier    Nutri-Score 2017 grade computed from USDA nutrients (fruit/veg share unknown -> 0)
  * in_off         whether the barcode also exists in OFF (those rows are the matched external set)
Writes data/processed/usda_labeled.parquet and reports/tables/usda_category_mapping.csv, and prints how often the
USDA nutrient tier agrees with the OFF tier on matched products (a sanity check of the computed labels).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "src"))
from nutriscan import config  # noqa: E402
from nutriscan.data import clean_products  # noqa: E402
from nutriscan.usda import label_usda, mapping_table, read_usda_nutrients  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Add independent category + health-tier labels to USDA.")
    ap.add_argument("--usda-zip", required=True, help="the FDC branded zip (needs food_nutrient.csv inside) or food_nutrient.csv")
    ap.add_argument("--usda", default=str(config.USDA_PROCESSED), help="step3 output")
    ap.add_argument("--off", default=str(config.OFF_CLEAN), help="for the in_off flag")
    a = ap.parse_args()

    usda_path = config.resolve(a.usda)
    if not usda_path.exists():
        sys.exit(f"{usda_path} missing: run step3_make_unseen_set.py first")
    usda = pd.read_parquet(usda_path)
    print(f"USDA products: {len(usda):,}")

    nutrients = read_usda_nutrients(config.resolve(a.usda_zip))
    print(f"USDA products with nutrient rows: {len(nutrients):,}")
    lab = label_usda(usda, nutrients)
    lab = clean_products(lab.drop(columns=["nutriscore_grade"], errors="ignore"))   # adds ingredients_norm, dedupes text
    print(f"after text cleaning: {len(lab):,}")
    print(f"mapped to an OFF food group: {lab['category'].notna().mean():.1%}")
    if "health_tier" in lab:
        print(f"with a nutrient-based tier: {lab['health_tier'].notna().mean():.1%}")
        print(lab["health_tier"].value_counts().to_string())

    config.TABLES.mkdir(parents=True, exist_ok=True)
    off_path = config.resolve(a.off)
    if off_path.exists():
        import pyarrow.parquet as pq
        from nutriscan.external import add_code_canon
        cols = [c for c in ("code", "code_canon", "health_tier") if c in pq.read_schema(off_path).names]
        off = add_code_canon(pd.read_parquet(off_path, columns=cols))
        lab["in_off"] = lab["code_canon"].isin(set(off["code_canon"].dropna()))
        both = lab[lab["in_off"]].merge(off[["code_canon", "health_tier"]].dropna().drop_duplicates("code_canon"),
                                        on="code_canon", suffixes=("", "_off"))
        both = both[both["health_tier"].notna() & both["health_tier_off"].isin(["good", "medium", "poor"])]
        if len(both):
            agree = (both["health_tier"] == both["health_tier_off"]).mean()
            print(f"tier agreement USDA-nutrients vs OFF on {len(both):,} matched products: {agree:.1%}")
            pd.crosstab(both["health_tier_off"], both["health_tier"]).to_csv(config.TABLES / "usda_vs_off_tier_agreement.csv")
    mapping_table(lab).to_csv(config.TABLES / "usda_category_mapping.csv", index=False)
    lab.to_parquet(config.USDA_LABELED, index=False)
    print(f"Wrote {config.USDA_LABELED}")


if __name__ == "__main__":
    main()
