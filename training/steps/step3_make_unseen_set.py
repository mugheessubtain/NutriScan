"""Step 3 - build the unseen USDA test set and the overlap-safe training pool.

USAGE
  python training/steps/step3_make_unseen_set.py \
      --usda-zip data/raw/FoodData_Central_branded_food_csv_<date>.zip \
      --off data/processed/off_clean.parquet

Reuses `external.read_usda_branded` + `normalize_usda`, then
`external_eval_sets` to produce:
  * usda_branded.parquet        - deduped USDA rows (one per barcode)
  * usda_matched.parquet        - USDA ingredient text + OFF labels (unseen test set)
  * off_train_disjoint.parquet  - OFF minus every USDA-overlapping barcode

The disjoint training pool is what step4 must use: `external.py` matches on
barcode only, so training on full OFF would make the "unseen" set partly seen.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "src"))
from nutriscan import config  # noqa: E402
from nutriscan.external import (  # noqa: E402
    add_code_canon,
    external_eval_sets,
    match_to_off,
    normalize_usda,
    read_usda_branded,
)

OFF_TRAIN_DISJOINT = config.PROCESSED / "off_train_disjoint.parquet"




def main() -> None:
    ap = argparse.ArgumentParser(description="Build the USDA unseen set and disjoint training pool.")
    ap.add_argument("--usda-zip", required=True,
                    help="branded_food.csv or the FoodData Central branded-food zip")
    ap.add_argument("--off", default="data/processed/off_clean.parquet")
    a = ap.parse_args()

    usda_path = config.resolve(a.usda_zip)
    off_path = config.resolve(a.off)
    if not usda_path.exists():
        sys.exit(f"USDA input not found: {usda_path}")
    if not off_path.exists():
        sys.exit(f"OFF table not found: {off_path} (run step2 first)")

    usda = normalize_usda(read_usda_branded(usda_path))
    print(f"USDA distinct barcodes with ingredients: {len(usda):,}")

    off = add_code_canon(pd.read_parquet(off_path))
    sets = external_eval_sets(off, usda)

    usda_codes = set(usda["code_canon"].dropna())
    off_codes = set(off["code_canon"].dropna())
    overlap = len(usda_codes & off_codes)
    overlap_pct = overlap / len(usda_codes) if usda_codes else 0.0
    print(f"USDA barcodes also in OFF: {overlap:,} ({overlap_pct:.1%} of USDA barcodes)")

    matched = sets["external_matched"]
    if "category" not in matched.columns:
        matched = match_to_off(usda, off, label_cols=("nova_group", "nutriscore_grade", "health_tier", "category"))
    for col in ("nova_group", "health_tier", "category"):
        if col in matched.columns:
            print(f"matched rows with non-null {col}: {matched[col].notna().sum():,}")

    print(f"train_disjoint rows: {len(sets['train_disjoint']):,}")
    print(f"usda_unlabeled (gallery only): {len(sets['usda_unlabeled']):,}")

    config.USDA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    usda.to_parquet(config.USDA_PROCESSED, index=False)
    matched.to_parquet(config.USDA_MATCHED, index=False)
    sets["train_disjoint"].to_parquet(OFF_TRAIN_DISJOINT, index=False)
    print(f"Wrote {config.USDA_PROCESSED}")
    print(f"Wrote {config.USDA_MATCHED}")
    print(f"Wrote {OFF_TRAIN_DISJOINT}")


if __name__ == "__main__":
    main()
