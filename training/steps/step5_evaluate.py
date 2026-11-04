"""Step 5 - score the saved artifacts on the unseen USDA set.

USAGE
  python training/steps/step5_evaluate.py --matched data/processed/usda_matched.parquet
  python training/steps/step5_evaluate.py --nova-merge12

Loads artifacts/{nova,tier,category}_model.joblib (via .load()), cleans the
matched set so it has `ingredients_norm`/`health_tier`, then reports:
  * NOVA: accuracy, macro-F1, QWK, Spearman (ordinal)
  * tier / category: accuracy, macro-F1 and a per-class report
Writes reports/tables/unseen_metrics.csv plus one per-class CSV per target.
If artifacts/train_meta.json says overlap_safe is false, it warns loudly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "src"))
from nutriscan import config  # noqa: E402
from nutriscan.data import clean_products  # noqa: E402
from nutriscan.evaluation import (  # noqa: E402
    confusion_df,
    merge_nova12,
    per_class_report,
    summary,
)
from nutriscan.models.category_predictor import CategoryPredictor  # noqa: E402
from nutriscan.models.nova_predictor import NovaPredictor  # noqa: E402
from nutriscan.models.tier_predictor import TierPredictor  # noqa: E402

TRAIN_META = config.ARTIFACTS / "train_meta.json"
TABLES = config.REPORTS / "tables"




def warn_contamination() -> None:
    if not TRAIN_META.exists():
        print("WARNING: artifacts/train_meta.json missing; cannot confirm overlap safety.")
        return
    meta = json.loads(TRAIN_META.read_text(encoding="utf-8"))
    if not meta.get("overlap_safe", False):
        print("!" * 78)
        print("WARNING: the saved models were NOT trained on the disjoint pool")
        print(f"  ({meta.get('data_path')}). Treat these 'unseen' numbers as contaminated.")
        print("!" * 78)


def labelled(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target == "health_tier":
        return df[df["health_tier"].notna() & (df["health_tier"] != "unknown")]
    return df[df[target].notna()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate saved artifacts on the unseen set.")
    ap.add_argument("--matched", default=str(config.USDA_MATCHED))
    ap.add_argument("--nova-merge12", action="store_true", help="merge NOVA 1+2 into one class")
    a = ap.parse_args()

    matched_path = config.resolve(a.matched)
    if not matched_path.exists():
        sys.exit(
            "Matched set missing. Build it first:\n"
            "  python training/steps/step3_make_unseen_set.py "
            "--usda-zip data/raw/FoodData_Central_branded_food_csv_<date>.zip "
            "--off data/processed/off_clean.parquet"
        )

    warn_contamination()
    df = clean_products(pd.read_parquet(matched_path))
    print(f"Matched set: {matched_path} | rows after clean: {len(df):,}")

    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []

    if "nova_group" in df.columns:
        test = labelled(df, "nova_group").copy()
        test["nova_group"] = test["nova_group"].astype(int)
        model = NovaPredictor.load()
        yt = test["nova_group"]
        yp = pd.Series(model.predict(test["ingredients_norm"].tolist()), index=test.index)
        if a.nova_merge12:
            yt, yp = merge_nova12(yt), merge_nova12(yp)
        metrics = summary(yt, yp, ordinal=True)
        rows.append({"target": "nova_group", "n": len(test), **metrics})
        report = per_class_report(yt, yp)
        report.to_csv(TABLES / "unseen_per_class_nova_group.csv", index=False)
        confusion_df(yt, yp).to_csv(TABLES / "unseen_confusion_nova_group.csv")
        print("\nNOVA (unseen)")
        print(report.to_string(index=False))
        print(pd.Series(metrics).round(3).to_string())

    if "health_tier" in df.columns:
        test = labelled(df, "health_tier")
        model = TierPredictor.load()
        yt = test["health_tier"]
        yp = pd.Series(model.predict(test["ingredients_norm"].tolist()), index=test.index)
        metrics = summary(yt, yp)
        rows.append({"target": "health_tier", "n": len(test), **metrics})
        report = per_class_report(yt, yp)
        report.to_csv(TABLES / "unseen_per_class_health_tier.csv", index=False)
        print("\nHealth tier (unseen)")
        print(report.to_string(index=False))
        print(pd.Series(metrics).round(3).to_string())

    if "category" in df.columns:
        test = labelled(df, "category").copy()
        test["category"] = test["category"].astype(str)
        model = CategoryPredictor.load()
        yt = test["category"]
        yp = pd.Series(model.predict(test["ingredients_norm"].tolist()), index=test.index)
        metrics = summary(yt, yp)
        rows.append({"target": "category", "n": len(test), **metrics})
        report = per_class_report(yt, yp)
        report.to_csv(TABLES / "unseen_per_class_category.csv", index=False)
        print("\nCategory (unseen)")
        print(report.to_string(index=False))
        print(pd.Series(metrics).round(3).to_string())

    if not rows:
        sys.exit("No labelled target columns found in the matched set.")

    table = pd.DataFrame(rows).round(3)
    out = TABLES / "unseen_metrics.csv"
    table.to_csv(out, index=False)
    print("\n" + table.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
