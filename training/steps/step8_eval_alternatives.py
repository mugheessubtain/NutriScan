"""Step 8 - evaluate the alternatives engine (plan v2 Sections 9 and 12).

USAGE
  python training/steps/step8_eval_alternatives.py [--queries 300] [--k 5]

Queries are gallery products that have room to improve (NOVA 3-4 or tier medium/poor). For each, the engine is run
with the PREDICTED category (as in the app) and its own known labels, and the top-k results are scored against the
query's TRUE labels:
  coverage          share of queries with at least one alternative
  same_category     share of results in the query's true food group
  better            share of results with a lower NOVA group or a better health tier than the query
  precision@k       share of results that are both (relevant = same group AND better)
  brand_diversity   distinct brands / results per query
  mean_similarity   ingredient-list similarity of the results to the query (higher = closer substitute)
Writes reports/tables/alternatives_eval.csv and alternatives_examples.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "src"))
from nutriscan import config  # noqa: E402
from nutriscan.alternatives import AlternativesEngine, build_gallery  # noqa: E402
from nutriscan.data import clean_products  # noqa: E402
from nutriscan.expert_system import ExpertSystem  # noqa: E402
from nutriscan.models import CategoryPredictor  # noqa: E402


def load_gallery() -> pd.DataFrame:
    if config.GALLERY.exists():
        return pd.read_parquet(config.GALLERY)
    src = next(p for p in (config.OFF_CLEAN, config.SAMPLE_CSV) if p.exists() or p == config.SAMPLE_CSV)
    df = pd.read_parquet(src) if src.suffix == ".parquet" else clean_products(pd.read_csv(src, dtype={"code": str}))
    return build_gallery(df)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    gallery = load_gallery()
    engine = AlternativesEngine(gallery, ExpertSystem())
    cat_model = CategoryPredictor.load() if CategoryPredictor.model_path.exists() else None
    print(f"gallery: {len(gallery):,} products | category model: {'saved artifact' if cat_model else 'true category used'}")

    pool = gallery[(gallery["nova_group"] >= 3) | gallery["health_tier"].isin(["medium", "poor"])]
    q = pool.sample(min(a.queries, len(pool)), random_state=config.RANDOM_STATE)
    rank = config.TIER_ORDER
    rows, examples = [], []
    for _, r in q.iterrows():
        cat = str(cat_model.predict(r["ingredients_norm"])[0]) if cat_model else r["category"]
        alts = engine.recommend(r["ingredients_norm"], cat, r["nova_group"], r["health_tier"], k=a.k)
        same = [x.category == r["category"] for x in alts]
        better = [(pd.notna(x.nova_group) and pd.notna(r["nova_group"]) and x.nova_group < r["nova_group"]) or
                  rank.get(x.health_tier, -1) > rank.get(r["health_tier"], -1) for x in alts]
        rel = [s and b for s, b in zip(same, better)]
        rows.append({"has_result": bool(alts), "n": len(alts), "same_category": np.mean(same) if alts else np.nan,
                     "better": np.mean(better) if alts else np.nan, f"precision@{a.k}": sum(rel) / a.k,
                     "brand_diversity": len({x.brand for x in alts}) / len(alts) if alts else np.nan,
                     "mean_similarity": np.mean([x.similarity for x in alts]) if alts else np.nan,
                     "category_predicted_correctly": cat == r["category"]})
        if len(examples) < 15 and alts:
            examples.append({"query": r["product_name"], "query_nova": r["nova_group"], "query_tier": r["health_tier"],
                             "alternative": alts[0].name, "alt_nova": alts[0].nova_group, "alt_tier": alts[0].health_tier,
                             "similarity": round(alts[0].similarity, 3), "why": alts[0].reason})
    per = pd.DataFrame(rows)
    summary = pd.DataFrame([{"queries": len(per), "k": a.k, "coverage": per["has_result"].mean(),
                             **per.drop(columns=["has_result", "n"]).mean(numeric_only=True).to_dict()}]).round(3)
    config.TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.TABLES / "alternatives_eval.csv", index=False)
    pd.DataFrame(examples).to_csv(config.TABLES / "alternatives_examples.csv", index=False)
    print(summary.T.to_string(header=False))
    print(f"Wrote {config.TABLES / 'alternatives_eval.csv'}")


if __name__ == "__main__":
    main()
