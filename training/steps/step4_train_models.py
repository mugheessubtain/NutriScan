"""Step 4 - train and save the three predictors (NOVA, health tier, category).

USAGE
  python training/steps/step4_train_models.py --data data/processed/off_train_disjoint.parquet
  python training/steps/step4_train_models.py --only nova
  python training/steps/step4_train_models.py --model voting             # tuned soft-voting ensemble (slower)
  python training/steps/step4_train_models.py --add-usda                 # + USDA-only labelled rows for tier/category

Default input is data/processed/off_train_disjoint.parquet (overlap-safe). If
that is missing it falls back to off_clean.parquet with a loud warning, because
training on full OFF makes the USDA "unseen" score partly seen data.
`--model` picks the estimator from runtime/src/nutriscan/models/zoo.py; with `--tuned` the best hyper-parameters
found by step6 (reports/tables/best_params_<target>.json) are applied. The script also writes the app's product
gallery (data/processed/gallery.parquet) so the UI starts without re-parsing the whole dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "src"))
from nutriscan import config  # noqa: E402
from nutriscan.alternatives import build_gallery  # noqa: E402
from nutriscan.data import clean_products  # noqa: E402
from nutriscan.models.zoo import IMPROVED, make_ensemble, make_model  # noqa: E402
from nutriscan.models.category_predictor import CategoryPredictor  # noqa: E402
from nutriscan.models.nova_predictor import NovaPredictor  # noqa: E402
from nutriscan.models.tier_predictor import TierPredictor  # noqa: E402

OFF_CLEAN = config.PROCESSED / "off_clean.parquet"
OFF_TRAIN_DISJOINT = config.PROCESSED / "off_train_disjoint.parquet"
TRAIN_META = config.ARTIFACTS / "train_meta.json"

PREDICTORS = {
    "nova": (NovaPredictor, "nova_group"),
    "tier": (TierPredictor, "health_tier"),
    "category": (CategoryPredictor, "category"),
}




def labelled_rows(df: pd.DataFrame, target: str) -> int:
    if target == "health_tier":
        return int((df["health_tier"].notna() & (df["health_tier"] != "unknown")).sum())
    return int(df[target].notna().sum())


def build_estimator(kind: str, target: str, tuned: bool):
    """Zoo estimator for `kind`, with step6's tuned hyper-parameters when requested and available."""
    params = {}
    path = config.TABLES / f"best_params_{target}.json"
    if tuned and path.exists():
        params = json.loads(path.read_text(encoding="utf-8"))
    def one(name):
        est = make_model(name)
        return est.set_params(**params[name]["params"]) if name in params else est
    if kind in ("voting", "stacking"):
        return make_ensemble(kind, {m: one(m) for m in IMPROVED})
    return one(kind)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and save the three predictors.")
    ap.add_argument("--data", default=None,
                    help="training parquet; defaults to off_train_disjoint.parquet if present")
    ap.add_argument("--only", choices=sorted(PREDICTORS), default=None)
    ap.add_argument("--model", default="tfidf_lr", choices=IMPROVED + ["voting", "stacking"])
    ap.add_argument("--tuned", action="store_true", help="apply step6's best_params_<target>.json")
    ap.add_argument("--add-usda", action="store_true",
                    help="also train tier/category on USDA products NOT in OFF (labels from step3b; never the matched test set)")
    a = ap.parse_args()

    if a.data:
        data_path = config.resolve(a.data)
    elif OFF_TRAIN_DISJOINT.exists():
        data_path = OFF_TRAIN_DISJOINT
    else:
        data_path = OFF_CLEAN
    if not data_path.exists():
        sys.exit(f"Training data not found: {data_path} (run step3 first)")

    overlap_safe = data_path.name == OFF_TRAIN_DISJOINT.name
    if not overlap_safe:
        print("!" * 78)
        print("WARNING: training on a non-disjoint pool. The USDA 'unseen' score is")
        print("NOT overlap-safe; use data/processed/off_train_disjoint.parquet for that.")
        print("!" * 78)

    df = pd.read_parquet(data_path)
    print(f"Training data: {data_path}")
    print(f"overlap_safe: {overlap_safe} | rows: {len(df):,}")
    if "ingredients_norm" not in df.columns:
        df = clean_products(df)

    try:
        data_label = str(data_path.relative_to(config.PROJECT_ROOT))
    except ValueError:
        data_label = str(data_path)

    config.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    meta = {
        "data_path": data_label,
        "overlap_safe": overlap_safe,
        "rows": int(len(df)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }

    usda_extra = None
    if a.add_usda and config.USDA_LABELED.exists():
        u = pd.read_parquet(config.USDA_LABELED)
        usda_extra = u[~u.get("in_off", pd.Series(False, index=u.index)).fillna(False).astype(bool)]
        print(f"USDA-only labelled rows available for tier/category: {len(usda_extra):,}")
    meta["model"], meta["tuned"], meta["add_usda"] = a.model, a.tuned, usda_extra is not None
    meta["synthetic"] = not config.OFF_META.exists()       # prepare_off.py writes off_en.meta.json for real OFF data

    names = [a.only] if a.only else list(PREDICTORS)
    for name in names:
        cls, target = PREDICTORS[name]
        if target not in df.columns:
            print(f"{name}: skipped, no '{target}' column")
            continue
        train_df = df
        if usda_extra is not None and target in ("health_tier", "category") and target in usda_extra:
            train_df = pd.concat([df, usda_extra[["ingredients_norm", target]]], ignore_index=True)
        n = labelled_rows(train_df, target)
        model = cls(kind=a.model).train(train_df, model=build_estimator(a.model, target, a.tuned))
        model.save()
        meta["models"][name] = {"target": target, "rows": n, "classes": model.classes_, "kind": a.model}
        print(f"{name}: rows={n:,} classes={model.classes_} -> {config.ARTIFACTS / f'{name}_model.joblib'}")

    gallery = build_gallery(pd.read_parquet(OFF_CLEAN) if OFF_CLEAN.exists() else df)
    config.GALLERY.parent.mkdir(parents=True, exist_ok=True)
    gallery.to_parquet(config.GALLERY, index=False)
    print(f"gallery: {len(gallery):,} products -> {config.GALLERY}")

    TRAIN_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {TRAIN_META}")


if __name__ == "__main__":
    main()
