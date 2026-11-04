"""Step 6 - model comparison (plan v2 Sections 7.2-7.4 and 12): baselines -> improved models -> tuned -> ensembles,
the imbalance-handling table, and the cross-dataset test on USDA.

USAGE
  python training/steps/step6_compare_models.py                                   # all targets, defaults below
  python training/steps/step6_compare_models.py --targets nova_group --max-train 100000 --n-iter 8
  python training/steps/step6_compare_models.py --save-best                       # also refit the winner -> artifacts/
  python training/steps/step6_compare_models.py --data data/sample/products_sample.csv --quick   # smoke test

Protocol (stated in the report):
  * training pool: off_train_disjoint.parquet (OFF minus every USDA barcode) so USDA results stay unseen
  * internal test: 25% of BRANDS held out (GroupShuffleSplit), never random rows
  * tuning: RandomizedSearchCV, StratifiedGroupKFold by brand, scoring = macro-F1, on a stratified subsample
  * headline metric: macro-F1 (accuracy, QWK and Spearman for NOVA as context)
Outputs (reports/tables, reports/figures):
  model_comparison_<target>.csv   imbalance_<target>.csv   best_params_<target>.json   cross_dataset_<target>.csv
  per_class_best_<target>.csv     confusion_best_<target>.csv   cmp_<target>.png   confusion_best_<target>.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "src"))
from nutriscan import config, eda  # noqa: E402
from nutriscan.data import clean_products, grouped_split  # noqa: E402
from nutriscan.evaluation import confusion_df, merge_nova12, per_class_report, summary  # noqa: E402
from nutriscan.models import CategoryPredictor, NovaPredictor, TierPredictor  # noqa: E402
from nutriscan.models.zoo import (BASELINES, IMPROVED, PARAM_SPACES, make_baseline, make_ensemble,  # noqa: E402
                                  make_imbalance_variant, make_model)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
PREDICTOR = {"nova_group": NovaPredictor, "health_tier": TierPredictor, "category": CategoryPredictor}
TIERS = {"good", "medium", "poor"}




def read(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"code": str})
    return df if "ingredients_norm" in df else clean_products(df)


def labelled(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df[df[target].notna()] if target in df else df.iloc[0:0]
    if target == "health_tier":
        out = out[out[target].isin(TIERS)]
    out = out.copy()
    out[target] = out[target].astype(int) if target == "nova_group" else out[target].astype(str)
    if "brand" not in out:
        out["brand"] = "unknown"
    out["brand"] = out["brand"].fillna("unknown")
    return out.reset_index(drop=True)


def stratified_cap(df: pd.DataFrame, target: str, n: int) -> pd.DataFrame:
    if not n or len(df) <= n:
        return df
    keep, _ = train_test_split(df, train_size=n, stratify=df[target], random_state=config.RANDOM_STATE)
    return keep.reset_index(drop=True)


def score(model, X, y, target: str) -> dict:
    pred = model.predict(X)
    return summary(y, pred, ordinal=target == "nova_group")


class Runner:
    def __init__(self, target: str, train: pd.DataFrame, test: pd.DataFrame):
        self.t, self.train, self.test = target, train, test
        self.Xtr, self.ytr = train["ingredients_norm"].tolist(), train[target]
        self.Xte, self.yte = test["ingredients_norm"].tolist(), test[target]
        self.rows: list[dict] = []
        self.fitted: dict[str, object] = {}

    def fit_eval(self, stage: str, name: str, est, X=None, y=None) -> object | None:
        t0 = time.perf_counter()
        try:
            est.fit(X if X is not None else self.Xtr, y if y is not None else self.ytr)
        except Exception as e:                       # one model failing must not lose the whole table
            print(f"  [{stage}] {name}: FAILED ({type(e).__name__}: {e})")
            return None
        fit_s = time.perf_counter() - t0
        m = score(est, self.Xte, self.yte, self.t)
        self.rows.append({"stage": stage, "model": name, **{k: round(float(v), 4) for k, v in m.items()},
                          "fit_seconds": round(fit_s, 1)})
        self.fitted[f"{stage}:{name}"] = est
        print(f"  [{stage:9s}] {name:22s} macro-F1={m['macro_f1']:.3f}  acc={m['accuracy']:.3f}  ({fit_s:.0f}s)")
        return est


def tune(name: str, train: pd.DataFrame, target: str, n_sample: int, n_iter: int, cv: int) -> dict:
    sub = stratified_cap(train, target, n_sample)
    n_splits = max(2, min(cv, sub["brand"].nunique(), int(sub[target].value_counts().min())))
    search = RandomizedSearchCV(make_model(name), PARAM_SPACES[name], n_iter=n_iter, scoring="f1_macro",
                                cv=StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE),
                                random_state=config.RANDOM_STATE, n_jobs=1, error_score=np.nan)
    t0 = time.perf_counter()
    search.fit(sub["ingredients_norm"].tolist(), sub[target], groups=sub["brand"])
    print(f"  [tune     ] {name:22s} cv macro-F1={search.best_score_:.3f} on {len(sub):,} rows "
          f"({time.perf_counter() - t0:.0f}s) best={search.best_params_}")
    return {"params": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
            "cv_macro_f1": float(search.best_score_), "tuning_rows": len(sub), "n_iter": n_iter, "cv_folds": n_splits}


def external_rows(target: str, models: dict, matched: pd.DataFrame | None, usda_lab: pd.DataFrame | None) -> list[dict]:
    rows = []
    sets = []
    if matched is not None and target in matched:
        sets.append(("USDA text, OFF label (product-disjoint barcodes)", labelled(matched, target)))
    if usda_lab is not None and target in ("health_tier", "category") and target in usda_lab:
        ind = usda_lab[~usda_lab.get("in_off", pd.Series(False, index=usda_lab.index)).fillna(False).astype(bool)]
        sets.append(("USDA text, USDA-derived label (products not in OFF)", labelled(ind, target)))
    for set_name, test in sets:
        if test.empty:
            continue
        for mname, model in models.items():
            m = score(model, test["ingredients_norm"].tolist(), test[target], target)
            rows.append({"test_set": set_name, "model": mname, "n": len(test), **{k: round(float(v), 4) for k, v in m.items()}})
            if target == "nova_group":
                pred = model.predict(test["ingredients_norm"].tolist())
                m2 = summary(merge_nova12(test[target]), merge_nova12(pred), ordinal=True)
                rows.append({"test_set": set_name + " [NOVA 1+2 merged]", "model": mname, "n": len(test),
                             **{k: round(float(v), 4) for k, v in m2.items()}})
            print(f"  [external ] {mname:22s} {set_name[:45]:45s} macro-F1={m['macro_f1']:.3f}")
    return rows


def run_target(target: str, df: pd.DataFrame, a, matched, usda_lab) -> None:
    print(f"\n=== {target} ===")
    data = labelled(df, target)
    if data[target].nunique() < 2:
        print("  fewer than 2 classes; skipped")
        return
    train, test = grouped_split(data, "brand", test_size=0.25)
    rare = [c for c, n in train[target].value_counts().items() if n < 10]
    if rare:                                   # calibration / stacking CV need a few rows of every class
        print(f"  dropping classes with < 10 training rows (reported as a limitation): {rare}")
        train, test = train[~train[target].isin(rare)], test[~test[target].isin(rare)]
    train = stratified_cap(train, target, a.max_train)
    print(f"  train {len(train):,} rows / {train['brand'].nunique():,} brands | test {len(test):,} rows / "
          f"{test['brand'].nunique():,} brands (brand-disjoint)")
    r = Runner(target, train, test)

    for b in BASELINES:
        r.fit_eval("baseline", b, make_baseline(b))
    models = [m for m in IMPROVED if not a.models or m in a.models]
    for m in models:
        r.fit_eval("default", m, make_model(m))

    best_params = {}
    tuned = {}
    for m in models:
        if a.skip_tuning:
            tuned[m] = make_model(m)
            continue
        try:
            best_params[m] = tune(m, train, target, a.tune_sample, a.n_iter, a.cv)
            est = make_model(m).set_params(**best_params[m]["params"])
        except Exception as e:
            print(f"  [tune     ] {m}: FAILED ({e}); using defaults")
            est = make_model(m)
        tuned[m] = est
        if not a.skip_tuning:
            r.fit_eval("tuned", m, clone(est))

    members = {m: clone(e) for m, e in tuned.items() if f"{'tuned' if not a.skip_tuning else 'default'}:{m}" in r.fitted}
    if len(members) >= 2:
        r.fit_eval("ensemble", "soft_voting", make_ensemble("voting", members))
        if not a.skip_stacking:
            r.fit_eval("ensemble", "stacking", make_ensemble("stacking", members))

    table = pd.DataFrame(r.rows)
    config.TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.TABLES / f"model_comparison_{target}.csv", index=False)
    (config.TABLES / f"best_params_{target}.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    table["label"] = table["stage"] + ": " + table["model"]
    eda.plot_bar_metric(table, "label", "macro_f1", f"cmp_{target}", f"{target}: macro-F1 on brand-held-out test",
                        highlight="ensemble")

    cand = table[table["stage"] != "baseline"].sort_values("macro_f1", ascending=False)
    best_key = f"{cand.iloc[0]['stage']}:{cand.iloc[0]['model']}"
    best = r.fitted[best_key]
    pred = best.predict(r.Xte)
    per_class_report(r.yte, pred).to_csv(config.TABLES / f"per_class_best_{target}.csv", index=False)
    cm = confusion_df(r.yte, pred)
    cm.to_csv(config.TABLES / f"confusion_best_{target}.csv")
    eda.plot_confusion(cm, f"confusion_best_{target}", f"{target}: {best_key} (row-normalised)")
    print(f"  best: {best_key} macro-F1={cand.iloc[0]['macro_f1']:.3f}")

    if target in ("nova_group", "health_tier") and not a.skip_imbalance:
        print("  -- imbalance handling --")
        imb = Runner(target, stratified_cap(train, target, a.imbalance_sample), test)
        k = min(5, int(imb.ytr.value_counts().min()) - 1)
        for model, strategies in (("tfidf_lr", ["none", "class_weights", "random_oversampling"]),
                                  ("rf_features", ["none", "class_weights", "random_oversampling"] + (["smote"] if k >= 1 else []))):
            for s in strategies:
                imb.fit_eval(s, model, make_imbalance_variant(model, s, smote_k=max(k, 1)))
        it = pd.DataFrame(imb.rows).rename(columns={"stage": "strategy"})
        if not it.empty:
            minority = per_minority_recall(imb, target)
            it = it.merge(minority, on=["strategy", "model"], how="left")
            it.to_csv(config.TABLES / f"imbalance_{target}.csv", index=False)

    ext_models = {"tfidf_lr (default)": r.fitted.get("default:tfidf_lr"), f"best: {best_key}": best}
    ext = external_rows(target, {k: v for k, v in ext_models.items() if v is not None}, matched, usda_lab)
    if ext:
        pd.DataFrame(ext).to_csv(config.TABLES / f"cross_dataset_{target}.csv", index=False)

    if a.save_best:
        name = cand.iloc[0]["model"]
        final = clone(best)
        pred_cls = PREDICTOR[target](kind=name)
        print(f"  refitting {best_key} on all {len(data):,} labelled rows -> {pred_cls.model_path.name}")
        pred_cls.train(data, model=final).save()


def per_minority_recall(r: Runner, target: str) -> pd.DataFrame:
    """Recall of the rarest class for each imbalance variant (the number imbalance handling is meant to move)."""
    rare = r.ytr.value_counts().idxmin()
    rows = []
    for key, est in r.fitted.items():
        s, m = key.split(":", 1)
        rep = per_class_report(r.yte, est.predict(r.Xte)).set_index("class")
        rows.append({"strategy": s, "model": m, f"recall_class_{rare}": float(rep.loc[rare, "recall"]) if rare in rep.index else np.nan})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare baselines, improved models, tuning and ensembles.")
    ap.add_argument("--data", default=None, help="training pool (default: off_train_disjoint > off_clean > sample CSV)")
    ap.add_argument("--targets", nargs="+", default=["nova_group", "health_tier", "category"])
    ap.add_argument("--models", nargs="+", default=None, choices=IMPROVED)
    ap.add_argument("--max-train", type=int, default=150000, help="cap on training rows (stratified); 0 = all")
    ap.add_argument("--tune-sample", type=int, default=20000)
    ap.add_argument("--n-iter", type=int, default=6)
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--imbalance-sample", type=int, default=60000)
    ap.add_argument("--skip-tuning", action="store_true")
    ap.add_argument("--skip-stacking", action="store_true")
    ap.add_argument("--skip-imbalance", action="store_true")
    ap.add_argument("--save-best", action="store_true", help="refit the winning model on all labelled rows -> artifacts/")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    a = ap.parse_args()
    if a.quick:
        a.n_iter, a.tune_sample, a.imbalance_sample, a.cv = 2, 2000, 3000, 2

    path = config.resolve(a.data) if a.data else next(p for p in (config.OFF_TRAIN_DISJOINT, config.OFF_CLEAN, config.SAMPLE_CSV)
                                               if p.exists() or p == config.SAMPLE_CSV)
    if not path.exists():
        sys.exit(f"No training data at {path}. Run steps 1-3 (or training/make_sample_data.py for a smoke test).")
    df = read(path)
    print(f"Training pool: {path} ({len(df):,} rows)"
          + ("  [overlap-safe]" if path.name == config.OFF_TRAIN_DISJOINT.name else "  [NOT overlap-safe for the USDA test]"))
    matched = read(config.USDA_MATCHED) if config.USDA_MATCHED.exists() else None
    usda_lab = pd.read_parquet(config.USDA_LABELED) if config.USDA_LABELED.exists() else None
    for t in a.targets:
        run_target(t, df, a, matched, usda_lab)
    print(f"\nTables -> {config.TABLES}\nFigures -> {config.FIGURES}")


if __name__ == "__main__":
    main()
