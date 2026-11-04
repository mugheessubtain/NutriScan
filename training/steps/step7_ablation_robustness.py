"""Step 7 - hybrid ablation, OCR-noise robustness, fairness and SHAP (plan v2 Sections 7.5, 8, 12, 13).

USAGE
  python training/steps/step7_ablation_robustness.py
  python training/steps/step7_ablation_robustness.py --data data/processed/off_train_disjoint.parquet --max-train 100000

Models are refitted here on a brand-held-out split (same protocol as step6) using the configuration of the saved
artifacts (sklearn `clone`), so no test product was seen in training. Outputs:
  ablation_nova.csv                 ML only vs rules only vs hybrid (internal test and USDA matched set)
  robustness_<target>.csv/.png      macro-F1 under injected OCR-style noise (0-20% of letters)
  fairness_<target>_<group>.csv     accuracy / macro-F1 per food group and per language or country
  shap_linear_<target>_<class>.csv  global mean |SHAP| (linear model) + bar chart
  shap_tree_<target>.csv            engineered-feature importance for rf_features (TreeExplainer)
  shap_examples_nova.csv            per-product top drivers for 20 test products (qualitative review)
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime" / "src"))
from nutriscan import config, eda  # noqa: E402
from nutriscan.data import clean_products, grouped_split  # noqa: E402
from nutriscan.evaluation import group_metrics, inject_ocr_noise, summary  # noqa: E402
from nutriscan.expert_system import ExpertSystem  # noqa: E402
from nutriscan.explain import linear_shap, linear_shap_summary, tree_shap_summary  # noqa: E402
from nutriscan.models import NovaPredictor, TierPredictor  # noqa: E402
from nutriscan.models.zoo import linear_member, make_model  # noqa: E402
from nutriscan.parser import normalized_string, parse_ingredients  # noqa: E402

warnings.filterwarnings("ignore")
NOISE = [0.0, 0.02, 0.05, 0.10, 0.20]
TIERS = {"good", "medium", "poor"}




def labelled(df, target):
    out = df[df[target].notna()]
    if target == "health_tier":
        out = out[out[target].isin(TIERS)]
    out = out.copy()
    out[target] = out[target].astype(int) if target == "nova_group" else out[target].astype(str)
    out["brand"] = out.get("brand", pd.Series("unknown", index=out.index)).fillna("unknown")
    return out.reset_index(drop=True)


def base_model(pred_cls, kind: str | None):
    """Unfitted copy of the saved artifact's model (or `kind` from the zoo when given / no artifact exists)."""
    if kind:
        return make_model(kind)
    try:
        return clone(pred_cls.load().model)
    except Exception:
        return make_model("tfidf_lr")


def fit(model, train, target, cap):
    if cap and len(train) > cap:
        train = train.sample(cap, random_state=config.RANDOM_STATE)
    return model.fit(train["ingredients_norm"].tolist(), train[target])


def save(tbl: pd.DataFrame, name: str):
    config.TABLES.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(config.TABLES / f"{name}.csv", index=False)
    print(f"  wrote {name}.csv")


# ---------------------------------------------------------------- 1. hybrid ablation
def ablation(model, test: pd.DataFrame, es: ExpertSystem, set_name: str) -> list[dict]:
    ml = np.asarray(model.predict(test["ingredients_norm"].tolist())).astype(int)
    rules, hybrid, raised = [], [], 0
    for text, m in zip(test["ingredients_text"], ml):
        ings = parse_ingredients(text)
        res = es.analyze(ings)
        rules.append(es.rules_only_nova(ings, res))
        h, why = es.hybrid_nova(int(m), res)
        hybrid.append(h)
        raised += bool(why)
    y = test["nova_group"].astype(int)
    rows = []
    for name, pred in (("ML only", ml), ("rules only (expert system)", rules), ("hybrid (ML + rules)", hybrid)):
        rows.append({"test_set": set_name, "system": name, "n": len(test),
                     **{k: round(float(v), 4) for k, v in summary(y, pred, ordinal=True).items()}})
    rows[-1]["rule_overrides"] = raised
    return rows


# ---------------------------------------------------------------- 2. robustness
def robustness(model, test: pd.DataFrame, target: str, n: int) -> pd.DataFrame:
    test = test.sample(min(n, len(test)), random_state=config.RANDOM_STATE)
    rng = np.random.default_rng(config.RANDOM_STATE)
    rows = []
    for rate in NOISE:
        noisy = [normalized_string(inject_ocr_noise(t, rate, rng)) for t in test["ingredients_text"]]
        m = summary(test[target], model.predict(noisy))
        rows.append({"noise_rate": rate, "n": len(test), **{k: round(float(v), 4) for k, v in m.items()}})
    tbl = pd.DataFrame(rows)
    tbl["macro_f1_drop"] = (tbl["macro_f1"].iloc[0] - tbl["macro_f1"]).round(4)
    return tbl


def plot_robustness(tables: dict[str, pd.DataFrame]):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for t, tbl in tables.items():
        ax.plot(100 * tbl["noise_rate"], tbl["macro_f1"], marker="o", label=t)
    ax.set_xlabel("% of letters corrupted (OCR-style)"); ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1)
    ax.set_title("Robustness to OCR noise"); ax.legend()
    fig.tight_layout(); config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / "robustness_ocr_noise.png", bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- 3. fairness
def fairness(model, test: pd.DataFrame, target: str) -> None:
    t = test.copy()
    t["pred"] = model.predict(t["ingredients_norm"].tolist())
    groups = [c for c in ("category", "lang") if c in t]
    if "countries" in t:
        t["country"] = t["countries"].astype(str).str.split(",").str[0].str.replace("en:", "", regex=False)
        groups.append("country")
    for g in groups:
        tbl = group_metrics(t.dropna(subset=[g]), g, target, "pred", min_n=30 if len(t) > 3000 else 10)
        if not tbl.empty:
            save(tbl, f"fairness_{target}_{g}")
            eda.plot_bar_metric(tbl.head(15), g, "macro_f1", f"fairness_{target}_{g}", f"{target}: macro-F1 per {g}")


# ---------------------------------------------------------------- 4. SHAP
def shap_reports(model, background, test: pd.DataFrame, target: str, classes, train: pd.DataFrame, n: int) -> None:
    sub = test.sample(min(n, len(test)), random_state=config.RANDOM_STATE)
    texts, y = sub["ingredients_norm"].tolist(), sub[target].tolist()
    for c in classes:
        tbl = linear_shap_summary(model, texts, c, background, top=20)
        if tbl.empty:
            print("  (no linear member; linear SHAP skipped)")
            break
        save(tbl, f"shap_linear_{target}_{c}")
        fig, ax = plt.subplots(figsize=(6, 5))
        t = tbl.iloc[::-1]
        ax.barh(t["feature"], t["mean_abs_shap"], color=["#c62828" if v else "#1e88e5" for v in t["toward_class"]])
        ax.set_xlabel("mean |SHAP|   (red: pushes toward the class when present, blue: away)")
        ax.set_title(f"SHAP summary: {target} = {c} (TF-IDF + LR)")
        fig.tight_layout(); fig.savefig(config.FIGURES / f"shap_linear_{target}_{c}.png", bbox_inches="tight"); plt.close(fig)
    # tree model on engineered features: fit a small rf_features model for the explanation
    rf = make_model("rf_features").fit(train["ingredients_norm"].tolist()[:30000], train[target].tolist()[:30000])
    tbl, method = tree_shap_summary(rf, texts[:300], y=y[:300], class_label=classes[0])
    tbl["method"] = method
    save(tbl, f"shap_tree_{target}")
    eda.plot_top_table(tbl.rename(columns={"importance": "pct"}), "feature", f"shap_tree_{target}",
                       f"{target} = {classes[0]}: engineered-feature importance ({method.split(' ')[0]})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid ablation, robustness, fairness and SHAP.")
    ap.add_argument("--data", default=None)
    ap.add_argument("--kind", default=None, help="model from the zoo (default: the saved artifact's configuration)")
    ap.add_argument("--max-train", type=int, default=150000)
    ap.add_argument("--noise-n", type=int, default=5000, help="test rows used per noise level")
    ap.add_argument("--shap-n", type=int, default=400)
    a = ap.parse_args()

    path = config.resolve(a.data) if a.data else next(p for p in (config.OFF_TRAIN_DISJOINT, config.OFF_CLEAN, config.SAMPLE_CSV)
                                               if p.exists() or p == config.SAMPLE_CSV)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, dtype={"code": str})
    df = df if "ingredients_norm" in df else clean_products(df)
    print(f"Data: {path} ({len(df):,} rows)")
    es = ExpertSystem()
    matched = clean_products(pd.read_parquet(config.USDA_MATCHED)) if config.USDA_MATCHED.exists() else None

    robust = {}
    for target, pred_cls in (("nova_group", NovaPredictor), ("health_tier", TierPredictor)):
        print(f"\n=== {target} ===")
        data = labelled(df, target)
        train, test = grouped_split(data, "brand", test_size=0.25)
        model = fit(base_model(pred_cls, a.kind), train, target, a.max_train)
        lin_bg = None
        lin = linear_member(model)
        if lin is not None:
            lin_bg = np.asarray(lin.named_steps["tfidf"].transform(train["ingredients_norm"].tolist()[:50000]).mean(axis=0)).ravel()

        if target == "nova_group":
            rows = ablation(model, test, es, "OFF brand-held-out test")
            if matched is not None:
                rows += ablation(model, labelled(matched, target), es, "USDA matched (unseen barcodes)")
            save(pd.DataFrame(rows), "ablation_nova")
            print(pd.DataFrame(rows).to_string(index=False))
            ex = test.sample(min(20, len(test)), random_state=1)
            preds = model.predict(ex["ingredients_norm"].tolist())
            save(pd.DataFrame([{"ingredients": t[:160], "true": y, "pred": int(p),
                                "top_drivers": "; ".join(f"{w} (+{v:.2f})" for w, v in linear_shap(model, t, p, lin_bg, k=5))}
                               for t, y, p in zip(ex["ingredients_norm"], ex[target], preds)]), "shap_examples_nova")

        robust[target] = robustness(model, test, target, a.noise_n)
        save(robust[target], f"robustness_{target}")
        print(robust[target].to_string(index=False))
        fairness(model, test, target)
        classes = [4, 1] if target == "nova_group" else ["poor", "good"]
        classes = [c for c in classes if c in set(train[target])]
        shap_reports(model, lin_bg, test, target, classes, train, a.shap_n)
    plot_robustness(robust)
    print(f"\nTables -> {config.TABLES}\nFigures -> {config.FIGURES}")


if __name__ == "__main__":
    main()
