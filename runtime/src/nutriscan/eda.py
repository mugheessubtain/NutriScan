"""Section 6 Exploratory Data Analysis: reusable summaries and the planned plots E1-E11.

Every plot function takes a dataframe, draws one figure, saves it under reports/figures/<name>.png and returns the
path. Summary functions return small dataframes that training/steps/step2b_eda.py writes to reports/tables/ and
quotes in reports/eda_findings.md. Notebooks import these functions instead of re-implementing them.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402
from .features import E_NUM, MARKERS, feature_frame  # noqa: E402

NOVA_COLORS = {1: "#2e7d32", 2: "#9e9d24", 3: "#ef6c00", 4: "#c62828"}
TIER_COLORS = {"good": "#2e7d32", "medium": "#f9a825", "poor": "#c62828"}
TIER_ORDER = ["good", "medium", "poor"]
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})


def _save(fig, name: str, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir or config.FIGURES)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ------------------------------------------------------------------ 6.1 summaries
def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the engineered features (n_ingredients, n_e_numbers, sugar_pos, ...) computed from ingredients_norm."""
    feats = feature_frame(df["ingredients_norm"].tolist())
    feats.index = df.index
    return pd.concat([df.drop(columns=[c for c in feats.columns if c in df.columns]), feats], axis=1)


def overview(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return pd.DataFrame([{"dataset": name, "rows": len(df), "columns": df.shape[1],
                          "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1)}])


def dtypes_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes],
                         "non_null": df.notna().sum().values, "unique": [df[c].nunique(dropna=True) for c in df.columns]})


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    if "ingredients_text" in df:
        miss["ingredients_text"] += int((df["ingredients_text"].astype(str).str.strip() == "").sum())
    out = pd.DataFrame({"column": miss.index, "missing": miss.values, "missing_pct": (100 * miss.values / max(len(df), 1)).round(2)})
    return out.sort_values("missing_pct", ascending=False).reset_index(drop=True)


def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df]
    return df[cols].describe().T.round(2) if cols else pd.DataFrame()


def class_balance(y: pd.Series) -> pd.DataFrame:
    vc = y.dropna().value_counts().sort_index()
    return pd.DataFrame({"class": vc.index, "count": vc.values, "pct": (100 * vc.values / max(vc.sum(), 1)).round(2)})


def imbalance_ratio(y: pd.Series) -> float:
    vc = y.dropna().value_counts()
    return float(vc.max() / vc.min()) if len(vc) > 1 else float("nan")


def group_summary(df: pd.DataFrame, by: str, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df]
    return df.groupby(by)[cols].agg(["mean", "median"]).round(2)


# ------------------------------------------------------------------ helpers for additive/ingredient counts
def additive_names() -> dict[str, str]:
    """E-number -> readable name from the knowledge base (falls back to the E-number itself)."""
    import yaml
    kb = yaml.safe_load((config.KB_DIR / "additives.yaml").read_text(encoding="utf-8"))
    return {e.lower(): f'{e.lower()} {x["name"].lower()}' for x in kb["entries"] for e in x.get("e_numbers", [])}


def top_additives(texts: pd.Series, n: int = 20) -> pd.DataFrame:
    """Most frequent E-numbers and named marker additives (counted once per product)."""
    names = additive_names()
    words = sorted({w for ws in MARKERS.values() for w in ws if not re.fullmatch(r"e\d+", w) and len(w) > 4})
    word_re = re.compile(r"(?<![a-z])(" + "|".join(map(re.escape, words)) + r")")
    c = Counter()
    for t in texts:
        found = {names.get(e, e) for e in E_NUM.findall(t)} | set(word_re.findall(t))
        c.update(found)
    total = max(len(texts), 1)
    return pd.DataFrame([(k, v, round(100 * v / total, 2)) for k, v in c.most_common(n)], columns=["additive", "products", "pct"])


def top_ingredients(texts: pd.Series, n: int = 15) -> pd.DataFrame:
    c = Counter()
    for t in texts:
        c.update({re.sub(r"\(.*", "", s).strip() for s in t.split(", ") if s.strip()})
    total = max(len(texts), 1)
    return pd.DataFrame([(k, v, round(100 * v / total, 2)) for k, v in c.most_common(n) if k],
                        columns=["ingredient", "products", "pct"])


# ------------------------------------------------------------------ 6.2 plots
def plot_missing(df: pd.DataFrame, name: str, title: str) -> Path:                                 # E1
    m = missing_table(df)
    m = m[m["missing_pct"] > 0].head(25)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.3 * len(m) + 1)))
    ax.barh(m["column"][::-1], m["missing_pct"][::-1], color="#5c6bc0")
    ax.set_xlabel("% missing"); ax.set_title(title); ax.set_xlim(0, 100)
    for y, v in enumerate(m["missing_pct"][::-1]):
        ax.text(v + 1, y, f"{v:.1f}%", va="center", fontsize=8)
    return _save(fig, name)


def plot_class_distribution(y: pd.Series, name: str, title: str, order=None, colors=None) -> Path:  # E2
    vc = y.dropna().value_counts()
    order = order or sorted(vc.index)
    vals = [vc.get(k, 0) for k in order]
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    bars = ax.bar([str(k) for k in order], vals, color=[(colors or {}).get(k, "#5c6bc0") for k in order])
    tot = max(sum(vals), 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{100 * v / tot:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("products"); ax.set_title(f"{title}\nimbalance ratio {imbalance_ratio(y):.1f} : 1")
    return _save(fig, name)


def plot_box_by(df: pd.DataFrame, cols: list[str], by: str, name: str, title: str, order=None, clip_q=0.99) -> Path:  # E3/E8
    cols = [c for c in cols if c in df]
    order = order or sorted(df[by].dropna().unique())
    fig, axes = plt.subplots(1, len(cols), figsize=(3.3 * len(cols), 3.4), squeeze=False)
    for ax, c in zip(axes[0], cols):
        hi = df[c].quantile(clip_q)
        data = [df.loc[df[by] == k, c].dropna().clip(upper=hi) for k in order]
        ax.boxplot(data, tick_labels=[str(k) for k in order], showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "#c5cae9"})
        ax.set_title(c, fontsize=10); ax.set_xlabel(by)
    fig.suptitle(title, fontweight="bold")
    return _save(fig, name)


def plot_top_table(tbl: pd.DataFrame, label_col: str, name: str, title: str, color="#00897b") -> Path:  # E4/E5
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.28 * len(tbl) + 1)))
    ax.barh(tbl[label_col][::-1], tbl["pct"][::-1], color=color)
    ax.set_xlabel("% of products"); ax.set_title(title)
    return _save(fig, name)


def plot_top_ingredients_by_class(df: pd.DataFrame, by: str, name: str, title: str, n: int = 10) -> Path:  # E5
    classes = sorted(df[by].dropna().unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(3.6 * len(classes), 3.6), squeeze=False)
    for ax, k in zip(axes[0], classes):
        t = top_ingredients(df.loc[df[by] == k, "ingredients_norm"], n)
        ax.barh(t["ingredient"][::-1], t["pct"][::-1], color=NOVA_COLORS.get(k, "#5c6bc0"))
        ax.set_title(f"{by} = {k}", fontsize=10); ax.tick_params(labelsize=8)
    fig.suptitle(title, fontweight="bold")
    return _save(fig, name)


def plot_share_by_category(df: pd.DataFrame, cat: str, by: str, name: str, title: str, colors=None, order=None) -> Path:  # E6
    ct = pd.crosstab(df[cat], df[by], normalize="index") * 100
    order = order or list(ct.columns)
    ct = ct.reindex(columns=[c for c in order if c in ct.columns])
    ct = ct.sort_values(ct.columns[-1])
    fig, ax = plt.subplots(figsize=(7.5, max(3, 0.32 * len(ct) + 1)))
    left = np.zeros(len(ct))
    for c in ct.columns:
        ax.barh(ct.index, ct[c], left=left, label=str(c), color=(colors or {}).get(c))
        left += ct[c].values
    ax.set_xlabel("% of products in category"); ax.set_title(title)
    ax.legend(title=by, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    return _save(fig, name)


def plot_counts(s: pd.Series, name: str, title: str, top: int = 20, color="#5c6bc0") -> Path:  # E6/E11
    vc = s.dropna().astype(str).value_counts().head(top)
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.28 * len(vc) + 1)))
    ax.barh(vc.index[::-1], vc.values[::-1], color=color)
    ax.set_xlabel("products"); ax.set_title(title)
    return _save(fig, name)


def plot_nutrients(df: pd.DataFrame, cols: list[str], name: str, title: str) -> Path:              # E7 histograms
    cols = [c for c in cols if c in df and df[c].notna().any()]
    fig, axes = plt.subplots(2, (len(cols) + 1) // 2, figsize=(3.2 * ((len(cols) + 1) // 2), 5.4), squeeze=False)
    for ax, c in zip(axes.ravel(), cols):
        v = df[c].dropna()
        ax.hist(v.clip(upper=v.quantile(0.99)), bins=40, color="#7e57c2")
        ax.set_title(c, fontsize=9)
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle(title, fontweight="bold")
    return _save(fig, name)


def plot_corr(df: pd.DataFrame, cols: list[str], name: str, title: str) -> Path:                   # E7 heatmap
    cols = [c for c in cols if c in df and df[c].notna().any()]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(0.75 * len(cols) + 2.5, 0.75 * len(cols) + 1.8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046); ax.set_title(title)
    return _save(fig, name)


def plot_compare(a: pd.Series, b: pd.Series, labels: tuple[str, str], name: str, title: str, bins=30) -> Path:  # E10
    hi = max(a.quantile(0.99), b.quantile(0.99))
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(a.clip(upper=hi), bins=bins, alpha=0.55, density=True, label=labels[0], color="#1e88e5")
    ax.hist(b.clip(upper=hi), bins=bins, alpha=0.55, density=True, label=labels[1], color="#e53935")
    ax.set_ylabel("density"); ax.legend(); ax.set_title(title)
    return _save(fig, name)


def plot_confusion(cm: pd.DataFrame, name: str, title: str, out_dir: Path | None = None) -> Path:  # used by step6/7
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(0.7 * len(cm) + 2.6, 0.7 * len(cm) + 2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(cm.shape[1]), [c.replace("pred_", "") for c in cm.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(cm.shape[0]), [r.replace("true_", "") for r in cm.index], fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{norm.iloc[i, j]:.2f}\n({cm.iloc[i, j]})", ha="center", va="center", fontsize=6.5,
                    color="white" if norm.iloc[i, j] > 0.6 else "black")
    ax.set_title(title)
    return _save(fig, name, out_dir)


def plot_bar_metric(tbl: pd.DataFrame, label_col: str, metric: str, name: str, title: str, highlight=None) -> Path:
    t = tbl.sort_values(metric)
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.32 * len(t) + 1)))
    colors = ["#c62828" if highlight and highlight in str(l) else "#5c6bc0" for l in t[label_col]]
    ax.barh(t[label_col].astype(str), t[metric], color=colors)
    for y, v in enumerate(t[metric]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlabel(metric); ax.set_xlim(0, 1.08); ax.set_title(title)
    return _save(fig, name)
