"""Shared metrics: macro-F1 + per-class report, confusion matrix, ordinal metrics for NOVA, OCR error rates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support


def merge_nova12(y) -> pd.Series:
    """3-class robustness variant: NOVA 2 is ~1.8% of the OFF modelling set, so merge 1+2 (relabel as 1)."""
    return pd.Series(y).replace({2: 1})


def summary(y_true, y_pred, ordinal: bool = False) -> dict:
    out = {"accuracy": accuracy_score(y_true, y_pred),
           "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}
    if ordinal:
        out.update(ordinal_metrics(y_true, y_pred))
    return out


def ordinal_metrics(y_true, y_pred) -> dict:
    """NOVA is ordinal: quadratic weighted kappa + Spearman complement macro-F1."""
    yt, yp = np.asarray(y_true, float), np.asarray(y_pred, float)
    qwk = cohen_kappa_score(yt, yp, weights="quadratic") if len(set(yt)) > 1 or len(set(yp)) > 1 else float("nan")
    rho = spearmanr(yt, yp).correlation if len(set(yt)) > 1 and len(set(yp)) > 1 else float("nan")
    return {"qwk": qwk, "spearman": rho}


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return pd.DataFrame({"class": labels, "precision": p, "recall": r, "f1": f, "support": s}).round(3)


def confusion_df(y_true, y_pred) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                        index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])


def group_metrics(df: pd.DataFrame, group_col: str, y_true: str, y_pred: str, min_n: int = 30) -> pd.DataFrame:
    """Fairness table: accuracy and macro-F1 per group (groups smaller than min_n are skipped)."""
    rows = []
    for g, part in df.groupby(group_col):
        if len(part) < min_n:
            continue
        rows.append({group_col: g, "n": len(part), **{k: round(v, 4) for k, v in summary(part[y_true], part[y_pred]).items()}})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


# ---------- OCR-style noise for the robustness test ----------
OCR_CONFUSIONS = {"o": "0", "l": "1", "i": "l", "e": "c", "a": "o", "s": "5", "b": "6", "g": "9", "u": "v", "n": "m",
                  "c": "e", "t": "f", "h": "b"}


def inject_ocr_noise(text: str, rate: float, rng: np.random.Generator) -> str:
    """Corrupt roughly `rate` of the letters like a bad OCR pass: look-alike substitution (60%), deletion (20%),
    duplicated character (10%) or swapped neighbours (10%)."""
    if rate <= 0 or not text:
        return text
    out, chars, i = [], list(text), 0
    while i < len(chars):
        ch = chars[i]
        if ch.isalpha() and rng.random() < rate:
            r = rng.random()
            if r < 0.6:
                out.append(OCR_CONFUSIONS.get(ch.lower(), ch))
            elif r < 0.8:
                pass                                  # deletion
            elif r < 0.9:
                out.append(ch + ch)
            elif i + 1 < len(chars):
                out.append(chars[i + 1] + ch)
                i += 1
            else:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


# ---------- OCR metrics (pure python, no extra dependency) ----------
def _edit_distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = reference.lower().split(), hypothesis.lower().split()
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = " ".join(reference.lower().split()), " ".join(hypothesis.lower().split())
    return _edit_distance(ref, hyp) / max(len(ref), 1)


def ingredient_extraction_accuracy(ref_names: list[str], hyp_names: list[str]) -> float:
    """Set-level F1 between reference and extracted ingredient names."""
    r, h = set(ref_names), set(hyp_names)
    if not r and not h:
        return 1.0
    tp = len(r & h)
    if tp == 0:
        return 0.0
    p, rec = tp / len(h), tp / len(r)
    return 2 * p * rec / (p + rec)
