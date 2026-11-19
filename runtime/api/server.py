"""NutriScan HTTP API for the Next.js frontend (web/). The Streamlit app stays available as a fallback.

Run (from Nutriscan_Final/):
  .venv/Scripts/python.exe -m uvicorn server:app --app-dir runtime/api --port 8000

Endpoints (all JSON unless noted)
  GET  /api/status          models loaded?, data source, model kind, gallery, OCR / chat availability, versions
  GET  /api/preferences     preference filters from rules.yaml
  POST /api/analyze         {text, prefs[]} -> verdict, flags, allergens, SHAP drivers, alternatives, audit log
  POST /api/ocr             multipart image -> extracted ingredient text (image kept in memory only)
  POST /api/chat            {id, question} -> grounded answer about a previous analysis
  POST /api/suggest         {id} -> Pakistani-market alternatives suggested by Gemini (assistant layer)
  GET  /api/metrics         result tables from reports/tables (or the earlier v0.2 run) for the performance page
  GET  /api/figures/{name}  PNG from reports/figures
"""
from __future__ import annotations

import math
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from nutriscan import __version__, agent, config, llm_suggest  # noqa: E402
from nutriscan import cleaning_log  # noqa: E402
from nutriscan.parser import extract_ingredient_section, normalize_text, parse_ingredients  # noqa: E402

app = FastAPI(title="NutriScan API", version=__version__)

_engine = None
_engine_error: str | None = None
_load_seconds: float | None = None
_results: "OrderedDict[str, dict]" = OrderedDict()   # recent analyses, for the chat endpoint
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MEMBER_V02 = config.REPORTS / "member_v0.2"


def _load_engine() -> None:
    global _engine, _engine_error, _load_seconds
    t0 = time.perf_counter()
    try:
        from nutriscan.pipeline import NutriScan
        _engine = NutriScan.load_or_train()
    except Exception as e:  # surfaced through /api/status
        _engine_error = f"{type(e).__name__}: {e}"
    _load_seconds = round(time.perf_counter() - t0, 1)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_load_engine, daemon=True).start()


def _require_engine():
    if _engine is None:
        raise HTTPException(503, _engine_error or "Models are still loading. Try again in a minute.")
    return _engine


def _clean(v):
    """JSON-safe values: NaN/inf -> None, numpy scalars -> Python."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _ocr_available() -> bool:
    try:
        from nutriscan.ocr import _rapidocr_engine
        _rapidocr_engine()
        return True
    except Exception:
        return False


@app.get("/api/status")
def status():
    e = _engine
    return {
        "ready": e is not None, "error": _engine_error, "load_seconds": _load_seconds, "version": __version__,
        "data_source": e.data_source if e else None,
        "model": e.nova_pred.kind if e else None,
        "trained_rows": (e.train_meta.get("rows") if e else None),
        "trained_at": (e.train_meta.get("trained_at") if e else None),
        "gallery_size": len(e.gallery) if e else 0,
        "gallery_synthetic": e.gallery_synthetic if e else None,
        "kb_entries": len(e.es.entries) if e else None, "rules": len(e.es.rules) if e else None,
        "ocr_available": _ocr_available(), "chat_online": agent.is_online(),
        "suggestions_available": llm_suggest.is_available(),
    }


@app.get("/api/preferences")
def preferences():
    e = _require_engine()
    return [{"id": p, "label": p.replace("_", " ")} for p in e.es.preferences]


class AnalyzeIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    prefs: list[str] = []


@app.post("/api/analyze")
def analyze(body: AnalyzeIn):
    e = _require_engine()
    t0 = time.perf_counter()
    r = e.analyze(body.text, body.prefs)
    if "error" in r:
        raise HTTPException(422, r["error"])
    rid = uuid.uuid4().hex[:12]
    _results[rid] = {**r, "prefs": body.prefs}
    while len(_results) > 50:
        _results.popitem(last=False)
    alts = [] if e.gallery_synthetic else [{k: _clean(v) for k, v in asdict(a).items()} for a in r["alternatives"]]
    return {
        "id": rid,
        "seconds": round(time.perf_counter() - t0, 2),
        "ingredients": r["ingredients"],
        "nova": {"final": r["nova_final"], "ml": r["nova_ml"], "confidence": r["nova_confidence"],
                 "rule_reasons": r["hybrid_reasons"]},
        "tier": {"value": r["health_tier"], "confidence": r["tier_confidence"]},
        "category": r["category"],
        "findings": [{**asdict(f)} for f in r["findings"] if f.kind != "allergen"],
        "allergens": [{"group": g, "words": w} for g, w in r["allergens"].items()],
        "drivers": {"nova": [{"term": t, "value": v} for t, v in r["drivers"]],
                    "tier": [{"term": t, "value": v} for t, v in r["tier_drivers"]]},
        "alternatives": alts,
        "alternatives_note": ("Alternatives are hidden: the product gallery has not been installed on this machine yet."
                              if e.gallery_synthetic else None),
        "versions": {k: _clean(v) for k, v in r["versions"].items() if not isinstance(v, dict)},
    }


@app.post("/api/ocr")
async def ocr(image: UploadFile = File(...)):
    data = await image.read()                      # in memory only; never written to disk
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image is larger than 8 MB.")
    try:
        from nutriscan.ocr import read_label
        raw = read_label(data)
    except ValueError as err:
        raise HTTPException(422, str(err))
    except Exception as err:
        raise HTTPException(503, f"OCR engine unavailable: {type(err).__name__}. Paste the ingredient list instead.")
    finally:
        del data
    section = extract_ingredient_section(raw) if raw else ""
    return {"text": section or raw, "raw": raw, "found_marker": bool(section and section != raw.strip())}


class ChatIn(BaseModel):
    id: str
    question: str = Field(min_length=1, max_length=500)


@app.post("/api/chat")
def chat(body: ChatIn):
    r = _results.get(body.id)
    if r is None:
        raise HTTPException(404, "Analysis expired. Run Analyze again.")
    answer, mode = agent.ask(body.question, r)
    return {"answer": answer, "mode": mode}


class SuggestIn(BaseModel):
    id: str


_suggestions: dict[str, dict] = {}


@app.post("/api/suggest")
def suggest(body: SuggestIn):
    r = _results.get(body.id)
    if r is None:
        raise HTTPException(404, "Analysis expired. Run Analyze again.")
    if body.id not in _suggestions:
        out = llm_suggest.suggest(r)
        if not out["available"]:
            return out                         # not cached, so adding a key later works without a restart
        _suggestions[body.id] = out
    return _suggestions[body.id]


# Measured on the real data: OFF/USDA export counts from docs/DATA_FACTS.md (independent validation of the
# exports), the training-pool size from artifacts/train_meta.json and the unseen-test size from reports/member_v0.2.
FUNNEL = [
    {"dataset": "Open Food Facts", "steps": [
        {"label": "Food products in the export", "value": 4_884_051, "source": "DATA_FACTS.md"},
        {"label": "With an ingredient list and a NOVA label", "value": 1_125_878, "source": "DATA_FACTS.md"},
        {"label": "English ingredient text", "value": 539_000, "approx": True, "source": "DATA_FACTS.md"},
        {"label": "Final training pool (cleaned, deduplicated, USDA barcodes removed)", "value": None,
         "source": "train_meta.json"},
    ]},
    {"dataset": "USDA FoodData Central", "steps": [
        {"label": "Rows in the Branded Foods file (all versions)", "value": 2_010_212, "source": "DATA_FACTS.md"},
        {"label": "Distinct products (one per barcode)", "value": 443_000, "approx": True, "source": "DATA_FACTS.md"},
        {"label": "Also present in Open Food Facts", "value": 363_767, "source": "DATA_FACTS.md"},
        {"label": "Unseen test products with a NOVA label", "value": 200_970, "source": "first full run"},
    ]},
]

EXAMPLES = [
    {"title": "Percentages, function words, E-number format and allergen tail",
     "raw": "INGREDIENTS: Sugar, Wheat Flour, Palm Oil (12%), Emulsifier (Soy Lecithin), Colour E 102. May contain traces of nuts."},
    {"title": "USDA style: capitals, \"contains 2% or less of\", footnote stars",
     "raw": "INGREDIENTS: ENRICHED FLOUR (WHEAT FLOUR, NIACIN, IRON), WATER, SUGAR, CONTAINS 2% OR LESS OF: SALT, YEAST, SOY LECITHIN**."},
    {"title": "OCR mistakes: digits and bars inside words",
     "raw": "Ingredients: wheat f1our, vegetab|e oil, c0coa powder, sa1t"},
    {"title": "Nested brackets kept as sub-ingredients",
     "raw": "Chocolate chips (sugar, cocoa mass, cocoa butter, emulsifier: soy lecithin), oat flakes, honey"},
]


def _example(ex: dict) -> dict:
    ings = parse_ingredients(extract_ingredient_section(ex["raw"]) if "ingredient" in ex["raw"].lower() else ex["raw"])
    return {**ex, "normalised": normalize_text(ex["raw"]),
            "ingredients": [{"name": i.name, "position": i.position + 1, "depth": i.depth} for i in ings]}


@app.get("/api/cleaning")
def cleaning():
    pool = None
    meta_path = config.ARTIFACTS / "train_meta.json"
    if meta_path.exists():
        import json
        pool = json.loads(meta_path.read_text(encoding="utf-8")).get("rows")
    funnel = [{**d, "steps": [{**s, "value": pool if s["source"] == "train_meta.json" else s["value"]} for s in d["steps"]]}
              for d in FUNNEL]
    return {"log": cleaning_log.entries() or None, "log_synthetic": cleaning_log.is_synthetic(),
            "funnel": funnel, "examples": [_example(e) for e in EXAMPLES]}


def _table(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {"columns": list(df.columns), "rows": [[_clean(v) for v in row] for row in df.itertuples(index=False)]}


def _confusion(path: Path):
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    return {"labels": [c.replace("pred_", "") for c in df.columns],
            "matrix": [[int(v) for v in row] for row in df.to_numpy()]}


@app.get("/api/metrics")
def metrics():
    t = config.TABLES
    targets = ["nova_group", "health_tier", "category"]
    current = {
        "comparison": {k: _table(t / f"model_comparison_{k}.csv") for k in targets},
        "per_class": {k: _table(t / f"per_class_best_{k}.csv") for k in targets},
        "confusion": {k: _confusion(t / f"confusion_best_{k}.csv") for k in targets},
        "imbalance": {k: _table(t / f"imbalance_{k}.csv") for k in targets[:2]},
        "cross_dataset": {k: _table(t / f"cross_dataset_{k}.csv") for k in targets},
        "unseen": _table(t / "unseen_metrics.csv"),
        "ablation": _table(t / "ablation_nova.csv"),
        "robustness": _table(t / "robustness_nova_group.csv"),
        "alternatives": _table(t / "alternatives_eval.csv"),
    }
    has_current = any(v for group in current.values() for v in (group.values() if isinstance(group, dict) else [group]))
    previous = None
    if MEMBER_V02.exists():
        previous = {
            "unseen": _table(MEMBER_V02 / "unseen_metrics.csv"),
            "baselines_nova": _table(MEMBER_V02 / "baselines_nova_group.csv"),
            "per_class_nova": _table(MEMBER_V02 / "unseen_per_class_nova_group.csv"),
            "per_class_tier": _table(MEMBER_V02 / "unseen_per_class_health_tier.csv"),
            "per_class_category": _table(MEMBER_V02 / "unseen_per_class_category.csv"),
            "confusion_nova": _confusion(MEMBER_V02 / "unseen_confusion_nova_group.csv"),
        }
    figures = sorted(p.stem for p in config.FIGURES.glob("*.png")) if config.FIGURES.exists() else []
    return {"current": current if has_current else None, "previous": previous, "figures": figures}


@app.get("/api/figures/{name}")
def figure(name: str):
    path = (config.FIGURES / f"{name}.png").resolve()
    if path.parent != config.FIGURES.resolve() or not path.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(path, media_type="image/png")
