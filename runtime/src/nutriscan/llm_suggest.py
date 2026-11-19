"""Healthier alternatives for the Pakistani market, suggested by Gemini (an LLM assistant layer).

What it is: an optional assistant feature. The NOVA group, health tier, food group and flags still come from the
trained models and the rule system; Gemini only receives that result and suggests similar products sold in
Pakistan that are likely less processed or healthier. The UI labels these as Gemini suggestions.
Setup: put GEMINI_API_KEY=<key> in Nutriscan_Final/.env (git-ignored) or the environment. GEMINI_MODEL overrides the
model name. Without a key, or when the call fails, the feature reports that it is unavailable.
"""
from __future__ import annotations

import json
import os

from . import config

DEFAULT_MODEL = "gemini-2.5-flash"
TIMEOUT_MS = 30_000
MAX_ITEMS = 4

SYSTEM = (
    "You help shoppers in Pakistan find healthier packaged-food choices. You receive the analysis of one product "
    "(its ingredients, NOVA processing group 1-4, Nutri-Score health tier and food group, from a trained model). "
    "Suggest up to 4 alternatives in the SAME food group that a shopper can buy in Pakistan (supermarkets such as "
    "Imtiaz, Carrefour, Al-Fatah, or local shops) and that are likely less processed or have a better nutrient "
    "profile. Only name a brand if you are confident the product exists in Pakistan; otherwise describe a generic "
    "option (for example 'plain roasted chana' or 'unsweetened dahi') and leave the brand empty. Give a short, "
    "concrete reason tied to the analysed ingredients. Do not make medical claims; use 'associated with' wording."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "brand": {"type": "string"},
                    "why_better": {"type": "string"},
                    "likely_nova": {"type": "integer", "minimum": 1, "maximum": 4},
                },
                "required": ["name", "brand", "why_better", "likely_nova"],
            },
        }
    },
    "required": ["alternatives"],
}


def _api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    env = config.PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY") and value.strip():
                return value.strip().strip('"').strip("'")
    return None


def is_available() -> bool:
    try:
        from google import genai  # noqa: F401
    except ImportError:
        return False
    return bool(_api_key())


def _context(r: dict) -> dict:
    return {
        "ingredients": r.get("ingredients", []),
        "nova_group": r.get("nova_final"),
        "health_tier": r.get("health_tier"),
        "food_group": r.get("category"),
        "flags": [f.message for f in r.get("findings", []) if f.kind in ("rule", "additive")][:6],
        "user_preferences": r.get("prefs", []),
    }


def suggest(r: dict) -> dict:
    """Returns {"available": bool, "items": [...], "model": str, "message": str | None}."""
    model = os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
    key = _api_key()
    if not key:
        return {"available": False, "items": [], "model": model,
                "message": "Add GEMINI_API_KEY to the .env file to get Pakistani market suggestions."}
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError:
        return {"available": False, "items": [], "model": model, "message": "google-genai is not installed."}

    client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    prompt = "Analysed product:\n" + json.dumps(_context(r), ensure_ascii=False)
    if r.get("prefs"):
        prompt += "\nRespect the user's preferences listed above (for example no palm oil)."
    try:
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                response_mime_type="application/json",
                response_json_schema=SCHEMA,
                temperature=0.3,
            ),
        )
        data = json.loads(resp.text or "{}")
    except errors.APIError as e:
        return {"available": False, "items": [], "model": model, "message": f"Gemini request failed ({e.code})."}
    except (json.JSONDecodeError, ValueError):
        return {"available": False, "items": [], "model": model, "message": "Gemini returned an unreadable answer."}
    except Exception as e:  # network errors, timeouts
        return {"available": False, "items": [], "model": model, "message": f"Gemini unavailable ({type(e).__name__})."}

    items = []
    for a in data.get("alternatives", [])[:MAX_ITEMS]:
        name = str(a.get("name", "")).strip()
        if not name:
            continue
        nova = a.get("likely_nova")
        items.append({
            "name": name[:120],
            "brand": str(a.get("brand", "")).strip()[:80],
            "why_better": str(a.get("why_better", "")).strip()[:300],
            "likely_nova": nova if isinstance(nova, int) and 1 <= nova <= 4 else None,
        })
    return {"available": True, "items": items, "model": model, "message": None if items else "No suggestions returned."}
