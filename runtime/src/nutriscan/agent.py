"""Plan v2 Section 14.1: optional external API, a chat helper for follow-up questions.

Purpose: explain NutriScan's OWN result in plain language ("why is this NOVA 4?", "what is E471?"). The API never
classifies products and never produces allergen warnings; those stay in the local ML + rule pipeline.
Integration: the result of `NutriScan.analyze` is turned into a small JSON context and sent with the question to the
Claude API (`anthropic` SDK). Key handling: the SDK reads ANTHROPIC_API_KEY from the environment (or Streamlit
secrets copied into it); nothing is stored in the repository. Fallback: when the SDK, the key or the network is
unavailable, or the request fails, a template answer built from the same result is returned instead.
"""
from __future__ import annotations

import json
import os

MODEL = "claude-opus-5-5"
TIMEOUT_S = 30.0
NOVA_TEXT = {1: "unprocessed or minimally processed", 2: "a processed culinary ingredient", 3: "processed",
             4: "ultra-processed"}
SYSTEM = (
    "You explain the output of NutriScan, a student project that analyses food ingredient labels. "
    "Answer ONLY from the JSON context you are given (parsed ingredients, predicted NOVA group and health tier, "
    "rules that fired with their sources, the words that most influenced the model, suggested alternatives). "
    "If the context does not contain the answer, say so. Use 'associated with' wording and cite the source given in "
    "the context. Do not give medical advice, diagnoses or treatment suggestions; if asked, say that NutriScan is not "
    "medical advice and suggest asking a doctor or dietitian. Do not add allergen warnings that are not in the "
    "context. Keep answers under 150 words."
)


def context_from_result(r: dict) -> dict:
    """The part of an analyze() result the model may see (no personal data exists in it)."""
    return {
        "ingredients": r.get("ingredients", []),
        "nova_group": r.get("nova_final"), "nova_ml_prediction": r.get("nova_ml"),
        "nova_confidence": round(r.get("nova_confidence", 0.0), 2),
        "rule_adjustments": r.get("hybrid_reasons", []),
        "health_tier": r.get("health_tier"), "category": r.get("category"),
        "findings": [{"kind": f.kind, "message": f.message, "source": f.source, "evidence": f.evidence}
                     for f in r.get("findings", [])],
        "top_words_toward_nova": [w for w, _ in r.get("drivers", [])],
        "alternatives": [{"name": a.name, "brand": a.brand, "why_better": a.reason} for a in r.get("alternatives", [])],
    }


def is_online() -> bool:
    """True when a Claude API call can be attempted (SDK installed and a key configured)."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def template_answer(r: dict) -> str:
    """Offline explanation built only from the result (used whenever the API is unavailable)."""
    nova = r.get("nova_final")
    parts = [f"NutriScan places this product in NOVA group {nova} ({NOVA_TEXT.get(nova, 'unknown')}) "
             f"with {r.get('nova_confidence', 0):.0%} model confidence, and in the '{r.get('health_tier')}' health tier."]
    if r.get("drivers"):
        parts.append("The words that pushed the model most were: " + ", ".join(w for w, _ in r["drivers"][:5]) + ".")
    parts += r.get("hybrid_reasons", [])
    flags = [f for f in r.get("findings", []) if f.kind in ("rule", "additive")]
    if flags:
        parts.append("Flags: " + " ".join(f"{f.message} (source: {f.source})" for f in flags[:3]))
    if r.get("alternatives"):
        a = r["alternatives"][0]
        parts.append(f"A closer-to-whole-food option in the same group is {a.name} ({a.brand}): {a.reason}.")
    parts.append("This is not medical advice.")
    return " ".join(parts)


def ask(question: str, result: dict) -> tuple[str, str]:
    """Answer `question` about `result`. Returns (answer, mode) where mode is 'claude' or 'offline template'."""
    if not is_online():
        return template_answer(result), "offline template"
    import anthropic

    client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=1)
    context = json.dumps(context_from_result(result), sort_keys=True)
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            output_config={"effort": "low"},                     # short grounded answers; chat does well at low effort
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},                 # server-side fallback model if the request is declined
            messages=[{"role": "user", "content": f"<context>{context}</context>\n\nQuestion: {question}"}],
        )
    except anthropic.APIConnectionError:
        return template_answer(result), "offline template (no network)"
    except anthropic.RateLimitError:
        return template_answer(result), "offline template (rate limited)"
    except anthropic.APIStatusError as e:
        return template_answer(result), f"offline template (API error {e.status_code})"
    if response.stop_reason == "refusal":
        return template_answer(result), "offline template (request declined)"
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return (text or template_answer(result)), "claude"
