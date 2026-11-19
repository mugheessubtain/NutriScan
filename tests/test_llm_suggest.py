import json
import types as pytypes

from nutriscan import llm_suggest
from nutriscan.expert_system import Finding

RESULT = {"ingredients": ["sugar", "wheat flour", "palm oil"], "nova_final": 4, "health_tier": "poor",
          "category": "sugary-snacks", "findings": [Finding("rule", "R", "Sugar first", 2, "WHO")], "prefs": []}


def test_without_key_reports_unavailable(monkeypatch):
    monkeypatch.setattr(llm_suggest, "_api_key", lambda: None)
    out = llm_suggest.suggest(RESULT)
    assert out["available"] is False and out["items"] == [] and "GEMINI_API_KEY" in out["message"]


def test_parses_and_cleans_gemini_answer(monkeypatch):
    answer = {"alternatives": [
        {"name": "Plain roasted chana", "brand": "", "why_better": "No added sugar or palm oil.", "likely_nova": 1},
        {"name": "", "brand": "x", "why_better": "dropped: no name", "likely_nova": 2},
        {"name": "Digestive biscuits", "brand": "Some Brand", "why_better": "More fibre.", "likely_nova": 9},
    ]}

    class FakeModels:
        def generate_content(self, **kw):
            assert "sugary-snacks" in kw["contents"]
            return pytypes.SimpleNamespace(text=json.dumps(answer))

    class FakeClient:
        def __init__(self, **kw):
            self.models = FakeModels()

    from google import genai
    monkeypatch.setattr(llm_suggest, "_api_key", lambda: "test-key")
    monkeypatch.setattr(genai, "Client", FakeClient)
    out = llm_suggest.suggest(RESULT)
    assert out["available"] is True
    assert [x["name"] for x in out["items"]] == ["Plain roasted chana", "Digestive biscuits"]
    assert out["items"][1]["likely_nova"] is None          # out-of-range value is discarded
