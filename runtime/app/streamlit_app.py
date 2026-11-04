"""NutriScan Streamlit UI (plan v2 Section 10). Run: streamlit run runtime/app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from nutriscan import agent, config  # noqa: E402
from nutriscan.pipeline import NutriScan  # noqa: E402

st.set_page_config(page_title="NutriScan", page_icon="🥫", layout="wide")


@st.cache_resource(show_spinner="Loading models and product gallery…")
def get_engine() -> NutriScan:
    return NutriScan.load_or_train()


NOVA_LABEL = {1: "Unprocessed / minimally processed", 2: "Processed culinary ingredients",
              3: "Processed foods", 4: "Ultra-processed"}
SEV_ICON = {3: "🔴", 2: "🟠", 1: "🟡", 0: "⚪"}
OCR_ENGINES = ["rapidocr", "tesseract", "easyocr"]

st.title("🥫 NutriScan")
st.caption("Ingredient label analyzer with healthier-alternative suggestions")
eng = get_engine()
if eng.data_source != "off":
    st.warning(f"Running on **{eng.data_source}** data. Results are demo-only until the models are trained on "
               "Open Food Facts (training/steps/step4_train_models.py).")

with st.sidebar:
    st.header("Preferences")
    prefs = [p for p in eng.es.preferences if st.checkbox(p.replace("_", " "), key=p)]
    st.divider()
    online = agent.is_online()
    st.caption(("🟢 Online: chat uses the Claude API" if online else "⚪ Offline: chat uses template answers")
               + " · core analysis always runs locally")
    st.caption(f"Model: `{eng.nova_pred.kind}` · gallery: {len(eng.gallery):,} products")

tab_scan, tab_data, tab_about = st.tabs(["🔍 Analyze", "📊 Data insights", "ℹ️ About & ethics"])

# ------------------------------------------------------------------ analyze
with tab_scan:
    src_text, src_img = st.tabs(["Paste text", "Upload photo"])
    with src_text:
        raw_text = st.text_area("Ingredient list", height=140,
                                placeholder="Sugar, wheat flour, palm oil, glucose syrup, emulsifier (soy lecithin), "
                                            "flavouring, colour (E102)")
    with src_img:
        up = st.file_uploader("Label photo (processed in memory, never stored)", type=["jpg", "jpeg", "png"])
        engine_choice = st.radio("OCR engine", OCR_ENGINES, horizontal=True,
                                 help="RapidOCR needs no system install; compare engines with training/eval_ocr.py")
        ocr_text = ""
        if up is not None:
            try:
                from nutriscan.ocr import read_label
                with st.spinner("Reading label…"):
                    ocr_text = read_label(up.getvalue(), engine=engine_choice)
            except Exception as e:  # OCR libs are optional
                st.error(f"OCR unavailable or failed: {e}")
        edited = st.text_area("Parsed text (edit OCR mistakes here)", value=ocr_text, height=140, key="ocr_edit")

    raw = raw_text or edited
    if st.button("Analyze", type="primary") and raw.strip():
        st.session_state["result"] = eng.analyze(raw, prefs)
        st.session_state["chat"] = []

    r = st.session_state.get("result")
    if r and "error" in r:
        st.error(r["error"])
    elif r:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("NOVA group", f'{r["nova_final"]}', help=NOVA_LABEL.get(r["nova_final"], ""))
        c1.caption(NOVA_LABEL.get(r["nova_final"], ""))
        c2.metric("Health tier", r["health_tier"].title())
        c2.caption(f'confidence {r["tier_confidence"]:.0%}')
        c3.metric("NOVA confidence", f'{r["nova_confidence"]:.0%}')
        c4.metric("Food group", r["category"].replace("-", " "))
        st.caption(f'ML NOVA prediction: {r["nova_ml"]} · final (after rules): {r["nova_final"]}')
        for why in r["hybrid_reasons"]:
            st.info(why)
        for g, words in r["allergens"].items():
            st.warning(f'Allergen ({g.replace("_", " ")}): {", ".join(words)} · rule-based keyword match')

        left, right = st.columns(2)
        with left:
            st.subheader("Flags (expert system)")
            flags = [f for f in r["findings"] if f.kind != "allergen"]
            if not flags:
                st.write("No flagged ingredients.")
            for f in flags:
                with st.container(border=True):
                    st.markdown(f'{SEV_ICON.get(f.severity, "⚪")} **{f.kind.title()}** · {f.message}')
                    st.caption(f'Evidence: {", ".join(f.evidence)} · Source: {f.source}')
            st.subheader("Parsed ingredients")
            st.write(" → ".join(r["ingredients"]))
        with right:
            st.subheader("Why the model decided (SHAP)")
            cols = st.columns(2)
            for col, key, title in ((cols[0], "drivers", f'toward NOVA {r["nova_ml"]}'),
                                    (cols[1], "tier_drivers", f'toward "{r["health_tier"]}"')):
                with col:
                    st.caption(title)
                    if r[key]:
                        st.bar_chart(pd.DataFrame(r[key], columns=["term", "SHAP"]).set_index("term"), horizontal=True)
                    else:
                        st.write("–")
            st.subheader("Healthier alternatives")
            if not r["alternatives"]:
                st.write("No better product found in the same food group.")
            for a in r["alternatives"]:
                with st.container(border=True):
                    st.markdown(f"**{a.name}** ({a.brand})")
                    nova = "?" if pd.isna(a.nova_group) else f"{a.nova_group:.0f}"
                    st.caption(f"Similarity {a.similarity:.0%} · NOVA {nova} · tier {a.health_tier} · {a.reason}")

        st.subheader("Ask about this result")
        for q, ans, mode in st.session_state.get("chat", []):
            st.chat_message("user").write(q)
            st.chat_message("assistant").write(ans)
            st.caption(f"answered by: {mode}")
        q = st.chat_input("e.g. Why is this ultra-processed? What is E471?")
        if q:
            with st.spinner("Thinking…"):
                ans, mode = agent.ask(q, r)
            st.session_state.setdefault("chat", []).append((q, ans, mode))
            st.rerun()
        with st.expander("Audit log"):
            st.json(r["versions"])

# ------------------------------------------------------------------ data insights
with tab_data:
    figs, tables = config.FIGURES, config.TABLES
    if not any(figs.glob("*.png")):
        st.info("No report figures yet. Run steps 2b, 6 and 7 in training/steps/ to generate them.")
    sections = [
        ("Dataset overview (EDA)", ["eda_E2_off_nova", "eda_E2_off_tier", "eda_E3_off_counts_by_nova",
                                    "eda_E4_off_top_additives_nova4", "eda_E6_off_nova_share_by_category",
                                    "eda_E10_off_vs_usda_n_ingredients"]),
        ("Model comparison", ["cmp_nova_group", "cmp_health_tier", "confusion_best_nova_group",
                              "confusion_best_health_tier"]),
        ("Explainability, robustness, fairness", ["shap_linear_nova_group_4", "shap_tree_nova_group",
                                                  "robustness_ocr_noise", "fairness_nova_group_category"]),
    ]
    for title, names in sections:
        paths = [figs / f"{n}.png" for n in names if (figs / f"{n}.png").exists()]
        if not paths:
            continue
        st.subheader(title)
        cols = st.columns(2)
        for i, p in enumerate(paths):
            cols[i % 2].image(str(p), width="stretch")
    for name, title in (("ablation_nova", "ML vs rules vs hybrid (NOVA)"), ("alternatives_eval", "Alternatives engine")):
        if (tables / f"{name}.csv").exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(tables / f"{name}.csv"), width="stretch", hide_index=True)

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown(
        "**How it works.** The ingredient list is cleaned and parsed, then three classifiers predict the NOVA "
        "processing group, the Nutri-Score health tier and the food group from the ingredient text. A rule-based "
        "expert system flags ingredients with a cited source and may raise the NOVA group to 4, always stating "
        "which rules fired. Alternatives are similar products in the same food group with a better tier or a "
        "lower NOVA group.\n\n"
        "**Privacy.** No accounts, no personal data. Uploaded photos are processed in memory and discarded. The "
        "optional chat sends only the analysis result (never the photo) to the Claude API.\n\n"
        "**Limitations.** Open Food Facts computes NOVA from ingredient markers, so the NOVA model partly re-learns "
        "that formula; the data over-represents some countries and languages; allergen detection is a keyword "
        "match on the parsed list and can miss or over-flag items. Always read the actual label.")
    st.caption(config.DISCLAIMER)

st.divider()
st.caption(config.DISCLAIMER)
st.markdown(config.ATTRIBUTION_MD)
