# Limitations to state in the report
1. **OFF NOVA is an approximation.** OFF computes NOVA automatically from ingredient lists (default NOVA 1, raised when markers are found), and NOVA 1 is the most error-prone group. Models therefore learn "OFF's NOVA", and largely re-learn its marker formula. Optional: hand-label 100-200 products to quantify the error.
2. **Hybrid mirrors OFF's logic.** `hybrid_nova` only raises to NOVA 4 with a stated reason, like OFF's own algorithm.
3. **Nutri-Score caveats.** Junk values removed; 2023 algorithm migration means mixed-vintage labels; the health-tier model uses ingredient text only. Nutriments are never features (see `leakage.py`).
4. **NOVA 2 scarcity.** ~1.8% of the modelling set: expect near-zero recall; per-class F1 and a 3-class (1+2 merged) variant are reported.
5. **USDA overlap.** ~85% of USDA products also exist in OFF; the headline external result removes overlapping barcodes from training. The matched-products variant tests transcription-source robustness, not generalisation.
6. **Language and country bias.** English is ~48% of the data, US products ~35%. Report per-category and per-language performance.
7. **Attribution.** "Contains data from Open Food Facts, available under the Open Database License" (ODbL) with a link, shown in the UI. Share-alike applies to any derived database you publish. USDA FoodData Central is public domain; cite it.
8. **No medical claims.** "Associated with" plus a source; allergen warnings are rule-based.
9. **USDA-derived labels.** The USDA health tier is computed with the Nutri-Score 2017 algorithm and a fruit/vegetable/nut
   share of 0 (USDA does not report it), so it can be stricter than OFF's grade; step3b reports the agreement on matched
   products. The USDA food-group label comes from keyword rules on `branded_food_category`
   (`reports/tables/usda_category_mapping.csv`).
10. **Rare classes.** step6 drops classes with fewer than 10 training rows before tuning/stacking; list them in the report.
11. **Explanations.** Linear SHAP uses an independent-feature background (the training mean). Tree SHAP explains only the
    engineered-feature model, not the TF-IDF members of the ensemble.
12. **Expert system coverage.** 58 KB entries and 12 rules cover common additives only; all entries are drafts
    (`verified: false`) until checked against their sources.
13. **Chat agent.** Optional; answers only from the analysis result and falls back to a template. It is not evaluated
    unless the 30-50 question groundedness check (plan Section 12) is done.

