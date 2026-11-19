from nutriscan.parser import extract_ingredient_section, ingredient_names, normalize_text, parse_ingredients, split_top_level


def test_strips_percentages_and_tail():
    names = ingredient_names("Ingredients: Wheat flour (65%), sugar, salt. May contain traces of nuts.")
    assert names == ["wheat flour", "sugar", "salt"]


def test_bracket_aware_split():
    assert split_top_level("chocolate (sugar, cocoa mass), milk") == ["chocolate (sugar, cocoa mass)", "milk"]


def test_function_word_dropped_but_subingredients_kept():
    ings = parse_ingredients("sugar, emulsifier (soy lecithin, e471)")
    names = [i.name for i in ings]
    assert "emulsifier" not in names and "soy lecithin" in names and "e471" in names
    assert all(i.position == 1 for i in ings if i.depth == 1)


def test_e_number_normalised():
    assert "e330" in ingredient_names("water, E 330, acidity regulator (E-330)")


def test_ocr_char_confusion():
    assert ingredient_names("c0coa powder") == ["cocoa powder"]
    assert ingredient_names("pa1m oil") == ["palm oil"]


def test_extract_section_from_ocr():
    raw = "NUTRITION INFO per 100g energy 500kcal INGREDIENTS: Sugar, cocoa butter. Store in a cool place. Best before: see pack"
    assert extract_ingredient_section(raw).lower().startswith("sugar, cocoa butter")


def test_us_contains_less_than():
    names = ingredient_names("Enriched flour, sugar, contains 2% or less of: salt, soy lecithin")
    assert "salt" in names and "soy lecithin" in names


def test_empty():
    assert ingredient_names("") == []


def test_usda_style_strings():
    raw = ("INGREDIENTS: ENRICHED FLOUR (WHEAT FLOUR, NIACIN), SUGAR, PORK**, CONTAINS LESS THAN 2% OF: SALT, "
           "NATURAL FLAVOR. - NO NITRATES OR NITRITES ADDED EXCEPT THOSE NATURALLY OCCURRING IN CELERY POWDER")
    names = ingredient_names(raw)
    assert "salt" in names and "natural flavor" in names and "pork" in names
    assert not any("*" in n or "contains" in n or "nitrates" in n for n in names)
    assert ingredient_names("SUGAR, 2% OR LESS OF: SALT") == ["sugar", "salt"]


def test_ocr_quality_score_prefers_clean_text():
    from nutriscan.ocr import text_quality
    clean = "INGREDIENTS: Tomato paste, water, sugar, vinegar, salt"
    garbled = "LAAL Tomato Ketchup maies e ce e doen s es"
    assert text_quality(clean) > text_quality(garbled)


def test_ocr_reads_sample_labels():
    import pytest
    from pathlib import Path
    from nutriscan import config
    try:
        from nutriscan.ocr import _rapidocr_engine, read_label
        _rapidocr_engine()
    except Exception:
        pytest.skip("RapidOCR / onnxruntime not available on this machine")
    from nutriscan.evaluation import ingredient_extraction_accuracy
    from nutriscan.parser import extract_ingredient_section, ingredient_names
    for name in ("03_cola_angled", "05_ketchup_glare"):     # the two that needed straightening / no preprocessing
        img = config.PROJECT_ROOT / "samples" / f"{name}.jpg"
        ref = img.with_suffix(".txt").read_text(encoding="utf-8")
        hyp = extract_ingredient_section(read_label(img.read_bytes()))
        assert ingredient_extraction_accuracy(ingredient_names(ref), ingredient_names(hyp)) >= 0.9


def test_function_word_prefixes_are_dropped():
    from nutriscan.parser import ingredient_names
    assert ingredient_names("sugar, emulsifier: soy lecithin, colour E 102") == ["sugar", "soy lecithin", "e102"]
    assert ingredient_names("acidity regulator e330, salt") == ["e330", "salt"]
    assert ingredient_names("sweetener syrup") == ["sweetener syrup"]      # not a function-word + name pattern
