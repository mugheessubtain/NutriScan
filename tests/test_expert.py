from nutriscan.expert_system import ExpertSystem
from nutriscan.parser import parse_ingredients

es = ExpertSystem()


def analyze(text):
    return es.analyze(parse_ingredients(text))


def ids(res, kind):
    return {f.id for f in res.findings if f.kind == kind}


def test_sweetener_and_colour_rule():
    res = analyze("carbonated water, aspartame, caramel colour iv, natural flavours")
    assert "R_SWEETENER_AND_COLOUR" in ids(res, "rule")
    assert res.nova4_evidence >= 2


def test_hydrogenated_fat_flagged_high_severity():
    res = analyze("wheat flour, partially hydrogenated vegetable oil, salt")
    hit = [f for f in res.findings if f.id == "R_HYDROGENATED_FAT"]
    assert hit and hit[0].severity == 3 and hit[0].source


def test_sugar_in_first_three():
    assert "R_HIGH_ADDED_SUGAR" in ids(analyze("sugar, wheat flour, salt"), "rule")
    assert "R_HIGH_ADDED_SUGAR" not in ids(analyze("wheat flour, water, yeast, salt, sugar"), "rule")


def test_allergens_and_false_positive():
    assert "milk" in es.detect_allergens(parse_ingredients("sugar, skimmed milk powder"))
    assert "milk" not in es.detect_allergens(parse_ingredients("sugar, cocoa butter"))


def test_clean_product_no_rules():
    res = analyze("rolled oats")
    assert not ids(res, "rule") and res.nova4_markers == []


def test_hybrid_only_raises_with_reason():
    res = analyze("water, aspartame, e150d")
    final, why = es.hybrid_nova(3, res)
    assert final == 4 and why
    clean = analyze("potatoes, salt")
    assert es.hybrid_nova(3, clean) == (3, [])
    assert es.hybrid_nova(1, clean)[0] == 1


def test_preferences():
    ings = parse_ingredients("flour, palm oil, salt")
    assert es.violates_preference(ings, ["no_palm_oil"]) == ["no_palm_oil"]
    assert es.violates_preference(ings, ["vegan"]) == []


def test_v02_rules():
    assert "R_CHILD_ACTIVITY_COLOURS" in ids(analyze("sugar, glucose syrup, colour (e104)"), "rule")
    assert "R_MULTIPLE_SUGAR_SOURCES" in ids(analyze("wheat flour, sugar, glucose syrup, dextrose, fat"), "rule")
    assert "R_MULTIPLE_SUGAR_SOURCES" not in ids(analyze("wheat flour, sugar, glucose syrup"), "rule")
    assert "R_HIGH_SALT" in ids(analyze("potatoes, salt, sunflower oil"), "rule")
    assert "R_PROCESSED_MEAT_CURING" in ids(analyze("pork, water, salt, sodium nitrite"), "rule")
    res = analyze("water, colour (e150d), flavouring, emulsifier (e471)")
    assert "R_COSMETIC_ADDITIVES" in ids(res, "rule") and res.nova4_evidence >= 2
    assert len(es.entries) >= 50 and len(es.rules) >= 10
