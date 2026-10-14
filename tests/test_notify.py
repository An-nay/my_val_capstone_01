from my_val_capstone_01.modeling.notify import CONTENTS, TYPICAL_COUNT, TYPICAL_FIRST_VALUE, TYPICAL_VALUE, suggest_next_item, suggest_with_fallback
from my_val_capstone_01.modeling.settings import SETTINGS


def ranked(*pairs):
    return [{"category": c, "probability": p} for c, p in pairs]


NONE_YET = {c: 0 for c in CONTENTS}
ALL_TYPICAL = {c: TYPICAL_COUNT[c] for c in CONTENTS}


def test_settings_come_from_the_dataset_and_exclude_vehicles():
    assert "Vehicles" not in SETTINGS["item_categories"]
    assert set(TYPICAL_VALUE) >= set(CONTENTS) and set(TYPICAL_FIRST_VALUE) == set(CONTENTS) and set(TYPICAL_COUNT) == set(CONTENTS)


def test_suggests_the_most_likely_category_that_has_no_item_yet():
    counts = {**NONE_YET, "Electronics": 2}  # Electronics is documented, so skip it
    result = suggest_next_item(ranked(("Electronics", 0.5), ("Furniture", 0.3), ("Appliances", 0.2)), counts)
    assert not result["done"] and result["stage"] == "new_category" and result["category"] == "Furniture"
    assert result["suggested_value"] == TYPICAL_FIRST_VALUE["Furniture"] and result["probability"] == 0.3  # new category: first-item median


def test_vehicles_are_never_suggested():
    result = suggest_next_item(ranked(("Vehicles", 0.9), ("Clothings", 0.1)), NONE_YET)
    assert result["category"] == "Clothings"


def test_when_every_category_has_an_item_it_asks_for_more_where_furthest_below_typical():
    counts = {c: 1 for c in CONTENTS}
    result = suggest_next_item(ranked(("Electronics", 0.9)), counts)
    shortfall = {c: TYPICAL_COUNT[c] - 1 for c in CONTENTS}
    assert result["stage"] == "more_of_category" and shortfall[result["category"]] == max(shortfall.values())
    assert result["have"] == 1 and result["typical"] == TYPICAL_COUNT[result["category"]]
    assert result["suggested_value"] == TYPICAL_VALUE[result["category"]]  # another one: median of all items


def test_loop_ends_when_every_category_is_at_its_typical_count():
    result = suggest_next_item(ranked(("Electronics", 0.9)), ALL_TYPICAL)
    assert result["done"]


def test_each_accepted_item_moves_the_loop_forward():
    counts = dict(NONE_YET)
    seen = []
    for _ in range(sum(TYPICAL_COUNT.values()) + 1):
        result = suggest_next_item(ranked(*[(c, 1 / (i + 1)) for i, c in enumerate(CONTENTS)]), counts)
        if result["done"]:
            break
        counts[result["category"]] += 1
        seen.append(result["category"])
    assert result["done"] and counts == ALL_TYPICAL and len(seen) == sum(TYPICAL_COUNT.values())


def test_fallback_is_not_used_when_the_top_3_can_be_suggested():
    def must_not_be_called():
        raise AssertionError("the full ranking should not be read")

    counts = {**NONE_YET, "Electronics": 1}
    result = suggest_with_fallback(ranked(("Electronics", 0.5), ("Furniture", 0.3), ("Appliances", 0.2)), counts, must_not_be_called)
    assert result["category"] == "Furniture" and result["stage"] == "new_category"


def test_fallback_reads_the_full_ranking_when_none_of_the_top_3_can_be_suggested():
    counts = {c: 1 for c in CONTENTS}
    counts.update({"Clothings": 0, "Appliances": 0})
    top_3 = ranked(("Vehicles", 0.4), ("Jewellery and Personal Items", 0.3), ("Other Items", 0.2))
    full = top_3 + ranked(("Clothings", 0.06), ("Appliances", 0.04))
    result = suggest_with_fallback(top_3, counts, lambda: full)
    assert result["stage"] == "new_category" and result["category"] == "Clothings"


def test_no_fallback_once_every_category_has_an_item():
    def must_not_be_called():
        raise AssertionError("every category has an item, so stage 2 applies")

    result = suggest_with_fallback(ranked(("Electronics", 0.9)), {c: 1 for c in CONTENTS}, must_not_be_called)
    assert result["stage"] == "more_of_category"
