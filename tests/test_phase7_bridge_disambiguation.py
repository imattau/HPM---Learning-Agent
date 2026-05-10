from hpm_ai_v5.adapter.atis import ATISBridgeAnchorAdapter
from hpm_ai_v5.adapter.snips import SNIPSBridgeAnchorAdapter
from hpm_ai_v5.adapter.packet import AdapterPacket


def test_atis_bridge_adapter_emits_ground_and_day_name_concepts():
    adapter = ATISBridgeAnchorAdapter()
    packet = AdapterPacket(
        raw="what day of the week do flights from nashville to tacoma fly on",
        context={
            "tokens": ["what", "day", "of", "the", "week", "do", "flights", "from", "GPE", "to", "GPE", "fly", "on"],
            "canonical_tokens": ["what", "day", "of", "the", "week", "do", "FLIGHT", "from", "GPE", "to", "GPE", "fly", "on"],
            "ent_types": ["", "", "", "", "", "", "", "", "GPE", "", "GPE", "", ""],
            "content_words": ["flights"],
            "skeleton": ["P", "N", "R", "D", "N", "V", "N", "R", "N", "R", "N", "V", "R"],
            "skeleton_ngrams": ["N_R", "R_N"],
        },
        states=[],
    )
    result = adapter.run(packet)
    concept_ids = {entry["concept_id"] for entry in result.context["atis_concepts"]}
    anchor_ids = set(result.context["atis_view_anchor_map"]["canonical_view"]["anchor_ids"])

    assert "concept::day_name_query" in concept_ids
    assert "concept::from_city" in concept_ids
    assert "concept::to_city" in concept_ids
    assert "route::day_name" in anchor_ids


def test_snips_bridge_adapter_emits_screening_and_restaurant_routes():
    adapter = SNIPSBridgeAnchorAdapter()
    packet = AdapterPacket(
        raw="book a restaurant table and find a screening tonight",
        context={
            "tokens": ["book", "a", "restaurant", "table", "and", "find", "a", "screening", "tonight"],
            "canonical_tokens": ["BOOK", "a", "RESTAURANT", "RESTAURANT", "and", "find", "a", "SCREENING", "tonight"],
            "ent_types": ["", "", "", "", "", "", "", "", "TIME"],
            "content_words": ["restaurant", "screening"],
            "skeleton": ["V", "D", "N", "N", "C", "V", "D", "N", "N"],
            "skeleton_ngrams": ["V_D", "D_N", "N_N"],
        },
        states=[],
    )
    result = adapter.run(packet)
    concept_ids = {entry["concept_id"] for entry in result.context["atis_concepts"]}
    anchor_ids = set(result.context["atis_view_anchor_map"]["canonical_view"]["anchor_ids"])

    assert "concept::restaurant_query" in concept_ids
    assert "concept::screening_query" in concept_ids
    assert "route::book_restaurant" in anchor_ids
    assert "route::search_screening_event" in anchor_ids
