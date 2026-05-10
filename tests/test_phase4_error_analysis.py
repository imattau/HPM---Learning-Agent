from hpm_ai_v5.experiments.intent_error_analysis import analyze_intent_errors, classify_error


def test_classify_error_detects_conflict_and_slot_phrase():
    record = classify_error(
        "find a flight from boston to denver",
        "flight",
        "airfare",
        {
            "intent_votes": {"flight": 0.9, "airfare": 1.0},
            "intent_vote_trace": [
                {"source": "primary", "intent": "flight", "contribution": 0.9},
                {"source": "content_view", "intent": "airfare", "contribution": 1.0},
            ],
        },
    )

    assert record.top_source == "content_view"
    assert record.source_conflict is True
    assert record.slot_sensitive is True
    assert "low_margin" in record.categories
    assert "source_conflict" in record.categories
    assert "slot_sensitive_phrase" in record.categories


def test_analyze_intent_errors_counts_categories():
    data = [
        {"text": "find a flight from boston to denver", "intent": "flight"},
        {"text": "play jazz", "intent": "play_music"},
    ]

    outputs = {
        "find a flight from boston to denver": (
            "airfare",
            {
                "intent_votes": {"flight": 0.8, "airfare": 0.9},
                "intent_vote_trace": [
                    {"source": "primary", "intent": "flight", "contribution": 0.8},
                    {"source": "content_view", "intent": "airfare", "contribution": 0.9},
                ],
            },
        ),
        "play jazz": ("play_music", {"intent_votes": {"play_music": 1.0}, "intent_vote_trace": []}),
    }

    analysis = analyze_intent_errors(data, lambda text: outputs[text])

    assert analysis["n_errors"] == 1
    assert analysis["category_counts"]["dominant_source:content_view"] == 1
    assert analysis["category_counts"]["source_conflict"] == 1
    assert analysis["top_confusions"][0]["gold"] == "flight"
