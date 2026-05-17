from hpm_ai_v6.quiz_bank_builder import _arc_example_to_item, _mmlu_example_to_item, _merge_unique


def test_arc_example_to_item_maps_answer_key():
    example = {
        "question": "What is the capital of France?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["London", "Paris", "Berlin", "Rome"],
        },
        "answerKey": "B",
    }

    item = _arc_example_to_item(example, "easy", "arc_easy")

    assert item is not None
    assert item["correct_index"] == 1
    assert item["source"] == "arc_easy"
    assert item["difficulty"] == "easy"


def test_mmlu_example_to_item_maps_integer_answer():
    example = {
        "question": "Which option is correct?",
        "choices": ["A1", "A2", "A3", "A4"],
        "answer": 2,
    }

    item = _mmlu_example_to_item(example, "abstract_algebra")

    assert item is not None
    assert item["correct_index"] == 2
    assert item["topic"] == "abstract algebra"
    assert item["source"] == "mmlu:abstract_algebra"


def test_merge_unique_skips_duplicate_questions():
    existing = [
        {
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "correct_index": 0,
        }
    ]
    additions = [
        {
            "question": "Q?",
            "options": ["A", "B", "C", "D"],
            "correct_index": 0,
        },
        {
            "question": "R?",
            "options": ["W", "X", "Y", "Z"],
            "correct_index": 3,
        },
    ]

    merged = _merge_unique(existing, additions)

    assert len(merged) == 2
    assert merged[-1]["question"] == "R?"
