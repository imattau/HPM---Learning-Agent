from hpm_ai_v6.web.web_demo import _update_gutenberg_cycle_state, gutenberg_cycle_state


def test_maintenance_done_updates_gutenberg_cycle_status_message():
    original = dict(gutenberg_cycle_state)
    try:
        gutenberg_cycle_state.update(
            {
                "active": True,
                "message": "Running.",
                "book_id": 11,
                "chapter": None,
                "phase": "book_done",
                "books_processed": 2,
                "chapter_added": 0,
            }
        )

        _update_gutenberg_cycle_state(
            {
                "type": "maintenance_done",
                "book_id": 11,
                "report": {
                    "_summary": {
                        "loaded": 17,
                        "improved_agents": ["word", "contextual"],
                    }
                },
            }
        )

        assert "book 11" in gutenberg_cycle_state["message"].lower()
        assert "loaded 17 patterns" in gutenberg_cycle_state["message"].lower()
        assert "word" in gutenberg_cycle_state["message"].lower()
    finally:
        gutenberg_cycle_state.clear()
        gutenberg_cycle_state.update(original)
