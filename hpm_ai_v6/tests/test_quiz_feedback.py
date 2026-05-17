from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_learn_from_quiz_attempt_reinforces_semantic_word_and_exact_memory():
    from hpm_ai_v6.quiz_feedback import learn_from_quiz_attempt, quiz_memory_name

    reader = MagicMock()
    reader.train_sequence = MagicMock()
    reader.flush_all = MagicMock()
    reader.reasoning_agent = MagicMock()

    semantic_pager = MagicMock()
    word_pager = MagicMock()
    phrase_pager = MagicMock()

    semantic_agent = SimpleNamespace(pattern_pager=semantic_pager, patterns=[])
    word_agent = SimpleNamespace(pattern_pager=word_pager, patterns=[])
    phrase_agent = SimpleNamespace(pattern_pager=phrase_pager, patterns=[])

    reader.semantic_agent = semantic_agent
    reader.word_agent = word_agent
    reader.phrase_agent = phrase_agent
    reader.agents = {
        "semantic": semantic_agent,
        "word": word_agent,
        "phrase": phrase_agent,
    }

    payload = {
        "question": "What is the capital of France?",
        "topic": "geography",
        "chosen": "C",
        "chosen_text": "Berlin",
        "correct": "B",
        "correct_text": "Paris",
        "options": {
            "A": "Madrid",
            "B": "Paris",
            "C": "Berlin",
            "D": "Rome",
        },
        "is_correct": False,
        "trace": {
            "chosen_path": {
                "nodes": [{"label": "capital"}, {"label": "paris"}],
                "steps": [
                    {
                        "source_label": "capital",
                        "target_label": "paris",
                        "relation": "lexical_transition",
                    }
                ],
            },
            "terms": ["capital", "france"],
            "answer": "B",
        },
    }

    result = learn_from_quiz_attempt(reader, payload)

    sentences = reader.train_sequence.call_args[0][0]
    assert len(sentences) == 2
    assert "Question: What is the capital of France?" in sentences[0]
    assert "Options: A) Madrid." in sentences[0]
    assert "Chosen answer: C) Berlin." in sentences[0]
    assert "Correct answer: B) Paris." in sentences[0]
    assert "Correct answer: B) Paris." in sentences[1]
    assert "Chosen answer: C) Berlin." in sentences[1]

    saved_names = [
        getattr(call.args[0], "name", "")
        for call in word_pager.enqueue_save.call_args_list
    ]
    assert quiz_memory_name(payload["question"], "B") in saved_names
    assert any(name.startswith("memory::fact::capital_france=>paris") for name in saved_names)
    assert any(name.startswith("memory::path::capital->paris") for name in saved_names)
    assert any(name.startswith("memory::path_step::capital->paris::lexical_transition") for name in saved_names)
    assert "w_capital->paris" in saved_names
    assert "w_france->paris" in saved_names
    assert reader.reasoning_agent.invalidate.called
    assert reader.flush_all.called
    assert result["trained"] == 1
    assert result["general"] >= 3


def test_cli_main_uses_shared_quiz_feedback_helper():
    from argparse import Namespace
    from hpm_ai_v6.cli import quiz_cli

    mock_reader = MagicMock()
    mock_reader.reasoning_agent = MagicMock()
    mock_reader.flush_all = MagicMock()

    mock_quiz_agent = MagicMock()

    def fake_run_quiz(*args, **kwargs):
        kwargs["feedback_hook"](
            {
                "question": "What is the capital of France?",
                "topic": "geography",
                "chosen": "C",
                "chosen_text": "Berlin",
                "correct": "B",
                "correct_text": "Paris",
                "options": {
                    "A": "Madrid",
                    "B": "Paris",
                    "C": "Berlin",
                    "D": "Rome",
                },
                "is_correct": False,
                "confident": False,
            }
        )
        return (0, [], set())

    with patch.object(quiz_cli, "parse_args", return_value=Namespace(source="bank", difficulty="easy", n=1, auto=True, loop=False)), \
         patch.object(quiz_cli, "_corpus_path", return_value="/tmp/fake_corpus.txt"), \
         patch("hpm_ai_v6.agents.multi_agent_reader.MultiAgentReader", return_value=mock_reader), \
         patch("hpm_ai_v6.agents.quiz_agent.QuizAgent", return_value=mock_quiz_agent), \
         patch.object(quiz_cli, "run_quiz", side_effect=fake_run_quiz), \
         patch("hpm_ai_v6.cli.quiz_cli.learn_from_quiz_attempt") as mock_feedback:
        quiz_cli.main([])

    mock_feedback.assert_called_once()
    assert mock_feedback.call_args.args[0] is mock_reader
    assert mock_reader.flush_all.called


def test_run_quiz_reads_question_before_reasoning():
    from hpm_ai_v6.cli.quiz_cli import run_quiz

    mock_reader = MagicMock()
    mock_reader.train_sequence = MagicMock()
    mock_reader.flush_all = MagicMock()
    mock_reader.agents = {}
    mock_quiz_agent = MagicMock()
    mock_reasoning_agent = MagicMock()

    question = MagicMock()
    question.id = "q1"
    question.question = "What is the capital of France?"
    question.options = ["Paris", "London", "Berlin", "Madrid"]
    question.correct_index = 0
    question.topic = "geography"
    mock_quiz_agent.generate_quiz.return_value = [question]

    call_order = []

    def _read_side_effect(reader, payload):
        call_order.append("read")
        return {"trained": 1, "saved": 0}

    def _reason_side_effect(question_text, method="auto"):
        call_order.append("reason")
        return {
            "answer": "A",
            "candidate_paths": [],
            "chosen_path": None,
            "evidence": [],
            "intent": "path",
            "mode": "connection",
            "method": method,
        }

    with patch("hpm_ai_v6.cli.quiz_cli.learn_from_quiz_question", side_effect=_read_side_effect):
        mock_reasoning_agent.reason_with_trace.side_effect = _reason_side_effect
        run_quiz(
            reader=mock_reader,
            quiz_agent=mock_quiz_agent,
            reasoning_agent=mock_reasoning_agent,
            n=1,
            difficulty="easy",
            source="bank",
            auto=True,
        )

    assert call_order == ["read", "reason"]


def test_web_submit_uses_shared_quiz_feedback_helper():
    from hpm_ai_v6.web import web_demo

    original_reader = web_demo.reader
    original_state = dict(web_demo._quiz_state)
    try:
        web_demo.reader = SimpleNamespace()
        web_demo._quiz_state = {
            "q1": SimpleNamespace(
                id="q1",
                question="What is the capital of France?",
                options=["Paris", "London", "Berlin", "Madrid"],
                correct_index=0,
                topic="geography",
                explanation="Paris is the capital of France.",
            )
        }

        with patch("hpm_ai_v6.web.web_demo.learn_from_quiz_attempt") as mock_feedback:
            client = web_demo.app.test_client()
            response = client.post("/api/quiz/submit", json={"question_id": "q1", "answer_index": 2})

        assert response.status_code == 200
        mock_feedback.assert_called_once()
        payload = mock_feedback.call_args.args[1]
        assert payload["chosen"] == "C"
        assert payload["chosen_text"] == "Berlin"
        assert payload["correct"] == "A"
        assert payload["correct_text"] == "Paris"
    finally:
        web_demo.reader = original_reader
        web_demo._quiz_state = original_state
