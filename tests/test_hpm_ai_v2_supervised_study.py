import json

import pytest

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import (
    SourceDocument,
    StudyEpisode,
    SupervisedStudyTrainer,
    load_curriculum_jsonl,
)
from hpm_ai_v2.training.ollama import OllamaChatBaseline


def _agent(tmp_path):
    repeated = "The shared symbol appears in this reference passage."
    config = TextDomainConfig.from_passages([repeated, repeated], max_vocab=100)
    return ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest"))


def test_supervision_reinforces_correct_evidence_and_test_is_read_only(tmp_path):
    agent = _agent(tmp_path)
    trainer = SupervisedStudyTrainer(agent)
    sources = [
        SourceDocument("source-a", "The shared symbol appears in this reference passage."),
        SourceDocument("source-b", "The shared symbol appears in this reference passage."),
    ]
    train = [
        StudyEpisode("teach-b", "train", "correction", "shared symbol", "source-b", expected_answer="shared symbol appears in this reference passage", correction_of="old-answer"),
        StudyEpisode("teach-b-again", "train", "grounded_qa", "shared symbol", "source-b", expected_answer="shared symbol appears in this reference passage"),
    ]
    validation = [
        StudyEpisode("known", "validation", "grounded_qa", "shared symbol", "source-b", expected_answer="shared symbol appears in this reference passage"),
        StudyEpisode("unknown", "validation", "abstain", "quasar thermodynamics", None),
    ]
    test = [
        StudyEpisode("held-out-known", "test", "grounded_qa", "shared symbol", "source-b", expected_answer="shared symbol appears in this reference passage"),
        StudyEpisode("held-out-unknown", "test", "abstain", "quasar thermodynamics", None),
    ]

    fit_result = trainer.fit(sources, train, validation)
    weights_before_test = {
        node: agent.observer.get_weight(node) for node in trainer.node_to_source
    }
    metrics = trainer.evaluate(test)
    weights_after_test = {
        node: agent.observer.get_weight(node) for node in trainer.node_to_source
    }

    assert fit_result["train_episodes"] == 2
    assert trainer.predict(test[0]).selected_evidence_id == "source-b"
    assert metrics["accuracy"] == 1.0
    assert metrics["evidence_accuracy"] == 1.0
    assert 0.0 < metrics["accuracy_ci95"][0] < 1.0
    assert metrics["unsupported_rejection_rate"] == 1.0
    assert metrics["false_abstention_rate"] == 0.0
    assert metrics["median_latency_ms"] >= 0.0
    assert metrics["answer_token_f1"] == 1.0
    assert metrics["episode_results"][0]["citation"]["source_id"] == "source-b"
    assert metrics["episode_results"][0]["citation"]["text"] == sources[1].text
    assert weights_after_test == weights_before_test
    assert agent.observer.get_weight(trainer.source_to_node["source-b"]) > 0.5

    trainer.save_checkpoint(tmp_path / "checkpoint")
    saved_weight = agent.observer.get_weight(trainer.source_to_node["source-b"])
    restored_agent = ReaderAgent.load_agent(str(tmp_path / "checkpoint"))
    restored = SupervisedStudyTrainer(restored_agent)
    restored.load_checkpoint_state(tmp_path / "checkpoint")
    assert restored_agent.observer.get_weight(restored.source_to_node["source-b"]) == saved_weight
    assert restored.predict(test[0]).selected_evidence_id == "source-b"
    restored_prediction = restored.predict(test[0])
    assert restored_prediction.answer == sources[1].text
    assert restored_prediction.citation["start"] == 0
    assert restored_prediction.citation["end"] == len(sources[1].text)


def test_fit_rejects_test_data_and_unknown_evidence(tmp_path):
    trainer = SupervisedStudyTrainer(_agent(tmp_path))
    source = SourceDocument("s1", "A source about memory consolidation.")
    test_episode = StudyEpisode("test", "test", "grounded_qa", "memory", "s1")
    with pytest.raises(ValueError, match="expected train"):
        trainer.fit([source], [test_episode])

    train_episode = StudyEpisode("train", "train", "grounded_qa", "memory", "missing")
    with pytest.raises(ValueError, match="unknown evidence"):
        trainer.fit([source], [train_episode])


def test_fit_prevents_episode_leakage_between_splits(tmp_path):
    trainer = SupervisedStudyTrainer(_agent(tmp_path))
    source = SourceDocument("s1", "A source about memory consolidation.")
    train = StudyEpisode("same-id", "train", "grounded_qa", "memory consolidation", "s1")
    validation = StudyEpisode("same-id", "validation", "grounded_qa", "memory consolidation", "s1")
    with pytest.raises(ValueError, match="reused across training/validation"):
        trainer.fit([source], [train], [validation])


def test_load_curriculum_jsonl(tmp_path):
    path = tmp_path / "curriculum.jsonl"
    rows = [
        {"kind": "source", "source_id": "s1", "text": "Memory can be consolidated during sleep."},
        {"kind": "episode", "episode_id": "q1", "split": "train", "task_type": "grounded_qa", "question": "What happens during sleep?", "evidence_id": "s1", "expected_answer": "Memory consolidation."},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    sources, episodes = load_curriculum_jsonl(path)
    assert sources == [SourceDocument("s1", "Memory can be consolidated during sleep.")]
    assert episodes[0].expected_answer == "Memory consolidation."


def test_ollama_baseline_records_fixed_generation_settings(monkeypatch):
    baseline = OllamaChatBaseline("qwen3.5:latest", temperature=0.0)
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"message":{"content":"{\\"answer\\":\\"A fact.\\",\\"evidence_id\\":\\"s1\\",\\"abstained\\":false}"},"prompt_eval_count":12,"eval_count":6}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    # The adapter imports urlopen directly, so patch its bound module symbol.
    monkeypatch.setattr("hpm_ai_v2.training.ollama.urlopen", fake_urlopen)
    answer = baseline.answer(
        "What is stated?", [SourceDocument("s1", "A fact.", "Source")]
    )

    assert answer.answer == "A fact."
    assert answer.evidence_id == "s1"
    assert not answer.abstained
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"] == {"temperature": 0.0, "num_predict": 160}
    assert baseline.configuration()["prompt_version"] == "grounded-study-v1"
