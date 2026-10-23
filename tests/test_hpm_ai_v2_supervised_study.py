import json
from pathlib import Path

import numpy as np
import pytest

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import (
    SourceDocument,
    StudyEpisode,
    SupervisedStudyTrainer,
    OllamaRetrievalBaseline,
    load_curriculum_jsonl,
)
from hpm_ai_v2.training.ollama import OllamaChatBaseline
from hpm_ai_v2.experiments.run_supervised_study import run as run_study
from hpm_ai_v2.domains.ollama_embedding import OllamaEmbeddingTextDomainConfig


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
        StudyEpisode("teach-a", "train", "grounded_qa", "shared symbol", "source-a", expected_answer="shared symbol appears in this reference passage"),
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
    with_facts = dict(trainer._rank("shared symbol"))
    learned_facts = trainer.fact_node_data
    trainer.fact_node_data = {}
    without_facts = dict(trainer._rank("shared symbol"))
    trainer.fact_node_data = learned_facts
    weights_before_test = {
        node: agent.observer.get_weight(node) for node in trainer.node_to_source
    }
    metrics = trainer.evaluate(test)
    weights_after_test = {
        node: agent.observer.get_weight(node) for node in trainer.node_to_source
    }

    assert fit_result["train_episodes"] == 3
    assert fit_result["validation_metrics"]["resubstitution"] is True
    assert fit_result["validation_metrics"]["unsupported_rejection_rate"] == 1.0
    assert fit_result["validation_metrics"]["false_abstention_rate"] == 0.0
    assert fit_result["knowledge_growth"]["new_fact_nodes"] == 1
    assert fit_result["knowledge_growth"]["source_support_edges_added"] == 2
    assert fit_result["knowledge_growth"]["promoted_cross_source_fact_nodes"] == 1
    assert with_facts["source-b"] > without_facts["source-b"]
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
    # The final source-a correction can penalize source-b after it was initially
    # reinforced; the cross-source fact promotion below is the retained-memory
    # assertion, while this check verifies test evaluation itself is read-only.
    fact_id = next(iter(trainer.fact_node_data))
    fact_node = agent.forest.get(fact_id)
    assert fact_node.relation_type == "study_fact"
    assert len(fact_node.children()) == 2
    assert {edge.relation for edge in fact_node.edges()} >= {"supported_by", "corrects"}
    assert trainer.fact_node_data[fact_id]["status"] == "promoted"

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


def test_fit_phase_reports_retention_without_training_on_heldout_suite(tmp_path):
    text = "The shared symbol appears in this reference passage."
    novel = "Neural signals travel between neurons across synapses."
    config = TextDomainConfig.from_passages([text, novel], max_vocab=100)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "phase-forest"))
    trainer = SupervisedStudyTrainer(agent)
    retained_source = SourceDocument("retained", novel)
    phase_source = SourceDocument("phase", text)
    retention = [
        StudyEpisode("retained-exam", "test", "grounded_qa", "signals between neurons", "retained", expected_answer=novel)
    ]
    phase_train = [
        StudyEpisode("phase-train", "train", "grounded_qa", "shared symbol", "phase", expected_answer=text)
    ]
    trainer.ingest([retained_source])
    retained_weight_before = agent.observer.get_weight(trainer.source_to_node["retained"])

    result = trainer.fit_phase(
        [retained_source, phase_source],
        phase_train,
        retention_episodes=retention,
        max_accuracy_drop=0.05,
    )

    assert result["retention_before"]["accuracy"] == 1.0
    assert result["retention_after"]["accuracy"] == 1.0
    assert result["retention_accuracy_delta"] == 0.0
    assert result["retention_gate_passed"] is True
    assert "retained-exam" not in trainer._fit_episode_ids
    assert agent.observer.get_weight(trainer.source_to_node["retained"]) == retained_weight_before


def test_retention_is_measured_after_each_cumulative_phase(tmp_path):
    retained_text = "Neural signals travel between neurons across synapses."
    phase_one_text = "Glass is produced by heating sand with soda ash and limestone."
    phase_two_text = "Bees use waggle dances to communicate food locations to hive mates."
    sources = [retained_text, phase_one_text, phase_two_text]
    config = TextDomainConfig.from_passages(sources, max_vocab=200)
    agent = ReaderAgent(
        config,
        dynamic_vocab=False,
        inject_blank_priors=False,
        cold_dir=str(tmp_path / "cumulative-forest"),
    )
    trainer = SupervisedStudyTrainer(agent)
    retention = [
        StudyEpisode(
            "retained-neural-exam", "test", "grounded_qa",
            "How do neural signals travel?", "retained", expected_answer=retained_text,
        )
    ]
    phase_one = trainer.fit_phase(
        [SourceDocument("retained", retained_text), SourceDocument("glass", phase_one_text)],
        [StudyEpisode("phase-one-example", "train", "grounded_qa", "How is glass produced?", "glass", expected_answer=phase_one_text)],
        retention_episodes=retention,
    )
    phase_two = trainer.fit_phase(
        [SourceDocument("bees", phase_two_text)],
        [StudyEpisode("phase-two-example", "train", "grounded_qa", "How do bees communicate food?", "bees", expected_answer=phase_two_text)],
        retention_episodes=retention,
    )

    assert phase_one["retention_before"]["accuracy"] == 1.0
    assert phase_one["retention_after"]["accuracy"] == 1.0
    assert phase_two["retention_before"]["accuracy"] == 1.0
    assert phase_two["retention_after"]["accuracy"] == 1.0
    assert phase_one["retention_gate_passed"] is True
    assert phase_two["retention_gate_passed"] is True
    assert "retained-neural-exam" not in trainer._fit_episode_ids


def test_fit_curriculum_reuses_knowledge_and_reports_each_retention_gate(tmp_path):
    retained_text = "Neural signals travel between neurons across synapses."
    phase_texts = [
        "Glass is produced by heating sand with soda ash and limestone.",
        "Bees use waggle dances to communicate food locations to hive mates.",
        "Bats locate insects by emitting ultrasonic calls and hearing echoes.",
    ]
    config = TextDomainConfig.from_passages([retained_text, *phase_texts], max_vocab=300)
    agent = ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "ordered-curriculum-forest"),
    )
    trainer = SupervisedStudyTrainer(agent)
    retention = [StudyEpisode(
        "curriculum-retention", "test", "grounded_qa",
        "How do neural signals travel?", "retained", expected_answer=retained_text,
    )]
    phases = [
        {
            "name": name,
            "sources": [SourceDocument(source_id, text)],
            "train_episodes": [StudyEpisode(
                f"{source_id}-teach", "train", "grounded_qa", question,
                source_id, expected_answer=text,
            )],
        }
        for name, source_id, text, question in [
            ("materials", "glass", phase_texts[0], "How is glass produced?"),
            ("communication", "bees", phase_texts[1], "How do bees communicate food?"),
            ("echolocation", "bats", phase_texts[2], "How do bats locate insects?"),
        ]
    ]
    phases[0]["sources"].insert(0, SourceDocument("retained", retained_text))

    report = trainer.fit_curriculum(phases, retention_episodes=retention)

    assert report["phase_count"] == 3
    assert [phase["name"] for phase in report["phases"]] == [
        "materials", "communication", "echolocation",
    ]
    assert report["all_retention_gates_passed"] is True
    assert [phase["fit"]["knowledge_growth"]["new_fact_nodes"] for phase in report["phases"]] == [1, 1, 1]
    assert report["total_fact_nodes"] == 3
    assert all(phase["retention_before"]["accuracy"] == 1.0 for phase in report["phases"])
    assert all(phase["retention_after"]["accuracy"] == 1.0 for phase in report["phases"])
    assert "curriculum-retention" not in trainer._fit_episode_ids


def test_runner_reports_separate_retention_curriculum(tmp_path, capsys):
    primary = tmp_path / "primary.jsonl"
    retention = tmp_path / "retention.jsonl"
    primary_rows = [
        {"kind": "source", "source_id": "water", "text": "Water freezes at 0 degrees Celsius."},
        {"kind": "episode", "episode_id": "train-water", "split": "train", "task_type": "grounded_qa", "question": "What temperature does water freeze at?", "evidence_id": "water", "expected_answer": "Water freezes at 0 degrees Celsius."},
        {"kind": "episode", "episode_id": "test-water", "split": "test", "task_type": "grounded_qa", "question": "At what temperature does water freeze?", "evidence_id": "water", "expected_answer": "Water freezes at 0 degrees Celsius."},
    ]
    retention_rows = [
        {"kind": "source", "source_id": "moon", "text": "The Moon orbits Earth approximately every 27 days."},
        {"kind": "episode", "episode_id": "retained-moon", "split": "test", "task_type": "grounded_qa", "question": "How often does the Moon orbit Earth?", "evidence_id": "moon", "expected_answer": "The Moon orbits Earth approximately every 27 days."},
    ]
    primary.write_text("\n".join(json.dumps(row) for row in primary_rows), encoding="utf-8")
    retention.write_text("\n".join(json.dumps(row) for row in retention_rows), encoding="utf-8")

    report = run_study(primary, tmp_path / "run", retention_curriculum=retention)
    capsys.readouterr()

    assert report["retention"]["before"]["accuracy"] == 1.0
    assert report["retention"]["after"]["accuracy"] == 1.0
    assert report["retention"]["accuracy_delta"] == 0.0
    assert report["retention"]["gate_passed"] is True
    assert report["retention"]["curriculum"] == str(retention)
    assert report["restart_consistency"]["same_predictions"] is True


def test_user_correction_transfers_after_restart_without_unrelated_regression(tmp_path):
    curriculum = Path(__file__).parents[1] / "hpm_ai_v2/training/curricula/correction_transfer_seed.jsonl"
    sources, episodes = load_curriculum_jsonl(curriculum)
    config = TextDomainConfig.from_passages([source.text for source in sources], max_vocab=200)
    agent = ReaderAgent(
        config,
        dynamic_vocab=False,
        inject_blank_priors=False,
        cold_dir=str(tmp_path / "correction-forest"),
    )
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    train = [episode for episode in episodes if episode.split == "train"]
    validation = [episode for episode in episodes if episode.split == "validation"]
    test = [episode for episode in episodes if episode.split == "test"]

    before = trainer.predict(test[0])
    assert before.selected_evidence_id == "visual"
    fit = trainer.fit([], train, validation)
    after = [trainer.predict(episode) for episode in test]

    assert after[0].selected_evidence_id == "waggle"
    assert after[1].selected_evidence_id == "bats"
    assert after[2].abstained
    assert fit["knowledge_growth"]["correction_links_added"] == 1
    fact_id = next(iter(trainer.fact_node_data))
    fact = trainer.fact_node_data[fact_id]
    assert fact["status"] == "validated"
    assert fact["correction_episode_ids"] == ["correction-bee"]
    assert fact["corrects_node_ids"] == [trainer.source_to_node["visual"]]
    fact_node = agent.forest.get(fact_id)
    assert any(edge.relation == "corrects" for edge in fact_node.edges())

    trainer.save_checkpoint(tmp_path / "correction-checkpoint")
    restored_agent = ReaderAgent.load_agent(str(tmp_path / "correction-checkpoint"))
    restored = SupervisedStudyTrainer(restored_agent)
    restored.load_checkpoint_state(tmp_path / "correction-checkpoint")
    restored_predictions = [restored.predict(episode) for episode in test]
    assert [p.selected_evidence_id for p in restored_predictions] == [
        p.selected_evidence_id for p in after
    ]
    assert restored_predictions[0].citation == after[0].citation


def test_embedding_encoder_transfers_and_restores_cached_vectors(tmp_path):
    class StubEmbedder:
        model = "test-embedding"
        base_url = "http://local"
        timeout = 1.0

        def embed_many(self, texts):
            vectors = []
            for text in texts:
                normalized = text.casefold()
                if "water" in normalized or "temperature" in normalized:
                    vectors.append(np.array([1.0, 0.1, 0.0], dtype="float32"))
                else:
                    vectors.append(np.array([0.0, 1.0, 0.0], dtype="float32"))
            return vectors

    config = OllamaEmbeddingTextDomainConfig(
        client=StubEmbedder(), dimension=3
    )
    source = SourceDocument("water", "Water freezes at zero degrees Celsius.")
    train = StudyEpisode(
        "water-train", "train", "grounded_qa", "What temperature freezes water?",
        "water", expected_answer="Water freezes at zero degrees Celsius.",
    )
    test = StudyEpisode(
        "water-test", "test", "grounded_qa", "At what temperature does water freeze?",
        "water", expected_answer="Water freezes at zero degrees Celsius.",
    )
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "embedding-forest"))
    trainer = SupervisedStudyTrainer(agent)
    config.precompute([source.text, train.question, test.question])
    trainer.fit([source], [train])
    assert trainer.evaluate([test])["accuracy"] == 1.0

    trainer.save_checkpoint(tmp_path / "embedding-checkpoint")
    restored_agent = ReaderAgent.load_agent(str(tmp_path / "embedding-checkpoint"))
    restored = SupervisedStudyTrainer(restored_agent)
    restored.load_checkpoint_state(tmp_path / "embedding-checkpoint")
    assert restored.predict(test).selected_evidence_id == "water"
    assert restored_agent.dynamic_vocab is False


def test_semantic_retrieval_uses_lexical_corroboration_for_close_distractors(tmp_path):
    query = "How does a waggle dance communicate food distance to hive mates?"
    unsupported = "How do dolphins locate caves with magnets?"

    class CloseSemanticDistractor:
        model = "test-embedding"
        base_url = "http://local"
        timeout = 1.0

        def embed_many(self, texts):
            result = []
            for text in texts:
                if text == query:
                    vector = np.array([1.0, 0.0, 0.0], dtype="float32")
                elif text == unsupported:
                    vector = np.array([0.0, 1.0, 0.0], dtype="float32")
                elif "distractor" in text:
                    vector = np.array([1.0, 0.0, 0.0], dtype="float32")
                else:
                    vector = np.array([0.995, 0.1, 0.0], dtype="float32")
                result.append(vector / np.linalg.norm(vector))
            return result

    config = OllamaEmbeddingTextDomainConfig(client=CloseSemanticDistractor(), dimension=3)
    relevant = SourceDocument(
        "relevant", "A waggle dance communicates food distance to hive mates."
    )
    distractor = SourceDocument(
        "distractor", "A semantically similar source about insects returning to colonies."
    )
    config.precompute([query, unsupported, relevant.text, distractor.text])
    trainer = SupervisedStudyTrainer(ReaderAgent(
        config, dynamic_vocab=False, cold_dir=str(tmp_path / "hybrid-retrieval-forest")
    ), semantic_lexical_rescue_margin=0.01)
    trainer.ingest([relevant, distractor])

    trainer.semantic_lexical_bonus = 0.0
    semantic_only = trainer._rank(query)
    trainer.semantic_lexical_bonus = 0.05
    hybrid = trainer._rank(query)
    trainer.abstain_threshold = hybrid[0][1] + 0.005
    rescued = trainer.predict_question(query)
    rejected = trainer.predict_question(unsupported)

    assert semantic_only[0][0] == "distractor"
    assert hybrid[0][0] == "relevant"
    assert rescued.selected_evidence_id == "relevant"
    assert rejected.abstained


def test_supervised_abstention_pattern_transfers_without_global_source_penalty(tmp_path):
    negative_train = "Do honeybees communicate food warmth with the waggle dance?"
    negative_test = "Does the waggle dance tell hive mates the temperature of distant food?"
    supported_test = "What aspect of movement gives nestmates the distance to food?"
    source = SourceDocument("bees", "Honeybees use a waggle dance to signal food direction and distance.")

    class ScopedAbstentionEncoder:
        model = "test-embedding"
        base_url = "http://local"
        timeout = 1.0

        def embed_many(self, texts):
            vectors = {
                negative_train: np.array([1.0, 0.0, 0.0], dtype="float32"),
                negative_test: np.array([0.85, 0.5267827, 0.0], dtype="float32"),
                supported_test: np.array([0.65, 0.7599342, 0.0], dtype="float32"),
                source.text: np.array([0.0, 1.0, 0.0], dtype="float32"),
            }
            return [
                (vectors.get(text, np.array([0.0, 1.0, 0.0], dtype="float32")) /
                 np.linalg.norm(vectors.get(text, np.array([0.0, 1.0, 0.0], dtype="float32"))))
                for text in texts
            ]

    config = OllamaEmbeddingTextDomainConfig(client=ScopedAbstentionEncoder(), dimension=3)
    config.precompute([negative_train, negative_test, supported_test, source.text])
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "negative-pattern-forest"))
    trainer = SupervisedStudyTrainer(agent, abstention_similarity_threshold=0.8)
    source_weight_before = None
    train = StudyEpisode(
        "train-negative", "train", "abstain", negative_train, None, user_id="student-a"
    )
    trainer.ingest([source])
    source_weight_before = agent.observer.get_weight(trainer.source_to_node["bees"])
    fit = trainer.fit([], [train])
    negative = trainer.predict(StudyEpisode(
        "negative-transfer", "test", "abstain", negative_test, None, user_id="student-a"
    ))
    supported = trainer.predict(StudyEpisode(
        "supported-transfer", "test", "grounded_qa", supported_test, "bees", user_id="student-a"
    ))
    other_user = trainer.predict(StudyEpisode(
        "other-user", "test", "grounded_qa", negative_test, "bees", user_id="student-b"
    ))

    assert fit["knowledge_growth"]["new_abstention_patterns"] == 1
    assert negative.abstained
    assert negative.abstention_reason == "learned_negative_pattern"
    assert supported.selected_evidence_id == "bees"
    assert other_user.selected_evidence_id == "bees"
    assert agent.observer.get_weight(trainer.source_to_node["bees"]) == source_weight_before

    trainer.save_checkpoint(tmp_path / "negative-pattern-checkpoint")
    restored_agent = ReaderAgent.load_agent(str(tmp_path / "negative-pattern-checkpoint"))
    restored = SupervisedStudyTrainer(restored_agent)
    restored.load_checkpoint_state(tmp_path / "negative-pattern-checkpoint")
    restored_negative = restored.predict(StudyEpisode(
        "negative-transfer-restored", "test", "abstain", negative_test, None,
        user_id="student-a",
    ))
    assert restored_negative.abstention_reason == "learned_negative_pattern"


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
    assert captured["payload"]["options"] == {"temperature": 0.0, "num_predict": 256}
    assert baseline.configuration()["prompt_version"] == "grounded-study-v2"


def test_ollama_runtime_metadata_records_digest(monkeypatch):
    baseline = OllamaChatBaseline("qwen3.5:latest")

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"models": [{
                "name": "qwen3.5:latest",
                "digest": "sha256:abc123",
                "size": 6600000000,
                "details": {"parameter_size": "9.7B", "quantization_level": "Q4_K_M"},
            }]}).encode("utf-8")

    monkeypatch.setattr("hpm_ai_v2.training.ollama.urlopen", lambda *_args, **_kwargs: Response())
    metadata = baseline.runtime_metadata()

    assert metadata["status"] == "found"
    assert metadata["digest"] == "sha256:abc123"
    assert metadata["details"]["parameter_size"] == "9.7B"


def test_ollama_retrieval_baseline_limits_context_and_records_sources(monkeypatch):
    baseline = OllamaRetrievalBaseline("qwen3.5:latest", top_k=1)
    sources = [
        SourceDocument("distractor", "Mars has two small moons named Phobos and Deimos."),
        SourceDocument("answer", "Water freezes at zero degrees Celsius."),
    ]
    captured = {}

    def fake_answer(question, selected_sources):
        captured["source_ids"] = [source.source_id for source in selected_sources]
        from hpm_ai_v2.training.ollama import ModelAnswer
        return ModelAnswer("Water freezes at zero degrees Celsius.", "answer", False, 1.0)

    monkeypatch.setattr(baseline.chat, "answer", fake_answer)
    monkeypatch.setattr(baseline, "runtime_metadata", lambda: {"status": "test"})
    result = baseline.answer("At what temperature does water freeze?", sources)

    assert captured["source_ids"] == ["answer"]
    assert result.retrieved_source_ids == ("answer",)
    assert baseline.configuration()["retrieval"] == {
        "method": "BM25", "top_k": 1, "k1": 1.5, "b": 0.75, "minimum_score": 0.0
    }
    from hpm_ai_v2.experiments.run_supervised_study import _evaluate_ollama
    metrics = _evaluate_ollama(
        baseline,
        sources,
        [StudyEpisode("rag-eval", "test", "grounded_qa", "At what temperature does water freeze?", "answer", expected_answer="Water freezes at zero degrees Celsius.")],
    )
    assert metrics["retrieval_diagnostics"]["answerable_source_recall_at_k"] == 1.0
    assert metrics["retrieval_diagnostics"]["mean_retrieved_sources_per_episode"] == 1.0
