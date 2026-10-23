import json
from pathlib import Path
from types import SimpleNamespace

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
from hpm_ai_v2.experiments import run_supervised_study
from hpm_ai_v2.domains.ollama_embedding import OllamaEmbeddingTextDomainConfig
from hfn.tiered_forest import TieredForest


def test_checkpoint_restore_bulk_registers_before_rebuilding_hierarchy(tmp_path, monkeypatch):
    source = SourceDocument(
        "biology",
        "Neurons communicate across synapses. The hippocampus supports memory formation.",
        "Biology",
    )
    trainer = SupervisedStudyTrainer(_agent(tmp_path, inject_blank_priors=False))
    trainer.fit([source], [StudyEpisode(
        "checkpoint-bulk-train", "train", "grounded_qa",
        "How do neurons communicate?", "biology",
        expected_answer="Neurons communicate across synapses.",
    )])
    checkpoint = trainer.save_checkpoint(tmp_path / "bulk-restore-checkpoint")

    def unexpected_parent_scan(self, node_id):
        raise AssertionError("checkpoint restore must not scan all parents per node")

    monkeypatch.setattr(TieredForest, "get_parents", unexpected_parent_scan)
    checkpoint_dir = checkpoint.parent
    restored_agent = ReaderAgent.load_agent(str(checkpoint_dir))
    restored = SupervisedStudyTrainer(restored_agent)
    restored.load_checkpoint_state(checkpoint_dir)

    assert len(restored_agent.forest) >= 3
    assert restored.fact_node_data


def test_fit_reports_when_validation_cannot_validate_trained_facts(tmp_path):
    source = SourceDocument(
        "biology",
        "Neurons communicate across synapses. The hippocampus supports memory formation.",
        "Biology",
    )
    trainer = SupervisedStudyTrainer(_agent(tmp_path))
    result = trainer.fit(
        [source],
        [StudyEpisode(
            "coverage-train", "train", "grounded_qa", "How do neurons communicate?",
            "biology", expected_answer="Neurons communicate across synapses.",
        )],
        [StudyEpisode(
            "coverage-validation", "validation", "grounded_qa", "What supports memory?",
            "biology", expected_answer="The hippocampus supports memory formation.",
        )],
    )

    assert result["fact_validation_coverage"] == {
        "training_fact_count": 1,
        "matched_validation_fact_count": 0,
        "fraction": 0.0,
    }
    assert result["fact_validation_gate"] == {
        "training_fact_count": 1,
        "matched_validation_fact_count": 0,
        "validated_training_fact_count": 0,
        "passed": False,
    }
    assert result["knowledge_growth"]["validated_fact_nodes"] == 0


def test_fit_passes_fact_validation_gate_for_matching_heldout_paraphrase(tmp_path):
    source = SourceDocument(
        "biology", "Xylem transports water and dissolved minerals from roots through a plant.",
        "Plant Transport",
    )
    config = TextDomainConfig.from_passages([source.text, source.text], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False, cold_dir=str(tmp_path)
    ))
    result = trainer.fit(
        [source],
        [StudyEpisode(
            "fact-gate-train", "train", "grounded_qa", "What does xylem transport?",
            "biology", expected_answer=source.text,
        )],
        [StudyEpisode(
            "fact-gate-validation", "validation", "grounded_qa",
            "Which vascular tissue moves water and minerals from roots?",
            "biology", expected_answer=source.text,
        )],
    )

    assert result["fact_validation_gate"] == {
        "training_fact_count": 1,
        "matched_validation_fact_count": 1,
        "validated_training_fact_count": 1,
        "passed": True,
    }


def test_validated_concept_does_not_overwrite_adjacent_fact_retrieval(tmp_path):
    curriculum = Path(__file__).parents[1] / (
        "hpm_ai_v2/training/curricula/concept_transfer_confirmation_fact_validation_45_seed.jsonl"
    )
    sources, episodes = load_curriculum_jsonl(curriculum)
    config = TextDomainConfig.from_passages([source.text for source in sources], max_vocab=20_000)
    agent = ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "concept-collateral-forest"),
    )
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    train = [episode for episode in episodes if episode.split == "train"]
    validation = [episode for episode in episodes if episode.split == "validation"]
    tests = [episode for episode in episodes if episode.split == "test"]
    latent = next(episode for episode in tests if episode.episode_id == "ct2-heat-test-04-03")
    conduction = next(episode for episode in tests if episode.episode_id == "ct2-heat-test-01-01")
    before = trainer.evaluate(tests)
    assert trainer.predict(latent).selected_evidence_id == "ct2_heat_latent"

    fit = trainer.fit(sources, train, validation)

    after = trainer.evaluate(tests)
    assert fit["fact_validation_gate"]["passed"] is True
    assert fit["knowledge_growth"]["validated_fact_nodes"] == len(train)
    assert trainer.predict(latent).selected_evidence_id == "ct2_heat_latent"
    assert trainer.predict(conduction).selected_evidence_id == "ct2_heat_conduction"
    assert after["accuracy"] >= before["accuracy"]


def _agent(tmp_path, *, inject_blank_priors=True):
    repeated = "The shared symbol appears in this reference passage."
    config = TextDomainConfig.from_passages([repeated, repeated], max_vocab=100)
    return ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=inject_blank_priors,
        cold_dir=str(tmp_path / "forest"),
    )


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


def test_grounded_answer_accuracy_requires_answer_match_not_only_source_match(tmp_path):
    source = SourceDocument(
        "biology",
        "Neurons communicate across synapses. The hippocampus supports memory formation.",
        "Biology",
    )
    config = TextDomainConfig.from_passages([source.text], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "answer-quality-forest"),
    ))
    trainer.fit([source], [StudyEpisode(
        "answer-quality-train", "train", "grounded_qa",
        "How do neurons communicate?", "biology",
        expected_answer="Neurons communicate across synapses.",
    )])
    metrics = trainer.evaluate([StudyEpisode(
        "answer-quality-test", "test", "grounded_qa",
        "How do neurons communicate?", "biology",
        expected_answer="The hippocampus supports memory formation.",
    )])

    assert metrics["accuracy"] == 1.0  # Evidence-source selection remains a separate score.
    assert metrics["grounded_answer_accuracy"] == 0.0
    assert metrics["grounded_answer_correct"] == 0
    assert metrics["grounded_answer_total"] == 1


def test_fact_validation_restores_metadata_missing_from_cold_loaded_node(tmp_path):
    source = SourceDocument("memory", "The hippocampus supports memory formation.")
    trainer = SupervisedStudyTrainer(ReaderAgent(
        TextDomainConfig.from_passages([source.text], max_vocab=100),
        dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "cold-metadata-forest"),
    ))
    trainer.fit([source], [StudyEpisode(
        "cold-metadata-train", "train", "grounded_qa",
        "What supports memory formation?", "memory",
        expected_answer=source.text,
    )])

    fact_id = next(iter(trainer.fact_node_data))
    trainer.fact_node_data[fact_id]["status"] = "candidate"
    fact_node = trainer.agent.forest.get(fact_id)
    if hasattr(fact_node, "metadata"):
        del fact_node.metadata
    validation = StudyEpisode(
        "cold-metadata-validation", "validation", "grounded_qa",
        "What supports memory formation?", "memory",
        expected_answer=source.text,
    )

    assert trainer._validate_fact_patterns([validation]) == 1
    assert trainer.fact_node_data[fact_id]["status"] == "validated"
    assert fact_node.metadata["status"] == "validated"


def test_fit_rejects_test_data_and_unknown_evidence(tmp_path):
    trainer = SupervisedStudyTrainer(_agent(tmp_path, inject_blank_priors=False))
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


def test_retention_gate_catches_per_family_regression_hidden_by_overall_score(tmp_path, monkeypatch):
    retained_text = "Neural signals travel between neurons across synapses."
    train_text = "Glass forms when silica is heated to a high temperature."
    config = TextDomainConfig.from_passages([retained_text, train_text], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "per-family-retention-forest"),
    ))
    retention = [StudyEpisode(
        "retained-family-qa", "test", "grounded_qa",
        "How do signals travel between neurons?", "retained",
    )]
    before = {
        "accuracy": 0.9,
        "per_task_type": {
            "grounded_qa": {"accuracy": 1.0},
            "procedure": {"accuracy": 0.8},
        },
    }
    after = {
        "accuracy": 0.86,
        "per_task_type": {
            "grounded_qa": {"accuracy": 0.7},
            "procedure": {"accuracy": 1.0},
        },
    }
    results = iter([before, after])
    monkeypatch.setattr(trainer, "evaluate", lambda _episodes: next(results))

    phase = trainer.fit_phase(
        [SourceDocument("retained", retained_text), SourceDocument("glass", train_text)],
        [StudyEpisode("glass-train", "train", "grounded_qa", "How does glass form?", "glass")],
        retention_episodes=retention,
        max_accuracy_drop=0.05,
    )

    assert phase["retention_overall_gate_passed"] is True
    assert phase["retention_task_accuracy_deltas"]["grounded_qa"] == pytest.approx(-0.3)
    assert phase["retention_task_gates_passed"]["grounded_qa"] is False
    assert phase["retention_gate_passed"] is False


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


def test_fit_curriculum_promotes_each_phase_exam_to_later_retention(tmp_path):
    glass = "Glass is made by heating sand with soda ash and limestone."
    bees = "Bees use waggle dances to share food direction and distance."
    config = TextDomainConfig.from_passages([glass, bees], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "phase-exam-forest"),
    ))
    phases = [
        {
            "name": "materials",
            "sources": [SourceDocument("glass", glass)],
            "train_episodes": [StudyEpisode(
                "glass-train", "train", "grounded_qa", "How is glass made?", "glass", glass,
            )],
            "test_episodes": [StudyEpisode(
                "glass-exam", "test", "grounded_qa", "What ingredients are heated to make glass?", "glass", glass,
            )],
        },
        {
            "name": "communication",
            "sources": [SourceDocument("bees", bees)],
            "train_episodes": [StudyEpisode(
                "bees-train", "train", "grounded_qa", "How do bees share food locations?", "bees", bees,
            )],
            "test_episodes": [StudyEpisode(
                "bees-exam", "test", "grounded_qa", "What do bees use to share food direction and distance?", "bees", bees,
            )],
        },
    ]

    report = trainer.fit_curriculum(phases)

    assert report["all_retention_gates_passed"] is True
    assert report["phase_test_episode_count"] == 2
    assert report["phases"][0]["retention_before"] is None
    assert report["phases"][0]["phase_test_metrics"]["accuracy"] == 1.0
    assert report["phases"][1]["retention_before"]["total"] == 1
    assert report["phases"][1]["retention_after"]["accuracy"] == 1.0
    assert report["phases"][1]["phase_test_metrics"]["accuracy"] == 1.0
    assert not ({"glass-exam", "bees-exam"} & trainer._fit_episode_ids)


def test_fit_curriculum_carries_validation_forward_for_calibration_only(tmp_path, monkeypatch):
    texts = [
        "Glass is made by heating sand with soda ash and limestone.",
        "Bees use waggle dances to share food direction and distance.",
    ]
    trainer = SupervisedStudyTrainer(ReaderAgent(
        TextDomainConfig.from_passages(texts, max_vocab=100),
        dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "historical-calibration-forest"),
    ))
    calls = []
    original = trainer._calibrate_threshold

    def capture(episodes):
        calls.append([episode.episode_id for episode in episodes])
        return original(episodes)

    monkeypatch.setattr(trainer, "_calibrate_threshold", capture)
    phases = [
        {
            "name": "materials",
            "sources": [SourceDocument("glass", texts[0])],
            "train_episodes": [StudyEpisode(
                "glass-train", "train", "grounded_qa", "How is glass made?", "glass", texts[0],
            )],
            "validation_episodes": [StudyEpisode(
                "glass-validation", "validation", "grounded_qa", "What makes glass?", "glass", texts[0],
            )],
            "test_episodes": [StudyEpisode(
                "glass-test", "test", "grounded_qa", "What is heated to make glass?", "glass", texts[0],
            )],
        },
        {
            "name": "communication",
            "sources": [SourceDocument("bees", texts[1])],
            "train_episodes": [StudyEpisode(
                "bees-train", "train", "grounded_qa", "How do bees share food locations?", "bees", texts[1],
            )],
            "validation_episodes": [StudyEpisode(
                "bees-validation", "validation", "grounded_qa", "How do bees communicate food?", "bees", texts[1],
            )],
            "test_episodes": [StudyEpisode(
                "bees-test", "test", "grounded_qa", "How do bees share food direction?", "bees", texts[1],
            )],
        },
    ]

    report = trainer.fit_curriculum(phases)

    assert calls == [["glass-validation"], ["glass-validation", "bees-validation"]]
    second = report["phases"][1]["fit"]
    assert second["validation_metrics"]["episode_count"] == 1
    assert second["calibration_episode_count"] == 2
    assert second["calibration_metrics"]["episode_count"] == 2
    assert not ({"glass-test", "bees-test"} & set(calls[-1]))


def test_abstention_pattern_transfers_to_oov_paraphrase_with_validated_anchors(tmp_path):
    source_text = "Coral reefs support diverse marine life in shallow tropical seas."
    trainer = SupervisedStudyTrainer(ReaderAgent(
        TextDomainConfig.from_passages([source_text], max_vocab=100),
        dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "negative-anchor-forest"),
    ))
    train = [StudyEpisode(
        "unsupported-train", "train", "abstain",
        "How do neutrinos turn into solid objects?", None, user_id="alice",
    )]
    validation = [
        StudyEpisode(
            "unsupported-validation", "validation", "abstain",
            "Can neutrinos become solid objects?", None, user_id="alice",
        ),
        StudyEpisode(
            "supported-validation", "validation", "grounded_qa",
            "How do corals support marine life?", "coral", source_text, user_id="alice",
        ),
    ]

    result = trainer.fit([SourceDocument("coral", source_text)], train, validation)
    unsupported = trainer.predict_question(
        "Do neutrinos change into solid matter?", user_id="alice"
    )
    unrelated = trainer.predict_question(
        "How do corals support marine life?", user_id="alice"
    )

    assert result["abstention_similarity_threshold"] < 0.8
    assert unsupported.abstained is True
    assert unrelated.selected_evidence_id == "coral"


def test_supervised_comparison_retrieves_both_sources_with_citations(tmp_path):
    solar = SourceDocument("solar", "Solar panels convert sunlight into electricity.")
    wind = SourceDocument("wind", "Wind turbines convert moving air into electricity.")
    distractor = SourceDocument("battery", "Batteries store electricity for later use.")
    sources = [solar, wind, distractor]
    config = TextDomainConfig.from_passages([source.text for source in sources], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "comparison-study-forest"),
    ))
    train = StudyEpisode(
        "compare-train", "train", "compare",
        "Compare electricity generation in solar and wind systems.", None,
        evidence_ids=["solar", "wind"], user_id="alice",
    )
    trainer.fit(sources, [train])
    exam = StudyEpisode(
        "compare-test", "test", "compare",
        "What distinguishes solar and wind power generation?", None,
        evidence_ids=("solar", "wind"), user_id="alice",
    )

    metrics = trainer.evaluate([exam])
    result = metrics["episode_results"][0]

    assert metrics["accuracy"] == 1.0
    assert metrics["comparison_evidence_precision"] == 1.0
    assert metrics["comparison_evidence_recall"] == 1.0
    assert set(result["selected_evidence_ids"]) == {"solar", "wind"}
    for citation in result["citations"]:
        source = trainer.sources[citation["source_id"]]
        assert source.text[citation["start"]:citation["end"]] == citation["text"]


def test_comparison_honors_explicit_source_titles_missing_from_passage_text(tmp_path):
    sources = [
        SourceDocument("solar", "Solar", "Light becomes a usable electrical current."),
        SourceDocument("solar_power", "Solar Power", "Photovoltaic devices generate electricity."),
        SourceDocument("astronomy", "Astronomy", "Stars form through gravitational collapse."),
    ]
    config = TextDomainConfig.from_passages([source.text for source in sources], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "explicit-comparison-forest"),
    ))
    trainer.fit(sources, [StudyEpisode(
        "explicit-compare-train", "train", "compare",
        "Compare solar power and astronomy.", None,
        evidence_ids=("solar_power", "astronomy"), user_id="alice",
    )])

    prediction = trainer.predict(StudyEpisode(
        "explicit-compare-test", "test", "compare",
        "Compare solar power and astronomy.", None,
        evidence_ids=("solar_power", "astronomy"), user_id="alice",
    ))

    assert set(prediction.selected_evidence_ids) == {"solar_power", "astronomy"}
    assert {citation["source_id"] for citation in prediction.citations} == {
        "solar_power", "astronomy"
    }


def test_ollama_evaluator_scores_multi_source_comparison_evidence():
    from hpm_ai_v2.experiments.run_supervised_study import _evaluate_ollama
    from hpm_ai_v2.training.ollama import ModelAnswer

    class Baseline:
        model = "fake-9b"

        def answer(self, *_args, **_kwargs):
            return ModelAnswer(
                "Solar and wind both generate electricity.", None, False, 1.0,
                evidence_ids=("wind", "solar"),
            )

        def configuration(self):
            return {"model": self.model}

        def runtime_metadata(self):
            return {"status": "test"}

    sources = [
        SourceDocument("solar", "Solar panels convert sunlight into electricity."),
        SourceDocument("wind", "Wind turbines convert moving air into electricity."),
    ]
    exam = [StudyEpisode(
        "compare-baseline-test", "test", "compare", "Compare solar and wind power.", None,
        evidence_ids=("solar", "wind"),
    )]

    metrics = _evaluate_ollama(Baseline(), sources, exam)

    assert metrics["task_accuracy"] == 1.0
    assert metrics["evidence_accuracy"] == 1.0
    assert metrics["comparison_evidence_precision"] == 1.0
    assert metrics["comparison_evidence_recall"] == 1.0


def test_ollama_evaluator_persists_and_resumes_only_matching_progress(tmp_path):
    from hpm_ai_v2.experiments.run_supervised_study import _evaluate_ollama
    from hpm_ai_v2.training.ollama import ModelAnswer

    class Baseline:
        model = "fake-resumable"

        def __init__(self):
            self.calls = 0

        def answer(self, question, *_args, **_kwargs):
            self.calls += 1
            return ModelAnswer(f"matched: {question}", "fact", False, 0.1)

        def configuration(self):
            return {"model": self.model}

        def runtime_metadata(self):
            return {"status": "test"}

    baseline = Baseline()
    sources = [SourceDocument("fact", "The fact is supported by this source.")]
    exam = [
        StudyEpisode(f"resumable-{i}", "test", "grounded_qa", f"Question {i}?", "fact")
        for i in range(2)
    ]
    progress = tmp_path / "baseline.jsonl"
    progress.write_text("", encoding="utf-8")

    first = _evaluate_ollama(baseline, sources, exam, progress_path=progress)
    resumed = _evaluate_ollama(baseline, sources, exam, progress_path=progress)

    assert baseline.calls == 2
    assert first["episodes"] == resumed["episodes"]
    assert progress.read_text(encoding="utf-8").count("\n") == 3

    changed_exam = [
        StudyEpisode("resumable-changed", "test", "grounded_qa", "Different question?", "fact")
    ]
    try:
        _evaluate_ollama(baseline, sources, changed_exam, progress_path=progress)
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("a mismatched evaluation must not reuse saved predictions")


def test_ollama_evaluator_reports_answer_quality_separately_from_source_selection():
    from hpm_ai_v2.experiments.run_supervised_study import _evaluate_ollama
    from hpm_ai_v2.training.ollama import ModelAnswer

    class Baseline:
        model = "fake-9b"

        def answer(self, *_args, **_kwargs):
            return ModelAnswer("An unrelated answer.", "biology", False, 1.0)

        def configuration(self):
            return {"model": self.model}

        def runtime_metadata(self):
            return {"status": "test"}

    sources = [SourceDocument("biology", "Neurons communicate across synapses.")]
    exam = [StudyEpisode(
        "bad-answer-test", "test", "grounded_qa", "How do neurons communicate?", "biology",
        expected_answer="Neurons communicate across synapses.",
    )]

    metrics = _evaluate_ollama(Baseline(), sources, exam)

    assert metrics["task_accuracy"] == 1.0
    assert metrics["grounded_answer_accuracy"] == 0.0
    assert metrics["grounded_answer_correct"] == 0
    assert metrics["grounded_answer_total"] == 1


def test_ollama_evaluator_counts_malformed_model_output_as_failed_item():
    from hpm_ai_v2.experiments.run_supervised_study import _evaluate_ollama

    class BrokenBaseline:
        model = "broken-model"

        def answer(self, *_args, **_kwargs):
            raise ValueError("truncated JSON response")

        def configuration(self):
            return {"model": self.model}

        def runtime_metadata(self):
            return {"status": "test"}

    sources = [SourceDocument("solar", "Solar panels convert sunlight into electricity.")]
    exam = [StudyEpisode(
        "malformed-output-test", "test", "grounded_qa",
        "How do solar panels generate electricity?", "solar",
    )]

    metrics = _evaluate_ollama(BrokenBaseline(), sources, exam)

    assert metrics["task_accuracy"] == 0.0
    assert metrics["episodes"][0]["evaluation_error"] == "ValueError: truncated JSON response"


def test_ollama_compare_prompt_requests_a_bounded_two_source_answer(monkeypatch):
    import json
    from hpm_ai_v2.training.ollama import OllamaChatBaseline

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "message": {"content": json.dumps({
                    "answer": "Solar panels use sunlight; wind turbines use moving air.",
                    "evidence_id": None,
                    "evidence_ids": ["solar", "wind"],
                    "abstained": False,
                    "workflow_key": None,
                })},
                "eval_count": 20,
            }).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr("hpm_ai_v2.training.ollama.urlopen", fake_urlopen)
    baseline = OllamaChatBaseline("fake-9b")
    sources = [
        SourceDocument("solar", "Solar panels convert sunlight into electricity."),
        SourceDocument("wind", "Wind turbines convert moving air into electricity."),
    ]

    result = baseline.answer(
        "Compare solar and wind power.", sources, task_type="compare"
    )

    assert result.evidence_ids == ("solar", "wind")
    assert captured["payload"]["options"]["num_predict"] == 128
    system = captured["payload"]["messages"][0]["content"]
    assert "evidence_ids" in system and "60 words" in system


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
    unrelated_scores_before = dict(trainer._rank(test[1].question, user_id=test[1].user_id))
    other_user_before = trainer.predict_question(test[0].question, user_id="another-user")
    source_weights_before = {
        node_id: agent.observer.get_weight(node_id)
        for node_id in trainer.source_to_node.values()
    }
    assert before.selected_evidence_id == "visual"
    fit = trainer.fit([], train, validation)
    after = [trainer.predict(episode) for episode in test]
    other_user_after = trainer.predict_question(test[0].question, user_id="another-user")
    unrelated_scores_after = dict(trainer._rank(test[1].question, user_id=test[1].user_id))

    assert after[0].selected_evidence_id == "waggle"
    assert after[0].answer == train[0].expected_answer
    assert other_user_before.selected_evidence_id == "visual"
    assert other_user_after.selected_evidence_id == "visual"
    assert {
        node_id: agent.observer.get_weight(node_id)
        for node_id in trainer.source_to_node.values()
    } == source_weights_before
    assert after[1].selected_evidence_id == "bats"
    assert unrelated_scores_after == unrelated_scores_before
    assert after[2].abstained
    assert fit["knowledge_growth"]["correction_links_added"] == 1
    fact_id = next(iter(trainer.fact_node_data))
    fact = trainer.fact_node_data[fact_id]
    assert fact["status"] == "validated"
    assert fact["correction_episode_ids"] == ["correction-bee"]
    assert len(fact["correction_query_vectors"]) == 1
    assert fact["correction_query_episode_ids"] == ["correction-bee"]
    assert fact["corrects_node_ids"] == [trainer.source_to_node["visual"]]
    fact_node = agent.forest.get(fact_id)
    assert any(edge.relation == "corrects" for edge in fact_node.edges())
    trainer.save_checkpoint(tmp_path / "correction-checkpoint")
    restored_agent = ReaderAgent.load_agent(str(tmp_path / "correction-checkpoint"))
    restored = SupervisedStudyTrainer(restored_agent)
    restored.load_checkpoint_state(tmp_path / "correction-checkpoint")
    assert restored.correction_query_similarity_threshold == trainer.correction_query_similarity_threshold
    assert restored.correction_answer_similarity_threshold == trainer.correction_answer_similarity_threshold
    assert restored.correction_transfer_score_threshold == trainer.correction_transfer_score_threshold
    assert restored.fact_retrieval_similarity_threshold == trainer.fact_retrieval_similarity_threshold
    assert restored.concept_retrieval_similarity_threshold == trainer.concept_retrieval_similarity_threshold
    restored_predictions = [restored.predict(episode) for episode in test]
    assert [p.selected_evidence_id for p in restored_predictions] == [
        p.selected_evidence_id for p in after
    ]
    assert restored_predictions[0].citation == after[0].citation


def test_correction_query_text_is_reencoded_after_vocabulary_growth(tmp_path):
    query = "How do bats use sound to locate insects?"
    config = TextDomainConfig.from_passages([query, "Bats are active at night."], max_vocab=100)
    agent = ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "correction-vocabulary-forest"),
    )
    trainer = SupervisedStudyTrainer(agent)
    original_dim = config.DIM
    old_vector = config.encode_passage(query)[config.S_DIM : config.S_DIM + config.DIM]
    data = {
        "correction_query_texts": [query],
        "correction_query_vectors": [old_vector.tolist()],
    }

    agent.expand_vocabulary(["ultrasonic echolocation frequencies"], max_new=10)

    assert config.DIM > original_dim
    assert len(data["correction_query_vectors"][0]) == original_dim
    assert trainer._correction_query_similarity(query, data) > 0.99


def test_correct_correction_prediction_does_not_create_a_false_corrects_edge(tmp_path):
    source = SourceDocument("facts", "Synapses pass signals between neurons.")
    config = TextDomainConfig.from_passages([source.text], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "non-correction-forest"),
    ))
    episode = StudyEpisode(
        "already-correct-correction", "train", "correction",
        "How do synapses pass signals?", "facts",
        expected_answer=source.text, correction_of="user-correction-001",
    )

    fit = trainer.fit([source], [episode])

    assert fit["knowledge_growth"]["correction_links_added"] == 0
    fact = next(iter(trainer.fact_node_data.values()))
    assert fact["corrects_node_ids"] == []


def test_correction_transfer_does_not_change_same_topic_collateral_questions(tmp_path):
    curriculum = Path(__file__).parents[1] / (
        "hpm_ai_v2/training/curricula/correction_transfer_confirmation_v2_48_seed.jsonl"
    )
    sources, episodes = load_curriculum_jsonl(curriculum)
    config = TextDomainConfig.from_passages([source.text for source in sources], max_vocab=400)
    agent = ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "scoped-correction-forest"),
    )
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    train = [episode for episode in episodes if episode.split == "train"]
    validation = [episode for episode in episodes if episode.split == "validation"]
    corrections = [
        episode for episode in episodes
        if episode.split == "test" and episode.task_type == "correction"
    ]
    collateral = [
        episode for episode in episodes
        if episode.split == "test" and "collateral" in episode.episode_id
    ]
    before = {episode.episode_id: trainer.predict(episode) for episode in collateral}

    trainer.fit(sources, train, validation)

    after = {episode.episode_id: trainer.predict(episode) for episode in collateral}
    assert collateral
    assert all(
        (before[key].selected_evidence_id, before[key].answer)
        == (after[key].selected_evidence_id, after[key].answer)
        for key in before
    )
    assert corrections
    assert any(
        trainer._correction_answer_similarity(
            episode.question,
            next(
                data for data in trainer.fact_node_data.values()
                if episode.evidence_id in data.get("source_ids", ())
                and data.get("correction_episode_ids")
            ),
        ) >= 0.32
        for episode in corrections
    )


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


def test_tfidf_retrieval_uses_validated_bm25_corroboration(tmp_path):
    from hpm_ai_v2.agents.reader_agent import ReaderAgent
    from hpm_ai_v2.domains.text_domain import TextDomainConfig
    from hpm_ai_v2.training import SourceDocument, SupervisedStudyTrainer

    sources = [
        SourceDocument("target", "Flywheel storage keeps energy in a spinning rotor."),
        SourceDocument("distractor", "Solar batteries store excess electricity."),
    ]
    config = TextDomainConfig.from_passages([source.text for source in sources])
    agent = ReaderAgent(config, cold_dir=str(tmp_path / "tfidf-lexical-corroboration"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    source_nodes = trainer.source_to_node
    agent.query_scored_nodes = lambda _question, k=10: [
        (source_nodes["distractor"], "distractor", 0.11),
        (source_nodes["target"], "target", 0.10),
    ]
    for node_id in source_nodes.values():
        agent.observer.set_weight(node_id, 1.0)
    trainer._bm25_source_scores = lambda _question: {"target": 2.0}

    trainer.semantic_lexical_bonus = 0.0
    sparse_only = trainer._rank("How does energy storage work?")
    trainer.semantic_lexical_bonus = 0.05
    sparse_with_bm25 = trainer._rank("How does energy storage work?")

    assert sparse_only[0][0] == "distractor"
    assert sparse_with_bm25[0][0] == "target"


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


def test_gpu_metadata_records_host_device_and_driver(monkeypatch):
    monkeypatch.setattr(run_supervised_study.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        run_supervised_study.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 5060 Ti, 16311 MiB, 580.173.02\n",
        ),
    )

    assert run_supervised_study._gpu_metadata() == {
        "status": "available",
        "devices": [{
            "name": "NVIDIA GeForce RTX 5060 Ti",
            "memory_total": "16311 MiB",
            "driver_version": "580.173.02",
        }],
    }


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


def test_baseline_history_contains_only_pretest_supervised_interactions():
    from hpm_ai_v2.experiments.run_supervised_study import _supervised_demonstrations

    episodes = [
        StudyEpisode("train-answer", "train", "grounded_qa", "Question?", "s1", "Answer."),
        StudyEpisode("validation-abstain", "validation", "abstain", "Unknown?", None),
        StudyEpisode("test-secret", "test", "grounded_qa", "Held out?", "s1", "Never reveal."),
    ]
    demonstrations = _supervised_demonstrations(episodes)

    assert [row["episode_id"] for row in demonstrations] == [
        "train-answer", "validation-abstain",
    ]
    assert demonstrations[0]["outcome"] == {
        "abstained": False, "evidence_id": "s1", "answer": "Answer.",
    }
    assert demonstrations[1]["outcome"] == {
        "abstained": True, "evidence_id": None, "answer": None,
    }
    assert all("Never reveal." not in json.dumps(row) for row in demonstrations)


def test_relevant_baseline_history_selects_matching_examples_before_recent_distractors():
    from hpm_ai_v2.experiments.run_supervised_study import _select_relevant_interactions

    history = [
        {"episode_id": "helpful", "task_type": "grounded_qa", "user_request": "How do bees show food location?",
         "outcome": {"answer": "Honeybees use the waggle dance to indicate food direction and distance."}},
        {"episode_id": "irrelevant", "task_type": "grounded_qa", "user_request": "What do pythons eat?",
         "outcome": {"answer": "Pythons are constricting snakes."}},
    ]
    selected = _select_relevant_interactions(
        "How do honeybees communicate where food is?", history,
        task_type="grounded_qa", preference_key=None, limit=1,
    )

    assert [row["episode_id"] for row in selected] == ["helpful"]


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
    prior = [{
        "episode_id": "earlier",
        "user_request": "What is stated?",
        "outcome": {"abstained": False, "evidence_id": "s1", "answer": "A fact."},
    }]
    answer = baseline.answer(
        "What is stated?", [SourceDocument("s1", "A fact.", "Source")],
        prior_interactions=prior,
    )

    assert answer.answer == "A fact."
    assert answer.evidence_id == "s1"
    assert not answer.abstained
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["think"] is False
    assert captured["payload"]["options"] == {"temperature": 0.0, "num_predict": 256}
    assert captured["payload"]["messages"][1]["content"]
    assert json.loads(captured["payload"]["messages"][1]["content"])["prior_supervised_interactions"] == prior
    assert baseline.configuration()["prompt_version"] == "grounded-study-v5-workflow-labels"


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

    def fake_answer(question, selected_sources, *, prior_interactions=(), task_type=None,
                    user_id=None, preference_key=None):
        captured["source_ids"] = [source.source_id for source in selected_sources]
        captured["prior_interactions"] = list(prior_interactions)
        captured["task_type"] = task_type
        captured["user_id"] = user_id
        from hpm_ai_v2.training.ollama import ModelAnswer
        return ModelAnswer("Water freezes at zero degrees Celsius.", "answer", False, 1.0)

    monkeypatch.setattr(baseline.chat, "answer", fake_answer)
    monkeypatch.setattr(baseline, "runtime_metadata", lambda: {"status": "test"})
    result = baseline.answer("At what temperature does water freeze?", sources, task_type="grounded_qa")

    assert captured["source_ids"] == ["answer"]
    assert captured["prior_interactions"] == []
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


def test_cumulative_runner_trains_across_sessions_and_audits_restart(tmp_path, capsys):
    from hpm_ai_v2.experiments.run_cumulative_study import run_cumulative

    phase_one = tmp_path / "session-one.jsonl"
    phase_two = tmp_path / "session-two.jsonl"
    retention = tmp_path / "retention.jsonl"
    pretraining = tmp_path / "pretraining.jsonl"
    pretraining.write_text(json.dumps({
        "kind": "source", "source_id": "mitochondria",
        "text": "Mitochondria generate much of the chemical energy that powers a cell.",
    }) + "\n", encoding="utf-8")
    phase_one.write_text("\n".join([
        json.dumps({"kind": "source", "source_id": "neural", "text": "Neural signals travel between neurons across synapses."}),
        json.dumps({"kind": "source", "source_id": "glass", "text": "Glass is produced by heating sand with soda ash and limestone."}),
        json.dumps({"kind": "episode", "episode_id": "teach-glass", "split": "train", "task_type": "grounded_qa", "question": "How is glass produced?", "evidence_id": "glass", "expected_answer": "Glass is produced by heating sand with soda ash and limestone."}),
    ]) + "\n", encoding="utf-8")
    phase_two.write_text("\n".join([
        json.dumps({"kind": "source", "source_id": "bees", "text": "Bees use waggle dances to communicate food locations to hive mates."}),
        json.dumps({"kind": "episode", "episode_id": "teach-bees", "split": "train", "task_type": "grounded_qa", "question": "How do bees communicate food locations?", "evidence_id": "bees", "expected_answer": "Bees use waggle dances to communicate food locations to hive mates."}),
    ]) + "\n", encoding="utf-8")
    retention.write_text("\n".join([
        json.dumps({"kind": "episode", "episode_id": "retained-neural", "split": "test", "task_type": "grounded_qa", "question": "Where do neural signals travel?", "evidence_id": "neural", "expected_answer": "Neural signals travel between neurons across synapses."}),
    ]) + "\n", encoding="utf-8")

    result = run_cumulative(
        [phase_one, phase_two], retention, tmp_path / "out", max_accuracy_drop=0.05,
        pretraining_corpus_paths=[pretraining],
    )
    capsys.readouterr()

    assert result["training"]["phase_count"] == 2
    assert [p["fit"]["knowledge_growth"]["new_fact_nodes"] for p in result["training"]["phases"]] == [1, 1]
    assert result["knowledge_growth"]["fact_nodes"] == 2
    assert result["knowledge_growth"]["source_documents"] == 4
    assert result["configuration"]["pretraining_corpus"]["source_documents"] == 1
    assert result["configuration"]["pretraining_corpus"]["ingested_before_first_phase_exam"] is True
    assert result["restart_consistency"]["same_predictions"] is True
    assert (tmp_path / "out" / "cumulative_study_report.json").exists()


def test_cumulative_runner_continues_checkpoint_and_retains_prior_exam(tmp_path, capsys):
    import hashlib
    from hpm_ai_v2.experiments.run_cumulative_study import run_cumulative

    glass = "Glass is made by heating sand with soda ash and limestone."
    bees = "Bees use waggle dances to share food direction and distance."
    initial = tmp_path / "initial.jsonl"
    continuation = tmp_path / "continuation.jsonl"
    history = tmp_path / "history.jsonl"
    retention = tmp_path / "retention.jsonl"
    initial.write_text("\n".join([
        json.dumps({"kind": "source", "source_id": "glass", "text": glass}),
        json.dumps({"kind": "episode", "episode_id": "glass-train-initial", "split": "train", "task_type": "grounded_qa", "question": "How is glass made?", "evidence_id": "glass", "expected_answer": glass}),
        json.dumps({"kind": "episode", "episode_id": "glass-validation-initial", "split": "validation", "task_type": "grounded_qa", "question": "How is glass produced?", "evidence_id": "glass", "expected_answer": glass}),
        json.dumps({"kind": "episode", "episode_id": "glass-exam-initial", "split": "test", "task_type": "grounded_qa", "question": "How do you make glass?", "evidence_id": "glass", "expected_answer": glass}),
    ]) + "\n", encoding="utf-8")
    history.write_text("\n".join([
        json.dumps({"kind": "episode", "episode_id": "glass-train-initial", "split": "train", "task_type": "grounded_qa", "question": "How is glass made?", "evidence_id": "glass", "expected_answer": glass}),
        json.dumps({"kind": "episode", "episode_id": "glass-validation-initial", "split": "validation", "task_type": "grounded_qa", "question": "How is glass produced?", "evidence_id": "glass", "expected_answer": glass}),
    ]) + "\n", encoding="utf-8")
    continuation.write_text("\n".join([
        json.dumps({"kind": "source", "source_id": "glass", "text": glass}),
        json.dumps({"kind": "source", "source_id": "bees", "text": bees}),
        json.dumps({"kind": "episode", "episode_id": "bees-train-continuation", "split": "train", "task_type": "grounded_qa", "question": "How do bees communicate food direction?", "evidence_id": "bees", "expected_answer": bees}),
        json.dumps({"kind": "episode", "episode_id": "bees-exam-continuation", "split": "test", "task_type": "grounded_qa", "question": "How do bees share food distance?", "evidence_id": "bees", "expected_answer": bees}),
    ]) + "\n", encoding="utf-8")
    retention.write_text(json.dumps({
        "kind": "episode", "episode_id": "glass-exam-retention", "split": "test",
        "task_type": "grounded_qa", "question": "How do you make glass?",
        "evidence_id": "glass", "expected_answer": glass,
    }) + "\n", encoding="utf-8")

    first = run_cumulative([initial], None, tmp_path / "initial-out")
    capsys.readouterr()

    def checkpoint_digests(directory):
        return {
            path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file()
        }

    original_checkpoint_digests = checkpoint_digests(tmp_path / "initial-out")
    with pytest.raises(ValueError, match="original validation calibration"):
        run_cumulative(
            [continuation], retention, tmp_path / "missing-history-out",
            initial_checkpoint=tmp_path / "initial-out",
        )
    second = run_cumulative(
        [continuation], retention, tmp_path / "continued-out",
        initial_checkpoint=tmp_path / "initial-out",
        baseline_history_paths=[history],
    )
    capsys.readouterr()

    assert checkpoint_digests(tmp_path / "initial-out") == original_checkpoint_digests

    assert second["configuration"]["initial_checkpoint"] == str(tmp_path / "initial-out")
    assert second["training"]["initial_validation_episode_count"] == 1
    assert second["training"]["phases"][0]["retention_before"]["total"] == 1
    assert second["training"]["phases"][0]["retention_gate_passed"] is True
    assert second["retention_after_all_phases"]["total"] == 2
    assert second["restart_consistency"]["same_predictions"] is True
    assert second["configuration"]["vocabulary_expansion"]["new_source_count"] == 1
    assert second["configuration"]["vocabulary_expansion"]["terms_added"] > 0
    assert len(first["knowledge_growth"]) >= 1


def test_prediction_signature_normalizes_json_tuple_list_roundtrip():
    from hpm_ai_v2.experiments.run_cumulative_study import _prediction_signature

    citation = {"source_id": "notes", "start": 0, "end": 12, "text": "Evidence text."}
    live = {"episode_results": [{
        "episode_id": "one", "selected_evidence_id": "notes", "answer": "Evidence text.",
        "citations": (citation,), "selected_evidence_ids": (),
        "selected_workflow_key": None, "abstained": False,
    }]}
    saved_json = {"episode_results": [{
        "episode_id": "one", "selected_evidence_id": "notes", "answer": "Evidence text.",
        "citations": [citation], "selected_evidence_ids": [],
        "selected_workflow_key": None, "abstained": False,
    }]}

    assert _prediction_signature(live) == _prediction_signature(saved_json)
    changed = {"episode_results": [{**saved_json["episode_results"][0], "answer": "Different answer."}]}
    assert _prediction_signature(live) != _prediction_signature(changed)


def test_cumulative_runner_uses_each_phase_exam_for_later_retention(tmp_path, capsys):
    from hpm_ai_v2.experiments.run_cumulative_study import run_cumulative

    glass = "Glass is made by heating sand with soda ash and limestone."
    bees = "Bees use waggle dances to share food direction and distance."
    phase_one = tmp_path / "materials.jsonl"
    phase_two = tmp_path / "communication.jsonl"
    phase_one.write_text("\n".join([
        json.dumps({"kind": "source", "source_id": "glass", "text": glass}),
        json.dumps({"kind": "episode", "episode_id": "glass-train", "split": "train", "task_type": "grounded_qa", "question": "How is glass made?", "evidence_id": "glass", "expected_answer": glass}),
        json.dumps({"kind": "episode", "episode_id": "glass-exam", "split": "test", "task_type": "grounded_qa", "question": "What ingredients are heated to make glass?", "evidence_id": "glass", "expected_answer": glass}),
    ]) + "\n", encoding="utf-8")
    phase_two.write_text("\n".join([
        json.dumps({"kind": "source", "source_id": "bees", "text": bees}),
        json.dumps({"kind": "episode", "episode_id": "bees-train", "split": "train", "task_type": "grounded_qa", "question": "How do bees share food locations?", "evidence_id": "bees", "expected_answer": bees}),
        json.dumps({"kind": "episode", "episode_id": "bees-exam", "split": "test", "task_type": "grounded_qa", "question": "What do bees use to share food direction and distance?", "evidence_id": "bees", "expected_answer": bees}),
    ]) + "\n", encoding="utf-8")

    result = run_cumulative([phase_one, phase_two], None, tmp_path / "cumulative-out")
    capsys.readouterr()

    assert result["training"]["phase_count"] == 2
    assert result["training"]["phase_test_episode_count"] == 2
    assert result["training"]["phases"][0]["retention_gate_passed"] is None
    assert result["training"]["phases"][0]["phase_test_metrics"]["accuracy"] == 1.0
    assert result["training"]["phases"][0]["phase_test_before_metrics"]["total"] == 1
    assert result["training"]["phases"][0]["phase_test_accuracy_delta"] == 0.0
    assert result["training"]["phases"][1]["retention_before"]["total"] == 1
    assert result["training"]["phases"][1]["retention_gate_passed"] is True
    assert result["training"]["phases"][1]["phase_test_metrics"]["accuracy"] == 1.0
    assert result["training"]["phases"][1]["phase_test_before_metrics"]["total"] == 1
    assert result["training"]["phases"][1]["phase_test_accuracy_delta"] == 0.0
    assert result["restart_consistency"]["same_predictions"] is True


def test_cumulative_phase_exam_reports_supervised_acquisition_gain(tmp_path):
    curriculum = Path(__file__).parents[1] / "hpm_ai_v2/training/curricula/correction_transfer_seed.jsonl"
    sources, episodes = load_curriculum_jsonl(curriculum)
    train = [episode for episode in episodes if episode.split == "train"]
    validation = [episode for episode in episodes if episode.split == "validation"]
    test_episode = next(episode for episode in episodes if episode.split == "test")
    config = TextDomainConfig.from_passages([source.text for source in sources], max_vocab=500)
    trainer = SupervisedStudyTrainer(ReaderAgent(
        config,
        dynamic_vocab=False,
        inject_blank_priors=False,
        cold_dir=str(tmp_path / "acquisition-forest"),
    ))

    result = trainer.fit_curriculum([{
        "name": "correction-acquisition",
        "sources": sources,
        "train_episodes": train,
        "validation_episodes": validation,
        "test_episodes": [test_episode],
    }])

    phase = result["phases"][0]
    before = phase["phase_test_before_metrics"]["episode_results"][0]
    after = phase["phase_test_metrics"]["episode_results"][0]
    assert before["selected_evidence_id"] != test_episode.evidence_id
    assert after["selected_evidence_id"] == test_episode.evidence_id
    assert phase["phase_test_accuracy_delta"] == 1.0
    assert result["total_fact_nodes"] > 0
    assert test_episode.episode_id not in trainer._fit_episode_ids


def test_user_preferences_update_by_scope_and_survive_checkpoint(tmp_path):
    source = SourceDocument("manual", "Study examples and review material.")
    config = TextDomainConfig.from_passages([source.text], max_vocab=100)
    agent = ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "preference-forest"),
    )
    trainer = SupervisedStudyTrainer(agent)
    first = StudyEpisode(
        "pref-a-1", "train", "preference", "Please use bullet summaries.", None,
        expected_answer="bullets", user_id="alice", preference_key="summary_format",
    )
    other_user = StudyEpisode(
        "pref-b-1", "train", "preference", "Please use short paragraphs.", None,
        expected_answer="paragraphs", user_id="bob", preference_key="summary_format",
    )
    updated = StudyEpisode(
        "pref-a-2", "train", "preference", "Use tables for summaries from now on.", None,
        expected_answer="tables", user_id="alice", preference_key="summary_format",
    )

    initial_fit = trainer.fit([source], [first, other_user])
    update_fit = trainer.fit([], [updated])
    assert initial_fit["knowledge_growth"]["new_preferences"] == 2
    assert update_fit["knowledge_growth"]["updated_preferences"] == 1
    assert trainer.get_preference("alice", "summary_format") == "tables"
    assert trainer.get_preference("bob", "summary_format") == "paragraphs"
    assert trainer.get_preference("eve", "summary_format") is None

    checkpoint = tmp_path / "preference-checkpoint"
    trainer.save_checkpoint(checkpoint)
    restored = SupervisedStudyTrainer(ReaderAgent.load_agent(str(checkpoint)))
    restored.load_checkpoint_state(checkpoint)
    exam = [
        StudyEpisode("pref-a-test", "test", "preference", "What is my summary format?", None,
                     expected_answer="tables", user_id="alice", preference_key="summary_format"),
        StudyEpisode("pref-b-test", "test", "preference", "What is my summary format?", None,
                     expected_answer="paragraphs", user_id="bob", preference_key="summary_format"),
    ]
    result = restored.evaluate(exam)
    assert result["accuracy"] == 1.0
    assert result["answer_token_f1"] == 1.0


def test_ollama_preference_evaluation_exposes_only_same_user_history():
    from hpm_ai_v2.experiments.run_supervised_study import _evaluate_ollama
    from hpm_ai_v2.training.ollama import ModelAnswer

    class Baseline:
        model = "fake-local-model"

        def answer(self, question, sources, *, prior_interactions=(), **context):
            self.history = list(prior_interactions)
            self.context = context
            return ModelAnswer("bullets", None, False, 1.0)

        def configuration(self):
            return {}

        def runtime_metadata(self):
            return {"status": "test"}

    baseline = Baseline()
    history = [
        {"user_id": "alice", "task_type": "preference", "preference_key": "summary_format", "outcome": {"answer": "bullets"}},
        {"user_id": "bob", "task_type": "preference", "preference_key": "summary_format", "outcome": {"answer": "paragraphs"}},
    ]
    episode = StudyEpisode(
        "pref-eval", "test", "preference", "What format do I prefer?", None,
        expected_answer="bullets", user_id="alice", preference_key="summary_format",
    )
    result = _evaluate_ollama(baseline, [], [episode], prior_interactions=history)

    assert result["task_accuracy"] == 1.0
    assert result["prior_interaction_count"] == 1
    assert [row["user_id"] for row in baseline.history] == ["alice"]
    assert baseline.context == {
        "task_type": "preference", "user_id": "alice", "preference_key": "summary_format",
    }


def test_ollama_preference_prompt_includes_user_and_key(monkeypatch):
    baseline = OllamaChatBaseline("qwen3.5:latest")
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"message":{"content":"{\\"answer\\":\\"bullets\\",\\"evidence_id\\":null,\\"abstained\\":false}"}}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return Response()

    monkeypatch.setattr("hpm_ai_v2.training.ollama.urlopen", fake_urlopen)
    baseline.answer(
        "What summary format do I prefer?", [], task_type="preference",
        user_id="alice", preference_key="summary_format",
        prior_interactions=[{
            "user_id": "alice", "task_type": "preference",
            "preference_key": "summary_format", "outcome": {"answer": "bullets"},
        }],
    )

    request = json.loads(captured["payload"]["messages"][1]["content"])
    assert request["task_type"] == "preference"
    assert request["user_id"] == "alice"
    assert request["preference_key"] == "summary_format"
    assert len(request["prior_supervised_interactions"]) == 1


def test_retrieval_discrimination_curriculum_has_heldout_transfer_and_grounded_answers():
    path = Path("hpm_ai_v2/training/curricula/retrieval_discrimination_90_seed.jsonl")
    sources, episodes = load_curriculum_jsonl(path)
    source_map = {source.source_id: source for source in sources}
    by_split = {
        split: [episode for episode in episodes if episode.split == split]
        for split in ("train", "validation", "test")
    }

    assert len(sources) == 10  # each document groups nine source passages
    assert len(episodes) == 29
    assert [len(by_split[split]) for split in ("train", "validation", "test")] == [11, 9, 9]
    assert all(
        episode.expected_answer in source_map[episode.evidence_id].text
        for episode in episodes
        if episode.evidence_id is not None
    )
    assert sum(episode.evidence_id is None for episode in by_split["test"]) == 1
    assert all(episode.user_id == "benchmark-user" for episode in episodes)


def test_grounded_qa_acquisition_curriculum_validates_trained_facts_and_tests_transfer():
    path = Path("hpm_ai_v2/training/curricula/grounded_qa_acquisition_growth_60_seed.jsonl")
    sources, episodes = load_curriculum_jsonl(path)
    source_map = {source.source_id: source for source in sources}
    by_split = {
        split: [episode for episode in episodes if episode.split == split]
        for split in ("train", "validation", "test")
    }
    train_by_source = {episode.evidence_id: episode for episode in by_split["train"]}
    validation_by_source = {episode.evidence_id: episode for episode in by_split["validation"]}

    assert len(sources) == 38
    assert [len(by_split[split]) for split in ("train", "validation", "test")] == [10, 10, 60]
    assert set(train_by_source) == set(validation_by_source)
    for source_id, train in train_by_source.items():
        validation = validation_by_source[source_id]
        assert train.expected_answer == validation.expected_answer
        assert train.question != validation.question
        assert train.expected_answer in source_map[source_id].text
        tests = [episode for episode in by_split["test"] if episode.evidence_id == source_id]
        assert len(tests) == 6
        assert sum(episode.expected_answer == train.expected_answer for episode in tests) == 2
        assert all(episode.expected_answer in source_map[source_id].text for episode in tests)

    # Source membership alone did not catch eight questions paired with the
    # neighboring source sentence instead of the sentence answering them.
    audited_answers = {
        "kg60-neuroscience-novel-01": "The hippocampus plays a central role in forming new long-term memories.",
        "kg60-neuroscience-novel-02": "Neurotransmitters such as dopamine and serotonin regulate mood and motivation in the brain.",
        "kg60-neuroscience-novel-03": "The cerebral cortex is responsible for higher-order functions like reasoning and language.",
        "kg60-neuroscience-novel-04": "Long-term potentiation strengthens synaptic connections when neurons fire together repeatedly.",
        "kg60-roman_catholic-novel-01": "Roman Catholic doctrine holds seven sacraments as central to religious practice.",
        "kg60-roman_catholic-novel-02": "The Second Vatican Council in the 1960s reformed many Roman Catholic liturgical practices.",
        "kg60-roman_catholic-novel-03": "The Catechism of the Roman Catholic Church summarizes core beliefs and moral teachings.",
        "kg60-roman_catholic-novel-04": "Roman Catholic monasteries preserved classical texts throughout the medieval period.",
    }
    test_by_id = {episode.episode_id: episode for episode in by_split["test"]}
    assert {episode_id: test_by_id[episode_id].expected_answer for episode_id in audited_answers} == audited_answers


def test_knowledge_acquisition_growth_curriculum_pairs_all_train_facts_with_validation():
    path = Path("hpm_ai_v2/training/curricula/knowledge_acquisition_growth_150_seed.jsonl")
    sources, episodes = load_curriculum_jsonl(path)
    source_map = {source.source_id: source for source in sources}
    by_split = {
        split: [episode for episode in episodes if episode.split == split]
        for split in ("train", "validation", "test")
    }
    train_facts = {
        (episode.user_id, episode.evidence_id, episode.expected_answer)
        for episode in by_split["train"]
    }
    validation_facts = {
        (episode.user_id, episode.evidence_id, episode.expected_answer)
        for episode in by_split["validation"]
    }

    assert [len(by_split[split]) for split in ("train", "validation", "test")] == [60, 70, 20]
    assert train_facts <= validation_facts
    assert len(train_facts) == 60
    assert all(
        episode.expected_answer in source_map[episode.evidence_id].text
        for episode in episodes
        if episode.evidence_id and episode.expected_answer
    )
    assert not (train_facts & {
        (episode.user_id, episode.evidence_id, episode.expected_answer)
        for episode in by_split["test"]
    })


def test_concept_transfer_curriculum_validates_across_sources_and_keeps_test_unlabeled():
    path = Path("hpm_ai_v2/training/curricula/concept_transfer_36_seed.jsonl")
    sources, episodes = load_curriculum_jsonl(path)
    source_map = {source.source_id: source for source in sources}
    by_split = {
        split: [episode for episode in episodes if episode.split == split]
        for split in ("train", "validation", "test")
    }

    assert len(sources) == 12
    assert [len(by_split[split]) for split in ("train", "validation", "test")] == [9, 3, 36]
    concepts = {episode.concept_key for episode in by_split["train"]}
    assert concepts == {"energy_storage", "error_correction", "feedback_control"}
    for concept in concepts:
        train_sources = {
            episode.evidence_id for episode in by_split["train"]
            if episode.concept_key == concept
        }
        validation = [episode for episode in by_split["validation"] if episode.concept_key == concept]
        tests = [episode for episode in by_split["test"] if episode.evidence_id in train_sources]
        assert len(train_sources) == 3
        assert len(validation) == 1
        assert validation[0].evidence_id not in train_sources
        assert len(tests) == 9
    assert all(episode.concept_key is None for episode in by_split["test"])
    assert all(
        episode.expected_answer in source_map[episode.evidence_id].text
        for episode in episodes
        if episode.expected_answer is not None
    )


def test_concept_transfer_confirmation_is_disjoint_and_keeps_test_unlabeled():
    path = Path("hpm_ai_v2/training/curricula/concept_transfer_confirmation_36_seed.jsonl")
    sources, episodes = load_curriculum_jsonl(path)
    prior_sources, _ = load_curriculum_jsonl(
        "hpm_ai_v2/training/curricula/concept_transfer_36_seed.jsonl"
    )
    source_map = {source.source_id: source for source in sources}
    by_split = {
        split: [episode for episode in episodes if episode.split == split]
        for split in ("train", "validation", "test")
    }

    assert len(sources) == 12
    assert not ({source.source_id for source in sources} & {
        source.source_id for source in prior_sources
    })
    assert [len(by_split[split]) for split in ("train", "validation", "test")] == [9, 3, 36]
    concepts = {episode.concept_key for episode in by_split["train"]}
    assert concepts == {"information_compression", "heat_transfer", "immune_defense"}
    for concept in concepts:
        train_sources = {
            episode.evidence_id for episode in by_split["train"]
            if episode.concept_key == concept
        }
        validation = [episode for episode in by_split["validation"] if episode.concept_key == concept]
        assert len(train_sources) == 3
        assert len(validation) == 1
        assert validation[0].evidence_id not in train_sources
        assert sum(episode.evidence_id in train_sources for episode in by_split["test"]) == 9
    assert all(episode.concept_key is None for episode in by_split["test"])
    assert all(
        episode.expected_answer in source_map[episode.evidence_id].text
        for episode in episodes
        if episode.expected_answer is not None
    )


def test_capability_transfer_confirmation_validates_every_trained_fact():
    path = Path(
        "hpm_ai_v2/training/curricula/capability_transfer_confirmation_36_seed.jsonl"
    )
    sources, episodes = load_curriculum_jsonl(path)
    source_map = {source.source_id: source for source in sources}
    by_split = {
        split: [episode for episode in episodes if episode.split == split]
        for split in ("train", "validation", "test")
    }
    train_facts = {
        (episode.user_id, episode.evidence_id, episode.expected_answer)
        for episode in by_split["train"]
    }
    validation_facts = {
        (episode.user_id, episode.evidence_id, episode.expected_answer)
        for episode in by_split["validation"]
    }

    assert len(sources) == 12
    assert [len(by_split[split]) for split in ("train", "validation", "test")] == [8, 8, 36]
    assert train_facts == validation_facts
    assert len(train_facts) == 8
    for train in by_split["train"]:
        validation = next(
            episode for episode in by_split["validation"]
            if episode.evidence_id == train.evidence_id
        )
        assert train.question != validation.question
        assert train.expected_answer in source_map[train.evidence_id].text
    assert all(
        episode.expected_answer in source_map[episode.evidence_id].text
        for episode in episodes
        if episode.expected_answer is not None
    )


def test_supervised_concepts_link_multiple_sources_and_promote_after_heldout_check(tmp_path):
    first_text = "Neurons transmit electrochemical signals across synapses."
    second_text = "Neural impulses pass between brain cells at synaptic junctions."
    sources = [
        SourceDocument("neuron-a", first_text, "Neuroscience notes A"),
        SourceDocument("neuron-b", second_text, "Neuroscience notes B"),
    ]
    config = TextDomainConfig.from_passages([first_text, second_text], max_vocab=100)
    agent = ReaderAgent(
        config, dynamic_vocab=False, inject_blank_priors=False,
        cold_dir=str(tmp_path / "concept-forest"),
    )
    trainer = SupervisedStudyTrainer(agent)
    train = [
        StudyEpisode(
            "teach-neuron-a", "train", "grounded_qa", "How do neurons signal across synapses?",
            "neuron-a", first_text, concept_key="neural communication",
        ),
        StudyEpisode(
            "teach-neuron-b", "train", "grounded_qa", "How do neural impulses pass between brain cells?",
            "neuron-b", second_text, concept_key="neural communication",
        ),
    ]
    validation = [StudyEpisode(
        "validate-neuron-b", "validation", "grounded_qa",
        "Where do neural impulses pass between cells?", "neuron-b", second_text,
        concept_key="neural communication",
    )]

    result = trainer.fit(sources, train, validation)

    assert result["knowledge_growth"]["new_fact_nodes"] == 2
    assert result["knowledge_growth"]["new_concept_nodes"] == 1
    assert result["knowledge_growth"]["validated_concept_nodes"] == 1
    assert result["knowledge_growth"]["concept_validation"] == {
        "checked": 1, "passed": 1, "promoted": 0,
    }
    concept_id = next(iter(trainer.concept_node_data))
    concept = trainer.concept_node_data[concept_id]
    assert concept["status"] == "validated"
    assert len(concept["fact_node_ids"]) == 2
    assert set(concept["source_ids"]) == {"neuron-a", "neuron-b"}
    assert agent.forest.get(concept_id).relation_type == "study_concept"
    query = "How do neurons transmit electrochemical messages?"
    unrelated_candidate_scores = dict(trainer._rank(query, user_id="other-user"))
    validated_scores = dict(trainer._rank(query))
    concept["status"] = "candidate"
    candidate_scores = dict(trainer._rank(query))
    unrelated_candidate_after_status_change = dict(trainer._rank(query, user_id="other-user"))
    concept["status"] = "validated"
    assert validated_scores["neuron-a"] > candidate_scores["neuron-a"]
    assert unrelated_candidate_scores == unrelated_candidate_after_status_change

    checkpoint = tmp_path / "concept-checkpoint"
    trainer.save_checkpoint(checkpoint)
    restored = SupervisedStudyTrainer(ReaderAgent.load_agent(str(checkpoint)))
    restored.load_checkpoint_state(checkpoint)
    assert restored.concept_node_data == trainer.concept_node_data
    assert restored.concept_key_to_node == trainer.concept_key_to_node
