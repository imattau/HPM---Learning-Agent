import json
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest

from hpm_ai_v2.training.language_prediction import (
    BOUNDARY,
    HierarchicalNextUnitModel,
    read_jsonl,
    split_documents,
    split_sentences,
    tokenize,
    _run_validation_sweeps,
    _run_metric_jobs,
    run_pilot,
)
from hpm_ai_v2.training.ngram_baseline import NGramBaseline


def test_hierarchy_and_predictive_sigma_are_learned_from_counts():
    model = HierarchicalNextUnitModel(max_order=2).fit([
        "cats chase mice", "cats chase birds", "dogs chase cats", "cats sleep"
    ])
    assert model.char_nodes
    assert model.char_context_nodes
    assert any(node.metadata["type"] == "next_character_distribution" for node in model.char_context_nodes.values())
    assert model.word_nodes
    assert () in model.context_nodes
    phrase_node = model.context_nodes[("cats",)]
    assert phrase_node.metadata["observations"] == 4
    assert phrase_node.mu[0] > 0  # learned conditional entropy
    assert np.all(phrase_node.sigma > 0)
    target_counts = phrase_node.metadata["target_counts"]
    total = sum(target_counts.values())
    expected_gini = 1 - sum((count / total) ** 2 for count in target_counts.values())
    assert np.isclose(phrase_node.sigma[0], expected_gini)
    assert all(child.metadata["type"] == "word" for child in phrase_node.children())
    assert BOUNDARY in model.vocab


def test_class_nodes_cluster_members_pool_contexts_and_round_trip(tmp_path):
    # Two subject words share the same follower class; the other nouns follow
    # a distinct verb, making profile clustering deterministic in this corpus.
    texts = (["cat sleeps"] * 3 + ["cat purrs"] * 3 + ["dog sleeps"] * 3 + ["dog purrs"] * 3
             + ["bird runs"] * 3 + ["bird flies"] * 3 + ["fish runs"] * 3 + ["fish flies"] * 3)
    model = HierarchicalNextUnitModel(max_order=2, class_nodes=True).fit(texts)
    # Exercise clustering independently of selector choice, then install a
    # known same-followers class to verify HFN structure and checkpointing.
    clusters = HierarchicalNextUnitModel._average_link_clusters(
        {"cat": np.array([1.0, 0.0]), "dog": np.array([0.95, 0.05]),
         "bird": np.array([0.0, 1.0])}, 0.2)
    assert any(set(group) == {"cat", "dog"} for group in clusters)
    profiles = model._profiles_from_counts(model.raw_word_context_counts)
    class_id = "language_class_test"
    model.class_labels = {"cat": class_id, "dog": class_id}
    member_profiles = np.asarray([profiles["cat"], profiles["dog"]])
    model.class_profiles = {class_id: member_profiles.mean(axis=0)}
    model.class_profile_variances = {class_id: member_profiles.var(axis=0) + .05 ** 2}
    model.class_profile_features = list(model._current_profile_features)
    model.class_lambda = .5
    model._class_membership_frozen = True
    model._fit_class_nodes_from_counts(refit=False)
    model._build_hfn_nodes()
    model._initialize_pattern_observer()
    assert model.class_nodes_enabled
    assert model.class_node_registry
    assert all(len(node.metadata["members"]) >= 2 for node in model.class_node_registry.values())
    assert all(child.metadata["type"] == "word"
               for node in model.class_node_registry.values() for child in node.children())
    assert all(child.metadata.get("follower_profile")
               for node in model.class_node_registry.values() for child in node.children())
    assert all(child.mu == pytest.approx(HierarchicalNextUnitModel._project_follower_profile(
        np.asarray(child.metadata["follower_profile"]),
        child.metadata["follower_profile_features"],
    )) for node in model.class_node_registry.values() for child in node.children())
    assert all(any(child.id == node.id for child in model.pattern_forest.active_nodes())
               for node in model.class_node_registry.values())
    for node in model.class_node_registry.values():
        assert node.metadata["follower_profile_dimensions"] > 0
        assert len(node.metadata["follower_profile_mean"]) == node.metadata["follower_profile_dimensions"]
        children = node.children()
        assert node.mu == pytest.approx(np.mean([child.mu for child in children], axis=0))
        assert node.sigma == pytest.approx(
            np.var([child.mu for child in children], axis=0) + .05 ** 2
        )
    before_members = dict(model.class_labels)
    model.partial_fit(["horse sleeps", "horse sleeps"])
    assert model.class_labels == before_members

    checkpoint = tmp_path / "class-nodes.json"
    model.save(checkpoint)
    restored = HierarchicalNextUnitModel.load(checkpoint)
    assert restored.class_labels == model.class_labels
    assert restored.class_lambda == model.class_lambda
    assert restored.class_profile_features == model.class_profile_features
    assert restored.class_selection == model.class_selection
    assert restored.predict_proba(["cat"]) == pytest.approx(model.predict_proba(["cat"]))


def test_class_nodes_are_off_by_default_and_validation_can_select_no_classes():
    texts = []
    for repeat in range(60):
        for index in range(8):
            texts.append(f"the b{index} w{index} a b{(index + 3 * repeat) % 8}")
            texts.append(f"the b{index} w{(index + 1) % 8} a b{(index + 5 * repeat) % 8}")
    plain = HierarchicalNextUnitModel(max_order=2).fit(texts)
    candidate = HierarchicalNextUnitModel(max_order=3, class_nodes=True).fit(texts)
    assert plain.class_nodes_enabled is False
    assert not plain.class_node_registry
    assert candidate.class_selection is not None
    assert candidate.class_lambda == 0.0
    assert candidate.class_selection["selection_reason"].startswith("no-class fallback")
    flat = NGramBaseline(order=3, alpha=.1, smoothing="interpolated").fit(texts)
    contexts = {()}
    for text in texts[:32]:
        sequence = [BOUNDARY, BOUNDARY] + tokenize(text) + [BOUNDARY]
        contexts.update(tuple(sequence[max(0, i - 2):i]) for i in range(2, len(sequence)))
    for context in contexts:
        hpm_probs = dict(zip(candidate.vocab, candidate.predict_proba(context)))
        flat_probs = dict(zip(flat.vocab, flat.predict_proba(context)))
        assert hpm_probs == pytest.approx(flat_probs)


def test_sentence_boundaries_reset_context_and_create_linked_nodes():
    text = "cats chase mice. dogs flee birds!"
    model = HierarchicalNextUnitModel(max_order=3).fit([text])
    assert split_sentences(text) == ["cats chase mice.", "dogs flee birds!"]
    assert model.counts[("chase",)]["dogs"] == 0
    sentence = model.sentence_nodes["cats chase mice ."]
    assert sentence.metadata["type"] == "sentence"
    assert any(child.metadata["type"] == "next_unit_distribution"
               for child in sentence.children())


def test_document_split_is_deterministic_and_disjoint():
    rows = [{"id": str(i), "text": f"doc {i}"} for i in range(50)]
    split = split_documents(rows, seed=4)
    assert split == split_documents(rows, seed=4)
    ids = [{r["id"] for r in split[k]} for k in split]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert all(ids)


def test_split_keeps_conversation_groups_together():
    rows = [{"id": f"msg-{i}-{turn}", "group_id": f"conversation-{i}", "text": f"message {i} {turn}"}
            for i in range(30) for turn in range(2)]
    split = split_documents(rows, seed=29)
    group_to_split = {row["group_id"]: name for name, items in split.items() for row in items}
    assert len(group_to_split) == 30
    for row in rows:
        assert group_to_split[row["group_id"]] == next(
            name for name, items in split.items() if any(item["id"] == row["id"] for item in items)
        )


def test_chunked_huggingface_articles_stay_in_one_language_split(tmp_path):
    corpus = tmp_path / "wiki.jsonl"
    rows = []
    for article in range(30):
        for chunk in range(2):
            rows.append({
                "kind": "source", "source_id": f"article-{article}-chunk-{chunk}",
                "text": f"article {article} chunk {chunk}",
                "metadata": {
                    "huggingface_dataset_id": "wikimedia/structured-wikipedia",
                    "huggingface_revision": "rev", "huggingface_split": "train",
                    "huggingface_row_id": str(article),
                },
            })
    corpus.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    documents = read_jsonl(corpus)
    split = split_documents(documents, seed=17)
    article_to_splits = {}
    for name, items in split.items():
        for item in items:
            article_to_splits.setdefault(item["metadata"]["huggingface_row_id"], set()).add(name)
    assert len(article_to_splits) == 30
    assert all(len(names) == 1 for names in article_to_splits.values())


def test_reader_and_heldout_evaluator(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    rows = [{"source_id": f"d{i}", "text": f"cats chase mice number {i}"} for i in range(50)]
    corpus.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert len(read_jsonl(corpus)) == 50
    model = HierarchicalNextUnitModel().fit([r["text"] for r in rows[:35]])
    metrics = model.evaluate(r["text"] for r in rows[35:])
    assert metrics["tokens"] > 0
    assert metrics["perplexity"] >= 1
    assert len(tokenize("Cats chase mice.")) == 4


def test_evaluation_reports_surface_accuracy_and_oov_coverage():
    texts = ["common common word", "common pattern"]
    heldout = ["common novelword"]
    model = HierarchicalNextUnitModel(min_word_count=2).fit(texts)
    baseline = NGramBaseline(order=3, min_word_count=2).fit(texts)
    for metrics in (model.evaluate(heldout), baseline.evaluate(heldout)):
        assert metrics["target_space"] == "vocabulary_mapped"
        assert metrics["oov_word_tokens"] > 0
        assert metrics["oov_word_rate"] > 0
        assert metrics["surface_top1_accuracy"] <= metrics["top1_accuracy"]
        assert metrics["surface_top5_accuracy"] <= metrics["top5_accuracy"]


def test_checkpoint_roundtrip_preserves_word_and_character_predictions(tmp_path):
    texts = ["a small corpus", "a small sample", "the sample corpus"]
    model = HierarchicalNextUnitModel(max_order=2, alpha=10).fit(texts)
    model.char_alpha = 1
    model.character_word_mix = 0.2
    model.training_manifest = {"objective": "next-character and next-word", "train_group_count": 3}
    checkpoint = tmp_path / "language.json"
    model.save(checkpoint)
    restored = HierarchicalNextUnitModel.load(checkpoint)
    assert restored.training_manifest == model.training_manifest
    assert restored.character_word_mix == model.character_word_mix
    assert len(restored.pattern_forest) == len(restored.nodes) + 2 * restored.max_order
    assert all(node.id in restored.pattern_forest for node in restored.nodes.values())
    assert model.evaluate(texts) == restored.evaluate(texts)
    assert model.evaluate_characters(texts) == restored.evaluate_characters(texts)
    assert model.score_cross_entropy(texts) == model.evaluate(texts)["cross_entropy_nats"]
    assert model.score_surface_cross_entropy(texts) == \
        model.evaluate(texts)["surface_cross_entropy_nats"]
    assert model.score_cross_entropy(texts, character=True) == \
        model.evaluate_characters(texts)["cross_entropy_nats"]
    assert restored.raw_word_context_counts == model.raw_word_context_counts
    assert {order: model.observer.get_weight(node.id)
            for order, node in model.word_level_nodes.items()} == {
        order: restored.observer.get_weight(node.id)
        for order, node in restored.word_level_nodes.items()
    }
    assert {order: model.observer.get_weight(node.id)
            for order, node in model.char_level_nodes.items()} == {
        order: restored.observer.get_weight(node.id)
        for order, node in restored.char_level_nodes.items()
    }


def test_observer_weights_context_levels_and_changes_predictions():
    texts = ["cats chase mice", "cats chase birds", "dogs chase cats", "cats sleep"]
    hpm = HierarchicalNextUnitModel(max_order=3, alpha=10).fit(texts)
    assert hpm.observer is not None
    assert len(hpm.pattern_forest) == len(hpm.nodes) + 2 * hpm.max_order
    assert all(node.id in hpm.pattern_forest for node in hpm.nodes.values())
    assert all(node.mu.shape == (4,) for node in hpm.pattern_forest.active_nodes())
    assert all(node.children() for node in hpm.word_level_nodes.values())
    assert hpm.word_nodes["cats"].children()
    assert any(child.metadata.get("type") == "sentence"
               for child in hpm.word_level_nodes[hpm.max_order - 1].children())
    assert all(0 <= hpm.observer.get_weight(node.id) <= 1
               for node in [*hpm.word_level_nodes.values(), *hpm.char_level_nodes.values()])
    assert any(hpm.observer.get_weight(node.id) != 0.5
               for node in [*hpm.word_level_nodes.values(), *hpm.char_level_nodes.values()])
    baseline = NGramBaseline(order=3, alpha=10, smoothing="interpolated").fit(texts)
    for node in hpm.word_level_nodes.values():
        hpm.observer.set_weight(node.id, 1.0)
    assert np.allclose(hpm.predict_proba(["cats", "chase"]),
                       baseline.predict_proba(["cats", "chase"]))
    before = hpm.predict_proba(["cats", "chase"])
    for order, node in hpm.word_level_nodes.items():
        hpm.observer.set_weight(node.id, 1.0 if order == 0 else 1e-6)
    after = hpm.predict_proba(["cats", "chase"])
    assert not np.allclose(before, after)
    progress = []
    validation_docs = ["cats chase birds", "dogs chase mice"]
    validation_tuning = hpm.tune_observer_weights(
        validation_docs, passes=1, progress_callback=progress.append)
    assert validation_tuning["validation_cross_entropy_after"] <= \
        validation_tuning["validation_cross_entropy_before"]
    assert validation_tuning["calibration_events"] == validation_tuning["validation_events"]
    assert progress
    assert np.isfinite(hpm.score_cross_entropy(validation_docs, max_events=1))


def test_training_epochs_optimize_observer_on_leave_one_event_out_loss():
    training_docs = ["cats chase mice", "cats chase birds", "dogs chase cats", "cats sleep"]
    model = HierarchicalNextUnitModel(max_order=3, alpha=10).fit(training_docs)
    before = {order: model.observer.get_weight(node.id)
              for order, node in model.word_level_nodes.items()}
    progress = []

    report = model.train_observer_weights(
        training_docs, passes=3, max_events=10_000, progress_callback=progress.append,
    )

    after = {order: model.observer.get_weight(node.id)
             for order, node in model.word_level_nodes.items()}
    assert report["objective"] == "training_leave_one_event_out_next_unit_cross_entropy"
    assert report["training_events"] > 0
    assert report["optimization_events"] == report["training_events"]
    assert report["training_loo_cross_entropy_after"] <= \
        report["training_loo_cross_entropy_before"]
    assert report["selected_weights"] == {str(order): weight for order, weight in after.items()}
    assert before != after
    assert progress

    character_report = model.train_observer_weights(training_docs, character=True, passes=2)
    assert character_report["training_loo_cross_entropy_after"] <= \
        character_report["training_loo_cross_entropy_before"]

    capped_report = model.train_observer_weights(training_docs, passes=1, max_events=2)
    assert capped_report["optimization_events"] <= 2
    assert 0 < capped_report["scored_event_contexts"] < capped_report["unique_event_contexts"]


def test_character_hierarchy_composes_into_word_distribution_and_validation_tunes_mix():
    texts = ["cats chase mice", "cats chase birds", "dogs chase cats", "cats sleep"]
    hpm = HierarchicalNextUnitModel(max_order=3, alpha=10).fit(texts)
    context = ("cats", "chase")
    base = hpm.predict_proba(context)
    hpm.character_word_mix = 0.4
    composed = hpm.predict_proba(context)
    assert np.isclose(composed.sum(), 1.0)
    assert not np.allclose(base, composed)
    validation = ["cats chase birds", "dogs chase mice", "cats sleep"]
    progress = []
    tuning = hpm.tune_character_word_mix(validation, progress_callback=progress.append)
    assert tuning["selection_metric"].startswith("sampled_validation_exact_surface_word_cross_entropy")
    assert tuning["validation_events"] > 0
    assert tuning["selected_mix"] == hpm.character_word_mix
    assert tuning["validation_surface_cross_entropy_after"] <= \
        tuning["validation_surface_cross_entropy_before"]
    assert progress[-1]["completed_candidates"] == progress[-1]["total_candidates"]


def test_surface_word_probability_scores_oov_spellings_and_matches_evaluator():
    model = HierarchicalNextUnitModel(max_order=3, min_word_count=2, alpha=1.0).fit(
        ["common rare", "common shared", "common shared"]
    )
    model.char_alpha = 0.1
    model.character_word_mix = 0.2
    context = ["common"]
    assert 0.0 < model.surface_word_probability(context, "novelword") < 1.0
    assert 0.0 < model.surface_word_probability(context, "rare") < 1.0
    assert 0.0 < model.surface_word_probability(context, BOUNDARY) < 1.0
    metrics = model.evaluate(["common novelword"])
    assert metrics["surface_target_space"] == "exact_surface_word_and_eos"
    assert metrics["surface_cross_entropy_nats"] == \
        model.score_surface_cross_entropy(["common novelword"])
    assert np.isfinite(metrics["surface_perplexity"])


def test_surface_decoder_can_generate_a_rare_word_outside_mapped_vocab():
    model = HierarchicalNextUnitModel(max_order=3, min_word_count=2, alpha=1.0).fit(
        ["common rare", "common shared", "common shared"]
    )
    model.char_alpha = 0.1
    model.character_word_mix = 0.1
    contextual_prior = model._oov_spelling_prior_probability(("common",), "rare")
    backoff_prior = model._oov_spelling_prior_probability(("unseen",), "rare")
    assert contextual_prior > backoff_prior
    candidates = model.predict_surface_next_words(["common"], k=10)
    rare = next((candidate for candidate in candidates if candidate["word"] == "rare"), None)
    assert rare is not None
    assert rare["is_oov"]
    assert 0.0 < rare["probability"] < 1.0
    assert all(candidates[index]["probability"] >= candidates[index + 1]["probability"]
               for index in range(len(candidates) - 1))
    assert sum(candidate["probability"] for candidate in candidates) <= 1.0


def test_rare_word_cutoff_maps_unseen_words_into_learned_unknown_class():
    texts = ["common rare", "common shared", "common shared"]
    model = HierarchicalNextUnitModel(max_order=2, min_word_count=2).fit(texts)
    baseline = NGramBaseline(order=2, min_word_count=2).fit(texts)
    assert model.word_counts["rare"] == 1
    assert model.counts[("common",)]["<unk>"] == 1
    assert baseline.counts[("common",)]["<unk>"] == 1
    assert model.evaluate(["common novel"])["tokens"] == 3
    assert baseline.evaluate(["common novel"])["tokens"] == 3


def test_partial_fit_accumulates_new_data_and_matches_cumulative_refit(tmp_path):
    initial = ["common rare", "common shared"]
    added = ["rare common", "common rare"]
    continued = HierarchicalNextUnitModel(max_order=2, min_word_count=2).fit(initial)
    old_observer_weights = {
        order: continued.observer.get_weight(node.id)
        for order, node in continued.word_level_nodes.items()
    }
    progress = []
    continued.partial_fit(added, workers=2, progress_callback=progress.append)
    # The repeated "common rare" row is an epoch replay, not fresh evidence.
    cumulative = HierarchicalNextUnitModel(max_order=2, min_word_count=2).fit(initial + ["rare common"])
    cumulative.train_observer_weights(["rare common"])
    assert continued.word_counts == cumulative.word_counts
    assert continued.raw_word_context_counts == cumulative.raw_word_context_counts
    assert continued.counts == cumulative.counts
    assert continued.char_counts == cumulative.char_counts
    assert continued.evaluate(["rare common"]) == cumulative.evaluate(["rare common"])
    assert {
        order: continued.observer.get_weight(node.id)
        for order, node in continued.word_level_nodes.items()
    } != old_observer_weights
    stages = [event["stage"] for event in progress]
    assert stages[:4] == [
        "count_new_document_statistics", "merge_raw_statistics", "rebuild_hfn_graph",
        "recompute_observer_reliability",
    ]
    assert stages[4] == "train_observer_weights"
    assert set(stages[4:-1]) == {"train_observer_weights"}
    assert stages[-1] == "complete"
    assert progress[0]["workers"] == 1
    observer_training = progress[-1]["observer_training"]
    for unit in ("word", "character"):
        assert observer_training[unit]["training_loo_cross_entropy_after"] <= \
            observer_training[unit]["training_loo_cross_entropy_before"]

    checkpoint = tmp_path / "continued.json"
    continued.save(checkpoint)
    restored = HierarchicalNextUnitModel.load(checkpoint)
    restored.partial_fit(["shared rare"])
    cumulative.partial_fit(["shared rare"])
    assert restored.counts == cumulative.counts
    assert restored.raw_word_context_counts == cumulative.raw_word_context_counts


def test_partial_fit_uses_replay_for_observer_without_recounting_it():
    initial = ["old domain common words", "old language context" ]
    fresh = ["new domain fresh words", "fresh language context"]
    replay = ["old domain common words", "old language context"]
    no_replay = HierarchicalNextUnitModel(max_order=2).fit(initial)
    with_replay = HierarchicalNextUnitModel(max_order=2).fit(initial)
    no_replay.training_manifest = {}
    with_replay.training_manifest = {}

    no_replay.partial_fit(fresh)
    with_replay.partial_fit(fresh, observer_replay_documents=replay)

    assert with_replay.word_counts == no_replay.word_counts
    assert with_replay.raw_word_context_counts == no_replay.raw_word_context_counts
    assert with_replay.char_counts == no_replay.char_counts
    trained = with_replay.training_manifest["incremental_training_batches"][-1]
    assert trained["observer_replay_documents"] == len(replay)
    assert (trained["observer_training"]["word"]["training_events"]
            > no_replay.training_manifest["incremental_training_batches"][-1]
            .get("observer_training", {}).get("word", {}).get("training_events", 0))


def test_incremental_vocabulary_cutoff_rebuilds_hfn_before_observer_fit(tmp_path):
    initial = ["stable context words", "stable shared pattern"]
    fresh = ["fresh rareword appears", "fresh rareword repeats"]
    model = HierarchicalNextUnitModel(max_order=2, min_word_count=1).fit(initial)
    model.training_manifest = {}

    model.partial_fit(fresh, min_word_count=2)
    reference = HierarchicalNextUnitModel(max_order=2, min_word_count=2).fit(initial + fresh)

    assert model.min_word_count == 2
    assert model.vocab == reference.vocab
    assert model.counts == reference.counts
    assert set(model.context_nodes) == set(model.counts)
    metrics = model.evaluate(["fresh rareword appears", "stable shared pattern"])
    checkpoint = tmp_path / "cutoff-updated.json"
    model.save(checkpoint)
    restored = HierarchicalNextUnitModel.load(checkpoint)
    assert restored.evaluate(["fresh rareword appears", "stable shared pattern"]) == metrics


def test_partial_fit_replay_is_idempotent_and_survives_checkpoint_reload(tmp_path):
    training_docs = ["common rare", "common shared", "common shared"]
    model = HierarchicalNextUnitModel(max_order=2, min_word_count=2).fit(training_docs)
    model.training_manifest = {"objective": "next-word"}
    before_counts = {context: Counter(counts) for context, counts in model.raw_word_context_counts.items()}
    before_words = Counter(model.word_counts)
    before_vocab = list(model.vocab)
    before_metrics = model.evaluate(["common rare", "common shared"])
    before_weights = {order: model.observer.get_weight(node.id)
                      for order, node in model.word_level_nodes.items()}
    before_nodes = len(model.nodes)
    progress = []

    model.partial_fit(training_docs, progress_callback=progress.append)

    assert progress == [{"stage": "replay_skipped", "documents": 3,
                         "new_documents": 0, "hfn_nodes": before_nodes}]
    assert model.raw_word_context_counts == before_counts
    assert model.word_counts == before_words
    assert model.vocab == before_vocab
    assert len(model.nodes) == before_nodes
    assert model.evaluate(["common rare", "common shared"]) == before_metrics
    assert {order: model.observer.get_weight(node.id)
            for order, node in model.word_level_nodes.items()} == before_weights
    assert model.training_manifest["replay_training_batches"][-1]["new_documents"] == 0

    checkpoint = tmp_path / "replay-safe.json"
    model.save(checkpoint)
    restored = HierarchicalNextUnitModel.load(checkpoint)
    restored.partial_fit(training_docs)
    assert restored.raw_word_context_counts == before_counts
    assert restored.vocab == before_vocab
    assert restored.evaluate(["common rare", "common shared"]) == before_metrics

    new_docs = ["new evidence", "common rare"]
    restored.partial_fit(new_docs)
    cumulative = HierarchicalNextUnitModel(max_order=2, min_word_count=2).fit(
        training_docs + ["new evidence"]
    )
    assert restored.raw_word_context_counts == cumulative.raw_word_context_counts
    assert restored.word_counts == cumulative.word_counts


def test_legacy_checkpoint_with_collapsed_rare_contexts_rejects_lossy_continuation(tmp_path):
    training_docs = ["common rare", "common shared", "common common"]
    model = HierarchicalNextUnitModel(max_order=2, min_word_count=2).fit(training_docs)
    checkpoint = tmp_path / "legacy.json"
    model.save(checkpoint)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["format"] = "hpm-hierarchical-next-unit-v2"
    payload.pop("raw_contexts")
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    restored = HierarchicalNextUnitModel.load(checkpoint)
    with pytest.raises(ValueError, match="lacks raw lexical context counts"):
        restored.partial_fit(["new data"])
    recovery = restored.recover_raw_training_contexts(training_docs)
    assert recovery["recovered"] == 1
    restored.partial_fit(["new data"])


def test_ngram_fast_log_loss_matches_full_evaluator():
    texts = ["a small test. another line", "a different test!"]
    for smoothing in ("add_alpha", "interpolated"):
        baseline = NGramBaseline(order=3, alpha=10, smoothing=smoothing).fit(texts)
        full = baseline.evaluate(texts)["cross_entropy_nats"]
        assert np.isclose(baseline.score_cross_entropy(texts), full, atol=1e-12)


def test_parallel_validation_sweep_matches_serial():
    texts = ["cats chase mice", "cats chase birds", "dogs chase cats", "cats sleep"]
    model = HierarchicalNextUnitModel(max_order=2).fit(texts)
    validation = ["cats chase cats", "dogs sleep"]
    points = [("words", alpha) for alpha in (0.1, 1, 10)]
    serial = _run_validation_sweeps(model, validation, points, workers=1)
    parallel = _run_validation_sweeps(model, validation, points, workers=2)
    assert parallel == serial
    baseline = NGramBaseline(order=2).fit(texts)
    baseline_points = [("words", alpha) for alpha in (0.1, 1, 10)]
    serial_baseline = _run_validation_sweeps(baseline, validation, baseline_points, workers=1)
    parallel_baseline = _run_validation_sweeps(baseline, validation, baseline_points, workers=2)
    assert parallel_baseline == serial_baseline
    models = {"hpm": model, "baseline": baseline}
    documents = {"validation": validation}
    jobs = [("hpm", "hpm", "words", "validation"),
            ("chars", "hpm", "characters", "validation"),
            ("baseline", "baseline", "words", "validation")]
    serial_metrics = dict(_run_metric_jobs(models, documents, jobs, workers=1))
    parallel_metrics = dict(_run_metric_jobs(models, documents, jobs, workers=2))
    assert parallel_metrics == serial_metrics


def test_parallel_fit_matches_serial_counts_and_predictions():
    texts = [f"document {i} cats chase mice; repeated corpus token {i % 3}. " * 12
             for i in range(40)]
    serial = HierarchicalNextUnitModel(max_order=2).fit(texts)
    parallel = HierarchicalNextUnitModel(max_order=2).fit(texts, workers=2)
    assert parallel.word_counts == serial.word_counts
    assert parallel.counts == serial.counts
    assert parallel.char_counts == serial.char_counts
    assert parallel.vocab == serial.vocab
    assert parallel.evaluate(texts[:3]) == serial.evaluate(texts[:3])


def test_pilot_persists_checkpoint_and_terminal_status(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"source_id": f"d{i}", "text": f"short text number {i}"})
                              for i in range(40)), encoding="utf-8")
    out = tmp_path / "run"
    result = run_pilot(corpus, out, seed=4, max_order=1, workers=1)
    status = json.loads((out / "run_status.json").read_text(encoding="utf-8"))
    checkpoint = HierarchicalNextUnitModel.load(out / "hpm_language_checkpoint.json")
    assert status["stage"] == "complete"
    assert status["state"] == "complete"
    assert checkpoint.training_manifest["train_group_count"] > 0
    assert (out / "metrics.json").is_file()
    assert result["checkpoint_roundtrip"]["word_test_metrics_match"]
    comparison = result["comparison_to_interpolated_ngram"]
    assert "cross_entropy_delta_nats_hpm_minus_control" in comparison
    assert result["observer_weights"]["observer_level_nodes"] == 2
    assert result["observer_weights"]["weighted_pattern_links"] > 0
    assert result["character_to_word_composition"]["validation_events"] > 0
    assert checkpoint.character_word_mix == result["character_to_word_composition"]["selected_mix"]
    assert "normalized character spelling likelihood" in checkpoint.training_manifest["objective"]
    assert "training-only leave-one-event-out loss" in checkpoint.training_manifest["objective"]


def test_language_pilot_cli_exits_with_compact_summary(tmp_path):
    corpus = tmp_path / "cli-corpus.jsonl"
    corpus.write_text("\n".join(json.dumps({"source_id": f"d{i}", "text": f"small corpus line {i}"})
                              for i in range(40)), encoding="utf-8")
    out = tmp_path / "cli-run"
    completed = subprocess.run(
        [sys.executable, "-m", "hpm_ai_v2.experiments.train_language_pilot", "--corpus", str(corpus),
         "--output", str(out), "--order", "1", "--workers", "2", "--min-word-count", "2"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    summary = json.loads(completed.stdout)
    assert summary["state"] == "complete"
    assert summary["min_word_count"] == 2
    assert summary["checkpoint_roundtrip"]["word_test_metrics_match"]
    assert (out / "metrics.json").is_file()
    assert (out / "hpm_language_checkpoint.json").is_file()


def test_l5_prequential_word_order_meta_learns_after_scoring_and_roundtrips(tmp_path):
    model = HierarchicalNextUnitModel(max_order=3, alpha=0.1).fit([
        "cats chase mice", "cats chase birds", "dogs chase mice", "cats sleep",
    ])
    calibration = model.fit_l5_meta(["cats chase mice. cats chase birds."], eta=0.2)
    assert calibration["events"] > 0
    assert calibration["calibration_split"] == "validation"
    assert calibration["state"]["updates"] == calibration["events"]
    assert set(calibration["weights"]) == {"observer", "word_order_0", "word_order_1", "word_order_2"}
    assert sum(calibration["weights"].values()) == pytest.approx(1.0)

    saved_state = json.dumps(model.l5_language_state, sort_keys=True)
    scored = model.evaluate_l5_prequential(["cats chase mice."])
    assert scored["ending_updates"] > scored["starting_updates"]
    assert json.dumps(model.l5_language_state, sort_keys=True) == saved_state

    checkpoint = tmp_path / "l5-language.json"
    model.save(checkpoint)
    restored = HierarchicalNextUnitModel.load(checkpoint)
    assert restored.l5_language_state == model.l5_language_state
    assert restored.l5_calibration_document_hashes == model.l5_calibration_document_hashes
    resumed = restored.evaluate_l5_prequential(["cats chase birds."])
    assert resumed["starting_updates"] == model.l5_language_state["updates"]


def test_l5_prequential_mixer_updates_only_after_forecast_and_rejects_bad_state():
    from hpm_ai_v2.training.l5_language_meta import PrequentialL5Mixer

    mixer = PrequentialL5Mixer(["good", "bad"], eta=0.5)
    forecasts = {"good": 0.8, "bad": 0.2}
    before = mixer.weights()
    expected = before["good"] * 0.8 + before["bad"] * 0.2
    assert mixer.update_after_score(forecasts) == pytest.approx(expected)
    assert mixer.updates == 1
    assert mixer.weights()["good"] > before["good"]
    restored = PrequentialL5Mixer.from_state_dict(mixer.state_dict())
    assert restored.state_dict() == mixer.state_dict()
    with pytest.raises(ValueError, match="every configured expert"):
        mixer.probability({"good": 0.5})


def test_l5_learning_rate_is_selected_on_validation_only():
    model = HierarchicalNextUnitModel(max_order=3, alpha=0.1).fit([
        "red birds fly south", "red birds sing", "blue birds fly north",
    ])
    selected = model.tune_l5_meta(
        ["red birds fly south", "blue birds fly north"],
        eta_candidates=(0.0001, 0.001),
    )
    assert selected["selection_split"] == "validation"
    assert selected["selected_eta"] in {0.0001, 0.001}
    assert selected["selected_prequential_cross_entropy_nats"] == min(
        item["prequential_cross_entropy_nats"] for item in selected["sweep"]
    )
    assert model.l5_language_state["eta"] == selected["selected_eta"]


def test_l5_calibration_continues_across_new_rounds_and_rejects_replay():
    model = HierarchicalNextUnitModel(max_order=2).fit([
        "amber birds travel", "blue birds sing",
    ])
    model.fit_l5_meta(["golden birds migrate"], eta=0.001)
    previous_updates = model.l5_language_state["updates"]
    model.partial_fit(["green birds fly", "green birds glide"])
    continued = model.continue_l5_meta(["azure birds migrate"])
    assert continued["previous_updates"] == previous_updates
    assert continued["cumulative_updates"] > previous_updates
    assert len(model.l5_calibration_document_hashes) == 2
    with pytest.raises(ValueError, match="exact-text replay"):
        model.continue_l5_meta(["golden birds migrate"])


def test_prequential_l5_fixed_share_is_opt_in_and_checkpointed():
    from hpm_ai_v2.training.l5_language_meta import PrequentialL5Mixer

    legacy = PrequentialL5Mixer(["recent", "stable"], eta=.4, fixed_share=0.0)
    assert legacy.fixed_share == 0.0
    restored_legacy = PrequentialL5Mixer.from_state_dict(legacy.state_dict())
    assert restored_legacy.fixed_share == 0.0

    assert PrequentialL5Mixer(["recent", "stable"]).fixed_share == .02

    mixer = PrequentialL5Mixer(["recent", "stable"], eta=.4, fixed_share=.05)
    for _ in range(100):
        mixer.update_after_score({"recent": .001, "stable": .8})
    assert mixer.weights()["stable"] >= .025
    restored = PrequentialL5Mixer.from_state_dict(mixer.state_dict())
    assert restored.fixed_share == .05
    assert restored.weights() == pytest.approx(mixer.weights())
