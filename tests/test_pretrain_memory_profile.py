import json

from hpm_ai_v2.experiments.profile_pretrain_memory import _bounded_input, profile


def test_bounded_input_hashes_only_the_selected_rows(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(' {"text":"one"}\n\n{"text":"two"}\n{"text":"three"}\n',
                      encoding="utf-8")

    sampled, digest, rows = _bounded_input(corpus, 2)

    assert rows == 2
    assert len(digest) == 64
    assert sampled.read_text(encoding="utf-8").splitlines() == [' {"text":"one"}', '{"text":"two"}']
    sampled.unlink()


def test_profile_reports_retained_indexes_and_peak_process_memory(tmp_path):
    corpus = tmp_path / "sources.jsonl"
    sentences = (
        "Mitochondria generate chemical energy for a cell.",
        "Mitochondria also help regulate energy use in the cell.",
    )
    corpus.write_text("\n".join(json.dumps({
        "kind": "source", "source_id": f"source-{index}", "text": sentence,
        "title": "Cell biology", "metadata": {"source_type": "book", "license": "Public domain"},
    }) for index, sentence in enumerate(sentences)) + "\n", encoding="utf-8")

    report = profile(corpus, tmp_path / "checkpoint", max_rows=2, hot_cap=2)
    memory = report["memory_profile"]

    assert report["source_documents"] == 2
    assert memory["forest_vector_index_nodes"] == report["forest_node_count"]
    assert memory["forest_vector_index_bytes"] > 0
    assert memory["forest_hot_node_count"] <= 2
    assert memory["pattern_index_loaded_object_count"] == 0
    assert memory["peak_process_rss_bytes"] > 0
    assert report["checkpoint_shared_term_links_verified"] is True
