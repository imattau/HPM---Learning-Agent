import json

from hpm_ai_v2.training.prepare_pretraining_corpus import prepare_corpus
from hpm_ai_v2.training.study import load_curriculum_jsonl


def test_pretraining_corpus_supports_books_wikis_conversations_web_and_code(tmp_path):
    books = tmp_path / "books"
    wiki = tmp_path / "wikipedia"
    conversations = tmp_path / "conversations"
    websites = tmp_path / "websites"
    code = tmp_path / "code"
    for folder in (books, wiki, conversations, websites, code):
        folder.mkdir()
    (books / "primer.txt").write_text("A book chapter explains cells and energy.", encoding="utf-8")
    (wiki / "page.jsonl").write_text(json.dumps({
        "title": "Cell", "text": "A wiki article describes the structure of a cell."
    }) + "\n", encoding="utf-8")
    (conversations / "dialogue.jsonl").write_text(json.dumps({
        "id": "dialogue-1", "messages": [
            {"role": "user", "content": "What powers the cell?"},
            {"role": "assistant", "content": "Mitochondria generate chemical energy."},
        ],
    }) + "\n", encoding="utf-8")
    (websites / "article.html").write_text(
        "<html><script>ignore this</script><body><h1>Cells</h1><p>Cells use energy.</p></body></html>",
        encoding="utf-8",
    )
    (code / "cell.py").write_text("def cell_energy():\n    return 'ATP'\n", encoding="utf-8")

    out = tmp_path / "corpus.jsonl"
    report = prepare_corpus(
        [tmp_path], output=out, license_name="CC-BY-4.0", max_chars=1000
    )
    sources, episodes = load_curriculum_jsonl(out)

    assert episodes == []
    assert report["source_document_count"] == 5
    assert set(report["source_type_counts"]) == {
        "book", "wiki", "conversation", "website", "code",
    }
    assert all(source.metadata["license"] == "CC-BY-4.0" for source in sources)
    assert all(source.metadata["content_sha256"] for source in sources)
    conversation = next(source for source in sources if source.metadata["source_type"] == "conversation")
    assert "user: What powers the cell?" in conversation.text
    website = next(source for source in sources if source.metadata["source_type"] == "website")
    assert "ignore this" not in website.text


def test_pretraining_corpus_chunks_large_documents_and_preserves_source_metadata(tmp_path):
    source = tmp_path / "books" / "long.txt"
    source.parent.mkdir()
    source.write_text("A useful sentence. " * 100, encoding="utf-8")
    out = tmp_path / "out.jsonl"

    report = prepare_corpus([source], output=out, license_name="Public domain", max_chars=300)
    sources, _ = load_curriculum_jsonl(out)

    assert report["source_document_count"] > 1
    assert len(sources) == report["source_document_count"]
    assert all(source.metadata["source_type"] == "book" for source in sources)
    assert all(len(source.text) <= 300 for source in sources)


def test_huggingface_rows_keep_dataset_revision_license_and_row_provenance(tmp_path, monkeypatch):
    import datasets

    def fake_load_dataset(dataset_id, **kwargs):
        assert dataset_id == "OpenAssistant/oasst1"
        assert kwargs["revision"] == "abc123"
        return [{
            "message_id": "message-7",
            "parent_id": "message-6",
            "message_tree_id": "tree-3",
            "text": "A user asks a question. An assistant answers it.",
            "role": "assistant",
        }]

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    output = tmp_path / "oasst.jsonl"
    report = prepare_corpus(
        output=output,
        hf_dataset="OpenAssistant/oasst1",
        hf_revision="abc123",
        license_name="Apache-2.0",
        source_type="conversation",
        hf_max_rows=3,
    )
    sources, episodes = load_curriculum_jsonl(output)

    assert episodes == []
    assert report["huggingface_rows_read"] == 1
    assert sources[0].metadata["huggingface_dataset_id"] == "OpenAssistant/oasst1"
    assert sources[0].metadata["huggingface_revision"] == "abc123"
    assert sources[0].metadata["huggingface_row_id"] == "message-7"
    assert sources[0].metadata["license"] == "Apache-2.0"
    assert sources[0].metadata["parent_id"] == "message-6"
    assert sources[0].metadata["conversation_id"] == "tree-3"


def test_huggingface_wikipedia_sections_and_row_license_are_preserved(tmp_path, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: [{
        "identifier": 123,
        "name": "Cell biology",
        "url": "https://en.wikipedia.org/wiki/Cell_biology",
        "license": "CC-BY-SA-4.0",
        "language": "en",
        "sections": json.dumps([
            {"heading": "Cell", "content": "A cell is the basic unit of life."},
            {"heading": "Energy", "content": "Mitochondria produce cellular energy."},
        ]),
    }])
    output = tmp_path / "wiki.jsonl"
    prepare_corpus(
        output=output, hf_dataset="wikimedia/structured-wikipedia",
        hf_config="enwiki_namespace_0", hf_revision="pinned-revision",
        license_name="per-row", source_type="wiki", hf_text_columns=("sections",),
        hf_language="en",
    )
    sources, _ = load_curriculum_jsonl(output)

    assert len(sources) == 1
    assert "Cell: A cell is the basic unit of life." in sources[0].text
    assert "Energy: Mitochondria produce cellular energy." in sources[0].text
    assert sources[0].metadata["license"] == "CC-BY-SA-4.0"
    assert sources[0].metadata["source_url"] == "https://en.wikipedia.org/wiki/Cell_biology"


def test_huggingface_license_can_come_from_a_declared_row_field(tmp_path, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: [{
        "id": "book-1", "title": "Public domain work", "text": "A book excerpt.",
        "rights": "Public domain in source jurisdiction",
    }])
    output = tmp_path / "book.jsonl"
    prepare_corpus(
        output=output, hf_dataset="example/books", hf_revision="pinned-revision",
        license_name="per-row", hf_license_column="rights", source_type="book",
    )
    sources, _ = load_curriculum_jsonl(output)
    assert sources[0].metadata["license"] == "Public domain in source jurisdiction"
    assert sources[0].metadata["license_source_column"] == "rights"


def test_huggingface_per_row_license_mode_rejects_unlicensed_rows(tmp_path, monkeypatch):
    import datasets
    import pytest

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: [{
        "id": "row-without-license", "text": "Text without a row-level license."
    }])
    with pytest.raises(ValueError, match="no row-level license"):
        prepare_corpus(
            output=tmp_path / "mixed.jsonl", hf_dataset="example/mixed-license",
            hf_revision="pinned-revision", license_name="per-row",
        )


def test_huggingface_conversation_parent_links_become_hfn_edges(tmp_path, monkeypatch):
    import datasets
    from hpm_ai_v2.agents.reader_agent import ReaderAgent
    from hpm_ai_v2.domains.text_domain import TextDomainConfig
    from hpm_ai_v2.training.study import SupervisedStudyTrainer

    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: [
        {"message_id": "root", "text": "How do plants use light?", "role": "prompter", "message_tree_id": "tree"},
        {"message_id": "reply", "parent_id": "root", "text": "They capture light for photosynthesis.", "role": "assistant", "message_tree_id": "tree"},
    ])
    corpus_path = tmp_path / "conversation.jsonl"
    prepare_corpus(
        output=corpus_path, hf_dataset="example/chat", hf_revision="rev-1",
        license_name="Apache-2.0", source_type="conversation",
    )
    sources, _ = load_curriculum_jsonl(corpus_path)
    agent = ReaderAgent(
        TextDomainConfig.from_passages([source.text for source in sources]),
        dynamic_vocab=False, inject_blank_priors=False, cold_dir=str(tmp_path / "hfn"),
    )
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)

    child = next(source for source in sources if source.metadata["huggingface_row_id"] == "reply")
    child_node = agent.forest.get(trainer.source_to_node[child.source_id])
    assert any(edge.relation == "conversation_reply_to" for edge in child_node.edges())


def test_source_only_hfn_checkpoint_can_be_continued_with_supervision(tmp_path):
    from hpm_ai_v2.experiments.run_cumulative_study import run_cumulative
    from hpm_ai_v2.agents.reader_agent import ReaderAgent
    from hpm_ai_v2.training.pretrain_hfn_corpus import pretrain_corpus

    corpus = tmp_path / "seed.jsonl"
    sentence = "Mitochondria generate much of the chemical energy that powers a cell."
    related_sentence = "Mitochondria also help regulate energy use in the cell."
    corpus.write_text(json.dumps({
        "kind": "source", "source_id": "mitochondria", "text": sentence,
        "title": "Cell biology", "metadata": {"source_type": "book", "license": "Public domain"},
    }) + "\n" + json.dumps({
        "kind": "source", "source_id": "mitochondria-regulation", "text": related_sentence,
        "title": "Cell biology", "metadata": {"source_type": "book", "license": "Public domain"},
    }) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "pretrained-checkpoint"
    report = pretrain_corpus([corpus], checkpoint)
    assert report["source_documents"] == 2
    assert report["fact_nodes_before_supervision"] == 0
    assert report["fit_episode_count"] == 0
    assert report["agent_state_bytes"] < 5_000_000
    assert report["compact_pattern_reference_count"] == report["pattern_count"]
    assert report["shared_term_node_count"] > 0
    assert report["shared_term_sentence_links"] >= 2
    restored_agent = ReaderAgent.load_agent(checkpoint)
    sentence_node = next(
        node for node in restored_agent.patterns.values()
        if getattr(node, "relation_type", None) == "sentence"
    )
    term_node = restored_agent.patterns["term_mitochondria"]
    assert sentence_node.children()
    assert len(term_node.children()) >= 2

    phase = tmp_path / "phase.jsonl"
    phase.write_text("\n".join([
        json.dumps({
            "kind": "source", "source_id": "mitochondria", "text": sentence,
            "title": "Cell biology", "metadata": {"source_type": "book", "license": "Public domain"},
        }),
        json.dumps({
            "kind": "episode", "episode_id": "mito-train", "split": "train",
            "task_type": "grounded_qa", "question": "What generates cell energy?",
            "evidence_id": "mitochondria", "expected_answer": sentence,
        }),
        json.dumps({
            "kind": "episode", "episode_id": "mito-validation", "split": "validation",
            "task_type": "grounded_qa", "question": "Which organelle makes chemical energy?",
            "evidence_id": "mitochondria", "expected_answer": sentence,
        }),
        json.dumps({
            "kind": "episode", "episode_id": "mito-test", "split": "test",
            "task_type": "grounded_qa", "question": "What powers a cell chemically?",
            "evidence_id": "mitochondria", "expected_answer": sentence,
        }),
    ]) + "\n", encoding="utf-8")
    result = run_cumulative(
        [phase], None, tmp_path / "supervised",
        initial_checkpoint=checkpoint,
    )

    assert result["knowledge_growth"]["source_documents"] == 2
    assert result["training"]["phases"][0]["fit"]["fact_validation_gate"]["passed"] is True


def test_huggingface_dataset_requires_an_explicit_license(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="declared license"):
        prepare_corpus(
            output=tmp_path / "out.jsonl", hf_dataset="example/dataset",
        )
