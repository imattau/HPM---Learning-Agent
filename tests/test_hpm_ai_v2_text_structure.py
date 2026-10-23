import json

import pytest

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.text_structure import StructureIndex, content_stems, stem
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import SourceDocument, SupervisedStudyTrainer
from hpm_ai_v2.training.storage import storage_problems

PHOTO = (
    "Photosynthesis is the process plants use to convert light energy into chemical energy. "
    "It takes place in the chloroplasts, which contain the pigment chlorophyll. "
    "The Calvin cycle then uses ATP and NADPH to fix carbon dioxide into glucose.\n\n"
    "Oxygen is released as a by-product when water molecules are split. "
    "Factors such as light intensity limit the rate."
)
RESP = "Cellular respiration releases energy from glucose using oxygen. It produces ATP and carbon dioxide."


def _trainer(tmp_path, sources=None, name="forest"):
    sources = sources or [SourceDocument("photo", PHOTO, "Photosynthesis"), SourceDocument("resp", RESP, "Respiration")]
    config = TextDomainConfig.from_passages([s.text for s in sources], max_vocab=300)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / name))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    return trainer


def test_stemming_unifies_word_forms():
    assert stem("released") == stem("release") == stem("releasing")
    assert content_stems("Oxygen is released as a by-product") == ["oxygen", "releas", "product"]


def test_sources_are_stored_as_a_passage_paragraph_sentence_polygraph(tmp_path):
    trainer = _trainer(tmp_path)
    assert storage_problems(trainer) == []
    passage = trainer.agent.patterns[trainer.source_to_node["photo"]]
    paragraphs = passage.children()
    assert [len(p.children()) for p in paragraphs] == [3, 2]            # blank line splits paragraphs
    assert [(e.source.id, e.target.id) for e in passage.edges()] == [(paragraphs[0].id, paragraphs[1].id)]
    for paragraph in paragraphs:
        sentences = paragraph.children()
        assert [e.relation for e in paragraph.edges()] == ["next"] * (len(sentences) - 1)
        for sentence in sentences:
            meta = sentence.metadata
            assert PHOTO[meta["start"]:meta["end"]] == trainer.answer_node_data[sentence.id]["text"]
            assert sentence.children() and meta["stems"]                  # words below, stems recorded


def test_there_is_one_node_per_sentence_not_a_flat_copy(tmp_path):
    trainer = _trainer(tmp_path)
    kinds = [n.relation_type for n in trainer.agent.patterns.values()]
    assert kinds.count("sentence") == sum(len(ids) for ids in trainer.source_to_answer_nodes.values()) == 7
    assert "answer_sentence" not in kinds
    for node_id, data in trainer.answer_node_data.items():
        node = trainer.agent.patterns[node_id]
        assert node.relation_type == "sentence" and node.metadata["start"] == data["start"]
        assert (node.mu == trainer.agent.config.encode_passage(data["text"])).all()
        assert trainer.agent.observer.get_weight(node_id) == pytest.approx(0.5)


def test_node_ids_are_deterministic_across_independent_builds(tmp_path):
    a, b = _trainer(tmp_path, name="a"), _trainer(tmp_path, name="b")
    shape = lambda t: sorted((n.id, [c.id for c in n.children()]) for n in t.agent.patterns.values()
                             if n.relation_type in {"sentence", "paragraph", "term"})
    assert shape(a) == shape(b)


def test_shared_stems_link_sentences_across_sources_as_term_nodes(tmp_path):
    trainer = _trainer(tmp_path)
    index = StructureIndex(trainer.agent)
    glucose = trainer.agent.patterns["term_" + stem("glucose")]
    assert {c.metadata["source_id"] for c in glucose.children()} == {"photo", "resp"}
    assert "term_" + stem("calvin") not in trainer.agent.patterns        # seen once: no link value yet
    assert {trainer.answer_node_data[i]["source_id"] for i in index.sentences_for_term(stem("energy"))} == {"photo", "resp"}
    calvin = next(i for i, d in trainer.answer_node_data.items() if d["text"].startswith("The Calvin"))
    related = index.related(calvin)
    assert {trainer.answer_node_data[i]["source_id"] for i in related} == {"photo", "resp"}
    assert all(count >= 1 for count in related.values())


def test_released_and_release_meet_in_one_concept(tmp_path):
    trainer = _trainer(tmp_path)
    term = trainer.agent.patterns["term_" + stem("release")]
    texts = sorted(trainer.answer_node_data[c.id]["text"][:20] for c in term.children())
    assert texts == ["Cellular respiration", "Oxygen is released a"]


def test_structure_index_reads_order_across_paragraphs(tmp_path):
    trainer = _trainer(tmp_path)
    index = StructureIndex(trainer.agent)
    order = index.sentence_ids("photo")
    assert order == trainer.source_to_answer_nodes["photo"] and len(order) == 5
    assert index.source_of(order[3]) == "photo" and index.source_of("nope") is None
    assert index.neighbours(order[2]) == [order[1], order[3]]            # crosses the paragraph break
    assert index.neighbours(order[0]) == [order[1]]
    assert index.neighbours(order[2], before=2, after=2) == [order[0], order[1], order[3], order[4]]


def test_incremental_ingest_links_to_existing_concepts_without_duplicates(tmp_path):
    first = SourceDocument("photo", PHOTO, "Photosynthesis")
    trainer = _trainer(tmp_path, [first, SourceDocument("resp", RESP)], name="inc")
    before = {n.id for n in trainer.agent.patterns.values() if n.relation_type == "term"}
    extra = SourceDocument("sugar", "Glucose is a simple sugar. Plants store energy as starch made from glucose.", "Sugar")
    trainer.ingest([extra])
    assert storage_problems(trainer) == []
    glucose = trainer.agent.patterns["term_" + stem("glucose")]
    assert {c.metadata["source_id"] for c in glucose.children()} == {"photo", "resp", "sugar"}
    ids = [c.id for c in glucose.children()]
    assert len(ids) == len(set(ids))
    assert before <= {n.id for n in trainer.agent.patterns.values() if n.relation_type == "term"}


def test_structure_survives_a_checkpoint_round_trip_and_keeps_growing(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.save_checkpoint(tmp_path / "ck")
    state = json.loads((tmp_path / "ck" / "study_training_state.json").read_text())
    assert state["storage_version"] == 2
    restored = SupervisedStudyTrainer(ReaderAgent.load_agent(str(tmp_path / "ck")))
    restored.load_checkpoint_state(tmp_path / "ck")
    assert restored.storage_version == 2 and storage_problems(restored) == []
    shape = lambda t: sorted((n.id, [c.id for c in n.children()], [(e.source.id, e.target.id, e.relation) for e in n.edges()])
                             for n in t.agent.patterns.values() if n.relation_type in {"sentence", "paragraph", "term", "passage"})
    assert shape(restored) == shape(trainer)
    assert StructureIndex(restored.agent).related(restored.source_to_answer_nodes["photo"][0]) == \
        StructureIndex(trainer.agent).related(trainer.source_to_answer_nodes["photo"][0])
    restored.ingest([SourceDocument("sugar", "Glucose is a simple sugar made by plants.", "Sugar")])
    assert storage_problems(restored) == []
    assert "sugar" in {c.metadata["source_id"] for c in restored.agent.patterns["term_" + stem("glucose")].children()}


def test_legacy_checkpoints_are_reported_not_misread(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.save_checkpoint(tmp_path / "ck")
    path = tmp_path / "ck" / "study_training_state.json"
    state = json.loads(path.read_text()); del state["storage_version"]; path.write_text(json.dumps(state))
    legacy = SupervisedStudyTrainer(ReaderAgent.load_agent(str(tmp_path / "ck")))
    legacy.load_checkpoint_state(tmp_path / "ck")
    assert legacy.storage_version == 1
    assert "legacy storage" in storage_problems(legacy)[0]


def test_the_checker_catches_each_kind_of_damage(tmp_path):
    def broken(mutate):
        trainer = _trainer(tmp_path, name=f"x{mutate.__name__}")
        mutate(trainer)
        return " | ".join(storage_problems(trainer))

    def drop_edge(t):
        paragraph = t.agent.patterns[t.source_to_node["photo"]].children()[0]
        paragraph._edges.pop()

    def bad_offset(t):
        next(iter(t.answer_node_data.values()))["text"] = "not the sentence"

    def orphan(t):
        node_id = t.source_to_answer_nodes["resp"][0]
        t.agent.patterns[t.source_to_node["resp"]].children()[0]._children.pop(0)
        t.agent.patterns.pop(node_id, None)

    def wrong_term_child(t):
        term = t.agent.patterns["term_" + stem("glucose")]
        term._children.append(t.agent.patterns[t.source_to_answer_nodes["resp"][1]])

    def stale_index(t):
        t.agent.term_sentences["brandnew"] = ["a", "b"]

    assert "next edges" in broken(drop_edge)
    assert "offsets" in broken(bad_offset)
    assert "do not match" in broken(orphan) or "missing" in broken(orphan)
    assert "does not mention the stem" in broken(wrong_term_child)
    assert "have no node" in broken(stale_index)


def test_stem_lookup_bridges_word_forms_even_for_single_mention_words(tmp_path):
    trainer = _trainer(tmp_path)
    index = StructureIndex(trainer.agent)
    oxygen = next(i for i, d in trainer.answer_node_data.items() if d["text"].startswith("Oxygen is released"))
    assert index.terms_of(oxygen) == [stem("oxygen"), stem("release")]   # shared with respiration
    water = index.sentences_with_stem(stem("water"))
    assert water == [oxygen] and "term_" + stem("water") not in trainer.agent.patterns
    hits = index.sentences_for_query("Why do plants release oxygen?")
    assert oxygen in hits and hits[oxygen] == 2                           # "release" ~ "released" and "oxygen"
    assert index.sentences_for_query("quantum chromodynamics") == {}
