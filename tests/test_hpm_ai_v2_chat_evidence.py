import tempfile
from pathlib import Path

import pytest

from hpm_ai_v2.chat import ChatSession
from hpm_ai_v2.chat.evidence import AccessParams, EvidenceSelector, ExpandingPlanner
from hpm_ai_v2.chat.planner import GroundedPlanner
from hpm_ai_v2.experiments.run_coverage_eval import DEFAULT_SET, build_trainer
from hpm_ai_v2.training.coverage import load_coverage_set, sentence_keys


@pytest.fixture(scope="module")
def trainer():
    sources, _ = load_coverage_set(DEFAULT_SET)
    with tempfile.TemporaryDirectory() as work:
        yield build_trainer(sources, work)


def _ask(trainer, question, params=None, scope=None):
    session = ChatSession(trainer, planner=ExpandingPlanner(trainer, params=params))
    reply = session.send(question, scope_source_id=scope)
    spans = [(c.source_id, c.start, c.end) for c in reply.citations]
    return reply, sentence_keys(trainer, spans)


def test_every_expanded_claim_is_an_exact_verified_source_sentence(trainer):
    for question in ["How does photosynthesis work?", "What is subduction?", "Summarize Neurons",
                     "How are photosynthesis and respiration related through oxygen?"]:
        reply, _ = _ask(trainer, question)
        assert reply.verified and reply.plan.act == "answer"
        for claim in reply.plan.claims:
            c = claim.citations[0]
            assert trainer.sources[c.source_id].text[c.start:c.end] == c.text == claim.text


def test_claims_come_back_in_reading_order(trainer):
    reply, keys = _ask(trainer, "How does photosynthesis work?")
    order = [(c.source_id, c.start) for c in reply.citations]
    assert order == sorted(order, key=lambda item: (item[0], item[1]))


def test_a_sentence_in_another_source_is_added_when_the_question_needs_it(trainer):
    _, keys = _ask(trainer, "How are photosynthesis and respiration related through oxygen?")
    assert {s for s, _ in keys} == {"photo", "resp"}


def test_a_stray_word_that_only_an_unrelated_source_uses_does_not_pull_it_in(trainer):
    # "work" appears only in the immune source ("Vaccines work by ...")
    _, keys = _ask(trainer, "How does photosynthesis work?")
    assert {s for s, _ in keys} == {"photo"}


def test_what_is_x_is_answered_where_x_is_introduced(trainer):
    _, keys = _ask(trainer, "What is subduction?")
    assert keys == {("tect", 5)}
    _, keys = _ask(trainer, "What is a synapse?")
    assert keys == {("neuro", 4)}


def test_process_questions_grow_but_plain_facts_do_not(trainer):
    _, process = _ask(trainer, "How does photosynthesis work?")
    assert len(process) >= 3 and {s for s, _ in process} == {"photo"}
    assert {i for _, i in process} >= {0, 2, 3}                       # the definition, light reactions and Calvin cycle
    _, fact = _ask(trainer, "What do phagocytes do?")
    assert len(fact) == 1
    _, capped = _ask(trainer, "How does photosynthesis work?", AccessParams(cap=2))
    assert len(capped) <= 2


def test_the_gain_threshold_limits_structural_growth(trainer):
    question = "How does photosynthesis work?"
    loose = _ask(trainer, question, AccessParams(theta_struct=0.01, cap=9))[1]
    default = _ask(trainer, question)[1]
    strict = _ask(trainer, question, AccessParams(theta_struct=0.9))[1]
    assert len(strict) < len(default) <= len(loose)
    assert len(strict) == 1                                           # nothing clears the bar: the seed alone


def test_ablation_switches_turn_each_signal_off(trainer):
    question = "How are photosynthesis and respiration related through oxygen?"
    assert {s for s, _ in _ask(trainer, question, AccessParams(expand_coverage=False, expand_structure=False))[1]} != {"photo", "resp"}
    process = "How does photosynthesis work?"
    none = _ask(trainer, process, AccessParams(expand_structure=False, expand_coverage=False))[1]
    assert len(none) == 1


def test_summaries_cover_every_paragraph_within_the_cap(trainer):
    for title, source_id in [("Neurons", "neuro"), ("Plate tectonics", "tect")]:
        reply, keys = _ask(trainer, f"Summarize {title}")
        n = len(trainer.source_to_answer_nodes[source_id])
        assert 3 <= len(keys) <= 6 and len(keys) < n
        assert {s for s, _ in keys} == {source_id}
        index = EvidenceSelector(trainer).index()
        paragraphs = index.paragraphs(source_id)
        ordinals = {i for _, i in keys}
        order = index.sentence_ids(source_id)
        for paragraph in paragraphs:
            members = {order.index(child.id) for child in paragraph.children()}
            assert members & ordinals, f"{title}: a paragraph is not represented"


def test_scope_keeps_every_sentence_in_the_chosen_source(trainer):
    _, keys = _ask(trainer, "How are photosynthesis and respiration related through oxygen?", scope="photo")
    assert {s for s, _ in keys} == {"photo"}


def test_a_users_correction_is_never_replaced_by_the_selector(trainer):
    selector = EvidenceSelector(trainer)
    node = trainer.source_to_answer_nodes["photo"][7]
    trainer.fact_node_data["study_fact_test"] = {
        "user_id": "alice", "correction_episode_ids": ["c"], "answer_sentence_ids": [node],
    }
    try:
        assert selector._is_correction_seed(node, "alice")
        assert not selector._is_correction_seed(node, "bob")
        assert not selector._is_correction_seed(trainer.source_to_answer_nodes["photo"][0], "alice")
    finally:
        del trainer.fact_node_data["study_fact_test"]


def test_abstain_clarify_and_compare_are_left_alone(trainer):
    plain = ChatSession(trainer)
    wide = ChatSession(trainer, planner=ExpandingPlanner(trainer))
    for question in ["Who painted the Mona Lisa?", "Compare photosynthesis and respiration"]:
        a, b = plain.send(question), wide.send(question)
        assert (a.plan.act, a.plan.fmt, [c.text for c in a.citations]) == (b.plan.act, b.plan.fmt, [c.text for c in b.citations])


def test_legacy_storage_falls_back_to_the_trainers_own_evidence(trainer, monkeypatch):
    monkeypatch.setattr(trainer, "storage_version", 1)
    base = ChatSession(trainer, planner=GroundedPlanner(trainer)).send("How does photosynthesis work?")
    wide = ChatSession(trainer, planner=ExpandingPlanner(trainer)).send("How does photosynthesis work?")
    assert [c.text for c in base.citations] == [c.text for c in wide.citations]


def test_expansion_is_deterministic(trainer):
    first = _ask(trainer, "How does the water cycle work?")[1]
    assert all(_ask(trainer, "How does the water cycle work?")[1] == first for _ in range(3))
