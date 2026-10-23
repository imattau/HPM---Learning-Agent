import json
from pathlib import Path

import pytest

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.chat import ChatSession
from hpm_ai_v2.chat.feedback import Ledger, LedgerMismatch, file_digest
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import SourceDocument, SupervisedStudyTrainer, load_curriculum_jsonl

SEED = Path(__file__).parents[1] / "hpm_ai_v2/training/curricula/correction_transfer_seed.jsonl"
QUESTION = "How do bees show others where food can be found?"         # asked, then corrected
PARAPHRASE = "How do bees communicate food location to hive mates?"   # held-out paraphrase
UNRELATED = "What senses help bats locate insects?"


def _base(tmp_path):
    sources, _episodes = load_curriculum_jsonl(SEED)
    config = TextDomainConfig.from_passages([s.text for s in sources], max_vocab=200)
    agent = ReaderAgent(config, dynamic_vocab=False, inject_blank_priors=False, cold_dir=str(tmp_path / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    trainer.save_checkpoint(tmp_path / "base")
    return trainer, tmp_path / "base"


def _restore(checkpoint):
    trainer = SupervisedStudyTrainer(ReaderAgent.load_agent(str(checkpoint)))
    trainer.load_checkpoint_state(checkpoint)
    return trainer


def _span(trainer, source_id, needle):
    text = trainer.sources[source_id].text
    start = text.index(needle)
    node = next(n for n in trainer.source_to_answer_nodes[source_id]
                if trainer.answer_node_data[n]["start"] <= start < trainer.answer_node_data[n]["end"])
    data = trainer.answer_node_data[node]
    return {"source_id": source_id, "start": data["start"], "end": data["end"]}


def _session(trainer, tmp_path, user="alice", checkpoint=None):
    ledger = Ledger(tmp_path / "learning.jsonl", file_digest(checkpoint / "study_training_state.json"))
    return ChatSession(trainer, user_id=user, event_log=tmp_path / "events.jsonl", ledger=ledger), ledger


def test_correction_transfers_to_paraphrase_after_restart_without_side_effects(tmp_path):
    trainer, checkpoint = _base(tmp_path)
    session, ledger = _session(trainer, tmp_path, checkpoint=checkpoint)
    unrelated_before = dict(trainer._rank(UNRELATED, user_id="alice"))

    first = session.send(QUESTION)
    assert {c.source_id for c in first.citations} == {"visual"}            # wrong answer
    result = session.feedback(first.turn_id, "correct", correction=_span(trainer, "waggle", "waggle dance"))
    assert result["learned"] and isinstance(result["effective"], bool) and result["message"]

    assert {c.source_id for c in session.send(PARAPHRASE).citations} == {"waggle"}
    assert dict(trainer._rank(UNRELATED, user_id="alice")) == unrelated_before
    assert {c.source_id for c in ChatSession(trainer, user_id="bob").send(PARAPHRASE).citations} == {"visual"}

    restarted = _restore(checkpoint)                                         # fresh process
    assert {c.source_id for c in ChatSession(restarted, user_id="alice").send(PARAPHRASE).citations} == {"visual"}
    Ledger(tmp_path / "learning.jsonl", file_digest(checkpoint / "study_training_state.json")).replay(restarted)
    assert {c.source_id for c in ChatSession(restarted, user_id="alice").send(PARAPHRASE).citations} == {"waggle"}
    assert dict(restarted._rank(UNRELATED, user_id="alice")) == unrelated_before
    assert {c.source_id for c in ChatSession(restarted, user_id="bob").send(PARAPHRASE).citations} == {"visual"}
    assert [e["type"] for e in ledger.entries] == ["correction"]


def test_ledger_refuses_a_different_base_checkpoint(tmp_path):
    trainer, checkpoint = _base(tmp_path)
    session, _ = _session(trainer, tmp_path, checkpoint=checkpoint)
    turn = session.send(QUESTION)
    session.feedback(turn.turn_id, "correct", correction=_span(trainer, "waggle", "waggle dance"))
    with pytest.raises(LedgerMismatch):
        Ledger(tmp_path / "learning.jsonl", "some-other-digest")


def test_correction_must_name_an_exact_source_sentence_and_only_once(tmp_path):
    trainer, checkpoint = _base(tmp_path)
    session, ledger = _session(trainer, tmp_path, checkpoint=checkpoint)
    turn = session.send(QUESTION)
    span = _span(trainer, "waggle", "waggle dance")
    for bad in (None, {}, {"source_id": "waggle", "start": 1, "end": 5}, {"source_id": "nope", "start": 0, "end": 3}):
        with pytest.raises(ValueError):
            session.feedback(turn.turn_id, "correct", correction=bad)
    assert ledger.entries == []
    session.feedback(turn.turn_id, "correct", correction=span)
    with pytest.raises(ValueError):
        session.feedback(turn.turn_id, "correct", correction=span)
    with pytest.raises(ValueError):
        session.feedback("missing-turn", "up")
    with pytest.raises(ValueError):
        session.feedback(turn.turn_id, "sideways")


def test_candidates_are_exact_source_sentences_with_the_cited_one_marked(tmp_path):
    trainer, checkpoint = _base(tmp_path)
    session, _ = _session(trainer, tmp_path, checkpoint=checkpoint)
    turn = session.send(QUESTION)
    rows = session.correction_candidates(turn.turn_id)
    assert rows and all(trainer.sources[r["source_id"]].text[r["start"]:r["end"]] == r["text"] for r in rows)
    assert [r["source_id"] for r in rows if r["current"]] == ["visual"]
    assert "waggle" in {r["source_id"] for r in rows}


def test_thumbs_adjust_only_cited_sentences_once_and_survive_replay(tmp_path):
    trainer, checkpoint = _base(tmp_path)
    session, ledger = _session(trainer, tmp_path, checkpoint=checkpoint)
    turn = session.send(UNRELATED)
    cited = turn.citations[0]
    node = next(n for n in trainer.source_to_answer_nodes[cited.source_id]
                if trainer.answer_node_data[n]["start"] == cited.start)
    other = [n for n in trainer.source_to_answer_nodes[cited.source_id] if n != node]
    weights = {n: trainer.agent.observer.get_weight(n) for n in [node, *other]}
    source_weight = trainer.agent.observer.get_weight(trainer.source_to_node[cited.source_id])

    result = session.feedback(turn.turn_id, "down")
    assert result["learned"]
    assert trainer.agent.observer.get_weight(node) < weights[node]
    assert all(trainer.agent.observer.get_weight(n) == weights[n] for n in other)
    assert trainer.agent.observer.get_weight(trainer.source_to_node[cited.source_id]) == source_weight
    with pytest.raises(ValueError):
        session.feedback(turn.turn_id, "up")

    restarted = _restore(checkpoint)
    ledger.replay(restarted)
    assert restarted.agent.observer.get_weight(node) == pytest.approx(trainer.agent.observer.get_weight(node))


def test_thumb_on_a_reply_that_cited_nothing_learns_nothing(tmp_path):
    trainer, checkpoint = _base(tmp_path)
    session, ledger = _session(trainer, tmp_path, checkpoint=checkpoint)
    turn = session.send("Quantum chromodynamics gluon confinement")
    assert turn.abstained
    assert session.feedback(turn.turn_id, "up")["learned"] is False
    assert ledger.entries == []


def test_summary_format_preference_is_user_scoped_and_replays(tmp_path):
    text = "Solar panels convert sunlight into electricity. Solar energy can reduce household emissions. Photovoltaic cells generate direct current."
    source = SourceDocument("solar", text, "Solar energy")
    config = TextDomainConfig.from_passages([text], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest")))
    trainer.ingest([source])
    trainer.save_checkpoint(tmp_path / "base")
    session, ledger = _session(trainer, tmp_path, checkpoint=tmp_path / "base")
    assert session.set_preference("summary_format", "bullets") == {"key": "summary_format", "value": "bullets"}
    assert session.send("Summarize solar energy").plan.format_style == "bullets"
    assert ChatSession(trainer, user_id="bob").send("Summarize solar energy").plan.format_style is None
    for bad in (("summary_format", "poem"), ("tone", "pirate")):
        with pytest.raises(ValueError):
            session.set_preference(*bad)

    restarted = _restore(tmp_path / "base")
    assert ChatSession(restarted, user_id="alice").send("Summarize solar energy").plan.format_style is None
    ledger.replay(restarted)
    assert ChatSession(restarted, user_id="alice").send("Summarize solar energy").plan.format_style == "bullets"


def test_feedback_is_logged_to_the_event_log(tmp_path):
    trainer, checkpoint = _base(tmp_path)
    session, _ = _session(trainer, tmp_path, checkpoint=checkpoint)
    turn = session.send(UNRELATED)
    session.feedback(turn.turn_id, "up")
    rows = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["chat_turn", "chat_feedback"]
    assert rows[1]["signal"] == "up" and rows[1]["learned"] is True
