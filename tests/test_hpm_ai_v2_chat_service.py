import json
import threading

import pytest

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.chat import Citation, Claim, ResponsePlan
from hpm_ai_v2.chat.presentation import UngroundedClaim, to_blocks
from hpm_ai_v2.chat.service import ChatService, NotFound
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import SourceDocument, SupervisedStudyTrainer

SOLAR = (
    "Solar panels convert sunlight into electricity. "
    "Solar energy can reduce household emissions. "
    "Photovoltaic cells generate direct current."
)
TIED = "Synaptic plasticity strengthens connections between neurons."


def _trainer(tmp_path, sources):
    config = TextDomainConfig.from_passages([s.text for s in sources], max_vocab=100)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    return trainer


def _service(tmp_path, sources=None):
    sources = sources or [SourceDocument("solar", SOLAR, "Solar energy")]
    return ChatService(_trainer(tmp_path, sources), tmp_path / "data")


def _plan(fmt, texts, *, style=None):
    claims = tuple(
        Claim(f"c{i}", t, (Citation("s", SOLAR.index(t), SOLAR.index(t) + len(t), t, "Solar"),))
        for i, t in enumerate(texts)
    )
    return ResponsePlan("answer", fmt, claims, 0.9, fmt=fmt, format_style=style)


def test_to_blocks_covers_every_format_with_exact_claim_text():
    sentences = ["Solar panels convert sunlight into electricity.", "Solar energy can reduce household emissions."]
    for fmt, kind in [("define", "paragraph"), ("summarize", "numbered_list"), ("outline", "numbered_list"),
                      ("flashcards", "flashcards"), ("compare", "compare"), ("grounded_qa", "paragraph")]:
        blocks, citations = to_blocks(_plan(fmt, sentences))
        assert blocks[0]["type"] == kind
        shown = [i["text"] for i in blocks[0]["items"]]
        assert shown == (sentences[:1] if kind == "paragraph" else sentences)
        for entry in blocks[0]["items"]:
            for n in entry["cites"]:
                c = citations[n - 1]
                assert SOLAR[c["start"]:c["end"]] == c["text"]
    assert to_blocks(_plan("summarize", sentences, style="bullets"))[0][0]["type"] == "bullet_list"


def test_to_blocks_abstain_clarify_and_ungrounded():
    assert to_blocks(ResponsePlan("abstain", "grounded_qa", (), 0.1, reason="x"))[0][0]["type"] == "abstain"
    clarify = ResponsePlan("clarify", "grounded_qa", (), 0.5, options=(("a", "A"), ("b", "B")))
    block = to_blocks(clarify)[0][0]
    assert block["type"] == "clarify" and [o["source_id"] for o in block["options"]] == ["a", "b"]
    bare = ResponsePlan("answer", "grounded_qa", (Claim("c", "t", ()),), 0.9)
    with pytest.raises(UngroundedClaim):
        to_blocks(bare)


def test_service_roundtrip_and_follow_up_after_reload(tmp_path):
    service = _service(tmp_path)
    conv = service.create_conversation()
    first = service.send(conv["id"], "What do solar panels convert into electricity?")
    assert first["act"] == "answer" and first["verified"] and first["citations"]

    fresh = ChatService(service.trainer, tmp_path / "data")
    loaded = fresh.get_conversation(conv["id"])
    assert [m["role"] for m in loaded["messages"]] == ["user", "assistant"]
    assert loaded["title"].startswith("What do solar")
    second = fresh.send(conv["id"], "Can you explain more about it?")
    assert "Context:" in second["why"]["effective_question"]
    events = [json.loads(l) for l in (tmp_path / "data" / "events.jsonl").read_text().splitlines()]
    assert len(events) == 2 and events[0]["session_id"] == events[1]["session_id"]


def test_service_scoped_clarify_flow(tmp_path):
    sources = [SourceDocument("notes-a", TIED, "Lecture notes"), SourceDocument("notes-b", TIED, "Textbook")]
    service = _service(tmp_path, sources)
    conv = service.create_conversation()
    question = "What does synaptic plasticity do?"
    clarify = service.send(conv["id"], question)
    assert clarify["blocks"][0]["type"] == "clarify"
    chosen = clarify["blocks"][0]["options"][1]["source_id"]
    answer = service.send(conv["id"], clarify["why"]["effective_question"], scope_source_id=chosen)
    assert answer["act"] == "answer" and {c["source_id"] for c in answer["citations"]} == {chosen}


def test_edit_truncates_later_turns_and_logs_drop(tmp_path):
    service = _service(tmp_path)
    conv = service.create_conversation()
    service.send(conv["id"], "What do solar panels convert?")
    second = service.send(conv["id"], "What do photovoltaic cells generate?")
    service.send(conv["id"], "Summarize solar energy")
    edited = service.edit(conv["id"], second["turn_id"], "What does solar energy reduce?")
    messages = service.get_conversation(conv["id"])["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"] * 2
    assert messages[-1]["turn_id"] == edited["turn_id"]
    assert len(service._session_for(service._read(conv["id"])).state.turns) == 2
    events = [json.loads(l) for l in (tmp_path / "data" / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "chat_turns_edited" and len(e["dropped_turn_ids"]) == 2 for e in events)
    with pytest.raises(NotFound):
        service.edit(conv["id"], "nope", "x")


def test_rename_delete_and_bad_ids(tmp_path):
    service = _service(tmp_path)
    conv = service.create_conversation()
    assert service.rename(conv["id"], "Renamed")["title"] == "Renamed"
    assert [c["title"] for c in service.list_conversations()] == ["Renamed"]
    service.delete(conv["id"])
    assert service.list_conversations() == []
    for bad in ("../x", "", "g" * 32):
        with pytest.raises(NotFound):
            service.get_conversation(bad)


def test_concurrent_sends_to_different_conversations_stay_consistent(tmp_path):
    service = _service(tmp_path)
    ids = [service.create_conversation()["id"] for _ in range(4)]
    errors = []

    def run(conv_id):
        try:
            for _ in range(3):
                service.send(conv_id, "What do solar panels convert?")
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in ids]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    for conv_id in ids:
        assert len(service.get_conversation(conv_id)["messages"]) == 6
    assert len((tmp_path / "data" / "events.jsonl").read_text().splitlines()) == 12
