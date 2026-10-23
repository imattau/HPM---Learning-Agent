import json

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import SourceDocument, StudyEpisode, SupervisedStudyTrainer
from hpm_ai_v2.chat import ChatSession


def _trainer(tmp_path):
    text = "The brain communicates through electrochemical signals across neurons and synapses."
    source = SourceDocument("brain", text, "Neuroscience")
    config = TextDomainConfig.from_passages([text], max_vocab=100)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.fit(
        [source],
        [StudyEpisode("teach", "train", "grounded_qa", "What carries brain messages?", "brain", text)],
        [StudyEpisode("validate", "validation", "grounded_qa", "How does the brain communicate?", "brain", text)],
    )
    return trainer, source


def test_chat_replies_with_verifiable_citation_and_abstains(tmp_path):
    trainer, source = _trainer(tmp_path)
    log = tmp_path / "turns.jsonl"
    chat = ChatSession(trainer, user_id="u1", event_log=log)

    reply = chat.send("What carries brain messages?")
    unknown = chat.send("What is the capital of Mars?")

    assert reply.verified and not reply.abstained
    assert reply.plan.claims[0].text == source.text
    assert reply.plan.claims[0].citations[0].text == source.text
    assert "Neuroscience" in reply.text
    assert unknown.verified and unknown.abstained
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["user_id"] == "u1"
    assert rows[0]["citations"][0]["source_id"] == "brain"


def test_chat_resolves_simple_reference_after_session_restart(tmp_path):
    trainer, _source = _trainer(tmp_path)
    state_path = tmp_path / "chat-state.json"
    first = ChatSession(trainer, session_id="session-1")
    first.send("What carries brain messages?")
    first.save(state_path)

    resumed = ChatSession.load(trainer, state_path)
    reply = resumed.send("How does it travel?")

    assert reply.verified and not reply.abstained
    assert "Context: What carries brain messages?" in reply.plan.effective_question
    assert reply.plan.claims[0].citations[0].source_id == "brain"


def test_chat_formats_summaries_as_multiple_cited_sentences(tmp_path):
    text = (
        "Solar panels convert sunlight into electricity. "
        "Solar energy can reduce household emissions. "
        "Photovoltaic cells generate direct current."
    )
    source = SourceDocument("solar", text, "Solar energy")
    config = TextDomainConfig.from_passages([text], max_vocab=100)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.fit(
        [source],
        [StudyEpisode("teach-solar", "train", "grounded_qa", "What do solar panels do?", "solar", "Solar panels convert sunlight into electricity.")],
        [StudyEpisode("validate-solar", "validation", "grounded_qa", "How do solar panels work?", "solar", "Solar panels convert sunlight into electricity.")],
    )

    reply = ChatSession(trainer).send("Summarize solar energy")

    assert reply.verified and not reply.abstained
    assert reply.plan.fmt == "summarize"
    assert len(reply.plan.claims) == 3
    for claim in reply.plan.claims:
        citation = claim.citations[0]
        assert text[citation.start:citation.end] == claim.text == citation.text
    assert reply.text.startswith("Summary\n")


