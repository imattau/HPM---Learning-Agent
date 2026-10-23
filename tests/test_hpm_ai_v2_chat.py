import json

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import SourceDocument, StudyEpisode, SupervisedStudyTrainer
from hpm_ai_v2.chat import ChatSession


def test_validated_supervised_workflow_routes_new_trigger_to_cited_summary(tmp_path):
    text = "Solar panels convert sunlight into electricity. Solar energy reduces emissions."
    source = SourceDocument("solar-workflow", text, "Solar energy")
    config = TextDomainConfig.from_passages([text], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest")))
    result = trainer.fit(
        [source],
        [
            StudyEpisode("workflow-a", "train", "procedure", "give me a recap of the notes", None,
                         user_id="alice", workflow_key="summarize"),
            StudyEpisode("workflow-b", "train", "procedure", "make a recap from these notes", None,
                         user_id="alice", workflow_key="summarize"),
        ],
        [StudyEpisode("workflow-val", "validation", "procedure", "please give a recap of the notes", None,
                      user_id="alice", workflow_key="summarize")],
    )
    assert result["procedure_validation_metrics"]["accuracy"] == 1.0
    assert trainer.predict_workflow("create a recap of solar notes", user_id="alice")[0] == "summarize"
    assert trainer.predict_workflow("create a recap of solar notes", user_id="bob") is None
    assert trainer.predict_workflow("What will the weather be tomorrow?", user_id="alice") is None
    workflow_exam = trainer.evaluate([
        StudyEpisode("workflow-test", "test", "procedure", "create a recap of solar notes", None,
                     user_id="alice", workflow_key="summarize")
    ])
    assert workflow_exam["accuracy"] == 1.0
    assert workflow_exam["per_task_type"]["procedure"]["accuracy"] == 1.0

    reply = ChatSession(trainer, user_id="alice").send("create a recap of solar notes")
    assert reply.verified and not reply.abstained
    assert reply.plan.task_family == "summarize"
    assert reply.plan.procedure_id is not None
    assert all(text[c.start:c.end] == c.text for c in reply.citations)

    trainer.save_checkpoint(tmp_path / "workflow-checkpoint")
    restored = SupervisedStudyTrainer(ReaderAgent.load_agent(str(tmp_path / "workflow-checkpoint")))
    restored.load_checkpoint_state(tmp_path / "workflow-checkpoint")
    assert restored.predict_workflow("create a recap of solar notes", user_id="alice")[0] == "summarize"


def test_procedure_candidate_is_not_chat_usable_before_heldout_validation(tmp_path):
    text = "Photosynthesis converts light energy into chemical energy."
    config = TextDomainConfig.from_passages([text], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest")))
    trainer.fit([SourceDocument("plants", text)], [
        StudyEpisode("proc-a", "train", "procedure", "summarize the notes", None, workflow_key="summarize"),
        StudyEpisode("proc-b", "train", "procedure", "give me a recap", None, workflow_key="summarize"),
    ])
    assert trainer.procedure_node_data[next(iter(trainer.procedure_node_data))]["status"] == "candidate"
    assert trainer.predict_workflow("make a recap of this", include_candidates=False) is None


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


def test_learned_summary_format_preference_is_applied_per_user(tmp_path):
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
        [StudyEpisode(
            "teach-preference", "train", "preference", "Use bullets for summaries.", None,
            expected_answer="bullets", user_id="alice", preference_key="summary_format",
        )],
    )

    alice = ChatSession(trainer, user_id="alice").send("Summarize solar energy")
    bob = ChatSession(trainer, user_id="bob").send("Summarize solar energy")

    assert alice.verified and alice.plan.format_style == "bullets"
    assert "\n• " in alice.text
    assert bob.verified and bob.plan.format_style is None
    assert "\n1. " in bob.text



def test_chat_asks_which_source_when_two_sources_tie(tmp_path):
    text = "Synaptic plasticity strengthens connections between neurons."
    config = TextDomainConfig.from_passages([text, text], max_vocab=100)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest([
        SourceDocument("notes-a", text, "Lecture notes"),
        SourceDocument("notes-b", text, "Textbook"),
    ])
    log = tmp_path / "turns.jsonl"

    reply = ChatSession(trainer, event_log=log).send("What does synaptic plasticity do?")

    assert reply.plan.act == "clarify" and reply.verified
    assert not reply.abstained and reply.citations == ()
    assert {sid for sid, _ in reply.plan.options} == {"notes-a", "notes-b"}
    assert "Lecture notes" in reply.text and "Textbook" in reply.text
    assert json.loads(log.read_text().splitlines()[0])["act"] == "clarify"


def test_chat_does_not_clarify_when_one_source_clearly_wins(tmp_path):
    trainer, _source = _trainer(tmp_path)
    reply = ChatSession(trainer).send("What carries brain messages?")
    assert reply.plan.act == "answer" and reply.plan.options == ()


def test_chat_compares_two_sources_with_exact_citations(tmp_path):
    solar = SourceDocument(
        "solar", "Solar panels convert sunlight into electricity.", "Solar power"
    )
    wind = SourceDocument(
        "wind", "Wind turbines convert moving air into electricity.", "Wind power"
    )
    config = TextDomainConfig.from_passages([solar.text, wind.text], max_vocab=100)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest([solar, wind])

    reply = ChatSession(trainer).send("Compare solar and wind power")

    assert reply.verified and not reply.abstained
    assert reply.plan.task_family == "compare"
    assert {citation.source_id for citation in reply.citations} == {"solar", "wind"}
    for claim in reply.plan.claims:
        citation = claim.citations[0]
        source = trainer.sources[citation.source_id]
        assert source.text[citation.start:citation.end] == claim.text == citation.text


def test_chat_abstains_on_comparison_with_only_one_supported_source(tmp_path):
    source = SourceDocument("solar", "Solar panels convert sunlight into electricity.", "Solar power")
    config = TextDomainConfig.from_passages([source.text], max_vocab=100)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest([source])

    reply = ChatSession(trainer).send("Compare solar and geothermal power")

    assert reply.verified and reply.abstained
    assert reply.plan.reason == "comparison_requires_two_supported_sources"


def _tied_trainer(tmp_path):
    text = "Synaptic plasticity strengthens connections between neurons."
    config = TextDomainConfig.from_passages([text, text], max_vocab=100)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest([
        SourceDocument("notes-a", text, "Lecture notes"),
        SourceDocument("notes-b", text, "Textbook"),
    ])
    return trainer


def test_scoped_send_answers_from_the_chosen_source_and_logs_it(tmp_path):
    trainer = _tied_trainer(tmp_path)
    log = tmp_path / "turns.jsonl"
    session = ChatSession(trainer, event_log=log)
    question = "What does synaptic plasticity do?"
    assert session.send(question).plan.act == "clarify"

    reply = session.send(question, scope_source_id="notes-b")

    assert reply.plan.act == "answer" and reply.verified and reply.plan.options == ()
    assert {c.source_id for c in reply.citations} == {"notes-b"}
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [row.get("scope_source_id") for row in rows] == [None, "notes-b"]


def test_scoped_send_rejects_unknown_source(tmp_path):
    session = ChatSession(_tied_trainer(tmp_path))
    try:
        session.send("What does synaptic plasticity do?", scope_source_id="missing")
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    assert session.state.turns == []
