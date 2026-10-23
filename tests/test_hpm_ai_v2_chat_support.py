import pytest

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.chat import ChatSession
from hpm_ai_v2.chat.language import QUESTION_WORDS, content_terms
from hpm_ai_v2.chat.planner import GroundedPlanner
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import SourceDocument, SupervisedStudyTrainer

ORACLE = "Ask the oracle: what is this? The oracle replies in riddles."
PLANTS = "Photosynthesis converts light energy into chemical energy in plants. Chlorophyll absorbs light."
JUNK = "What is the capital of France?"          # only "what" overlaps the corpus
LEGIT = "How does chlorophyll absorb light?"


def _trainer(tmp_path, sources):
    config = TextDomainConfig.from_passages([s.text for s in sources], max_vocab=200)
    agent = ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    return trainer


@pytest.fixture
def trainer(tmp_path):
    return _trainer(tmp_path, [SourceDocument("oracle", ORACLE, "Oracle"), SourceDocument("plants", PLANTS, "Plants")])


def test_question_words_carry_no_subject_matter():
    assert content_terms("What is the capital of France?") == {"capit", "franc"}
    assert content_terms("How does chlorophyll absorb light?") == {"chlorophyl", "absorb", "light"}
    assert content_terms("What is this?") == set()
    assert content_terms("Why, how and what?") == set()                     # the set holds stems, so test behaviour
    assert all(isinstance(word, str) for word in QUESTION_WORDS)


def test_without_the_check_a_match_on_a_question_word_is_answered(trainer):
    plan = GroundedPlanner(trainer, require_content_support=False).plan("t", JUNK)
    assert plan.act == "answer" and "oracle" in plan.claims[0].text      # the match rests on "what" alone


def test_an_uncalibrated_lexical_checkpoint_abstains_when_the_match_rests_on_question_words(trainer):
    assert trainer.abstain_threshold == 0.0
    plan = GroundedPlanner(trainer).plan("t", JUNK)
    assert plan.act == "abstain" and plan.reason == "no_content_support"
    assert plan.claims == () and plan.options == ()
    reply = ChatSession(trainer).send(JUNK)                                 # the chat default has it on
    assert reply.abstained and reply.verified and reply.citations == ()


def test_a_supported_answer_is_untouched(trainer):
    plan = GroundedPlanner(trainer).plan("t", LEGIT)
    assert plan.act == "answer" and "Chlorophyll" in plan.claims[0].text


def test_a_clarification_is_judged_by_the_sources_it_offers(tmp_path):
    twins = _trainer(tmp_path, [SourceDocument("a", ORACLE, "A"), SourceDocument("b", ORACLE, "B")])
    loose = GroundedPlanner(twins, require_content_support=False).plan("t", JUNK)
    assert loose.act == "clarify"
    strict = GroundedPlanner(twins).plan("t", JUNK)
    assert strict.act == "abstain" and strict.reason == "no_content_support"
    on_topic = GroundedPlanner(twins).plan("t", "What does the oracle reply?")
    assert on_topic.act == "clarify" and len(on_topic.options) == 2         # a real tie is still asked about


def test_a_question_with_no_subject_matter_cannot_be_supported(trainer):
    assert GroundedPlanner(trainer).plan("t", "What is this?").act == "abstain"


def test_a_calibrated_checkpoint_is_left_to_its_own_threshold(trainer):
    trainer.abstain_threshold = 1e-6                                        # calibrated: the trainer owns abstention
    assert GroundedPlanner(trainer).plan("t", JUNK).act == "answer"


def test_a_semantic_encoder_may_match_without_a_shared_word(trainer, monkeypatch):
    monkeypatch.setattr(trainer.agent.config, "embedding_model", "frozen-encoder", raising=False)
    assert GroundedPlanner(trainer).plan("t", JUNK).act == "answer"


def test_the_check_can_be_forced_either_way(trainer):
    trainer.abstain_threshold = 1e-6
    assert GroundedPlanner(trainer, require_content_support=True).plan("t", JUNK).act == "abstain"
    trainer.abstain_threshold = 0.0
    assert GroundedPlanner(trainer, require_content_support=False).plan("t", JUNK).act == "answer"


def test_a_follow_up_is_supported_by_the_context_it_inherits(trainer):
    session = ChatSession(trainer)
    assert session.send(LEGIT).plan.act == "answer"
    follow_up = session.send("What about it?")
    assert follow_up.plan.act == "answer" and "Context:" in follow_up.plan.effective_question


def test_the_optional_confidence_floor_abstains_below_it_and_is_off_by_default(trainer):
    base = GroundedPlanner(trainer).plan("t", LEGIT)
    assert base.act == "answer" and 0 < base.confidence < 0.9
    floored = GroundedPlanner(trainer, min_confidence=0.9).plan("t", LEGIT)
    assert floored.act == "abstain" and floored.reason == "below_confidence_floor" and floored.claims == ()
    assert GroundedPlanner(trainer, min_confidence=base.confidence / 2).plan("t", LEGIT).act == "answer"


def test_an_abstention_reason_is_logged_with_the_turn(trainer, tmp_path):
    import json
    log = tmp_path / "turns.jsonl"
    ChatSession(trainer, event_log=log).send(JUNK)
    row = json.loads(log.read_text().splitlines()[0])
    assert row["act"] == "abstain" and row["abstained"] is True and row["citations"] == []


def test_the_check_judges_the_final_evidence_not_the_seed(tmp_path):
    # The trainer's best sentence matches only on "what". Another sentence says "observing ... weight",
    # which shares a stem with "observer ... weights" but no token, so the selector can find it and the
    # seed cannot.
    store = _trainer(tmp_path, [SourceDocument("oracle", "Ask the oracle what is this.", "Oracle"),
                                SourceDocument("lab", "Observing what adjusts weight.", "Lab")])
    question = "What does the observer do with weights?"
    seed_only = GroundedPlanner(store).plan("t", question)
    assert seed_only.act == "abstain" and seed_only.reason == "no_content_support"     # the seed alone fails the check
    from hpm_ai_v2.chat.evidence import ExpandingPlanner
    final = ExpandingPlanner(store).plan("t", question)
    assert final.act == "answer" and [c.text for c in final.claims] == ["Observing what adjusts weight."]
    assert ChatSession(store).send(question).plan.act == "answer"                        # and the chat default agrees
