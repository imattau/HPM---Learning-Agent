from dataclasses import replace

from hpm_ai_v2.chat.realizers import PlanBoundRealizer
from hpm_ai_v2.chat.session import ChatSession
from hpm_ai_v2.chat.schema import Citation, Claim, ResponsePlan
from hpm_ai_v2.chat.verifier import GroundingVerifier


class _Source:
    def __init__(self, text):
        self.text = text


def _plan():
    texts = ("Water evaporates from the surface.", "Water vapour cools and condenses into clouds.")
    claims = tuple(
        Claim(f"claim-{i}", text, (Citation("source", start, start + len(text), text, "Water cycle"),))
        for i, (start, text) in enumerate(((0, texts[0]), (len(texts[0]) + 1, texts[1])), start=1)
    )
    return ResponsePlan("answer", "grounded_qa", claims, .9)


def test_each_rendered_claim_has_a_sentence_to_claim_and_citation_mapping():
    plan = _plan()
    realization = PlanBoundRealizer().realize_with_map(plan)
    verifier = GroundingVerifier({"source": _Source(plan.claims[0].text + " " + plan.claims[1].text)})

    assert verifier.verify_realization(plan, realization)
    assert len(realization.sentences) == 2
    assert [item.text for item in realization.sentences] == [claim.text for claim in plan.claims]
    assert all(item.claim_ids and item.citation_ids for item in realization.sentences)
    assert "[1]" in realization.text and "[2]" in realization.text


def test_realization_verification_rejects_unmapped_or_changed_factual_text():
    plan = _plan()
    realizer = PlanBoundRealizer()
    realization = realizer.realize_with_map(plan)
    verifier = GroundingVerifier({"source": _Source(plan.claims[0].text + " " + plan.claims[1].text)})

    changed = replace(realization.sentences[0], text="Water evaporates quickly.")
    assert not verifier.verify_realization(plan, replace(realization, sentences=(changed, *realization.sentences[1:])))
    unmapped = replace(realization.sentences[1], claim_ids=())
    assert not verifier.verify_realization(plan, replace(realization, sentences=(realization.sentences[0], unmapped)))


def test_language_model_ranks_fixed_attributions_without_rewriting_claims():
    plan = _plan()
    verifier = GroundingVerifier({"source": _Source(plan.claims[0].text + " " + plan.claims[1].text)})
    scored = []

    def scorer(candidate):
        scored.append(candidate)
        return 0.0 if candidate.startswith("According to the source:") else 1.0

    realization = PlanBoundRealizer(phrase_scorer=scorer).realize_with_map(plan)

    assert len(scored) == 8  # three attributions and a direct form per claim
    assert realization.text.startswith("According to the source:")
    assert [sentence.text for sentence in realization.sentences] == [claim.text for claim in plan.claims]
    assert verifier.verify_realization(plan, realization)


def test_language_model_can_choose_unquoted_direct_cited_claim_without_changing_it():
    plan = _plan()
    verifier = GroundingVerifier({"source": _Source(plan.claims[0].text + " " + plan.claims[1].text)})
    realization = PlanBoundRealizer(
        phrase_scorer=lambda candidate: 0.0 if candidate.startswith("Water evaporates") else 1.0
    ).realize_with_map(plan)

    assert realization.text.startswith("Water evaporates from the surface. [1]")
    assert [sentence.text for sentence in realization.sentences] == [claim.text for claim in plan.claims]
    assert verifier.verify_realization(plan, realization)


def test_chat_session_accepts_opt_in_ranked_realizer_end_to_end():
    plan = _plan()

    class Trainer:
        sources = {"source": _Source(plan.claims[0].text + " " + plan.claims[1].text)}

    class Planner:
        def plan(self, *_args, **_kwargs):
            return plan

    realizer = PlanBoundRealizer(
        phrase_scorer=lambda candidate: 0.0 if candidate.startswith("According to the source:") else 1.0
    )
    session = ChatSession(Trainer(), planner=Planner(), realizer=realizer)
    reply = session.send("Explain the water cycle")

    assert reply.realizer == "plan-bound-lm-ranked"
    assert reply.text.startswith("According to the source:")
    assert reply.verified
    assert [item.text for item in reply.sentence_map] == [claim.text for claim in plan.claims]
