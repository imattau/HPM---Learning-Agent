from dataclasses import replace

from hpm_ai_v2.chat.realizers import PlanBoundRealizer
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
