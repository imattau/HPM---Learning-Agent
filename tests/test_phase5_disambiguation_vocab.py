from hpm_ai_v5.adapter.nlp import CanonicalPhraser
from hpm_ai_v5.adapter.packet import AdapterPacket
from hpm_ai_v5.adapter.snips import SNIPSCanonicalIntentAdapter


def _canonicalize(tokens: list[str]) -> list[str]:
    adapter = CanonicalPhraser()
    packet = AdapterPacket(raw=" ".join(tokens), context={"lemmas": tokens}, states=[])
    result = adapter.run(packet)
    return list(result.context["canonical_tokens"])


def test_canonical_phraser_marks_atis_disambiguation_terms():
    canonical = _canonicalize(["airline", "fare", "monday", "transportation"])

    assert canonical[:3] == ["AIRLINE", "AIRFARE", "MONDAY"]
    assert canonical[3].startswith("TRANSPORTATION")


def test_snips_canonical_adapter_marks_snips_disambiguation_terms():
    base = _canonicalize(["screening", "restaurant", "playlist", "play"])
    adapter = SNIPSCanonicalIntentAdapter()
    packet = AdapterPacket(raw="screening restaurant playlist play", context={"lemmas": ["screening", "restaurant", "playlist", "play"], "canonical_tokens": base}, states=[])
    result = adapter.run(packet)
    canonical = list(result.context["canonical_tokens"])

    assert canonical == ["SCREENING", "RESTAURANT", "MUSIC", "PLAY"]
