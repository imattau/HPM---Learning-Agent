from hpm_ai_v4.tools.text_signals import TextSignalExtractor


def test_math_like_text_scores_mixed_domain_signals():
    extractor = TextSignalExtractor(use_spacy=False)

    pack = extractor.analyze("x = 2 + 3 and y = x * 4")

    assert pack.math_symbol_score > 0.15
    assert pack.equation_score > 0.4
    assert pack.mixed_domain_score > 0.3


def test_mixed_domain_text_outscores_plain_text():
    extractor = TextSignalExtractor(use_spacy=False)

    plain = extractor.analyze("The quick brown fox jumps over the lazy dog.")
    mixed = extractor.analyze("The area is A = pi r^2 and grows with n.")

    assert mixed.math_symbol_score > plain.math_symbol_score
    assert mixed.mixed_domain_score > plain.mixed_domain_score
    assert mixed.equation_score >= plain.equation_score


def test_non_math_text_keeps_mixed_score_low():
    extractor = TextSignalExtractor(use_spacy=False)

    pack = extractor.analyze("This is ordinary prose without equations.")

    assert pack.math_symbol_score < 0.2
    assert pack.mixed_domain_score < 0.2
