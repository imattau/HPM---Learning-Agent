from hpm_ai_v2.domains.text_domain import TextDomainConfig


def test_expand_vocab_does_not_let_frequent_known_terms_hide_rare_new_terms():
    config = TextDomainConfig.from_passages(
        ["alpha beta gamma delta"] * 4,
        max_vocab=4,
    )

    added = config.expand_vocab(
        ["alpha beta gamma delta"] * 4 + ["zorb quux"],
        max_new=2,
    )

    assert added == 2
    assert {"zorb", "quux"} <= set(config.concepts)
