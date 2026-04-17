from hpm_ai_v2.utils.text_fetcher import fetch_passages, strip_html

def test_strip_html_removes_tags():
    html = "<p>Hello <b>world</b></p>"
    result = strip_html(html)
    assert "Hello world" in result
    assert "<" not in result

def test_strip_html_removes_scripts():
    html = "<script>alert('x')</script><p>Content</p>"
    result = strip_html(html)
    assert "alert" not in result
    assert "Content" in result

def test_fetch_passages_splits_paragraphs():
    text = "First sentence. Second sentence.\n\nNew paragraph here."
    passages = fetch_passages(text=text, min_length=5)
    assert len(passages) >= 2
    assert all(isinstance(p, str) for p in passages)

def test_fetch_passages_filters_short():
    text = "Hi.\n\nThis is a longer and more meaningful passage."
    passages = fetch_passages(text=text, min_length=20)
    assert len(passages) == 1
    assert "meaningful" in passages[0]


from hpm_ai_v2.domains.text_domain import TextDomainConfig

def test_text_domain_builds_vocab():
    passages = ["the cat sat on the mat", "the dog barked loudly"]
    config = TextDomainConfig.from_passages(passages, max_vocab=10)
    assert len(config.concepts) <= 10
    assert "cat" in config.concepts or "dog" in config.concepts

def test_text_domain_encodes_passage():
    passages = ["the cat sat on the mat", "the dog barked loudly"]
    config = TextDomainConfig.from_passages(passages, max_vocab=10)
    vec = config.encode_passage("cat sat mat")
    assert vec.shape == (config.m_dim,)
    assert vec.sum() > 0

def test_text_domain_similar_passages_closer():
    # Query shares terms with passage 1 but not passage 2 — compare query-passage similarity
    passages = ["machine learning models train data science",
                "cat sat mat dog barked loudly outside"]
    config = TextDomainConfig.from_passages(passages, max_vocab=20)
    import numpy as np
    query = config.encode_passage("machine learning data models")
    p1 = config.encode_passage(passages[0])
    p2 = config.encode_passage(passages[1])
    sim_relevant = np.dot(query, p1) / (np.linalg.norm(query) * np.linalg.norm(p1) + 1e-9)
    sim_irrelevant = np.dot(query, p2) / (np.linalg.norm(query) * np.linalg.norm(p2) + 1e-9)
    assert sim_relevant > sim_irrelevant


from hpm_ai_v2.utils.oracle.text_oracle import TextOracle

def test_text_oracle_state_shape():
    passages = ["machine learning trains models on data"]
    config = TextDomainConfig.from_passages(passages, max_vocab=20)
    oracle = TextOracle(config)
    state = oracle.compute_state(outputs=["machine learning"], errors=[None])
    assert state.shape == (config.S_DIM,)

def test_text_oracle_similar_query_high_score():
    passages = ["machine learning trains models on data",
                "machine learning data science models"]
    config = TextDomainConfig.from_passages(passages, max_vocab=20)
    for p in passages:
        config.register_passage(p)
    oracle = TextOracle(config)
    # Query shares terms with stored passages — should score > 0
    state = oracle.compute_state(outputs=["machine learning models"], errors=[None])
    assert state[0] > 0.0


from hpm_ai_v2.agents.reader_agent import ReaderAgent

def test_reader_agent_observe_and_query():
    passages = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by the human brain.",
        "The cat sat on the mat near the window.",
        "Deep learning uses many layers of neural networks.",
    ]
    config = TextDomainConfig.from_passages(passages, max_vocab=50)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)
    result = agent.query("machine learning neural networks")
    assert result is not None
    assert isinstance(result, str)

def test_reader_agent_observe_document():
    text = (
        "First paragraph about machine learning and neural networks.\n\n"
        "Second paragraph about cats and dogs and animals."
    )
    all_passages = [p.strip() for p in text.split("\n\n") if p.strip()]
    config = TextDomainConfig.from_passages(all_passages, max_vocab=30)
    agent = ReaderAgent(config)
    agent.observe_document(text, min_length=20)
    result = agent.query("machine learning")
    assert result is not None

def test_end_to_end_multi_document():
    doc1 = (
        "Python is a high-level programming language.\n\n"
        "It is widely used in data science and machine learning.\n\n"
        "Python supports object-oriented and functional programming."
    )
    doc2 = (
        "The Roman Empire was one of the largest empires in history.\n\n"
        "Rome was founded in 753 BC according to tradition.\n\n"
        "Latin was the official language of the Roman Empire."
    )
    all_passages = (
        [p for p in doc1.split("\n\n") if len(p) > 10] +
        [p for p in doc2.split("\n\n") if len(p) > 10]
    )
    config = TextDomainConfig.from_passages(all_passages, max_vocab=100)
    agent = ReaderAgent(config)
    agent.observe_document(doc1)
    agent.observe_document(doc2)
    result_tech = agent.query("programming language python data science")
    result_hist = agent.query("roman empire latin history")
    assert result_tech is not None
    assert result_hist is not None

def test_query_top_k_returns_multiple():
    passages = [
        "machine learning trains models",
        "deep learning uses neural networks",
        "the cat sat on the mat",
        "artificial intelligence is broad field",
    ]
    config = TextDomainConfig.from_passages(passages, max_vocab=40)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)
    results = agent.query_top_k("machine learning artificial intelligence", k=2)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
