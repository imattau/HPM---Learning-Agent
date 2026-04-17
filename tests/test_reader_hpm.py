import pytest
import numpy as np
import shutil
from pathlib import Path
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.utils.text_fetcher import fetch_passages
from hpm_ai_v2.experiments.experiment_sp_reader1_retrieval import run_experiment

# --- Helper Tests ---

def test_fetch_passages_splits_paragraphs():
    text = "Paragraph one is long enough.\n\nParagraph two is also long enough."
    ps = fetch_passages(text=text, min_length=10)
    assert len(ps) >= 2

def test_fetch_passages_filters_short():
    text = "This is a long enough passage. Too short. Another long enough passage."
    ps = fetch_passages(text=text, min_length=15)
    # "Too short." (10 chars) should be filtered out
    assert len(ps) == 2

# --- Domain & Config Tests ---

def test_text_domain_builds_vocab():
    passages = ["Machine learning is fun", "Artificial intelligence is deep"]
    config = TextDomainConfig.from_passages(passages, max_vocab=100)
    assert "learning" in config.concepts
    assert "intelligence" in config.concepts

# --- Agent Tests ---

def test_reader_agent_observe_and_query():
    passages = [
        "The quick brown fox jumps over the lazy dog.",
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs."
    ]
    config = TextDomainConfig.from_passages(passages, max_vocab=100)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)
    
    # Query for something specifically in one of the passages
    result = agent.query("brown fox")
    assert result is not None
    assert "fox" in result.lower()

def test_query_scored_returns_tuples():
    passages = ["cats eat mice", "dogs chase cats"]
    config = TextDomainConfig.from_passages(passages)
    agent = ReaderAgent(config)
    for p in passages:
        agent.observe_passage(p)
    
    scored = agent.query_scored("cats")
    assert len(scored) == 2
    assert scored[0][1] >= scored[1][1]

def test_full_agent_persistence():
    save_dir = "test_reader_save"
    if Path(save_dir).exists(): shutil.rmtree(save_dir)
        
    passages = ["Persistence is a valuable trait.", "Learning agents are the future."]
    config = TextDomainConfig.from_passages(passages)
    agent = ReaderAgent(config)
    agent.observe_passage(passages[0])
    agent.observe_passage(passages[1])
    agent.save_agent(save_dir)
    
    agent2 = ReaderAgent.load_agent(save_dir)
    assert agent2.config.concepts == agent.config.concepts
    res = agent2.query("valuable")
    assert res is not None
    shutil.rmtree(save_dir)

def test_automated_vocab_growth():
    passages = ["machine learning", "deep learning"]
    config = TextDomainConfig.from_passages(passages, max_vocab=50)
    agent = ReaderAgent(config)
    for p in passages: agent.observe_passage(p)
    
    initial_dim = agent.config.DIM
    # Expansion with new novel text
    agent.expand_vocabulary(["quantum computing is the next frontier of computation"], max_new=10)
    
    assert agent.config.DIM > initial_dim
    # Retrieval should still work (no shape mismatch)
    res = agent.query("learning")
    assert res is not None

def test_build_topic_clusters_creates_nodes():
    passages = ["topic apple orange", "topic grape melon", "topic banana peach"]
    config = TextDomainConfig.from_passages(passages)
    agent = ReaderAgent(config)
    for p in passages: agent.observe_passage(p)
    
    agent.build_topic_clusters(n_clusters=2)
    topic_nodes = [k for k in agent.patterns if k.startswith("topic_")]
    assert len(topic_nodes) == 2

def test_cross_doc_patterns_finds_shared_terms():
    passages = ["The sun is bright.", "Bright stars are far away."]
    config = TextDomainConfig.from_passages(passages)
    agent = ReaderAgent(config)
    for p in passages: agent.observe_passage(p)
        
    links = agent.find_cross_doc_patterns(threshold=0.1)
    assert len(links) >= 1

def test_experiment_runs_without_error():
    # run_experiment in experiment_sp_reader1_retrieval.py takes 0-1 args
    res = run_experiment(verbose=False)
    assert res["precision_at_1"] > 0

def test_sequential_mapping_and_themes():
    passages = ["Chapter one.", "Chapter two.", "Chapter three."]
    config = TextDomainConfig.from_passages(passages)
    agent = ReaderAgent(config)
    agent.observe_document("\n\n".join(passages))
    agent.build_topic_clusters(n_clusters=2)
    transitions = agent.learn_thematic_transitions(0)
    assert transitions >= 0

def test_universal_concepts_l5():
    passages = ["Science is broad.", "History is deep.", "Art is subjective.", "Music is universal."]
    config = TextDomainConfig.from_passages(passages)
    agent = ReaderAgent(config)
    for p in passages: agent.observe_passage(p)
    agent.build_topic_clusters(n_clusters=4)
    concepts = agent.stabilize_universal_concepts(n_concepts=2)
    assert concepts >= 1
