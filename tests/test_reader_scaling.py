import pytest
import numpy as np
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig

def test_structural_wiring_depth():
    """Verify that ReaderAgent builds a connected L2->L3->L5 hierarchy."""
    passages = [
        "Quantum mechanics is a fundamental theory in physics.",
        "Classical mechanics describes the motion of macroscopic objects.",
        "Biology is the natural science that studies life.",
        "Zoology is the branch of biology that studies the animal kingdom."
    ]
    config = TextDomainConfig.from_passages(passages)
    agent = ReaderAgent(config)
    for p in passages: agent.observe_passage(p)
    
    # Build Hierarchy
    agent.build_topic_clusters(n_clusters=2) # Physics vs Biology
    agent.stabilize_universal_concepts(n_concepts=1) # "Science"
    
    # Check L5 -> L3
    concept = agent.patterns["concept_0"]
    topics = concept.children()
    assert len(topics) >= 1
    
    # Check L3 -> L2
    topic = topics[0]
    passages_nodes = topic.children()
    assert len(passages_nodes) >= 1
    assert "passage_" in passages_nodes[0].id

def test_predictive_curiosity():
    """Verify that predictive curiosity measures surprise against narrative transitions."""
    passages = [
        "The hero starts his journey.",
        "The hero finds a magical sword.",
        "The hero fights a terrible dragon.",
        "The hero saves the kingdom."
    ]
    config = TextDomainConfig.from_passages(passages)
    agent = ReaderAgent(config)
    agent.observe_document("\n\n".join(passages))
    
    agent.build_topic_clusters(n_clusters=2)
    agent.learn_thematic_transitions(0)
    
    # Prime last_topic_mu by observing one more familiar passage
    agent.observe_passage("The hero continues his quest.")
    
    # Familiar text (continuation of theme)
    familiar = "The hero returns home in glory."
    # Surprise text (abrupt shift)
    surprise = "Quantum computing uses superposition and entanglement."
    
    s_familiar = agent.predictive_curiosity_score(familiar)
    s_surprise = agent.predictive_curiosity_score(surprise)
    
    print(f"      [DEBUG] Predictive Curiosity: familiar={s_familiar:.4f}, surprise={s_surprise:.4f}")
    assert s_surprise > s_familiar

def test_summarization():
    """Verify centroid summarization extracts relevant keywords."""
    passages = [
        "artificial intelligence machine learning neural networks",
        "deep learning convolutional networks transformers"
    ]
    config = TextDomainConfig.from_passages(passages)
    agent = ReaderAgent(config)
    for p in passages: agent.observe_passage(p)
    
    agent.build_topic_clusters(n_clusters=1)
    summary = agent.summarize_node("topic_0", top_n=3)
    
    assert "Topic" in summary
    # Check if at least one keyword is present
    keywords = ["learning", "networks", "intelligence", "machine", "deep"]
    assert any(k in summary.lower() for k in keywords)

def test_hierarchical_retrieval():
    """Verify that hierarchical search finds the correct passage."""
    passages = [
        "Physics involves forces and motion.",
        "Chemistry involves atoms and molecules.",
        "Art involves painting and sculpture.",
        "Music involves rhythm and melody."
    ]
    config = TextDomainConfig.from_passages(passages)
    agent = ReaderAgent(config)
    for p in passages: agent.observe_passage(p)
    
    agent.build_topic_clusters(n_clusters=2) # Science vs Arts
    agent.stabilize_universal_concepts(n_concepts=2)
    
    res = agent.query_hierarchical("What is rhythm and melody?")
    assert res is not None
    assert "Music" in res or "rhythm" in res.lower()
