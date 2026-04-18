"""Unit tests for Wikipedia ingestion and sentence splitting."""
import pytest
from hpm_ai_v2.utils.sentence_splitter import SentenceSplitter
from hpm_ai_v2.utils.text_chunker import chunk_passages
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

def test_sentence_splitter():
    splitter = SentenceSplitter()
    text = "Dr. Smith went to U.S. for work. He arrived at 3 p.m. The weather was nice."
    sentences = splitter.split(text)
    assert len(sentences) == 3
    assert sentences[0] == "Dr. Smith went to U.S. for work."
    assert sentences[1] == "He arrived at 3 p.m."
    assert sentences[2] == "The weather was nice."

def test_chunking():
    sentences = ["A."] * 10
    chunks = chunk_passages(sentences, window_size=5, overlap=2)
    # i=0: [0,5] -> "A. A. A. A. A."
    # i=3: [3,8] -> "A. A. A. A. A."
    # i=6: [6,11] -> "A. A. A. A."
    assert len(chunks) == 3
    assert len(chunks[0].split(" ")) == 5
    assert len(chunks[2].split(" ")) == 4

def test_dynamic_vocab():
    config = TextDomainConfig.from_passages(["the quick dog"], max_vocab=10, include_char_primitives=True)
    agent = ReaderAgent(config, dynamic_vocab=True)
    
    # Observe a passage with a new word 'fox'
    agent.observe_passage("the quick fox")
    
    # Should have added 'fox' to concepts
    assert "fox" in agent.config.concepts
    # Should have created a word macro for 'fox'
    assert "word_spelling_fox" in agent.patterns
    
    # Reindex should have been triggered
    assert agent.config.m_dim == agent.forest._D

def test_wikipedia_ingestion_mocked(mocker):
    # Mock wikipedia.page
    mock_page = mocker.Mock()
    mock_page.content = "Python is a programming language. It is very popular. HPM is a theory of learning."
    mocker.patch("wikipedia.page", return_value=mock_page)
    
    config = TextDomainConfig.from_passages(["initial"], max_vocab=10, include_char_primitives=True)
    agent = ReaderAgent(config, dynamic_vocab=True)
    
    indices = agent.ingest_wikipedia_page("Python (programming language)")
    assert len(indices) > 0
    assert any("Python" in p for p in agent.config._passages)
