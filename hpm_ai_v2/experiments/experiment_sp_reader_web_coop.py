"""
SP-Reader-Web: Cooperative Knowledge Acquisition via Reader + Web Agents.
"""
import os
import shutil
from typing import Dict, List
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.web_domain import WebDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.web_agent import WebAgent
from hfn.tiered_forest import TieredForest
import hpm_ai_v2.agents.mixins.web_fetch as wf
from unittest.mock import MagicMock

def run_experiment():
    print("================================================================================")
    print("SP-Reader-Web: Cooperative Knowledge Acquisition")
    print("================================================================================")

    # Clean up previous data
    if os.path.exists("data/coop_forest"):
        shutil.rmtree("data/coop_forest")

    # Text config first to determine dimension
    # We use a large enough fixed vocab to prevent dimension growth during the experiment
    text_config = TextDomainConfig.from_passages(["initial knowledge"], max_vocab=1000, include_char_primitives=True)
    m_dim = text_config.m_dim
    
    shared_forest = TieredForest(D=m_dim, cold_dir="data/coop_forest", hot_cap=1000)

    # Initialise Web Agent with matching DIM
    web_config = WebDomainConfig(force_dim=text_config.DIM)
    web_agent = WebAgent(web_config, forest=shared_forest)

    # Initialise Reader Agent (disable dynamic vocab growth for this test)
    reader_agent = ReaderAgent(text_config, forest=shared_forest, web_agent=web_agent, dynamic_vocab=False)

    # 2. Mocking Web for stability
    MOCK_WEB = {
        "artificial": "Artificial intelligence is intelligence demonstrated by machines.",
        "intelligence": "Intelligence has been defined in many ways: the capacity for logic, understanding, self-awareness, learning.",
        "logic": "Logic is the study of correct reasoning.",
        "computer": "A computer is a machine that can be programmed.",
        "initial": "Initial knowledge is the starting point of learning."
    }
    
    def mock_fetch(url):
        topic = url.split("/")[-1].lower()
        return MOCK_WEB.get(topic, f"Content about {topic}.")
    
    wf.fetch_url = MagicMock(side_effect=mock_fetch)

    # 3. Seed Knowledge
    print("\nPhase 2: Seeding Initial Knowledge...")
    reader_agent.observe_passage("Artificial intelligence is a field of computer science that emphasizes logic.")
    reader_agent.build_topic_clusters(n_clusters=2)
    
    print(f"  Forest size: {len(shared_forest._hot)} nodes")

    # 4. Cooperative Curiosity Loop
    print("\nPhase 3: Starting Cooperative Curiosity Loop...")
    visited_topics = set()
    for i in range(3):
        curious_topic = reader_agent.get_most_curious_topic()
        
        # Avoid repeating topics for better variety in the experiment
        if curious_topic in visited_topics:
            for c in reader_agent.config.concepts:
                if c not in visited_topics and not c.startswith("CHAR_") and len(c) > 3:
                    curious_topic = c
                    break
        
        visited_topics.add(curious_topic)
        print(f"\n      [LOOP {i+1}] Reader is curious about: '{curious_topic}'")
        
        # Web search
        results = web_agent.search(curious_topic, num_results=1)
        if not results:
            print(f"      [LOOP {i+1}] No web results found.")
            continue
            
        webpage_node = results[0]
        print(f"      [LOOP {i+1}] Found webpage: {webpage_node.metadata.get('url')}")
        
        # Fetch page
        text = web_agent.fetch_page(webpage_node)
        
        # Ingest text
        print(f"      [LOOP {i+1}] Reader ingesting content...")
        doc_node = reader_agent.ingest_text(text, title=curious_topic, webpage_node=webpage_node)
        
        print(f"      [LOOP {i+1}] Created document: {doc_node.id}")
        
        # Update topics
        reader_agent.build_topic_clusters(n_clusters=3 + i)
        
        print(f"      [LOOP {i+1}] Forest size: {len(shared_forest._hot)} nodes")

    # 5. Verification
    print("\nPhase 4: Verifying Knowledge Graph Integrity...")
    # Check for 'derived_from' edges
    docs = [n for k, n in reader_agent.patterns.items() if k.startswith("document_")]
    linked_docs = 0
    for d in docs:
        derived_edges = [e for e in d._edges if e.relation == "derived_from"]
        if derived_edges:
            linked_docs += 1
            web_node = derived_edges[0].target
            print(f"  Document '{d.id}' derived from '{web_node.id}' ({web_node.metadata.get('url')})")
            
    if linked_docs > 0:
        print(f"\n  SUCCESS: {linked_docs} documents successfully linked to web resources.")
    else:
        print("\n  FAILURE: No derived_from links found.")

    # Final Knowledge State
    print(f"\nPhase 5: Final Knowledge State")
    print(f"  Total nodes in forest: {len(shared_forest._hot)}")
    print(f"  ReaderAgent patterns: {len(reader_agent.patterns)}")
    print(f"  WebAgent patterns: {len(web_agent.patterns)}")

    print("\n[SUCCESS] SP-Reader-Web experiment completed.")

if __name__ == "__main__":
    run_experiment()
