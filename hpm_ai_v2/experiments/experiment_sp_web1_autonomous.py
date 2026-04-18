"""
SP-Web 1: Autonomous Web Interaction and Integration.
Demonstrates WebAgent's autonomous discovery and integration with ReaderAgent.
"""
from hpm_ai_v2.domains.web_domain import WebDomainConfig
from hpm_ai_v2.agents.web_agent import WebAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
import numpy as np
from unittest.mock import MagicMock

def run_experiment():
    print("================================================================================")
    print("SP-Web 1: Autonomous Web Interaction and Integration")
    print("================================================================================")

    # 1. Setup Web Agent
    print("Phase 1: Initialising Web Agent...")
    web_config = WebDomainConfig()
    web_agent = WebAgent(web_config)

    # 2. Setup Reader Agent and link with Web Agent
    print("Phase 2: Initialising Reader Agent with Web Agent integration...")
    text_config = TextDomainConfig.from_passages(["initial"], max_vocab=20, include_char_primitives=True)
    # ReaderAgent uses the same forest/observer for shared knowledge
    reader_agent = ReaderAgent(text_config, web_agent=web_agent)

    # 3. Autonomous Search and Fetch
    print("\nPhase 3: Autonomous Web Search and Discovery...")
    # Mock search results and fetching
    import hpm_ai_v2.agents.mixins.web_fetch as wf
    
    # We mock the function IN the mixin module
    wf.fetch_url = MagicMock(return_value="HPM is a Hierarchical Pattern Modelling framework. It learns patterns from data.")
    
    query = "HPM learning framework"
    print(f"  Searching for: '{query}'")
    query_node = web_agent.search_web(query)
    
    print(f"  Search Query Node ID: {query_node.id}")
    print(f"  Number of results discovered: {len(query_node.children())}")
    
    # 4. Content Ingestion (Reader + Web)
    print("\nPhase 4: Content Ingestion with Web-to-Text Linking...")
    result_webpage = query_node.children()[0]
    print(f"  Fetching content from: {result_webpage.metadata.get('url')}")
    
    # We'll use the ReaderAgent's ingest logic which will now use WebAgent
    doc_node, _ = reader_agent.ingest_wikipedia_page("Hierarchical Pattern Modelling")
    
    if doc_node:
        print(f"  Document Node created: {doc_node.id}")
        
        # Verify 'derived_from' edge
        edges = doc_node.edges() # This might not be where the edge is stored if added to doc_node
        # Wait, ReaderAgent does: doc_node.add_edge(doc_node, webpage_node, "derived_from")
        # Let's check the doc_node's internal _edges
        derived_edges = [e for e in doc_node._edges if e.relation == "derived_from"]
        
        if derived_edges:
            edge = derived_edges[0]
            print(f"  SUCCESS: Document linked to Webpage via '{edge.relation}' edge.")
            print(f"    Source: {edge.source.id} (Type: {edge.source.metadata.get('type')})")
            print(f"    Target: {edge.target.id} (Type: {edge.target.metadata.get('type')}, URL: {edge.target.metadata.get('url')})")
        else:
            print("  FAILURE: 'derived_from' edge missing.")
    else:
        print("  FAILURE: Document node not created.")

    # 5. Relational Query
    print("\nPhase 5: Relational Rendering...")
    if doc_node and derived_edges:
        webpage_node = derived_edges[0].target
        print(f"  Renderer output for Webpage: {reader_agent.renderer.render(webpage_node)}")
        print(f"  Renderer output for Search Query: {reader_agent.renderer.render(query_node)}")

    print("\n[SUCCESS] SP-Web 1 experiment completed.")

if __name__ == "__main__":
    run_experiment()
