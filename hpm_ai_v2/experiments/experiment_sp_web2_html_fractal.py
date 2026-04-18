"""
SP-Web 2: Fractal HTML Understanding.
Demonstrates HtmlReaderAgent parsing raw HTML into a deep HFN hierarchy.
"""
import os
import shutil
import numpy as np
from hfn.tiered_forest import TieredForest
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.html_reader_agent import HtmlReaderAgent
from hpm_ai_v2.agents.web_agent import WebAgent
from hpm_ai_v2.domains.web_domain import WebDomainConfig

def run_experiment():
    print("================================================================================")
    print("SP-Web 2: Fractal HTML Understanding")
    print("================================================================================")

    if os.path.exists("data/html_forest"):
        shutil.rmtree("data/html_forest")

    # 1. Setup Shared Environment
    print("Phase 1: Initialising HTML Reader and Shared Forest...")
    text_config = TextDomainConfig.from_passages(["initial"], max_vocab=1000, include_char_primitives=True)
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir="data/html_forest", hot_cap=2000)

    web_config = WebDomainConfig(force_dim=text_config.DIM)
    web_agent = WebAgent(web_config, forest=shared_forest)
    
    html_agent = HtmlReaderAgent(text_config, forest=shared_forest, web_agent=web_agent, dynamic_vocab=True)

    # 2. Ingest Sample HTML
    print("\nPhase 2: Ingesting Sample HTML Structure...")
    html_sample = """
    <div class="container">
        <h1>Fractal AI</h1>
        <p>HPM learns by discoverng patterns.</p>
    </div>
    """
    url = "https://example.com/fractal"
    
    # Create a webpage node first to link
    webpage_node = web_agent.fetch_webpage(url) # This creates a stub webpage node
    
    doc_node = html_agent.ingest_html(html_sample, url, webpage_node=webpage_node)
    
    print(f"  Root Document Node: {doc_node.id}")
    print(f"  Forest size: {len(shared_forest._hot)} nodes")

    # 3. Verification of Fractal Structure
    print("\nPhase 3: Verifying Fractal Structure...")
    
    # Check for document node
    if doc_node.relation_type == "html_document":
        print("  SUCCESS: Document node found.")
    
    # Check for element nodes (div, h1, p)
    elements = [n for n in html_agent.patterns.values() if n.relation_type == "html_element"]
    tags = [n.metadata.get("tag") for n in elements]
    print(f"  Found element tags: {tags}")
    for t in ["div", "h1", "p"]:
        if t in tags:
            print(f"  SUCCESS: Found <{t}> element.")
        else:
            print(f"  FAILURE: Missing <{t}> element.")

    # Check for attribute node (class="container")
    attrs = [n for n in html_agent.patterns.values() if n.relation_type == "html_attribute"]
    print(f"  Found attributes: {[n.metadata.get('name') for n in attrs]}")
    if any(n.metadata.get("name") == "class" and n.metadata.get("value") == "container" for n in attrs):
        print("  SUCCESS: Found class='container' attribute.")
    else:
        print("  FAILURE: Missing class='container' attribute.")

    # Check for Word Macros (Shared across HTML and Text)
    word_div = html_agent.patterns.get("word_spelling_div")
    if word_div:
        print(f"  SUCCESS: Word macro '{word_div.id}' created for 'div' tag.")
    
    # 4. Text Content Verification
    print("\nPhase 4: Verifying Integrated Text Understanding...")
    # The 'h1' element should have a sentence child
    h1_elem = next(n for n in elements if n.metadata.get("tag") == "h1")
    text_children = [c for c in h1_elem.children() if c.relation_type == "sentence"]
    if text_children:
        print(f"  SUCCESS: <h1> has sentence child: '{html_agent.renderer.render(text_children[0])}'")
    else:
        print("  FAILURE: <h1> missing sentence child.")

    # 5. Semantic Retrieval over Structure
    print("\nPhase 5: Structural & Semantic Retrieval...")
    # Query for 'Fractal AI'
    q = "What is fractal ai?"
    res = html_agent.query(q)
    print(f"  Query: '{q}'")
    print(f"  Result: '{res}'")

    print("\n[SUCCESS] SP-Web 2 experiment completed.")

if __name__ == "__main__":
    run_experiment()
