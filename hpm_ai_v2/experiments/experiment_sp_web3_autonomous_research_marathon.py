"""
SP-Web 3: Autonomous Research Marathon (5-Minute Cycle).
Demonstrates Web, HtmlReader, Reader, and Writer agents collaborating to
autonomously research a topic and synthesize a final NLP summary.
"""
import os
import shutil
import time
import random
from typing import List, Set
from hfn.tiered_forest import TieredForest
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.web_domain import WebDomainConfig
from hpm_ai_v2.agents.web_agent import WebAgent
from hpm_ai_v2.agents.html_reader_agent import HtmlReaderAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent

MOCK_HTML = """
<html>
  <head><title>Transformer (machine learning model)</title></head>
  <body>
    <main>
      <h1>Transformer (machine learning model)</h1>
      <p>
        A transformer is a deep learning architecture that uses self-attention
        to weigh relationships between tokens in a sequence. It was introduced
        for neural machine translation and later became the dominant foundation
        for large language models, retrieval augmented generation, and many
        multimodal systems.
      </p>
      <p>
        Unlike recurrent neural networks, transformers process tokens in
        parallel and use positional encodings to represent word order. The
        attention mechanism allows each token representation to incorporate
        evidence from other tokens, which helps the model capture long-range
        dependencies in text, code, images, and audio.
      </p>
      <p>
        Encoder-only transformer models are often used for classification,
        search, and embedding tasks. Decoder-only transformer models are widely
        used for autoregressive text generation. Encoder-decoder transformers
        remain common in translation, summarization, and structured sequence
        transformation.
      </p>
      <p>
        Training large transformer systems requires large datasets, matrix
        multiplication on accelerators, optimization methods such as Adam, and
        careful evaluation for factuality, robustness, and bias. Research into
        sparse attention, mixture-of-experts routing, and retrieval systems aims
        to reduce computational cost while preserving model quality.
      </p>
      <section>
        <h2>Related topics</h2>
        <ul>
          <li><a href="https://en.wikipedia.org/wiki/Attention_(machine_learning)">Attention mechanism</a></li>
          <li><a href="https://en.wikipedia.org/wiki/Large_language_model">Large language model</a></li>
          <li><a href="https://en.wikipedia.org/wiki/Neural_machine_translation">Neural machine translation</a></li>
          <li><a href="https://en.wikipedia.org/wiki/Retrieval-augmented_generation">Retrieval augmented generation</a></li>
          <li><a href="https://en.wikipedia.org/wiki/Mixture_of_experts">Mixture of experts</a></li>
        </ul>
      </section>
    </main>
  </body>
</html>
"""

def run_marathon(
    seed_topic: str,
    duration_seconds: int = 300,
    use_mock: bool = False,
    max_sentences_per_page: int = 30,
    text_char_limit: int = 20000,
):
    print("================================================================================")
    print(f"SP-Web 3: Autonomous Research Marathon - Topic: '{seed_topic}'")
    print(f"Duration: {duration_seconds} seconds")
    print(f"Fetch mode: {'mock' if use_mock else 'live'}")
    print("================================================================================")

    knowledge_base = "data/research_marathon_forest"

    # 1. Initialize Cooperative triad
    print("Phase 1: Initializing Research Triad...")
    text_config = TextDomainConfig.from_passages(["initial"], max_vocab=1000, include_char_primitives=True)
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=10000)

    web_config = WebDomainConfig(force_dim=text_config.DIM)
    web_agent = WebAgent(web_config, forest=shared_forest, skip_priors=True)
    
    # Use HtmlReaderAgent for deep structural ingestion
    html_agent = HtmlReaderAgent(text_config, forest=shared_forest, web_agent=web_agent, dynamic_vocab=True, skip_priors=True)
    
    # Writer Agent for final synthesis
    writer_agent = WriterAgent(text_config, reader_agent=html_agent, forest=shared_forest, skip_priors=True)

    # 2. Research Loop
    print(f"\nPhase 2: Starting Research Marathon (Time limit: {duration_seconds}s)...")
    
    visited_urls: Set[str] = set()
    topic_queue: List[str] = [seed_topic]
    start_time = time.time()
    
    iteration = 1
    while (time.time() - start_time) < duration_seconds and topic_queue:
        current_topic = topic_queue.pop(0)
        elapsed = time.time() - start_time
        
        print(f"\n--- Iteration {iteration} (T+{elapsed:.1f}s) ---")
        print(f"Targeting: '{current_topic}'")
        
        # A. Search
        search_results = web_agent.search(current_topic, num_results=1)
        if not search_results:
            print(f"  [!] No search results for '{current_topic}'. Skipping.")
            continue
            
        webpage_node = search_results[0]
        url = webpage_node.metadata.get("url")
        if url in visited_urls:
            print(f"  [!] Already visited {url}. Skipping.")
            continue
            
        # B. Fetch
        print(f"  [WEB] Fetching: {url}")
        # Note: We use raw HTML for HtmlReaderAgent
        # We need to bypass the default strip_html for the deep parse
        try:
            if use_mock:
                raw_html = MOCK_HTML
            else:
                from hpm_ai_v2.utils.text_fetcher import fetch_url_raw
                raw_html = fetch_url_raw(url)
            webpage_node.metadata["text"] = raw_html
            webpage_node.metadata["status"] = 200
            visited_urls.add(url)
        except Exception as e:
            print(f"  [!] Fetch failed: {e}")
            continue

        # C. Fractal Ingestion (HTML + Text)
        print(f"  [HTML] Fractal Ingestion (Deep Structure)...")
        # We limit the HTML parsing for the marathon to keep it moving
        # but the HtmlReaderAgent naturally builds a hierarchy.
        ingest_start_time = time.time()
        doc_node = html_agent.ingest_html(
            raw_html,
            url,
            webpage_node=webpage_node,
            element_limit=max_sentences_per_page,
            text_char_limit=text_char_limit,
        )
        ingest_duration = time.time() - ingest_start_time
        print(f"  [TIMER] Ingestion took {ingest_duration:.2f} seconds.")
        
        
        # D. Curiosity-Driven Discovery (Finding next topics)
        # We look for links within the HTML structural nodes
        links = web_agent.extract_links_hierarchical(webpage_node)
        print(f"  [WEB] Extracted {len(links)} links from structure.")
        
        # Heuristic: Pick a few diverse links from the document
        new_links = []
        for ln in links:
            target_url = ln.metadata.get("url") if hasattr(ln, "metadata") else None
            if not target_url:
                # Links are pairs of source/target in children
                children = ln.children()
                if len(children) >= 2:
                    target_url = children[1].metadata.get("url")
            
            if target_url and target_url not in visited_urls and "wikipedia.org/wiki/" in target_url:
                # Extract topic from URL
                new_topic = target_url.split("/")[-1].replace("_", " ")
                if new_topic not in topic_queue:
                    new_links.append(new_topic)
        
        # Shuffle and take top 2 for breadth
        random.shuffle(new_links)
        topic_queue.extend(new_links[:2])
        
        print(f"  [CURIOSITY] Queue size: {len(topic_queue)} topics. Added: {new_links[:2]}")
        print(f"  [FOREST] Total nodes: {len(shared_forest._hot)}")
        
        iteration += 1
        # Brief pause to ensure we don't spam
        time.sleep(1)

    # 3. Final Synthesis
    print("\nPhase 3: Synthesizing Final Research Report...")
    print(f"Total time elapsed: {time.time() - start_time:.1f}s")
    print(f"Total pages researched: {len(visited_urls)}")
    
    # Get all html_document nodes
    doc_nodes = [n for n in html_agent.forest.active_nodes() if n.relation_type == "html_document"]
    
    if not doc_nodes:
        print("  [ERROR] No document knowledge acquired.")
        return

    print("\n--- FINAL NLP RESEARCH SUMMARY ---")
    
    summary = writer_agent.generate_research_summary(doc_nodes, max_sentences=8)
    
    print(f"Topic: {seed_topic}")
    print("-" * 40)
    print(summary)
    print("-" * 40)

    # Demonstrate QA over learned knowledge
    print("\nPhase 4: Research Verification (QA)...")
    q1 = f"What is {seed_topic}?"
    print(f"  Q: {q1}")
    a1 = writer_agent.answer_natural(q1)
    print(f"  A: {a1}")

    import pickle
    config_path = os.path.join(knowledge_base, "config.pkl")
    with open(config_path, "wb") as f:
        pickle.dump(text_config, f)
    print(f"\n  [SUCCESS] Saved TextDomainConfig to {config_path}")

    print(f"\nFinal Forest size: {len(shared_forest._hot)} nodes")
    print("[SUCCESS] SP-Web 3 Research Marathon completed.")

if __name__ == "__main__":
    import sys
    topic = sys.argv[1] if len(sys.argv) > 1 else "Transformer (machine learning model)"
    use_mock = "--mock" in sys.argv
    run_marathon(topic, duration_seconds=120, use_mock=use_mock)
