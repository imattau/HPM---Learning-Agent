"""
SP-Reader 11: Autonomous Wikipedia Exploration & Knowledge Graph Building.
Demonstrates curiosity-driven crawling and cross-document link analysis.
"""
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from pathlib import Path
import shutil

KNOWLEDGE_BASE_DIR = "data/knowledge_base/reader_exploration"

def run_experiment():
    print("================================================================================")
    print("SP-Reader 11: Autonomous Wikipedia Exploration")
    print("================================================================================")

    # 1. Setup Agent
    print("Phase 1: Initialising Reader Agent...")
    if Path(KNOWLEDGE_BASE_DIR).exists():
        shutil.rmtree(KNOWLEDGE_BASE_DIR)
        
    config = TextDomainConfig.from_passages(["initial vocab"], max_vocab=50, include_char_primitives=True)
    agent = ReaderAgent(config, dynamic_vocab=True, cold_dir=KNOWLEDGE_BASE_DIR)

    # 2. Autonomous Exploration
    print("\nPhase 2: Starting Autonomous Exploration (Seed: 'Artificial intelligence')...")
    # We'll mock the wikipedia responses for the experiment to ensure stability and speed
    import wikipedia
    from unittest.mock import MagicMock
    
    # Mock page data
    MOCK_DATA = {
        "Artificial intelligence": {
            "content": "Artificial intelligence (AI) is intelligence demonstrated by machines. It is related to computer science and machine learning.",
            "links": ["Computer science", "Machine learning"]
        },
        "Machine learning": {
            "content": "Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn'.",
            "links": ["Neural networks", "Statistics"]
        },
        "Neural networks": {
            "content": "Neural networks are computing systems inspired by biological neural networks.",
            "links": ["Machine learning"]
        },
        "Computer science": {
            "content": "Computer science is the study of computation, information, and automation.",
            "links": ["Machine learning"]
        },
        "Statistics": {
            "content": "Statistics is the discipline that concerns data analysis.",
            "links": ["Machine learning"]
        }
    }
    
    def mock_page(title):
        if title in MOCK_DATA:
            p = MagicMock()
            p.title = title
            p.content = MOCK_DATA[title]["content"]
            p.links = MOCK_DATA[title]["links"]
            return p
        raise Exception("Page not found")
        
    def mock_summary(title, sentences=1):
        if title in MOCK_DATA:
            return MOCK_DATA[title]["content"]
        return "Generic summary for " + title
        
    wikipedia.page = MagicMock(side_effect=mock_page)
    wikipedia.summary = MagicMock(side_effect=mock_summary)

    path_visited = agent.explore_wikipedia("Artificial intelligence", max_iterations=4)
    print(f"\n      [RESULT] Exploration path: {' -> '.join(path_visited)}")

    # 3. Verify Knowledge Graph
    print("\nPhase 3: Verifying Knowledge Graph...")
    if agent.corpus_node:
        print(f"  Corpus Node ID: {agent.corpus_node.id}")
        print(f"  Number of Documents in Corpus: {len(agent.corpus_node.children())}")
        print(f"  Number of Links (Edges) in Corpus: {len(agent.corpus_node.edges())}")
        
        # Verify specific edges
        for edge in agent.corpus_node.edges():
            print(f"    Edge: {edge.source.id} --{edge.relation}--> {edge.target.id}")
    else:
        print("  FAILURE: Corpus node missing.")

    # 4. Path Finding Query
    print("\nPhase 4: Testing Path Finding Query...")
    if len(path_visited) >= 2:
        start = path_visited[0]
        end = path_visited[-1]
        print(f"  Finding path from '{start}' to '{end}'...")
        path = agent.find_path_between_docs(start, end)
        if path:
            print(f"  SUCCESS: Found path: {' -> '.join(path)}")
        else:
            print(f"  FAILURE: Path not found.")

    # 5. Fractal Uniformity Check
    print("\nPhase 5: Verifying Fractal Uniformity of discovered document...")
    doc_id = f"document_{path_visited[0].replace(' ', '_')}"
    if doc_id in agent.patterns:
        doc_node = agent.patterns[doc_id]
        print(f"  Document '{path_visited[0]}' node exists.")
        print(f"  Children (Paragraphs): {len(doc_node.children())}")
        for p in doc_node.children():
            print(f"    Paragraph children (Sentences): {len(p.children())}")
            for s in p.children():
                 print(f"      Sentence children (Words): {len(s.children())}")
                 break
            break
    else:
        print(f"  FAILURE: Document node '{doc_id}' not found.")

    # 6. Persistence
    print("\nPhase 6: Saving Exploration Knowledge Base...")
    agent.save_agent(KNOWLEDGE_BASE_DIR)
    print(f"  Knowledge base saved to {KNOWLEDGE_BASE_DIR}")

if __name__ == "__main__":
    run_experiment()
