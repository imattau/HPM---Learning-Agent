"""
SP-Web8: Book Reading & Exam Resit.
Demonstrates the society's ability to read a multi-chapter text, identify knowledge gaps via an exam,
and perform targeted research to 'resit' and improve performance.
"""
import os
import time
from pathlib import Path
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.librarian_agent import LibrarianAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hpm_ai_v2.agents.executive_agent import ExecutiveAgent
from hpm_ai_v2.agents.orchestrator import AgentOrchestrator
from hfn.tiered_forest import TieredForest

# --- THE BOOK ---
BOOK_TITLE = "The Evolution of Intelligent Machines"
BOOK_CHAPTERS = {
    "Chapter 1: The Dawn of Logic": """
        Early AI research focused on symbolic logic and formal reasoning. 
        Alan Turing proposed the Turing Test as a measure of machine intelligence in 1950. 
        John McCarthy coined the term Artificial Intelligence in 1956 at the Dartmouth Conference.
    """,
    "Chapter 2: The First Winter": """
        High expectations in the 1960s led to disappointment when translation and reasoning proved difficult. 
        Funding for AI projects was significantly reduced in the 1970s. 
        This period is known as the First AI Winter.
    """,
    "Chapter 3: The Connectionist Turn": """
        In the 1980s, neural networks and connectionism gained popularity. 
        The backpropagation algorithm allowed for the training of multi-layer perceptrons. 
        Expert systems also flourished during this decade in commercial applications.
    """,
    "Chapter 4: Statistical Learning": """
        The 1990s saw a shift toward statistical models and probabilistic reasoning. 
        Support Vector Machines and Hidden Markov Models became dominant techniques. 
        IBM's Deep Blue defeated world chess champion Garry Kasparov in 1997.
    """,
    "Chapter 5: The Deep Learning Era": """
        Starting around 2012, deep neural networks achieved breakthroughs in image recognition. 
        Transformers, introduced in 2017, revolutionized natural language processing. 
        Large Language Models use massive amounts of data to predict the next word in a sequence.
    """
}

# --- THE EXAM ---
EXAM_QUESTIONS = [
    {
        "id": 1,
        "question": "Who coined the term Artificial Intelligence and when?",
        "keywords": ["McCarthy", "1956", "Dartmouth"],
        "chapter": "Chapter 1: The Dawn of Logic"
    },
    {
        "id": 2,
        "question": "What happened during the First AI Winter?",
        "keywords": ["funding", "disappointment", "1970s"],
        "chapter": "Chapter 2: The First Winter"
    },
    {
        "id": 3,
        "question": "Which algorithm enabled training multi-layer perceptrons in the 1980s?",
        "keywords": ["backpropagation"],
        "chapter": "Chapter 3: The Connectionist Turn"
    },
    {
        "id": 4,
        "question": "When did Deep Blue defeat Garry Kasparov?",
        "keywords": ["1997"],
        "chapter": "Chapter 4: Statistical Learning"
    },
    {
        "id": 5,
        "question": "What architecture revolutionized natural language processing in 2017?",
        "keywords": ["Transformer"],
        "chapter": "Chapter 5: The Deep Learning Era"
    }
]

def grade_answer(answer: str, question_meta: dict) -> bool:
    """Simple keyword-based grader."""
    if "I don't know" in answer or "don't know yet" in answer:
        return False
    
    match_count = 0
    for kw in question_meta["keywords"]:
        if kw.lower() in answer.lower():
            match_count += 1
            
    return match_count >= 1 # Pass if at least one key fact is present

from hpm_ai_v2.agents.web_agent import WebAgent
from hpm_ai_v2.domains.web_domain import WebDomainConfig

# ... (rest of the definitions)

from hpm_ai_v2.agents.dictionary_agent import DictionaryAgent
from hpm_ai_v2.agents.knowledge_graph_agent import KnowledgeGraphAgent
from hpm_ai_v2.domains.knowledge_graph_domain import KnowledgeGraphDomainConfig

def run_book_exam_workflow():
    print("================================================================================")
    print(f"SP-Web8: Book Reading & Exam Resit - '{BOOK_TITLE}'")
    print("================================================================================")

    # 1. Initialize Society
    print("Phase 1: Initializing Student Society (Loading Cumulative Forest)...")
    t_init = time.time()
    
    # RESUME from the large scientific knowledge base
    knowledge_base = Path("data/knowledge_base/scientific_curiosity_v2")
    if not knowledge_base.exists():
        knowledge_base.mkdir(parents=True, exist_ok=True)
        print(f"  [BOOTSTRAP] Knowledge base {knowledge_base} not found. Initializing new standalone forest.")
        # Create a basic initial forest meta if it doesn't exist
        shared_forest = TieredForest(D=100, cold_dir=knowledge_base)
        # Create a basic config
        text_config = TextDomainConfig.from_passages(["artificial intelligence"], s_dim=4)
        text_config.save_to_forest(shared_forest)
    else:
        # Load D from forest_meta.json automatically
        shared_forest = TieredForest(D=None, cold_dir=knowledge_base, library_dirs=["data/library"])
    
    # BOOTSTRAP: Load the exact configuration used during SP-Web7
    from hpm_ai_v2.domains.base import DomainConfig
    text_config = DomainConfig.load_from_forest(shared_forest)
    if text_config is None:
        print("  [ERROR] Could not load config from forest. Using default fallback.")
        text_config = TextDomainConfig.from_passages(["initial knowledge"], s_dim=4)
    
    D = shared_forest._D
    s_dim = text_config.S_DIM
    
    orchestrator = AgentOrchestrator(shared_forest)
    
    # 1. Specialist Agents
    librarian = LibrarianAgent(text_config, forest=shared_forest)
    reader = ReaderAgent(text_config, forest=shared_forest, librarian_agent=librarian)
    dictionary = DictionaryAgent(text_config, reader_agent=reader, forest=shared_forest)
    reader.dictionary_agent = dictionary
    
    kg_config = KnowledgeGraphDomainConfig(s_dim=s_dim)
    kg_agent = KnowledgeGraphAgent(kg_config, forest=shared_forest)
    
    web_config = WebDomainConfig(s_dim=s_dim)
    web = WebAgent(web_config, forest=shared_forest)
    reader.web_agent = web # Enable Reader to use WebAgent during ingestion
    
    writer = WriterAgent(text_config, reader_agent=reader, forest=shared_forest)
    writer.librarian_agent = librarian
    writer.dictionary_agent = dictionary
    
    # 2. Executive Agent (Planning & Coordination)
    executive = ExecutiveAgent(text_config, reader, writer, librarian, forest=shared_forest)
    
    # Register agents with orchestrator for unified reindexing and persistence
    orchestrator.register(librarian)
    orchestrator.register(reader)
    orchestrator.register(dictionary)
    orchestrator.register(kg_agent)
    orchestrator.register(writer)
    orchestrator.register(web)
    orchestrator.register(executive)
    
    print(f"  [TIME] Society Initialized (D={D}): {time.time()-t_init:.2f}s")
    
    # 3-5. Execute Autonomous Workflow via ExecutiveAgent
    results = executive.run_exam_workflow(
        corpus=BOOK_CHAPTERS,
        questions=EXAM_QUESTIONS,
        grader_func=grade_answer
    )

    # 6. Final Persistence
    print("\nPhase 6: Final Society Persistence...")
    orchestrator.shutdown()
    print("  [SUCCESS] All agent state persisted.")

    print("\n[SUCCESS] SP-Web8 Book Exam & Resit test completed.")

if __name__ == "__main__":
    run_book_exam_workflow()
