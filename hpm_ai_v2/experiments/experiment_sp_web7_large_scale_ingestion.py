"""
SP-Web7: Large-Scale Modular Ingestion & Thematic Synthesis.
Ingests a diverse scientific corpus using the refactored ReaderAgent (Perception) and LibrarianAgent (Ideation).
"""
import os
import time
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.librarian_agent import LibrarianAgent
from hpm_ai_v2.agents.dictionary_agent import DictionaryAgent
from hpm_ai_v2.agents.physics_agent import PhysicsAgent
from hpm_ai_v2.agents.math_agent import MathAgent
from hpm_ai_v2.agents.sentiment_agent import SentimentAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hpm_ai_v2.agents.orchestrator import AgentOrchestrator
from hfn.tiered_forest import TieredForest

EXPANDED_SCIENTIFIC_CORPUS = {
    "Biology: Evolution & Genetics": """
        Natural selection is a key mechanism of evolution. 
        Genes are units of heredity made of DNA sequences. 
        Mutations introduce genetic variation in a population. 
        Adaptation allows species to survive in changing environments.
    """,
    "Chemistry: Atomic Structure": """
        Atoms are the basic building blocks of matter. 
        Electrons orbit a nucleus containing protons and neutrons. 
        Chemical bonds form when atoms share or transfer electrons. 
        Reactions involve the breaking and forming of these bonds.
    """,
    "Computer Science: Algorithms": """
        Algorithms are step-by-step procedures for solving problems. 
        Big O notation measures the time and space complexity of an algorithm. 
        Sorting algorithms organize data in a specific order. 
        Recursion is a technique where a function calls itself to solve smaller subproblems.
    """,
    "Psychology: Cognitive Learning": """
        Memory is the cognitive process of encoding, storing, and retrieving information. 
        Perception involves interpreting sensory input to understand the environment. 
        Social behavior explores how individuals interact within groups. 
        Conditioning is a type of learning through association and reinforcement.
    """,
    "Astronomy: Cosmology": """
        Cosmology is the study of the origin and evolution of the universe. 
        The Big Bang theory explains the initial expansion of space and time. 
        Black holes are regions of space with immense gravitational pull. 
        Galaxies are massive systems of stars, gas, and dark matter.
    """,
    "Sociology: Social Structures": """
        Social institutions are established patterns of beliefs and behaviors. 
        Culture encompasses the shared values and norms of a society. 
        Inequality arises from the unequal distribution of resources and status. 
        Socialization is the process by which individuals learn societal expectations.
    """,
    "Physics: Thermodynamics": """
        Thermodynamics is the branch of physics dealing with heat and temperature. 
        Energy cannot be created or destroyed, only transformed. 
        Entropy is a measure of disorder in a physical system. 
        Systems reach equilibrium when there is no net flow of energy.
    """,
    "Mathematics: Topology": """
        Topology is the study of geometric properties that are preserved under continuous deformation. 
        Homeomorphism is a continuous mapping between two topological spaces. 
        Manifolds are spaces that locally resemble Euclidean space. 
        Invariants like Euler characteristic help distinguish different spaces.
    """
}

def run_large_scale_ingestion():
    print("================================================================================")
    print("SP-Web7: Large-Scale Modular Ingestion & Thematic Synthesis")
    print("================================================================================")

    knowledge_base = "data/knowledge_base/scientific_curiosity_v2"
    if not os.path.exists(knowledge_base):
        os.makedirs(knowledge_base, exist_ok=True)
        print(f"  [INIT] Created new knowledge base at {knowledge_base}")
    else:
        print(f"  [RESUME] Using existing knowledge base at {knowledge_base}")

    # 1. Initialize Society
    print("Phase 1: Initializing Scientific Society...")
    t_init = time.time()
    
    # Text config with initial concepts
    # IMPORTANT: We initialize forest FIRST with D=None to autodetect from disk
    shared_forest = TieredForest(D=None, cold_dir=knowledge_base, hot_cap=5000)
    
    # Then we ensure text_config matches the forest dimension
    D = shared_forest._D
    s_dim = 4
    concepts = [f"concept_{i}" for i in range(D - 2*s_dim)]
    text_config = TextDomainConfig(concepts=concepts, idf={}, s_dim=s_dim)
    
    orchestrator = AgentOrchestrator(shared_forest)
    
    # Specialist Agents
    librarian = LibrarianAgent(text_config, forest=shared_forest)
    reader = ReaderAgent(text_config, forest=shared_forest, librarian_agent=librarian)
    dictionary = DictionaryAgent(text_config, reader_agent=reader, forest=shared_forest)
    reader.dictionary_agent = dictionary
    
    physics = PhysicsAgent(text_config, forest=shared_forest)
    math = MathAgent(text_config, forest=shared_forest)
    sentiment = SentimentAgent(text_config, forest=shared_forest)
    writer = WriterAgent(text_config, reader_agent=reader, forest=shared_forest)
    
    # Register all
    orchestrator.register(librarian)
    orchestrator.register(reader)
    orchestrator.register(dictionary)
    orchestrator.register(physics)
    orchestrator.register(math)
    orchestrator.register(sentiment)
    orchestrator.register(writer)
    
    # Set up collaborations
    writer.reader_agent = reader
    writer.sentiment_agent = sentiment
    writer.math_agent = math
    writer.physics_agent = physics
    writer.dictionary_agent = dictionary
    
    print(f"  [TIME] Society Initialized: {time.time()-t_init:.2f}s")

    # 2. Ingestion Phase
    print("\nPhase 2: Ingesting Expanded Scientific Corpus...")
    t_ingest = time.time()
    
    for title, text in EXPANDED_SCIENTIFIC_CORPUS.items():
        doc_id = f"document_{title.replace(' ', '_')}"
        if doc_id in shared_forest:
            print(f"  [SKIPPING] {title} (already in forest)")
            continue
            
        print(f"  [READING] {title}...")
        doc_node = reader.ingest_text(text, title=title)
        print(f"    - Perception complete. Document Node ID: {doc_node.id}")
        
    print(f"\n  [TIME] Ingestion Complete: {time.time()-t_ingest:.2f}s")

    # 3. Verification of Librarian Discovery
    print("\nPhase 3: Verifying Librarian Concept Discovery...")
    topics = [n for n in shared_forest.active_nodes() if n.relation_type == "topic"]
    print(f"  Total Topics Discovered: {len(topics)}")
    
    # Cross-domain search for themes that should span papers
    key_themes = ["energy", "system", "evolution", "space", "behavior"]
    print("\nPhase 4: Cross-Domain Semantic Search & Synthesis...")
    for theme in key_themes:
        results = librarian.search_knowledge(theme)
        print(f"  Theme: '{theme}'")
        for res in results:
            label = res.metadata.get('word') or res.metadata.get('title') or res.id
            print(f"    - Associated: {label} [{res.relation_type}]")
            
        # If it's a topic, summarize it
        topic_node = shared_forest.get(f"topic_{theme}")
        if topic_node:
            summary = librarian.summarize_theme(topic_node)
            print(f"    - SUMMARY: {summary}")

    # 5. Persistence
    print("\nPhase 5: Persisting Multi-Domain Knowledge Base...")
    # Use agent's save_state to ensure DomainConfig is persisted as an HFN node
    reader.save_state()
    # Also save the forest to be sure
    shared_forest.save_to_cold()

    # 6. Final Diagnostics
    print("\nPhase 6: Society Performance & Diagnostics...")
    nodes = shared_forest.active_nodes()
    print(f"  - Total Forest Patterns: {len(nodes)}")
    print(f"  - Final Dimensionality (D): {shared_forest._D}")
    
    types = {}
    for n in nodes:
        t = getattr(n, "relation_type", "unknown")
        types[t] = types.get(t, 0) + 1
    
    print("\nKnowledge Breakdown:")
    for t, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {t}: {count}")

    print("\n[SUCCESS] SP-Web7 Large-Scale Ingestion & Synthesis completed.")

if __name__ == "__main__":
    run_large_scale_ingestion()
