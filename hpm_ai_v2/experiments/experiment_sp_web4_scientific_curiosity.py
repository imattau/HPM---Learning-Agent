"""
SP-Web4: Project "Scientific Curiosity".
A large-scale multi-agent learning session to populate the HFN forest with scientific knowledge.
"""
import os
import time
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.agents.dictionary_agent import DictionaryAgent
from hpm_ai_v2.agents.physics_agent import PhysicsAgent
from hpm_ai_v2.agents.math_agent import MathAgent
from hpm_ai_v2.agents.sentiment_agent import SentimentAgent
from hpm_ai_v2.agents.writer_agent import WriterAgent
from hpm_ai_v2.agents.orchestrator import AgentOrchestrator
from hfn.tiered_forest import TieredForest

SCIENTIFIC_CORPUS = {
    "Physics: Classical Mechanics": """
        Classical mechanics is the study of the motion of bodies under the influence of forces.
        Newton's first law states that an object remains at rest unless acted upon by a force.
        The second law defines force as the product of mass and acceleration (F = m * a).
        Gravity is a fundamental attraction between all masses in the universe.
        Momentum is the product of mass and velocity (p = m * v).
        Kinetic energy is the energy of motion, calculated as 0.5 * m * v^2.
    """,
    "Mathematics: Calculus & Logic": """
        Calculus is the mathematical study of continuous change.
        Differential calculus deals with the rate of change of functions.
        The derivative of x^2 is 2*x.
        The derivative of sin(x) is cos(x).
        Integral calculus focuses on the accumulation of quantities and areas under curves.
        Mathematical logic explores the formal principles of reasoning and deduction.
    """,
    "Cognitive Science: Sentiment & Affect": """
        Affective computing is the study of systems that can recognize and process human emotions.
        Sentiment analysis is a subfield that classifies the polarity of text as positive or negative.
        Emotions like love and joy are characterized by high positive valence.
        Negative feelings like hate and anger are often intense and complex.
        Affective states influence decision making and social interaction in human agents.
    """
}

def run_curiosity_session():
    print("================================================================================")
    print("Project 'Scientific Curiosity': Large-Scale HPM Learning")
    print("================================================================================")

    knowledge_base = "data/scientific_curiosity"
    
    # 1. Initialize Society
    print("Phase 1: Initializing Scientific Society...")
    text_config = TextDomainConfig.from_passages(["HPM Society"], max_vocab=1000)
    shared_forest = TieredForest(D=text_config.m_dim, cold_dir=knowledge_base, hot_cap=2000)
    orchestrator = AgentOrchestrator(shared_forest)
    
    # Specialized Agents
    reader = ReaderAgent(text_config, forest=shared_forest)
    dictionary = DictionaryAgent(text_config, reader_agent=reader, forest=shared_forest)
    reader.dictionary_agent = dictionary
    
    physics = PhysicsAgent(text_config, forest=shared_forest)
    math = MathAgent(text_config, forest=shared_forest)
    sentiment = SentimentAgent(text_config, forest=shared_forest)
    writer = WriterAgent(text_config, reader_agent=reader, forest=shared_forest)
    
    # Register all with orchestrator for manifold sync
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
    
    # Seed sentiment lexicon
    sentiment.seed_lexicon()

    # 2. Deep Reading Phase
    print("\nPhase 2: Deep Reading Corpus...")
    start_time = time.time()
    
    for title, text in SCIENTIFIC_CORPUS.items():
        print(f"\n  [READING] {title}...")
        doc_node = reader.ingest_text(text, title=title)
        print(f"    - Knowledge Acquired. Node ID: {doc_node.id}")
        
    duration = time.time() - start_time
    print(f"\nPhase 2 Complete. Ingested {len(SCIENTIFIC_CORPUS)} papers in {duration:.2f}s.")

    # 3. Persistence
    print("\nPhase 3: Persisting Global Knowledge Base...")
    shared_forest.save_to_cold()

    # 4. Final Diagnostics
    print("\nPhase 4: Society Performance & Diagnostics...")
    nodes = shared_forest.active_nodes()
    print(f"  - Total Forest Patterns: {len(nodes)}")
    print(f"  - Final Dimensionality (D): {shared_forest._D}")
    
    # Type count
    types = {}
    for n in nodes:
        t = getattr(n, "relation_type", "unknown")
        types[t] = types.get(t, 0) + 1
    
    print("\nKnowledge Breakdown:")
    for t, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"    {t}: {count}")

    print("\n[SUCCESS] Project 'Scientific Curiosity' Session 1 Completed.")

if __name__ == "__main__":
    run_curiosity_session()
