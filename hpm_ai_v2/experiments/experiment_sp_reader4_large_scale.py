"""
SP-Reader4: Hierarchical Scaling with Large Document.
Verifies L2-L5 structural hierarchy, predictive curiosity, and top-down retrieval.
"""
import numpy as np
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.agents.reader_agent import ReaderAgent

# --- Synthetic Large Document ---
AI_TEXT = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans.
AI applications include advanced web search engines (e.g., Google Search), recommendation systems (used by YouTube, Amazon, and Netflix), understanding human speech (such as Siri and Alexa), self-driving cars (e.g., Waymo), generative or creative tools (ChatGPT and AI art), and playing games (such as chess and Go).
The various sub-fields of AI research are centered around particular goals and the use of particular tools.
Typical goals of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and support for robotics.
General intelligence (the ability to solve an arbitrary problem) is among the field's long-term goals.
To solve these problems, AI researchers have adapted and integrated a wide range of problem-solving techniques, including search and mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, probability, and economics.
AI also draws upon computer science, psychology, linguistics, philosophy, and many other fields.
The field was founded on the assumption that human intelligence can be so precisely described that a machine can be made to simulate it.
This raises philosophical arguments about the mind and the ethical consequences of creating artificial beings endowed with human-like intelligence.
"""

BIO_TEXT = """
Molecular biology is the branch of biology that seeks to understand the molecular basis of biological activity in and between cells, including biomolecular synthesis, modification, mechanisms, and interactions.
The study of chemical and physical structure of biological macromolecules is known as molecular biology.
Molecular biology was first described as an approach focused on the underpinnings of biological phenomena - uncovering the structures of biological molecules as well as their interactions, and how these interactions explain observations of classical biology.
In 1945 the term molecular biology was used by physicist William Astbury.
The development in molecular biology happened very late as to understand that the complex system or efficient instrument would be easy to understand by using simple way of study.
By using bacteria and bacteriophages this instrument is more simple than animal cell.
In 1953, two young men named James Watson and Francis Crick, working at Medical Research Council unit, Cavendish Laboratory, Cambridge, made a double helix model of DNA which changed the whole research scenario.
They proposed the DNA structure based on previous research done by Rosalind Franklin and Maurice Wilkins.
Then this research leads to finding DNA material in other organisms like plants and animals.
"""

HIST_TEXT = """
The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, that occurred during the period from around 1760 to about 1820–1840.
This transition included going from hand production methods to machines, new chemical manufacturing and iron production processes, the increasing use of steam power and water power, the development of machine tools and the rise of the mechanized factory system.
The Industrial Revolution also led to an unprecedented rise in the rate of population growth.
Textiles were the dominant industry of the Industrial Revolution in terms of employment, value of output and capital invested.
The textile industry was also the first to use modern production methods.
The Industrial Revolution began in Great Britain, and many of the technological and architectural innovations were of British origin.
By the mid-18th century, Britain was the world's leading commercial nation, controlling a global trading empire with colonies in North America and the Caribbean.
Britain had major military and political hegemony on the Indian subcontinent; particularly with the proto-industrialized Mughal Bengal, through the activities of the East India Company.
The Industrial Revolution marks a major turning point in history; almost every aspect of daily life was influenced in some way.
"""

def run_experiment():
    print("================================================================================")
    print("SP-Reader4: Hierarchical Scaling & Large Document Retrieval")
    print("================================================================================")

    # 1. Setup Large Multi-Domain Corpus
    full_text = AI_TEXT + "\n\n" + BIO_TEXT + "\n\n" + HIST_TEXT
    passages = [p.strip() for p in full_text.split("\n") if len(p.strip()) > 40]
    
    print(f"Phase 1: Initializing Reader...")
    import shutil
    from pathlib import Path
    kb_dir = Path("data/knowledge_base/reader_scaling_test")
    if kb_dir.exists():
        shutil.rmtree(kb_dir)

    config = TextDomainConfig.from_passages(passages, max_vocab=500)
    agent = ReaderAgent(config, cold_dir=str(kb_dir))

    # Observe documents to enable L4 thematic mapping
    agent.observe_document(AI_TEXT)
    agent.observe_document(BIO_TEXT)
    agent.observe_document(HIST_TEXT)

    
    # 2. Build Deep Hierarchy (L2 -> L3 -> L5)
    print("\nPhase 2: Building Structural Hierarchy (L2 -> L5)...")
    # We expect roughly 3-6 topics and 3 concepts (AI, Bio, Hist)
    agent.build_topic_clusters(n_clusters=6)
    agent.stabilize_universal_concepts(n_concepts=3)
    
    print(f"  Forest Status: {len(agent.forest)} nodes.")
    concept_ids = [k for k in agent.patterns if k.startswith("concept_")]
    topic_ids = [k for k in agent.patterns if k.startswith("topic_")]
    print(f"  Created {len(concept_ids)} Concepts and {len(topic_ids)} Topics.")

    # 3. Recursive Summarization
    print("\nPhase 3: Recursive Summarization of Universal Concepts:")
    for cid in concept_ids:
        summary = agent.summarize_node(cid, top_n=5)
        print(f"  [Concept {cid}] {summary}")

    # 4. Top-Down Hierarchical Retrieval
    print("\nPhase 4: Top-Down Hierarchical Retrieval (L5 -> L3 -> L2):")
    queries = [
        "How do neural networks and statistics help AI?",
        "Who discovered the double helix structure of DNA?",
        "What was the impact of steam power on factories?"
    ]
    
    for q in queries:
        print(f"\n  Query: '{q}'")
        # Direct retrieval (flat)
        flat_res = agent.query(q)
        # Hierarchical retrieval (structured)
        hier_res = agent.query_hierarchical(q)
        
        print(f"  Flat Result: {flat_res[:100] if flat_res else 'NOT FOUND'}...")
        print(f"  Hierarchical: {hier_res[:100] if hier_res else 'NOT FOUND'}...")

    # 5. Predictive Curiosity (Structural Surprise)
    print("\nPhase 5: Predictive Curiosity (Structural Surprise):")
    # We prime the agent with the "History" document narrative (Doc 2)
    agent.learn_thematic_transitions(2) 
    
    # Observe one more history passage to "lock in" the current topic/state
    agent.observe_passage("The Industrial Revolution began in Great Britain and spread to Europe.")
    
    # Continuation of History (Low Surprise)
    history_cont = "Steam engines revolutionized transport and manufacturing in the 19th century."
    # Abrupt shift to AI (High Surprise)
    ai_shift = "Modern artificial intelligence uses deep learning and large language models."
    
    s_history = agent.predictive_curiosity_score(history_cont)
    s_ai = agent.predictive_curiosity_score(ai_shift)
    
    print(f"  Surprise (History Continuation): {s_history:.4f}")
    print(f"  Surprise (AI Shift):            {s_ai:.4f}")
    
    if s_ai > s_history:
        print("\n  SUCCESS: Agent detected the structural shift across Universal Concepts!")
    else:
        print("\n  FAILURE: Surprise metrics were not discriminative.")

if __name__ == "__main__":
    run_experiment()
