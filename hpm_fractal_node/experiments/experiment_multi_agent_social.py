"""
SP40: Experiment 16 — Multi-Agent HFN Interaction (Social Learning)

Validates whether HFN supports knowledge transfer between independent agents.
Setup:
1. Two Specialized Agents:
   - Agent B trains on Concept 1 (Dim 0-5).
   - Agent C trains on Concept 2 (Dim 5-10).
2. Social Exchange: They broadcast 'dreams' (decoded high-weight nodes) to each other.
3. Observe: Does this social exchange allow them to master both concepts faster
   than a Solo Agent (Agent A) training on the raw stream?

Metrics:
- Surprise Reduction: Speed of mastering the full curriculum.
- Cross-Domain Accuracy: Success on the 'other' agent's specialty after exchange.
"""
import numpy as np
import time
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.decoder import Decoder
from hfn.evaluator import Evaluator

def generate_concepts(dim=20, n_examples=20):
    """
    Creates two distinct pattern concepts.
    Concept 1: Sharp features in dims 0-5.
    Concept 2: Sharp features in dims 10-15.
    """
    rng = np.random.RandomState(42)
    
    concept_1 = []
    for _ in range(n_examples):
        vec = np.zeros(dim)
        vec[0:5] = rng.normal(5.0, 0.5, size=5)
        concept_1.append(vec)
        
    concept_2 = []
    for _ in range(n_examples):
        vec = np.zeros(dim)
        vec[10:15] = rng.normal(5.0, 0.5, size=5)
        concept_2.append(vec)
        
    return concept_1, concept_2

def run_experiment():
    print("--- SP40: Experiment 16 — Multi-Agent HFN Interaction ---\n")
    
    DIM = 20
    EXAMPLES = 25
    c1, c2 = generate_concepts(DIM, EXAMPLES)
    full_curriculum = c1 + c2
    
    # 1. Agent A: Solo Baseline (Learns everything raw)
    forest_a = Forest()
    obs_a = Observer(forest=forest_a, tau=1.0, residual_surprise_threshold=1.5, node_use_diag=True)
    
    print("PHASE 1: Training Solo Agent A on full curriculum...")
    surprises_a = []
    for x in full_curriculum:
        res = obs_a.observe(x)
        surprises_a.append(res.residual_surprise)
    
    # 2. Agents B & C: Specialists
    # SPECIALIST CONFIG: Lower tau and cooccurrence threshold to integrate social data faster
    forest_b = Forest(); obs_b = Observer(forest=forest_b, tau=0.5, residual_surprise_threshold=1.0, compression_cooccurrence_threshold=2, node_use_diag=True)
    forest_c = Forest(); obs_c = Observer(forest=forest_c, tau=0.5, residual_surprise_threshold=1.0, compression_cooccurrence_threshold=2, node_use_diag=True)
    
    print("\nPHASE 2: Training Specialists B (Concept 1) and C (Concept 2)...")
    for x in c1: obs_b.observe(x)
    for x in c2: obs_c.observe(x)
    
    print(f"  Agent B Forest Size: {len(list(forest_b.active_nodes()))}")
    print(f"  Agent C Forest Size: {len(list(forest_c.active_nodes()))}")
    
    # 3. Social Interaction: Knowledge Exchange via Dreaming
    print("\nPHASE 3: Social Exchange (High-Bandwidth Dreaming)...")
    dec_b = Decoder(target_forest=forest_b, sigma_threshold=1.5)
    dec_c = Decoder(target_forest=forest_c, sigma_threshold=1.5)
    
    # Agent B dreams its concepts for Agent C
    dreams_from_b = []
    top_nodes_b = sorted(list(forest_b.active_nodes()), key=lambda n: obs_b.get_weight(n.id), reverse=True)[:15]
    for node in top_nodes_b:
        # Generate multiple diverse dreams from each top node
        for _ in range(5):
            goal = HFN(mu=node.mu + np.random.normal(0, 0.1, size=DIM), sigma=np.ones(DIM)*5.0, id="dream_b", use_diag=True)
            dec_res = dec_b.decode(goal)
            if isinstance(dec_res, list) and dec_res:
                dreams_from_b.append(dec_res[0].mu)
            
    # Agent C dreams its concepts for Agent B
    dreams_from_c = []
    top_nodes_c = sorted(list(forest_c.active_nodes()), key=lambda n: obs_c.get_weight(n.id), reverse=True)[:15]
    for node in top_nodes_c:
        for _ in range(5):
            goal = HFN(mu=node.mu + np.random.normal(0, 0.1, size=DIM), sigma=np.ones(DIM)*5.0, id="dream_c", use_diag=True)
            dec_res = dec_c.decode(goal)
            if isinstance(dec_res, list) and dec_res:
                dreams_from_c.append(dec_res[0].mu)
            
    # Exchange Ingestion with Refinement
    print(f"  Agent B ingesting and refining {len(dreams_from_c)} dreams from Agent C...")
    for _ in range(3): # Refinement Epochs
        for dream in dreams_from_c: obs_b.observe(dream)
    
    print(f"  Agent C ingesting and refining {len(dreams_from_b)} dreams from Agent B...")
    for _ in range(3):
        for dream in dreams_from_b: obs_c.observe(dream)
    
    # 4. Evaluation: Test Social Agents on the 'other' concept
    print("\nPHASE 4: Cross-Domain Probing...")
    
    def probe(obs, curriculum):
        surprises = []
        orig_tau = obs.tau
        obs.tau = 2.0 # Broader gating for probe
        try:
            for x in curriculum:
                res = obs.expand(x)
                if res.accuracy_scores:
                    best_id = max(res.accuracy_scores, key=res.accuracy_scores.get)
                    best_node = obs.forest.get(best_id)
                    s = obs._kl_surprise(x, best_node)
                    surprises.append(s)
                else:
                    surprises.append(2.0) 
        finally:
            obs.tau = orig_tau
        return np.mean(surprises)

    # Solo Baseline Final performance on full curriculum (last 10 examples)
    solo_final_surprise = np.mean(surprises_a[-10:])
    
    # Social Agents performance on the concepts they only saw via dreams
    surprise_b_on_c2 = probe(obs_b, c2)
    surprise_c_on_c1 = probe(obs_c, c1)
    
    print("\n--- RESULTS ANALYSIS ---")
    print(f"Solo Agent Final Surprise:    {solo_final_surprise:.4f}")
    print(f"Agent B Surprise on Concept 2: {surprise_b_on_c2:.4f} (Learned via Social)")
    print(f"Agent C Surprise on Concept 1: {surprise_c_on_c1:.4f} (Learned via Social)")
    
    social_avg = (surprise_b_on_c2 + surprise_c_on_c1) / 2.0
    
    if social_avg < 1.0:
        print("\n[SUCCESS] Social Knowledge Transfer Achieved!")
        print("Agents successfully mastered foreign concepts via generated dreams without raw data.")
    else:
        print("\n[FAIL] Social learning did not significantly reduce surprise on foreign domains.")

if __name__ == "__main__":
    run_experiment()
