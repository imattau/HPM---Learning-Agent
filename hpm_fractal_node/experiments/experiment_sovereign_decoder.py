"""
Experiment: Sovereign Decoder (SP23).
Multi-process top-down synthesis for a mixed-modal "Say and Point" task.
"""
from __future__ import annotations
import multiprocessing as mp
import numpy as np
import time
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hfn.decoder import Decoder

# --- Constants ---
D = 10 # Small latent space for test

@dataclass
class WorkerConfig:
    name: str
    forest_id: str
    cold_dir: Path
    sigma_threshold: float
    source_nodes: list[HFN] = field(default_factory=list)

class GenerativeWorker(mp.Process):
    def __init__(self, config: WorkerConfig, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__(name=f"GenWorker-{config.name}")
        self.config = config
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        if self.config.cold_dir.exists(): shutil.rmtree(self.config.cold_dir)
        self.config.cold_dir.mkdir(parents=True, exist_ok=True)
        
        self.forest = TieredForest(D=D, forest_id=self.config.forest_id, cold_dir=self.config.cold_dir)
        
        for node in self.config.source_nodes:
            self.forest.register(node)
            
        self.decoder = Decoder(target_forest=self.forest, sigma_threshold=self.config.sigma_threshold)

        while True:
            task = self.task_queue.get()
            if task is None: break
            
            cmd = task.get("cmd")
            if cmd == "DECODE":
                goal_node = task["goal"]
                results = self.decoder.decode(goal_node)
                self.result_queue.put({
                    "name": self.config.name, 
                    "task_id": task["id"],
                    "results": [n.id for n in results]
                })

def run_experiment():
    print("SP23: Sovereign Decoder Experiment (Multi-Process)\n")
    mp.set_start_method("spawn", force=True)

    # --- 1. Build Priors ---
    # Shared concepts (Topology bridges)
    concept_red = HFN(mu=np.zeros(D), sigma=np.zeros(D), id="concept_red")
    concept_blue = HFN(mu=np.zeros(D), sigma=np.zeros(D), id="concept_blue")

    # L1 Lexical (Strings)
    word_red = HFN(mu=np.ones(D)*0.1, sigma=np.zeros(D), id="word_red")
    word_red.add_edge(word_red, concept_red, "REFERS_TO")
    
    word_blue = HFN(mu=np.ones(D)*0.2, sigma=np.zeros(D), id="word_blue")
    word_blue.add_edge(word_blue, concept_blue, "REFERS_TO")
    
    # L1 Motor (Coordinates)
    pos_red = HFN(mu=np.ones(D)*2.0, sigma=np.zeros(D), id="coord_2.0")
    pos_red.add_edge(pos_red, concept_red, "LOCATED_AT")

    pos_blue = HFN(mu=np.ones(D)*5.0, sigma=np.zeros(D), id="coord_5.0")
    pos_blue.add_edge(pos_blue, concept_blue, "LOCATED_AT")

    # L2 Narrative (Abstract Goals)
    goal_identify_red = HFN(mu=np.zeros(D), sigma=np.ones(D)*10.0, id="goal_identify_red")
    
    # Sub-goals (High variance, no children, but topological edges)
    sub_say_red = HFN(mu=np.ones(D)*0.15, sigma=np.ones(D)*5.0, id="sub_say_red")
    sub_say_red.add_edge(sub_say_red, concept_red, "REFERS_TO")
    
    sub_point_red = HFN(mu=np.ones(D)*2.5, sigma=np.ones(D)*5.0, id="sub_point_red")
    sub_point_red.add_edge(sub_point_red, concept_red, "LOCATED_AT")
    
    goal_identify_red.add_child(sub_say_red)
    goal_identify_red.add_child(sub_point_red)

    # --- 2. Spawn Workers ---
    configs = [
        WorkerConfig(name="L1_Lexical", forest_id="lex", cold_dir=Path("data/sp23_lex"), sigma_threshold=0.01, source_nodes=[word_red, word_blue]),
        WorkerConfig(name="L1_Motor", forest_id="mot", cold_dir=Path("data/sp23_mot"), sigma_threshold=0.01, source_nodes=[pos_red, pos_blue]),
        # L2 acts as the router here, but in a full system it would decode to subgoals first
    ]

    task_queues = {c.name: mp.Queue() for c in configs}
    result_queue = mp.Queue()
    workers = {c.name: GenerativeWorker(c, task_queues[c.name], result_queue) for c in configs}
    for w in workers.values(): w.start()

    # --- 3. The Generative Governor Loop ---
    print("--- Generative Dispatch ---")
    print(f"Goal: {goal_identify_red.id}")
    
    # Step 1: L2 Narrative Expansion (Done locally by Governor for this experiment)
    sub_goals = goal_identify_red.children()
    print(f"Expanded into {len(sub_goals)} sub-goals:\n")

    # Step 2: Parallel Dispatch
    # Governor routes based on mu heuristics or predefined domain ownership
    t0 = time.perf_counter()
    
    # Route Say -> Lexical
    print(f"Routing {sub_goals[0].id} -> L1_Lexical")
    task_queues["L1_Lexical"].put({"cmd": "DECODE", "id": 1, "goal": sub_goals[0]})
    
    # Route Point -> Motor
    print(f"Routing {sub_goals[1].id} -> L1_Motor")
    task_queues["L1_Motor"].put({"cmd": "DECODE", "id": 2, "goal": sub_goals[1]})

    # Step 3: Aggregate Results
    final_output = {}
    for _ in range(2):
        res = result_queue.get()
        final_output[res["name"]] = res["results"]

    print("\n--- Synchronized Execution Script ---")
    print(f"Speaker (L1_Lexical): {final_output.get('L1_Lexical', [])}")
    print(f"Actor (L1_Motor):     {final_output.get('L1_Motor', [])}")
    print(f"Constraint Verify: Both must target the 'RED' concept entity.")
    
    # Shutdown
    for name in workers: task_queues[name].put(None)
    for name in workers: workers[name].join()
    print(f"\nExperiment concluded in {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    run_experiment()
