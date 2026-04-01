"""
Thematic Anchor Extraction Experiment (SP20).

Processes a page of Peter Rabbit text through a multi-tier hierarchy:
- L1 Lexical: Recognizes individual word context.
- L2 Thematic: Synthesizes rolling windows of L1 identities into narrative anchors.
"""
from __future__ import annotations

import gc
import multiprocessing as mp
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parents[2]))

from hfn import Observer, calibrate_tau
from hfn.hfn import HFN
from hfn.tiered_forest import TieredForest
from hpm_fractal_node.nlp.nlp_loader import (
    VOCAB, VOCAB_SIZE, D as LEX_D, _tokenize, compose_corpus_context, _WORD_TO_CATEGORY
)
from hpm_fractal_node.nlp.nlp_world_model import build_nlp_world_model

SEED = 42
OFFSLICE_VAR = 1.0
WINDOW_SIZE = 2 
L2_D = LEX_D * WINDOW_SIZE 

@dataclass
class WorkerConfig:
    name: str
    forest_id: str
    cold_dir: Path
    max_hot: int
    degree: float
    tau_sigma: float
    common_d: int
    source_nodes: list[HFN] = field(default_factory=list)
    source_prior_ids: set[str] = field(default_factory=set)
    prefix: str = ""

def _clone_node(node: HFN, prefix: str, common_d: int) -> HFN:
    mu = np.zeros(common_d, dtype=np.float64)
    sigma = np.full(common_d, OFFSLICE_VAR, dtype=np.float64)
    src_mu = np.asarray(node.mu, dtype=np.float64)
    src_diag = np.asarray(node.sigma, dtype=np.float64) if node.use_diag else np.diag(node.sigma)
    r = min(common_d, src_mu.shape[0])
    mu[:r], sigma[:r] = src_mu[:r], src_diag[:r]
    return HFN(mu=mu, sigma=sigma, id=f"{prefix}{node.id}", use_diag=True)

class SovereignWorker(mp.Process):
    def __init__(self, config: WorkerConfig, task_queue: mp.Queue, result_queue: mp.Queue):
        super().__init__(name=f"SovereignWorker-{config.name}")
        self.config = config
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        if self.config.cold_dir.exists(): shutil.rmtree(self.config.cold_dir)
        self.config.cold_dir.mkdir(parents=True, exist_ok=True)
        
        self.forest = TieredForest(D=self.config.common_d, forest_id=self.config.forest_id,
                                   cold_dir=self.config.cold_dir, max_hot=self.config.max_hot)
        
        prior_ids: set[str] = set()
        for node in self.config.source_nodes:
            clone = _clone_node(node, self.config.prefix, self.config.common_d)
            self.forest.register(clone)
            if node.id in self.config.source_prior_ids:
                prior_ids.add(clone.id)
        
        if self.config.degree > 0.0: self.forest.set_protected(prior_ids)

        # Custom tau for L2 to force creation
        tau = 0.5 if self.config.degree == 0.0 else calibrate_tau(self.config.common_d, sigma_scale=self.config.tau_sigma, margin=5.0)
        
        self.observer = Observer(
            forest=self.forest, tau=tau, protected_ids=prior_ids if self.config.degree > 0.0 else set(),
            recombination_strategy="nearest_prior", hausdorff_absorption_threshold=0.35,
            persistence_guided_absorption=True, lacunarity_guided_creation=True,
            lacunarity_creation_radius=0.1, node_use_diag=True,
            residual_surprise_threshold=0.5,
            node_prefix=self.config.prefix + "leaf_",
            alpha_gain=0.5, # Faster learning
            beta_loss=0.1   # Faster forgetting of overlaps
        )

        while True:
            task = self.task_queue.get()
            if task is None: break
            
            cmd = task.get("cmd")
            if cmd == "OBSERVE":
                # Apply slow decay to all non-protected weights
                for nid in self.observer._weights:
                    if nid not in self.observer.protected_ids:
                        self.observer._weights[nid] *= 0.99
                
                res = self.observer.observe(task["x"])
                self.forest._on_observe()
                winner_mu = res.explanation_tree[0].mu if res.explanation_tree else None
                winner_id = res.explanation_tree[0].id if res.explanation_tree else None
                self.result_queue.put({"name": self.config.name, "winner_mu": winner_mu, "winner_id": winner_id})
                
            elif cmd == "STATS":
                self.result_queue.put({"name": self.config.name, "total_nodes": len(self.forest)})
            
            elif cmd == "GET_TOP_ANCHORS":
                # Returns nodes sorted by weight
                nodes = []
                # Use _mu_index to see ALL nodes (hot and cold)
                prefix = self.config.prefix + "leaf_"
                for nid in self.forest._mu_index:
                    weight = self.observer._weights.get(nid, 0.0)
                    if nid.startswith(prefix): # Learned anchors
                        node = self.forest.get(nid)
                        if node:
                            nodes.append({"id": nid, "mu": node.mu, "weight": weight})
                nodes.sort(key=lambda x: x["weight"], reverse=True)
                self.result_queue.put({"name": self.config.name, "anchors": nodes[:10]})

def main():
    mp.set_start_method("spawn", force=True)
    print("SP20: Thematic Anchor Extraction Experiment")
    
    corpus_path = Path("data/corpus/peter_rabbit.txt")
    if not corpus_path.exists():
        print(f"Error: {corpus_path} not found.")
        return

    text = corpus_path.read_text(encoding="utf-8", errors="ignore")
    tokens = _tokenize(text)
    print(f"Loaded {len(tokens)} tokens from Peter Rabbit.\n")

    # 1. Build L1 Lexical World Model
    lex_forest, lex_prior_ids = build_nlp_world_model(forest_cls=TieredForest, cold_dir=Path("data/sanchor_lex_cold"), max_hot=500)

    # 2. Build L2 Background Priors (Negative Selection)
    # These HFNs represent sequences of common function words (the, and, to, was, in)
    # By protecting these in L2, the Observer will "explain" function-word sequences 
    # and NOT create new leaf nodes for them.
    l2_priors = []
    l2_prior_ids = set()
    function_indices = [3, 4, 11, 14, 15, 16] # indices for the, a, to, and, is, was
    
    # Create more permutations of function word sequences
    for i in range(10):
        mu_bg = np.zeros(L2_D)
        for k in range(WINDOW_SIZE):
            idx = function_indices[(i + k) % len(function_indices)]
            mu_bg[k * LEX_D + idx] = 1.0
        
        bg_node = HFN(mu=mu_bg, sigma=np.ones(L2_D) * 0.2, id=f"bg_seq_{i}", use_diag=True)
        l2_priors.append(bg_node)
        l2_prior_ids.add(f"anc::bg_seq_{i}")

    # 3. Spawn Workers
    configs = [
        WorkerConfig(name="L1_Lexical", forest_id="sanchor_l1", cold_dir=Path("data/sanchor_run_l1"), 
                     max_hot=1000, degree=1.0, tau_sigma=1.0, common_d=LEX_D, 
                     source_nodes=list(lex_forest.active_nodes()), source_prior_ids=lex_prior_ids, prefix="w::"),
        WorkerConfig(name="L2_Thematic", forest_id="sanchor_l2", cold_dir=Path("data/sanchor_run_l2"), 
                     max_hot=500, degree=0.5, tau_sigma=0.05, common_d=L2_D, 
                     source_nodes=l2_priors, source_prior_ids=l2_prior_ids, prefix="anc::")
    ]

    task_queues = {c.name: mp.Queue() for c in configs}
    result_queue = mp.Queue()
    workers = {c.name: SovereignWorker(c, task_queues[c.name], result_queue) for c in configs}
    for w in workers.values(): w.start()

    t0 = time.perf_counter()
    l1_winners = [] # Circular buffer of L1 winner mus
    
    print("--- Processing Text Stream ---")
    # Process a larger subset (~6 pages)
    limit = 2000
    for i in range(2, limit - 2):
        # Compose context for the current token
        context_vec = compose_corpus_context(tokens[i-2], tokens[i-1], tokens[i+1], tokens[i+2])
        
        # Step A: L1 (Identify the Word Concept)
        task_queues["L1_Lexical"].put({"cmd": "OBSERVE", "x": context_vec})
        res = result_queue.get()
        mu_w = res["winner_mu"]
        
        if mu_w is not None:
            l1_winners.append(mu_w)
            if len(l1_winners) > WINDOW_SIZE:
                l1_winners.pop(0)
            
            # Step B: L2 (Synthesize Anchor)
            if len(l1_winners) == WINDOW_SIZE:
                msg_l2 = np.concatenate(l1_winners)
                task_queues["L2_Thematic"].put({"cmd": "OBSERVE", "x": msg_l2})
                result_queue.get() # Consume L2 result

        if (i+1) % 50 == 0:
            print(f"  Token {i+1}/{limit} processed...")

    # 3. Extract Summary
    print("\n--- Structural Summary Extraction ---")
    task_queues["L2_Thematic"].put({"cmd": "GET_TOP_ANCHORS"})
    anchors = result_queue.get()["anchors"]

    # Reverse Map L2 mu back to L1 word labels
    # mu_l2 is (WINDOW_SIZE * LEX_D)
    for j, anchor in enumerate(anchors):
        mu_l2 = anchor["mu"]
        weight = anchor["weight"]
        components = []
        for k in range(WINDOW_SIZE):
            slice_mu = mu_l2[k * LEX_D : (k + 1) * LEX_D]
            # Find nearest word in L1 Lexical world model
            # For simplicity, we'll just check the max component if it's one-hot-like
            # or we could query L1. Here we'll use VOCAB indices.
            idx = np.argmax(slice_mu)
            word = VOCAB[idx] if idx < len(VOCAB) else "???"
            components.append(word)
        
        print(f"  Anchor {j+1} (Weight {weight:.3f}): {' '.join(components)}")

    # Shutdown
    for name in workers: task_queues[name].put(None)
    for name in workers: workers[name].join()
    print(f"\nExperiment concluded in {time.perf_counter() - t0:.2f}s")

if __name__ == "__main__":
    main()
