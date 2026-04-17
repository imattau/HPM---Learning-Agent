"""
ChessMacroLearner — HFN-native temporal abstraction (L5 Macros) for chess (SP-Chess6).
Follows HPM layer 2/3: Pattern dynamics (Stabilization) and Evaluators (Score).
"""
from __future__ import annotations
import numpy as np
import chess
from typing import List, Tuple, Dict, Optional
from hfn.hfn import HFN
from hfn.evaluator import Evaluator

class ChessMacro:
    """
    Represents an L5 Temporal Macro as an HFN node.
    mu: concatenated [BoardSummary (20D), MoveVec (3D)].
    Tracks its own HPM weight/score via the Evaluator.
    """
    def __init__(self, node: HFN, moves: List[str], initial_weight: float = 0.1):
        self.node = node
        self.moves = moves
        self.weight = initial_weight
        self.accuracy_ema = 0.0 # Running average of utility
        self.uses = 0

    @property
    def id(self) -> str:
        return self.node.id

    def get_context_mu(self) -> np.ndarray:
        return self.node.mu[:20]

    def get_full_mu(self) -> np.ndarray:
        return self.node.mu

    def similarity(self, current_mu: np.ndarray) -> float:
        """Cosine similarity of context."""
        ctx = self.get_context_mu()
        norm_a = np.linalg.norm(ctx)
        norm_b = np.linalg.norm(current_mu)
        if norm_a == 0 or norm_b == 0: return 0.0
        return np.dot(ctx, current_mu) / (norm_a * norm_b)

class ChessMacroLearner:
    """
    Discovers and stabilizes L5 Macros using HPM principles.
    Uses an Evaluator for structural coherence checks.
    """
    def __init__(self, evaluator: Evaluator, sim_threshold: float = 0.90):
        self.evaluator = evaluator
        self.macros: List[ChessMacro] = []
        self.sim_threshold = sim_threshold
        self.next_id = 0

    def discover(self, board_mu: np.ndarray, move_uci: str, delta_v: float):
        """Discover or stabilize a macro based on utility delta_v."""
        # 1. Similarity search
        best_m = None
        best_sim = -1.0
        for m in self.macros:
            if m.moves[0] == move_uci:
                sim = m.similarity(board_mu)
                if sim > best_sim:
                    best_sim = sim
                    best_m = m

        if best_m and best_sim > self.sim_threshold:
            # 2. Stabilization (Layer 2)
            best_m.node.mu[:20] = 0.9 * best_m.node.mu[:20] + 0.1 * board_mu
            # EMA of accuracy
            best_m.accuracy_ema = 0.9 * best_m.accuracy_ema + 0.1 * delta_v
            best_m.uses += 1
            best_m.weight += 0.05
        elif delta_v > 0.001:
            # 3. Discovery (New Pattern)
            mu = np.zeros(23)
            mu[:20] = board_mu
            move = chess.Move.from_uci(move_uci)
            mu[20] = move.from_square / 64.0
            mu[21] = move.to_square / 64.0
            mu[22] = (move.promotion or 0) / 6.0
            
            node = HFN(mu=mu, sigma=np.full(23, 0.1))
            node.id = f"macro_{self.next_id}"
            node.relation_type = "macro"
            self.next_id += 1
            
            new_m = ChessMacro(node, [move_uci])
            new_m.accuracy_ema = delta_v
            new_m.uses = 1
            self.macros.append(new_m)

    def get_best_macro(self, board_mu: np.ndarray, legal_moves_uci: List[str]) -> Optional[ChessMacro]:
        """Weighted selection over candidate macros (Meta-Pattern Rule)."""
        candidates = []
        for m in self.macros:
            if m.moves[0] in legal_moves_uci:
                sim = m.similarity(board_mu)
                if sim > self.sim_threshold:
                    # Construct query point for Evaluator
                    # We want to see how well the 'current' board fits the macro
                    x = np.zeros(23)
                    x[:20] = board_mu
                    # Use macro's own move for the action part of the vector
                    x[20:] = m.node.mu[20:]
                    
                    # Accuracy is a mix of geometric fit and empirical utility
                    geom_acc = self.evaluator.accuracy(x, m.node)
                    utility_acc = m.accuracy_ema
                    
                    # HPM Score: (Accuracy - Complexity + Coherence)
                    coh = self.evaluator.coherence(m.node)
                    score = (0.4 * geom_acc + 0.4 * utility_acc + 0.2 * coh) - 0.1 * self.evaluator.description_length(m.node)
                    candidates.append((m, score))

        if not candidates: return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
