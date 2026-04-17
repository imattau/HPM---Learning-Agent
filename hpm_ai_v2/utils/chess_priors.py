"""
ChessRulePriors — HFN-native piece movement priors for chess (SP-Chess7).
Provides the "Innate Structure" required by the HPM framework.
"""
from __future__ import annotations
import numpy as np
import chess
from typing import List, Dict, Tuple
from hfn.hfn import HFN

class ChessRulePriors:
    """
    Generates HFN nodes representing the deterministic rules of chess.
    These nodes encode the L4 transition (delta) for each piece type.
    """
    
    @staticmethod
    def get_piece_move_priors() -> List[HFN]:
        """
        Returns a list of protected HFN nodes for each piece type.
        These are 'template' priors that the agent uses to recognize legal moves.
        """
        priors = []
        
        # 1-6: White P, N, B, R, Q, K
        # 7-12: Black P, N, B, R, Q, K
        piece_names = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"]
        
        for p_idx in range(1, 13):
            p_type = ((p_idx - 1) % 6) + 1
            is_black = p_idx > 6
            name = f"{'BLACK' if is_black else 'WHITE'}_{piece_names[p_type-1]}"
            
            # We create a prior that recognizes a piece move
            # mu: [SummaryDelta (20), SquaresDelta (832)]
            mu = np.zeros(20 + 832)
            # Piece identity in summary delta (approximate)
            mu[p_type + (6 if is_black else 0)] = 0.1
            
            node = HFN(mu=mu, sigma=np.full(mu.size, 0.1))
            node.id = f"prior_rule_MOVE_{name}"
            node.relation_type = "move_prior"
            priors.append(node)
            
        return priors

    @staticmethod
    def inject_priors(forest: 'TieredForest', observer: 'Observer'):
        """Inject chess rules into the forest and mark as protected."""
        priors = ChessRulePriors.get_piece_move_priors()
        prior_ids = set()
        for node in priors:
            forest.register(node)
            prior_ids.add(node.id)
        # Mark as protected in forest (HFN 2.0 standard)
        if hasattr(forest, 'set_protected'):
            forest.set_protected(prior_ids)
        # Also protect in observer for dynamic updates
        if hasattr(observer, 'protected_ids'):
            observer.protected_ids.update(prior_ids)
        print(f"Injected {len(priors)} Chess Rule Priors.")
