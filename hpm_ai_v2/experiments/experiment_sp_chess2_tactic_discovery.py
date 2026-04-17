"""
SP-Chess2: Autonomous Tactic Discovery via Self-Play
"""
import chess
import numpy as np
import time
import os
from typing import List, Any, Tuple, Optional
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.l2_macro import L2MacroMixin
from hpm_ai_v2.domains.chess_domain import ChessDomainConfig
from hpm_ai_v2.domains.chess_renderer import ChessRenderer
from hpm_ai_v2.utils.oracle.chess_oracle import ChessOracle

# ----------------------------------------------------------------------
# 1. Physical Chess Logic (Non-HPM)
# ----------------------------------------------------------------------

def get_piece_value(piece: Optional[chess.Piece]) -> float:
    if piece is None: return 0.0
    values = {chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
              chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 100.0}
    return values[piece.piece_type]

def material_balance(board: chess.Board) -> float:
    white = sum(get_piece_value(board.piece_at(s)) for s in chess.SQUARES if board.piece_at(s) and board.piece_at(s).color == chess.WHITE)
    black = sum(get_piece_value(board.piece_at(s)) for s in chess.SQUARES if board.piece_at(s) and board.piece_at(s).color == chess.BLACK)
    return white - black

# ----------------------------------------------------------------------
# 2. HPM Chess Agent
# ----------------------------------------------------------------------

class ChessAgent(L2MacroMixin, BaseHFNAgent):
    """
    Agent that uses HPM discovery to improve chess evaluation.
    """
    def __init__(self, name: str = "Learner"):
        config = ChessDomainConfig()
        renderer = ChessRenderer(config)
        super().__init__(config=config, renderer=renderer)
        self.name = name
        self.oracle = ChessOracle(config)
        self.counting_oracle.oracle = self.oracle
        
        # Prime priors for guidance
        offset = self.s_dim + self.dim
        node_to = self.forest.get("prior_rule_CHESS_PIECE_VALUE_AT_TO")
        if node_to: node_to.mu[offset + 4] = 10.0
        node_att = self.forest.get("prior_rule_CHESS_SQUARE_ATTACKED_BY_ME_AT_TO")
        if node_att: node_att.mu[offset + 5] = 10.0
            
        # Registration strategies
        self.add_strategy("exact", self._try_exact)
        self.add_strategy("bfs", self._try_bfs)

        # Tactical library (registered macros)
        self.tactical_macros = []
        
        # Explicitly set candidate ops
        self._candidate_ops = []
        for c in config.concepts:
            node = self.forest.get(f"prior_rule_{c}")
            if node: self._candidate_ops.append(node)

    def evaluate_board(self, board: chess.Board, color: chess.Color) -> float:
        """Score the board from the perspective of 'color'."""
        score = material_balance(board)
        if color == chess.BLACK:
            score = -score
        return score

    def select_move(self, board: chess.Board, depth: int = 1) -> Optional[chess.Move]:
        """Minimax move selection."""
        best_move = None
        best_val = -np.inf
        turn = board.turn
        
        moves = list(board.legal_moves)
        if not moves: return None
        np.random.shuffle(moves)
        
        for move in moves:
            t_bonus = 0.0
            if self.tactical_macros:
                for macro in self.tactical_macros:
                    code = self.renderer.render(macro)
                    results, errors = self.executor.run_batch(code, [(board, move)])
                    if not errors[0]:
                        t_bonus += float(results[0])
            
            board.push(move)
            val = -self.negamax(board, depth - 1, -np.inf, np.inf, not turn)
            board.pop()
            
            total_val = val + 0.5 * t_bonus # High weight to see impact
            if total_val > best_val:
                best_val = total_val
                best_move = move
                
        return best_move

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, color: chess.Color) -> float:
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board, color)
        
        value = -np.inf
        for move in board.legal_moves:
            board.push(move)
            value = max(value, -self.negamax(board, depth - 1, -beta, -alpha, not color))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

# ----------------------------------------------------------------------
# 3. Experiment Loop
# ----------------------------------------------------------------------

def play_game(white_agent: ChessAgent, black_agent: ChessAgent) -> Tuple[float, List[Tuple[chess.Board, chess.Move]]]:
    """Play one game and return (white_result, material_gaining_moves)."""
    board = chess.Board()
    examples = []
    
    while not board.is_game_over() and board.fullmove_number < 30:
        mover = white_agent if board.turn == chess.WHITE else black_agent
        move = mover.select_move(board)
        if move is None: break
        
        mover_color = board.turn
        b_before = board.copy()
        mat_before = material_balance(b_before)
        if mover_color == chess.BLACK: mat_before = -mat_before
        
        board.push(move)
        
        mat_after = material_balance(board)
        if mover_color == chess.BLACK: mat_after = -mat_after
        
        gain = mat_after - mat_before
        if gain > 0.1:
            examples.append((b_before, move, gain))
            
    res = board.result()
    if res == "1-0": outcome = 1.0
    elif res == "0-1": outcome = 0.0
    else: outcome = 0.5
    
    return outcome, examples

def run_generation(agent: ChessAgent, generation: int, n_games: int = 4) -> float:
    print(f"\n--- [Generation {generation}] ---")
    baseline = ChessAgent(name="Baseline")
    
    wins = 0
    all_examples = []
    
    for i in range(n_games):
        if i % 2 == 0:
            res, ex = play_game(agent, baseline)
            wins += res
            all_examples.extend(ex)
        else:
            res, ex = play_game(baseline, agent)
            wins += (1.0 - res)
            
    win_rate = wins / n_games
    print(f"  Win Rate: {win_rate:.2f} ({wins}/{n_games})")
    
    if all_examples:
        if len(all_examples) > 10:
            indices = np.random.choice(len(all_examples), 10, replace=False)
            sampled_examples = [all_examples[i] for i in indices]
        else:
            sampled_examples = all_examples
            
        print(f"  Gathered {len(all_examples)} positive examples. Running discovery on {len(sampled_examples)}...")
        inputs = [(ex[0], ex[1]) for ex in sampled_examples]
        outputs = [ex[2] for ex in sampled_examples]
        
        success, code, strat, path = agent.solve(inputs, outputs, goal_type="map", beam_width=10, max_depth=3)
        if success and path:
            # Check for duplicates
            is_new = True
            for m in agent.tactical_macros:
                if agent.renderer.render(m) == code:
                    is_new = False
                    break
            
            if is_new:
                print(f"  DISCOVERED NEW TACTIC using {strat}. Path length: {len(path)}")
                macro = agent._compose_sequence(path)
                if macro:
                    macro.id = f"tactic_gen{generation}"
                    agent.tactical_macros.append(macro)
                    agent.observer.register(macro)
            else:
                print(f"  Discovery found existing tactic. Skipping registration.")
                
    return win_rate

def main():
    print("================================================================================")
    print("SP-Chess2: Autonomous Tactic Discovery via Self-Play")
    print("================================================================================")
    
    agent = ChessAgent("Learner")
    
    history = []
    for gen in range(1, 4): 
        wr = run_generation(agent, gen, n_games=4)
        history.append(wr)
        
    print("\n================================================================================")
    print("Summary:")
    print(f"  Win Rate Progression: {history}")
    print(f"  Final Tactical Library Size: {len(agent.tactical_macros)}")
    print("================================================================================")

if __name__ == "__main__":
    main()
