"""
SP-Chess4: Multi-Agent Competition with Fractal Forward Models.
Competition between HPM agents using shared L4 mental simulation.
"""
import numpy as np
import chess
import random
from typing import Tuple, List, Optional
from hpm_ai_v2.domains.chess_transition import ChessTransitionLearner
from hpm_ai_v2.agents.mental_chess_agent import MentalChessAgent

def generate_training_data(n: int = 5000) -> List[Tuple[chess.Board, chess.Move, chess.Board]]:
    """Gather (board, move, next_board) triples for training."""
    triples = []
    board = chess.Board()
    for _ in range(n):
        if board.is_game_over() or board.fullmove_number > 50:
            board = chess.Board()
        moves = list(board.legal_moves)
        if not moves: break
        move = random.choice(moves)
        b_curr = board.copy()
        board.push(move)
        b_next = board.copy()
        triples.append((b_curr, move, b_next))
    return triples

class BaselineAgent:
    """Simple agent for comparison."""
    def __init__(self, mode: str = "random"):
        self.mode = mode
        self.name = f"Baseline({mode})"

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        moves = list(board.legal_moves)
        if not moves: return None
        return random.choice(moves)

def play_match(agent_white, agent_black, n_games: int = 10) -> float:
    """Play a match and return White's win rate."""
    white_wins = 0
    for i in range(n_games):
        board = chess.Board()
        while not board.is_game_over() and board.fullmove_number < 60:
            mover = agent_white if board.turn == chess.WHITE else agent_black
            move = mover.select_move(board)
            if move is None: break
            board.push(move)
            
        res = board.result()
        if res == "1-0": white_wins += 1.0
        elif res == "1/2-1/2" or res == "*": white_wins += 0.5
        
    return white_wins / n_games

def run_experiment():
    print("================================================================================")
    print("SP-Chess4: Multi-Agent Competition with Fractal Forward Models")
    print("================================================================================")

    # 1. Training "Domain Physics" (Shared L4 Knowledge)
    print("Phase 1: Training shared L4 forward model on 5,000 triples...")
    learner = ChessTransitionLearner()
    data = generate_training_data(5000)
    for b_curr, move, b_next in data:
        learner.collect_triple(b_curr, move, b_next)
    learner.train()

    # 2. Match: Mental (L4) Agent vs. Random Baseline
    print("\nPhase 2: Match - MentalAgent (L4) vs. RandomBaseline")
    mental_agent = MentalChessAgent(learner, name="MentalAgent")
    random_agent = BaselineAgent(mode="random")
    
    wr_v_rand = play_match(mental_agent, random_agent, n_games=10)
    print(f"  MentalAgent (White) vs. RandomBaseline: Win Rate {wr_v_rand*100:.1f}%")
    
    wr_v_rand_rev = play_match(random_agent, mental_agent, n_games=10)
    print(f"  RandomBaseline (White) vs. MentalAgent: MentalAgent Win Rate {(1.0-wr_v_rand_rev)*100:.1f}%")

    # 3. Match: Mental vs. Mental (Symmetric Play)
    print("\nPhase 3: Match - MentalAgent vs. MentalAgent (Shared Space)")
    wr_v_self = play_match(mental_agent, mental_agent, n_games=6)
    print(f"  MentalAgent vs. MentalAgent: Win Rate {wr_v_self*100:.1f}%")

    # 4. Match: Depth-2 vs. Depth-1
    print("\nPhase 4: Match - MentalAgent(Depth 2) vs. MentalAgent(Depth 1)")
    
    def play_match_depth(n_games: int = 10) -> float:
        d2_wins = 0
        for i in range(n_games):
            board = chess.Board()
            if i % 2 == 0:
                while not board.is_game_over() and board.fullmove_number < 60:
                    mover_move = mental_agent.select_move(board, depth=2) if board.turn == chess.WHITE else mental_agent.select_move(board, depth=1)
                    if mover_move is None: break
                    board.push(mover_move)
                res = board.result()
                if res == "1-0": d2_wins += 1.0
                elif res == "1/2-1/2" or res == "*": d2_wins += 0.5
            else:
                while not board.is_game_over() and board.fullmove_number < 60:
                    mover_move = mental_agent.select_move(board, depth=1) if board.turn == chess.WHITE else mental_agent.select_move(board, depth=2)
                    if mover_move is None: break
                    board.push(mover_move)
                res = board.result()
                if res == "0-1": d2_wins += 1.0
                elif res == "1/2-1/2" or res == "*": d2_wins += 0.5
        return d2_wins / n_games

    wr_depth = play_match_depth(n_games=10)
    print(f"  MentalAgent(Depth 2) Win Rate: {wr_depth*100:.1f}%")

    print("\n================================================================================")
    print("Conclusion:")
    if wr_v_rand > 0.6 and wr_depth >= 0.5:
        print("  SUCCESS: HPM agents compete using shared fractal forward models and planning.")
    else:
        print("  PARTIAL: Win rates are lower than expected; check L4 delta quality.")

if __name__ == "__main__":
    run_experiment()
