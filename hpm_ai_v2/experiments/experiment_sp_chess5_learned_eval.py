"""
SP-Chess5: Learned Evaluation via HFN.
Validation of HPM's ability to learn strategic board assessment from game outcomes.
"""
import numpy as np
import chess
import random
from typing import Tuple, List, Optional
from hpm_ai_v2.domains.chess_hfn import ChessHFNEncoder
from hpm_ai_v2.domains.chess_transition import ChessTransitionLearner
from hpm_ai_v2.agents.mental_chess_agent import MentalChessAgent
from hpm_ai_v2.utils.chess_evaluator import ChessValueLearner

def generate_games(n_games: int = 100) -> List[Tuple[List[np.ndarray], float]]:
    """Play random games and return (board_mus, outcome) pairs."""
    data = []
    for _ in range(n_games):
        board = chess.Board()
        mus = []
        while not board.is_game_over() and board.fullmove_number < 50:
            moves = list(board.legal_moves)
            if not moves: break
            move = random.choice(moves)
            mus.append(ChessHFNEncoder.get_board_summary(board))
            board.push(move)
            
        res = board.result()
        if res == "1-0": outcome = 1.0
        elif res == "0-1": outcome = 0.0
        else: outcome = 0.5
        data.append((mus, outcome))
    return data

def run_experiment():
    print("================================================================================")
    print("SP-Chess5: Learned Evaluation via HFN")
    print("================================================================================")

    # 1. Gather Physics (Forward Model)
    print("Phase 1: Training forward model (L4 physics)...")
    learner = ChessTransitionLearner()
    board = chess.Board()
    for _ in range(5000):
        if board.is_game_over() or board.fullmove_number > 50: board = chess.Board()
        moves = list(board.legal_moves)
        if not moves: break
        move = random.choice(moves)
        b_curr = board.copy()
        board.push(move)
        learner.collect_triple(b_curr, move, board)
    learner.train()

    # 2. Gather Experience
    print("\nPhase 2: Generating 1,000 random games for outcome learning...")
    game_data = generate_games(1000)
    
    # 3. Train Value Function
    print("Phase 3: Training HFN Value Learner (10 epochs)...")
    evaluator = ChessValueLearner(learning_rate=0.03)
    for epoch in range(10):
        random.shuffle(game_data)
        for mus, outcome in game_data:
            # Focus on terminal states (last 15)
            for mu in mus[-15:]:
                evaluator.update(mu, outcome)
    
    print(f"  [Debug] Evaluator Weight for Material (mu[0]): {evaluator.value_node.mu[0]:.4f}")
    print(f"  [Debug] Evaluator Weight for White Queen (mu[5]): {evaluator.value_node.mu[5]:.4f}")
            
    # 4. Evaluation: Validation on Test Positions
    print("\nPhase 4: Evaluating Value Function Alignment...")
    test_boards = [chess.Board() for _ in range(20)]
    for b in test_boards:
        for _ in range(random.randint(5, 30)):
            m = random.choice(list(b.legal_moves))
            b.push(m)
            if b.is_game_over(): break
            
    correlations = []
    for b in test_boards:
        mu = ChessHFNEncoder.get_board_summary(b)
        pred = evaluator.predict(mu)
        def get_val(p):
            if p is None: return 0.0
            vals = {1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 0.0}
            return vals[p.piece_type]
        white = sum(get_val(b.piece_at(s)) for s in chess.SQUARES if b.piece_at(s) and b.piece_at(s).color == chess.WHITE)
        black = sum(get_val(b.piece_at(s)) for s in chess.SQUARES if b.piece_at(s) and b.piece_at(s).color == chess.BLACK)
        material = (white - black)
        correlations.append((pred, material))
        
    preds = [c[0] for c in correlations]
    mats = [c[1] for c in correlations]
    corr = np.corrcoef(preds, mats)[0, 1]
    print(f"  Correlation (Learned Value vs. Material Balance): {corr:.4f}")

    # 5. Competition: Learned Evaluator vs. True Random
    print("\nPhase 5: Competition - Learned Agent (Depth 2) vs. RandomBaseline")
    from hpm_ai_v2.experiments.experiment_sp_chess4_multi_agent import BaselineAgent
    
    learned_agent = MentalChessAgent(learner, evaluator=evaluator, name="LearnedAgent")
    random_agent = BaselineAgent(mode="random")
    
    def play_match_v_random(white, black, n=20):
        wins = 0
        for _ in range(n):
            b = chess.Board()
            while not b.is_game_over() and b.fullmove_number < 60:
                mover = white if b.turn == chess.WHITE else black
                
                # Safer dispatch
                if isinstance(mover, MentalChessAgent):
                    m = mover.select_move(b, depth=2)
                else:
                    m = mover.select_move(b)
                
                if m is None: break
                b.push(m)
            res = b.result()
            if res == "1-0": wins += 1.0
            elif res == "1/2-1/2" or res == "*": wins += 0.5
        return wins / n

    wr = play_match_v_random(learned_agent, random_agent, n=10)
    print(f"  LearnedAgent (White) vs. RandomBaseline: Win Rate {wr*100:.1f}%")
    
    wr_rev = play_match_v_random(random_agent, learned_agent, n=10)
    print(f"  RandomBaseline (White) vs. LearnedAgent: LearnedAgent Win Rate {(1.0-wr_rev)*100:.1f}%")

    print("\n================================================================================")
    print("Conclusion:")
    if corr > 0.4 and (wr > 0.6 or (1.0-wr_rev) > 0.6):
        print("  SUCCESS: HPM successfully learned a strategic evaluation function.")
    else:
        print("  PARTIAL: Win rates are lower than expected; check training parameters.")

if __name__ == "__main__":
    run_experiment()
