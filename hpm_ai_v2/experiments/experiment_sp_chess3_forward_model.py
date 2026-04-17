"""
SP-Chess3: Fractal Board Representation and L4 Forward Model.
Validation of HPM's ability to learn board dynamics from examples.
"""
import numpy as np
import chess
import random
from hpm_ai_v2.domains.chess_hfn import ChessHFNEncoder
from hpm_ai_v2.domains.chess_transition import ChessTransitionLearner

def generate_random_triples(n: int = 1000):
    """Generate (board, move, next_board) triples."""
    triples = []
    board = chess.Board()
    for _ in range(n):
        if board.is_game_over() or board.fullmove_number > 50:
            board = chess.Board()
        
        # Random move
        moves = list(board.legal_moves)
        if not moves:
            board = chess.Board()
            moves = list(board.legal_moves)
            
        move = random.choice(moves)
        b_curr = board.copy()
        board.push(move)
        b_next = board.copy()
        
        triples.append((b_curr, move, b_next))
    return triples

def run_experiment():
    print("================================================================================")
    print("SP-Chess3: Fractal Forward Model Validation")
    print("================================================================================")

    # 1. Data Collection
    print("Phase 1: Generating move triples...")
    n_triples = 5000
    all_data = generate_random_triples(n_triples)
    split = int(0.8 * n_triples)
    train_data = all_data[:split]
    test_data = all_data[split:]
    print(f"  Training: {len(train_data)}, Testing: {len(test_data)}")

    # 2. Training
    print("\nPhase 2: Training L4 transition deltas...")
    learner = ChessTransitionLearner()
    for b_curr, move, b_next in train_data:
        learner.collect_triple(b_curr, move, b_next)
    learner.train()

    # 3. Evaluation: Prediction Accuracy
    print("\nPhase 3: Evaluating prediction accuracy on test data...")
    mse_summary = []
    mse_squares = []
    material_sign_correct = 0
    
    for b_curr, move, b_next in test_data:
        node_curr = ChessHFNEncoder.board_to_node(b_curr)
        node_next = ChessHFNEncoder.board_to_node(b_next)
        
        # Pass b_curr as context for capture identification
        pred_summary, pred_squares = learner.predict(node_curr, move, board_curr=b_curr)
        
        # MSE for 20-D summary
        mse_summary.append(np.mean((pred_summary - node_next.mu)**2))
        
        # MSE for 64x13 squares
        true_squares = np.concatenate([sq.mu for sq in node_next.inputs])
        mse_squares.append(np.mean((pred_squares - true_squares)**2))
        
        # Material balance sign prediction (pred_summary[0] is scaled material)
        true_gain = node_next.mu[0] - node_curr.mu[0]
        pred_gain = pred_summary[0] - node_curr.mu[0]
        if np.sign(true_gain) == np.sign(pred_gain) or (abs(true_gain) < 1e-6 and abs(pred_gain) < 1e-6):
            material_sign_correct += 1
            
    avg_mse_summary = np.mean(mse_summary)
    avg_mse_squares = np.mean(mse_squares)
    acc_sign = material_sign_correct / len(test_data)
    
    print(f"  Summary MSE: {avg_mse_summary:.6f}")
    print(f"  Squares MSE: {avg_mse_squares:.6f}")
    print(f"  Material Gain Sign Accuracy: {acc_sign*100:.2f}%")

    # 4. Tactical Alignment
    print("\nPhase 4: Tactical Alignment Test...")
    # Select boards with captures from the test set
    tactical_examples = []
    for b_curr, move, b_next in test_data:
        node_curr = ChessHFNEncoder.board_to_node(b_curr)
        node_next = ChessHFNEncoder.board_to_node(b_next)
        # Tactical gain > 0.1 (scaled)
        if (node_next.mu[0] - node_curr.mu[0]) * (1.0 if b_curr.turn == chess.WHITE else -1.0) > 0.05:
            tactical_examples.append((b_curr, move))
    
    if tactical_examples:
        print(f"  Found {len(tactical_examples)} tactical captures in test set.")
        alignment_count = 0
        for b_curr, true_move in tactical_examples:
            # Score all legal moves using learned forward model
            moves = list(b_curr.legal_moves)
            node_curr = ChessHFNEncoder.board_to_node(b_curr)
            
            scores = []
            for m in moves:
                pred_sum, _ = learner.predict(node_curr, m, board_curr=b_curr)
                gain = pred_sum[0] - node_curr.mu[0]
                if b_curr.turn == chess.BLACK: gain = -gain
                scores.append((m, gain))
                
            scores.sort(key=lambda x: x[1], reverse=True)
            best_move = scores[0][0]
            if best_move == true_move:
                alignment_count += 1
            elif abs(scores[0][1] - (scores[[m[0] for m in scores].index(true_move)][1])) < 1e-6:
                # If predicted gains are the same, count as aligned (e.g., capture any same-value piece)
                alignment_count += 1
                
        print(f"  Top Move Tactical Alignment: {alignment_count / len(tactical_examples) * 100:.2f}%")
    else:
        print("  No clear tactical captures found in random test sample for alignment check.")

    print("\n================================================================================")
    print("Conclusion:")
    if avg_mse_summary < 0.1 and acc_sign > 0.8:
        print("  SUCCESS: HPM successfully learned chess transitions via L4 deltas.")
    else:
        print("  PARTIAL: Learned model shows high MSE; consider larger dataset or board context.")

if __name__ == "__main__":
    run_experiment()
