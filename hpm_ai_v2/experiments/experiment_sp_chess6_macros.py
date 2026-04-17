"""
SP-Chess6: Temporal Abstraction (L5 Macros) - HPM-Native Version.
Validated with Chess Rule Priors and Native Evaluators.
"""
import numpy as np
import chess
import random
from hfn.forest import Forest
from hfn.observer import Observer
from hfn.evaluator import Evaluator
from hpm_ai_v2.domains.chess_hfn import ChessHFNEncoder
from hpm_ai_v2.domains.chess_transition import ChessTransitionLearner
from hpm_ai_v2.utils.chess_evaluator import ChessValueLearner
from hpm_ai_v2.utils.chess_macro_learner import ChessMacroLearner
from hpm_ai_v2.utils.chess_priors import ChessRulePriors
from hpm_ai_v2.agents.macro_chess_agent import MacroChessAgent
from hpm_ai_v2.experiments.experiment_sp_chess5_learned_eval import generate_games

def run_experiment():
    print("================================================================================")
    print("SP-Chess6: HPM-Native Temporal Abstraction (L5 Macros)")
    print("================================================================================")

    # 1. Setup HFN Infrastructure (Physics Layer - L4)
    print("Phase 1: Initializing L4 Forest and Injecting Chess Rule Priors...")
    # D=852 for [SummaryDelta(20), SquaresDelta(832)]
    physics_forest = Forest(D=852)
    observer = Observer(physics_forest)
    evaluator_native = Evaluator()
    
    # Inject "Innate" Rules
    ChessRulePriors.inject_priors(physics_forest, observer)

    # 2. Bootstrap Forward Model (L4 Physics)
    print("\nPhase 2: Training Forward Model (L4) in Forest...")
    learner = ChessTransitionLearner(forest=physics_forest, observer=observer)
    board = chess.Board()
    # Smoke test: 500 triples
    for _ in range(500):
        if board.is_game_over() or board.fullmove_number > 50: board = chess.Board()
        m = random.choice(list(board.legal_moves))
        b_curr = board.copy()
        board.push(m)
        learner.collect_triple(b_curr, m, board)
    learner.train()

    # 3. Train Value Function (Strategic Layer - L1/L2)
    print("\nPhase 3: Training Strategic Evaluator (L1/L2)...")
    evaluator = ChessValueLearner(learning_rate=0.03)
    game_data = generate_games(100)
    for epoch in range(1):
        for mus, outcome in game_data:
            for mu in mus[-10:]: evaluator.update(mu, outcome)

    # 4. Macro Discovery (Abstraction Layer - L5)
    print("\nPhase 4: Discovering L5 Macros with Native HPM Dynamics (30 Games)...")
    # Macros have dimension 23, so they need their own Forest
    macro_forest = Forest(D=23)
    macro_learner = ChessMacroLearner(evaluator=evaluator_native, sim_threshold=0.88)
    agent = MacroChessAgent(learner, evaluator, macro_learner, name="DiscoveryAgent")
    
    for g_idx in range(30):
        b = chess.Board()
        while not b.is_game_over() and b.fullmove_number < 25:
            m = agent.select_move(b, depth=2)
            if m is None: break
            b.push(m)
        if (g_idx+1) % 10 == 0:
            print(f"  Game {g_idx+1} | Macros Stabilized: {len(macro_learner.macros)}")

    # 5. Validation
    print("\nPhase 5: Validating Macro Abstractions...")
    if len(macro_learner.macros) > 0:
        macros_sorted = sorted(macro_learner.macros, key=lambda x: x.accuracy_ema, reverse=True)
        print(f"  Total Macros Discovered: {len(macros_sorted)}")
        for i, m in enumerate(macros_sorted[:3]):
            x_query = np.zeros(23)
            x_query[:20] = m.get_context_mu()
            x_query[20:] = m.node.mu[20:]
            score = evaluator_native.score(x_query, m.node, lambda_complexity=0.1)
            print(f"  Macro {i+1}: {m.moves} | Utility (EMA): {m.accuracy_ema:.4f} | HPM Score: {score:.4f}")
            
        # Check Prior Presence
        prior_nodes = [n for n in physics_forest.active_nodes() if n.relation_type == "move_prior"]
        print(f"  Physics Forest Status: {len(physics_forest)} nodes active | {len(prior_nodes)} Move Priors preserved.")
    else:
        print("  WARNING: No macros discovered.")

    # 6. Competition
    print("\nPhase 6: Competition - MacroAgent vs. SearchAgent")
    from hpm_ai_v2.agents.mental_chess_agent import MentalChessAgent
    search_agent = MentalChessAgent(learner, evaluator=evaluator, name="SearchAgent")
    
    def play_match(white, black, n=10):
        wins = 0
        for _ in range(n):
            b = chess.Board()
            while not b.is_game_over() and b.fullmove_number < 40:
                mover = white if b.turn == chess.WHITE else black
                m = mover.select_move(b, depth=2)
                if m is None: break
                b.push(m)
            res = b.result()
            if res == "1-0": wins += 1.0
            elif res == "1/2-1/2" or res == "*": wins += 0.5
        return wins / n

    wr = play_match(agent, search_agent, n=10)
    print(f"  MacroAgent (White) vs. SearchAgent: Win Rate {wr*100:.1f}%")

    print("\n================================================================================")
    print("Conclusion:")
    if len(macro_learner.macros) > 0 and len(prior_nodes) > 0:
        print("  SUCCESS: HPM successfully integrated Chess Priors and abstracted L5 Macros.")
    else:
        print("  PARTIAL: Framework alignment incomplete; check prior injection logic.")

if __name__ == "__main__":
    run_experiment()
