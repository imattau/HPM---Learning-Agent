"""
run_physics_benchmark.py - Evaluate the Physics Manager Agent on Formula Discovery (Induction).
"""

import sys, os
# Add parent to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hpm_ai_v3.agents.manager import PhysicsManagerAgent
import torch
import numpy as np

def generate_problems():
    """
    Generate induction problems where the agent sees observations and must find the law.
    """
    return [
        {
            "id": 1,
            "law": "s = 0.5 * g * t**2",
            "description": "Free fall from rest.",
            "observations": [
                {"t": 1.0, "s": 4.9},
                {"t": 2.0, "s": 19.6},
                {"t": 3.0, "s": 44.1}
            ],
            "query": {"t": 4.0},
            "expected": 78.4,
            "tolerance": 0.5
        },
        {
            "id": 2,
            "law": "F = m * a",
            "description": "Newton's second law.",
            "observations": [
                {"m": 2.0, "a": 5.0, "F": 10.0},
                {"m": 5.0, "a": 2.0, "F": 10.0},
                {"m": 10.0, "a": 1.0, "F": 10.0}
            ],
            "query": {"m": 4.0, "a": 3.0},
            "expected": 12.0,
            "tolerance": 0.1
        },
        {
            "id": 3,
            "law": "v = u + a * t",
            "description": "Constant acceleration velocity.",
            "observations": [
                {"u": 0.0, "a": 2.0, "t": 1.0, "v": 2.0},
                {"u": 0.0, "a": 2.0, "t": 5.0, "v": 10.0},
                {"u": 5.0, "a": 1.0, "t": 2.0, "v": 7.0}
            ],
            "query": {"u": 10.0, "a": 2.0, "t": 3.0},
            "expected": 16.0,
            "tolerance": 0.1
        }
    ]

def evaluate():
    print("=== Physics Benchmark: Formula Discovery from Observations ===\n")
    manager = PhysicsManagerAgent()
    problems = generate_problems()
    
    correct = 0
    for p in problems:
        print(f"Problem {p['id']}: {p['description']}")
        print(f"  Observations: {p['observations']}")
        
        try:
            # We wrap the problem in an 'induction' task
            task = {
                "mode": "induction",
                "observations": p["observations"],
                "query": p["query"]
            }
            
            res = manager.process(task)
            pred = res.get("answer")
            discovered_law = res.get("discovered_law")
            
            if pred is not None:
                if isinstance(pred, torch.Tensor): pred = pred.item()
                
                diff = abs(pred - p["expected"])
                if diff <= p["tolerance"]:
                    status = "✓ CORRECT"
                    correct += 1
                else:
                    status = "✗ INCORRECT"
                
                print(f"  Discovered Law: {discovered_law}")
                print(f"  Prediction for {p['query']}: {pred:.3f} (Expected: {p['expected']}) -> {status}")
            else:
                print(f"  Result: FAILED (No prediction found) -> ✗ INCORRECT")
                if res.get("error"): print(f"    Error: {res['error']}")
            
            print(f"  Workflow: {' -> '.join(res.get('workflow', []))}")
        except Exception as e:
            import traceback
            # traceback.print_exc()
            print(f"  Error processing problem: {e}")
        print("-" * 50)
        
    print(f"\nFinal Score: {correct}/{len(problems)} ({correct/len(problems)*100:.1f}%)")

if __name__ == "__main__":
    evaluate()
