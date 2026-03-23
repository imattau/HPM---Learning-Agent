"""SP16: Geometric Rosetta — Relational Alignment via Shared Concept Discovery."""

import sys
import os
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))))

from benchmarks.rosetta_geometric_sim import CartesianAgent, TurtleAgent, RelayBuffer
from hpm.agents.l5_monitor import L5MetaMonitor

def run_geometric_rosetta():
    # 1. Setup Agents and the latent Relay
    # Both agents have the SAME internal concept of a 'Unit Square' (side=10)
    agent_a = CartesianAgent(side=10.0)
    agent_b = TurtleAgent(side=10.0)
    
    # The Relay adds a latent 45° rotation and 1.2x scale
    relay = RelayBuffer(rotation_deg=45.0, scale=1.2)
    monitor_b = L5MetaMonitor()
    
    print("\nSP16 Geometric Rosetta — Concept-Driven Discovery")
    print(f"Agent A: 'I am drawing a Square (side=10).'")
    print(f"Agent B: 'I know what a square is, but your points look different.'")
    print("-" * 55)

    # --- Phase 1: The Surprise (Semantic Mismatch) ---
    points_a = agent_a.get_square_points()
    
    # A sends first vertex
    p1_raw = points_a[0]
    p1_received = relay.transmit(p1_raw)
    
    # A sends second vertex
    p2_raw = points_a[1]
    p2_received = relay.transmit(p2_raw)
    
    # Agent B tries to guess where the second point SHOULD be if they spoke the same math
    # B's internal model says: side is 10, so point 2 should be (10, 0)
    prediction_b = np.array([10.0, 0.0])
    
    # Measure Surprise in L5
    monitor_b.update(prediction_b, p2_received)
    surprise = monitor_b._surprises[-1]
    print(f"Phase 1: Observation vs Internal Model")
    print(f"  Received Point 2: {p2_received.round(2)}")
    print(f"  Internal Model:   {prediction_b}")
    print(f"  L5 Surprise:      {surprise:.4f}")
    
    # --- Phase 2: The Litmus Discovery ---
    # Agent B realizes: "Wait, if A says this is a square, then the segment p1->p2
    # MUST be the first side. I will use this segment to align our maths."
    
    # Discovery Logic:
    # 1. Find the length of the received segment
    received_side_len = np.linalg.norm(p2_received - p1_received)
    # 2. Find the angle of the received segment
    delta = p2_received - p1_received
    received_angle = np.arctan2(delta[1], delta[0])
    
    # 3. Calculate Scale and Rotation
    # We know A's internal side is 10.0
    discovered_scale = received_side_len / 10.0
    discovered_rotation_rad = received_angle # Relative to B's 0-degree assumption
    
    print(f"\nPhase 2: Concept-Driven Discovery")
    print(f"  Discovered Scale factor: {discovered_scale:.2f}")
    print(f"  Discovered Rotation:     {np.degrees(discovered_rotation_rad):.1f}°")
    
    # --- Phase 3: Verified Alignment ---
    # Agent B builds a Discovery Matrix (Rosetta Stone) from these parameters
    c, s = np.cos(discovered_rotation_rad), np.sin(discovered_rotation_rad)
    M_discovery = np.array([
        [c, -s],
        [s,  c]
    ]) * discovered_scale
    
    # Now Agent B can zero-shot predict the rest of A's square
    correct_count = 0
    for i in range(2, 4):
        # A sends the point
        pi_received = relay.transmit(points_a[i])
        
        # B predicts using discovered matrix
        pi_predicted = points_a[i] @ M_discovery.T
        
        error = np.linalg.norm(pi_predicted - pi_received)
        print(f"Point {i+1}: Pred {pi_predicted.round(2)}, Recv {pi_received.round(2)}, Err {error:.4e}")
        
        if error < 1e-10:
            correct_count += 1
            
    success = (correct_count == 2)
    print(f"\nPhase 3: Shared Understanding")
    print(f"  Zero-shot accuracy on remaining points: {correct_count/2:.1%}")
    print(f"  RESULT: {'SUCCESS ✅' if success else 'FAILURE ❌'}")
    
    return success

if __name__ == "__main__":
    run_geometric_rosetta()
