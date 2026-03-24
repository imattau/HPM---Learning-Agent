"""Geometric Rosetta Simulator: Cartesian vs Turtle substrates with latent relay transformations."""

import numpy as np
from dataclasses import dataclass

@dataclass
class GeometricMessage:
    agent_id: str
    data: np.ndarray # The vector being sent
    metadata: str # Optional context (e.g. 'point_1')

class RelayBuffer:
    """The shared medium that adds a latent transformation to all messages."""
    def __init__(self, rotation_deg: float = 45.0, scale: float = 1.2):
        theta = np.radians(rotation_deg)
        c, s = np.cos(theta), np.sin(theta)
        # 2D Rotation + Scaling matrix
        self.M_latent = np.array([
            [c, -s],
            [s,  c]
        ]) * scale

    def transmit(self, data: np.ndarray) -> np.ndarray:
        """Data from A passes through the latent transformation before reaching B."""
        # Ensure data is (N, 2)
        return data @ self.M_latent.T

class CartesianAgent:
    """Agent A: Absolute (x, y) coordinates."""
    def __init__(self, side: float = 10.0):
        self.side = side
        
    def get_square_points(self) -> np.ndarray:
        """Returns the 4 vertices of a square."""
        return np.array([
            [0, 0],
            [self.side, 0],
            [self.side, self.side],
            [0, self.side]
        ], dtype=float)

class TurtleAgent:
    """Agent B: Relative (distance, angle_delta) movements."""
    def __init__(self, side: float = 10.0):
        self.side = side
        
    def get_square_moves(self) -> np.ndarray:
        """Returns the 4 moves of a square."""
        return np.array([
            [self.side, 0],
            [self.side, 90],
            [self.side, 90],
            [self.side, 90]
        ], dtype=float)
    
    def predict_next_move(self, history: list[np.ndarray]) -> np.ndarray:
        """Agent B's internal 'Square Law': Move 's', then Turn 90."""
        if not history:
            return np.array([self.side, 0])
        return np.array([self.side, 90])

def cartesian_to_relative(points: np.ndarray) -> np.ndarray:
    """Helper to convert A's format to B's format for comparison."""
    moves = []
    for i in range(len(points)):
        p_curr = points[i]
        p_prev = points[i-1] if i > 0 else np.array([0, -1.0]) # Dummy start
        
        dist = np.linalg.norm(p_curr - p_prev)
        # This is a simplification; real turtle math is more complex
        moves.append([dist, 90 if i > 0 else 0])
    return np.array(moves)
