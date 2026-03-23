"""Rosetta Simulator: Divergent mathematical substrates for shape reasoning."""

import numpy as np
from dataclasses import dataclass

@dataclass
class Shape:
    name: str
    euclidean: np.ndarray # [n_sides, l1, l2, l3, l4, angle_sum]
    coordinate: np.ndarray # [x1, y1, x2, y2, x3, y3, x4, y4]

def get_square(side: float = 1.0, x_off: float = 0.0, y_off: float = 0.0) -> Shape:
    coords = np.array([
        x_off, y_off, 
        x_off + side, y_off, 
        x_off + side, y_off + side, 
        x_off, y_off + side
    ], dtype=float)
    # Center the coordinates
    coords[0::2] -= coords[0::2].mean()
    coords[1::2] -= coords[1::2].mean()
    return Shape(
        name="square",
        euclidean=np.array([4, side, side, side, side, 360], dtype=float),
        coordinate=coords
    )

def get_rectangle(w: float = 1.0, h: float = 2.0, x_off: float = 0.0, y_off: float = 0.0) -> Shape:
    coords = np.array([
        x_off, y_off, 
        x_off + w, y_off, 
        x_off + w, y_off + h, 
        x_off, y_off + h
    ], dtype=float)
    coords[0::2] -= coords[0::2].mean()
    coords[1::2] -= coords[1::2].mean()
    return Shape(
        name="rectangle",
        euclidean=np.array([4, w, h, w, h, 360], dtype=float),
        coordinate=coords
    )

def get_triangle(side: float = 1.0, x_off: float = 0.0, y_off: float = 0.0) -> Shape:
    h = (np.sqrt(3)/2) * side
    coords = np.array([
        x_off, y_off, 
        x_off + side, y_off, 
        x_off + side/2, y_off + h, 
        x_off, y_off
    ], dtype=float)
    coords[0::2] -= coords[0::2].mean()
    coords[1::2] -= coords[1::2].mean()
    return Shape(
        name="triangle",
        euclidean=np.array([3, side, side, side, 0, 180], dtype=float),
        coordinate=coords
    )

def generate_shared_shapes(n: int = 20, seed: int = 42) -> list[Shape]:
    """Generate a sequence of shapes for Rosetta alignment."""
    rng = np.random.default_rng(seed)
    shapes = []
    types = ["square", "rectangle", "triangle"]
    
    for _ in range(n):
        t = rng.choice(types)
        size = rng.uniform(0.5, 5.0)
        x = rng.uniform(-10, 10)
        y = rng.uniform(-10, 10)
        
        if t == "square":
            shapes.append(get_square(size, x, y))
        elif t == "rectangle":
            shapes.append(get_rectangle(size, size * rng.uniform(1.1, 3.0), x, y))
        else:
            shapes.append(get_triangle(size, x, y))
            
    return shapes
