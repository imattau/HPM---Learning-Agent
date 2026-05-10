from typing import List, Optional, Callable
import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell

class Polygraph:
    """
    Manages a collection of Cells and provides polygraph operations.
    Covers composition (Section 3.4) and boundary matching.
    """
    def __init__(self, cells: Optional[List[Cell]] = None):
        self.cells = cells or []

    def add_cell(self, cell: Cell):
        if cell not in self.cells:
            self.cells.append(cell)

    def get_cells_by_dim(self, dim: int) -> List[Cell]:
        return [c for c in self.cells if c.dim == dim]

    def compose_1cells(self, c1: Cell, c2: Cell, 
                        compose_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = lambda u, v: (u + v) / 2,
                        name: Optional[str] = None) -> Cell:
        """
        Vertical composition of two 1-cells (f: A->B, g: B->C => g.f: A->C).
        Section 3.4.
        """
        if c1.dim != 1 or c2.dim != 1:
            raise ValueError("Composition only implemented for 1-cells in this version.")
        
        if c1.target.name != c2.source.name:
            raise ValueError(f"Boundary mismatch: {c1.target.name} != {c2.source.name}")
        
        new_emb = compose_fn(c1.embedding, c2.embedding)
        new_name = name or f"({c1.name}∘{c2.name})"
        
        return Cell(
            name=new_name,
            dim=1,
            embedding=new_emb,
            source=c1.source,
            target=c2.target
        )

    def find_analogies(self, threshold: float = 0.8) -> List[Cell]:
        """
        Discovers 2-cells (analogies) based on embedding similarity between 1-cells.
        Section 2.3.
        """
        analogies = []
        one_cells = self.get_cells_by_dim(1)
        for i, c1 in enumerate(one_cells):
            for j, c2 in enumerate(one_cells):
                if i >= j: continue
                sim = c1.similarity(c2)
                if sim > threshold:
                    name = f"analogy({c1.name},{c2.name})"
                    # 2-cell represents the transformation/analogy
                    analogies.append(Cell(
                        name=name,
                        dim=2,
                        embedding=(c1.embedding + c2.embedding) / 2,
                        source=c1,
                        target=c2
                    ))
        return analogies
