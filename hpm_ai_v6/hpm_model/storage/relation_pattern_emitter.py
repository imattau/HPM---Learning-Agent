from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell


class RelationPatternEmitter:
    """
    Accumulates (source, relation_name, target) observations and emits
    dim-2 Cells whose embedding converges to the mean (target-source) offset
    for each relation type.

    These dim-2 cells enter the pattern polygraph like any other analogy cell,
    making relation patterns subject to replicator dynamics and pager persistence.
    """

    def __init__(self, embedding_dim: int = 64, lr: float = 0.05):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self._cells: Dict[str, Cell] = {}
        self._counts: Dict[str, int] = {}

    def observe(self, source: Cell, relation_name: str, target: Cell) -> Cell:
        src = np.asarray(source.as_numpy(), dtype=np.float32)
        tgt = np.asarray(target.as_numpy(), dtype=np.float32)
        if src.shape[0] != self.embedding_dim or tgt.shape[0] != self.embedding_dim:
            if relation_name in self._cells:
                return self._cells[relation_name]
            zero_emb = np.zeros(self.embedding_dim, dtype=np.float32)
            cell = Cell(name=f"rel_{relation_name}", dim=2, embedding=zero_emb.tolist())
            self._cells[relation_name] = cell
            self._counts[relation_name] = 0
            return cell

        offset = tgt - src
        if relation_name not in self._cells:
            cell = Cell(
                name=f"rel_{relation_name}",
                dim=2,
                embedding=offset.tolist(),
            )
            self._cells[relation_name] = cell
            self._counts[relation_name] = 1
        else:
            old_emb = np.asarray(self._cells[relation_name].as_numpy(), dtype=np.float32)
            new_emb = old_emb + self.lr * (offset - old_emb)
            self._cells[relation_name] = Cell(
                name=f"rel_{relation_name}",
                dim=2,
                embedding=new_emb.tolist(),
            )
            self._counts[relation_name] += 1

        return self._cells[relation_name]

    def get_relation_cells(self) -> List[Tuple[Cell, float]]:
        return [
            (cell, float(self._counts[rel_name]))
            for rel_name, cell in self._cells.items()
        ]
