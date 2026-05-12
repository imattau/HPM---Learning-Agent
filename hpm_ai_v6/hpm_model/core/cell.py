from typing import Optional, List, Union, Any
import numpy as np
from pydantic import BaseModel, Field, ConfigDict
import torch

class Cell(BaseModel):
    """
    Core HPM Cell (Polygraph Node).
    Supports hierarchical structure (dim), vector embeddings, and relational links.
    All matching and predictions are vector-based.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    dim: int = Field(default=0, ge=0)
    embedding: Any
    
    # Hierarchical boundaries (source/target links for dim > 0)
    source: Optional['Cell'] = None
    target: Optional['Cell'] = None
    
    # Replicator dynamics weight
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    @staticmethod
    def _unwrap_embedding(value: Union['Cell', np.ndarray, torch.Tensor, Any]) -> Any:
        if isinstance(value, Cell):
            return value.embedding
        return value

    @staticmethod
    def _to_numpy(value: Union['Cell', np.ndarray, torch.Tensor, Any]) -> np.ndarray:
        value = Cell._unwrap_embedding(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value, dtype=float)

    @staticmethod
    def _to_tensor(value: Union['Cell', np.ndarray, torch.Tensor, Any], *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        value = Cell._unwrap_embedding(value)
        if isinstance(value, torch.Tensor):
            return value.detach().clone().to(dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)

    def as_numpy(self) -> np.ndarray:
        return self._to_numpy(self.embedding)

    def as_tensor(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self._to_tensor(self.embedding, dtype=dtype)

    def similarity(self, other: Union['Cell', np.ndarray, torch.Tensor]) -> float:
        """Cosine similarity between this cell's embedding and another cell or vector."""
        self_emb = self._to_numpy(self.embedding)
        other_emb = self._to_numpy(other)
        denom = (np.linalg.norm(self_emb) * np.linalg.norm(other_emb) + 1e-9)
        return float(np.dot(self_emb, other_emb) / denom)

    def similarity_tensor(self, other: Union['Cell', np.ndarray, torch.Tensor], *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        self_emb = self.as_tensor(dtype=dtype)
        other_emb = self._to_tensor(other, dtype=dtype)
        denom = torch.norm(self_emb) * torch.norm(other_emb) + 1e-9
        return torch.dot(self_emb, other_emb) / denom

    def predict_probs(self, population: List['Cell'], temperature: float = 1.0) -> np.ndarray:
        """
        Generalized hierarchical prediction.
        Returns a softmax distribution over the provided population based on vector similarity.
        """
        if not population:
            return np.array([])
        
        # Vector-based matching scores
        scores = np.array([self.similarity(c) for c in population], dtype=float)
        
        # Softmax
        exp_scores = np.exp(scores / temperature)
        return exp_scores / (np.sum(exp_scores) + 1e-9)

    def predict_probs_tensor(
        self,
        population: List['Cell'],
        temperature: float = 1.0,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if not population:
            return torch.zeros(0, dtype=dtype)

        base = self.as_tensor(dtype=dtype)
        pop = torch.stack([cell.as_tensor(dtype=dtype) for cell in population])
        base = base / (torch.norm(base) + 1e-9)
        pop = pop / (torch.norm(pop, dim=1, keepdim=True) + 1e-9)
        scores = torch.matmul(pop, base)
        return torch.softmax(scores / temperature, dim=0)

    def apply_transformation(self, input_vector: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        For dim=1 cells, treats the embedding as a relational shift.
        Prediction = input_vector + self.embedding
        """
        if self.dim != 1:
            return input_vector
        return self._to_numpy(input_vector) + self._to_numpy(self.embedding)

    def get_context_match_score(self, context_cell: 'Cell') -> float:
        """
        Computes how well this pattern's source matches the current context.
        Used for pattern selection in the forest.
        """
        if self.dim == 0 or self.source is None:
            return 1.0 # 0-cells or root patterns always match
        return self.source.similarity(context_cell)

    def __repr__(self):
        return f"Cell(name='{self.name}', dim={self.dim}, weight={self.weight:.2f})"
