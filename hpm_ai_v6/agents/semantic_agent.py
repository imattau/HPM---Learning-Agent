from typing import List, Dict, Any, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.social_agent import SocialAgent
from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField


class _FallbackSentenceEncoder:
    """Deterministic local encoder used when transformer weights are unavailable."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(self, sentence: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=float)
        if not sentence:
            return vec

        for idx, byte in enumerate(sentence.encode("utf-8")):
            vec[idx % self._dim] += (byte % 31) / 30.0

        norm = np.linalg.norm(vec)
        return vec if norm == 0.0 else vec / norm

class SemanticAgent(SocialAgent):
    """
    Specialized agent for high-level semantic/thematic patterns.
    Inputs: Sentence embeddings from a fixed encoder.
    Learns: 1-cells (semantic transitions), 2-cells (thematic analogies).
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', shared_field: Optional[DynamicPatternField] = None, **kwargs):
        # Prefer a local/cached sentence-transformers model, but stay runnable offline.
        try:
            self.encoder = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.encoder = _FallbackSentenceEncoder()
        if hasattr(self.encoder, "get_embedding_dimension"):
            self.emb_dim = self.encoder.get_embedding_dimension()
        else:
            self.emb_dim = self.encoder.get_sentence_embedding_dimension()
        
        self.sent_cells = {}
        super().__init__(patterns=[], shared_field=shared_field, **kwargs)

    def _get_or_create_sent_cell(self, sentence: str):
        # Use first 100 chars as key to handle potential long sentences
        key = sentence.strip()
        if not key: return None
        
        if key not in self.sent_cells:
            emb = self.encoder.encode(key)
            self.sent_cells[key] = Cell(name=f"sent_{key[:30]}", dim=0, embedding=emb)
        return self.sent_cells[key]

    def _ensure_pattern(self, src_cell: Cell, tgt_cell: Cell):
        name = f"sem_{src_cell.name}->{tgt_cell.name}"
        for p in self.patterns:
            if p.name == name: return p
        
        new_p = Cell(name=name, dim=1, embedding=tgt_cell.embedding - src_cell.embedding, 
                     source=src_cell, target=tgt_cell)
        self.patterns.append(new_p)
        self._refresh_learner()
        return new_p

    def _refresh_learner(self):
        from hpm_ai_v6.hpm_model.dynamics.meta_rule import MetaPatternRule
        from hpm_ai_v6.hpm_model.dynamics.learning import HPMLearner
        
        old_weights = self.meta_rule.get_weights_tensor() if hasattr(self, 'meta_rule') else torch.zeros(0, dtype=torch.float32)
        self.meta_rule = MetaPatternRule(patterns=self.patterns, learning_rate=0.2)
        
        if len(old_weights) > 0:
            new_weights = torch.ones(len(self.patterns), dtype=torch.float32) / (len(self.patterns) + 1e-9)
            new_weights[:len(old_weights)] = old_weights
            self.meta_rule.set_weights_tensor(new_weights / (new_weights.sum() + 1e-9))
            
        self.learner = HPMLearner(meta_rule=self.meta_rule)

    def process_sentences(self, sentences: List[str]):
        """Learns transitions between sentence embeddings."""
        seq = []
        for s in sentences:
            cell = self._get_or_create_sent_cell(s)
            if cell: seq.append(cell)
            
        for i in range(len(seq)-1):
            self._ensure_pattern(seq[i], seq[i+1])
            
        if len(seq) > 1:
            # For the semantic agent, we use a larger context window or specific population
            # Here we use all sentences seen so far as the population
            self.perceive(seq, list(self.sent_cells.values()), context={})
