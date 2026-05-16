from typing import List, Dict, Any, Optional
import hashlib
import numpy as np
import torch
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.social_agent import SocialAgent
from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField

class WordAgent(SocialAgent):
    """
    Specialized agent for word-level patterns.
    Inputs: Segmented words (0-cells).
    Learns: 1-cells (word transitions), 2-cells (phrases).
    """
    def __init__(self, shared_field: Optional[DynamicPatternField] = None, **kwargs):
        self.word_cells = {}
        super().__init__(patterns=[], shared_field=shared_field, **kwargs)

    def _get_or_create_word_cell(self, word: str):
        if word not in self.word_cells:
            # Word embeddings are higher dimension than character embeddings
            self.word_cells[word] = Cell(name=f"word_{word}", dim=0, embedding=np.random.randn(16) * 0.1)
        return self.word_cells[word]

    @staticmethod
    def _seed_embedding(token: str, dim: int = 16) -> np.ndarray:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        if values.size == 0:
            return np.zeros(dim, dtype=np.float32)
        tiled = np.resize(values / 255.0 - 0.5, dim)
        return tiled.astype(np.float32) * 0.2

    def rebuild_word_cells_from_patterns(self) -> None:
        for pattern in self.patterns:
            name = getattr(pattern, "name", "")
            if not name.startswith("w_") or "->" not in name:
                continue
            body = name.removeprefix("w_")
            source_word, target_word = body.split("->", 1)
            source = self.word_cells.get(source_word) or Cell(
                name=f"word_{source_word}",
                dim=0,
                embedding=self._seed_embedding(source_word),
            )
            target = self.word_cells.get(target_word) or Cell(
                name=f"word_{target_word}",
                dim=0,
                embedding=self._seed_embedding(target_word),
            )
            self.word_cells[source_word] = source
            self.word_cells[target_word] = target
            if pattern.source is None:
                pattern.source = source
            if pattern.target is None:
                pattern.target = target

    def _ensure_pattern(self, src_word: str, tgt_word: str):
        name = f"w_{src_word}->{tgt_word}"
        for p in self.patterns:
            if p.name == name: return p
        
        src = self._get_or_create_word_cell(src_word)
        tgt = self._get_or_create_word_cell(tgt_word)

        query_embedding = tgt.as_tensor() - src.as_tensor()
        restored = self.restore_pattern_from_archive(name, query_embedding=query_embedding)
        if restored is not None:
            self._refresh_learner()
            return restored
        
        new_p = Cell(name=name, dim=1, embedding=tgt.embedding - src.embedding, 
                     source=src, target=tgt)
        self.patterns.append(new_p)
        self._refresh_learner()
        return new_p

    def _paging_lookup(self):
        return {cell.name: cell for cell in self.word_cells.values()}

    def _refresh_learner(self):
        from hpm_ai_v6.hpm_model.dynamics.meta_rule import MetaPatternRule
        from hpm_ai_v6.hpm_model.dynamics.learning import HPMLearner
        
        old_weights = self.get_weights_dict() if hasattr(self, 'meta_rule') else {}
        self.meta_rule = MetaPatternRule(patterns=self.patterns, learning_rate=0.2)
        
        if old_weights:
            new_weights = torch.ones(len(self.patterns), dtype=torch.float32) / (len(self.patterns) + 1e-9)
            for i, pattern in enumerate(self.patterns):
                if pattern.name in old_weights:
                    new_weights[i] = float(old_weights[pattern.name])
            self.meta_rule.set_weights_tensor(new_weights / (new_weights.sum() + 1e-9))
            
        self.learner = HPMLearner(meta_rule=self.meta_rule)

    def process_words(self, words: List[str]):
        seq = [self._get_or_create_word_cell(w) for w in words]
        if seq:
            self.drop_incompatible_patterns(seq[0].as_numpy().shape[0])
        for i in range(len(words)-1):
            self._ensure_pattern(words[i], words[i+1])
            
        if len(seq) > 1:
            self.perceive(seq, list(self.word_cells.values()), context={})
