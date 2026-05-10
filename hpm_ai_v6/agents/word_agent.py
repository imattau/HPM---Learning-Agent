from typing import List, Dict, Any, Optional
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

    def _ensure_pattern(self, src_word: str, tgt_word: str):
        name = f"w_{src_word}->{tgt_word}"
        for p in self.patterns:
            if p.name == name: return p
        
        src = self._get_or_create_word_cell(src_word)
        tgt = self._get_or_create_word_cell(tgt_word)
        
        new_p = Cell(name=name, dim=1, embedding=tgt.embedding - src.embedding, 
                     source=src, target=tgt)
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

    def process_words(self, words: List[str]):
        seq = [self._get_or_create_word_cell(w) for w in words]
        for i in range(len(words)-1):
            self._ensure_pattern(words[i], words[i+1])
            
        if len(seq) > 1:
            self.perceive(seq, list(self.word_cells.values()), context={})
