from typing import List, Dict, Any, Optional
import numpy as np
import torch
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.social_agent import SocialAgent
from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField

class PhraseAgent(SocialAgent):
    """
    Specialized agent for phrase-level/syntactic patterns.
    Inputs: Parts of speech or chunk labels (0-cells).
    Learns: 1-cells (POS transitions), 2-cells (grammar rules).
    """
    def __init__(self, shared_field: Optional[DynamicPatternField] = None, **kwargs):
        self.pos_cells = {}
        # Pre-defined common POS tags
        tags = ["NOUN", "VERB", "ADJ", "ADV", "DET", "PREP", "CONJ", "PRON"]
        for t in tags:
            self.pos_cells[t] = Cell(name=f"pos_{t}", dim=0, embedding=np.random.randn(12) * 0.1)
        
        super().__init__(patterns=[], shared_field=shared_field, **kwargs)

    def _ensure_pattern(self, src_tag: str, tgt_tag: str):
        name = f"s_{src_tag}->{tgt_tag}"
        for p in self.patterns:
            if p.name == name: return p
        
        src = self.pos_cells.get(src_tag)
        tgt = self.pos_cells.get(tgt_tag)
        if not src or not tgt: return None
        
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

    def process_tags(self, tags: List[str]):
        seq = [self.pos_cells[t] for t in tags if t in self.pos_cells]
        for i in range(len(tags)-1):
            if tags[i] in self.pos_cells and tags[i+1] in self.pos_cells:
                self._ensure_pattern(tags[i], tags[i+1])
            
        if len(seq) > 1:
            self.perceive(seq, list(self.pos_cells.values()), context={})
