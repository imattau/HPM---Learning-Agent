from typing import List, Dict, Any, Optional
import numpy as np
import string
import torch
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.social_agent import SocialAgent
from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField

class CharacterAgent(SocialAgent):
    """
    Specialized agent for character-level patterns.
    Inputs: Raw characters (0-cells).
    Learns: 1-cells (char transitions), 2-cells (analogies).
    """
    def __init__(self, shared_field: Optional[DynamicPatternField] = None, **kwargs):
        # Initialize with a standard alphabet + symbols
        self.alphabet = string.ascii_lowercase + " .,:;!?'\""
        self.char_cells = {
            c: Cell(name=f"char_{c}", dim=0, embedding=np.random.randn(8) * 0.1)
            for c in self.alphabet
        }
        # Initial empty patterns
        super().__init__(patterns=[], shared_field=shared_field, **kwargs)

    def _ensure_pattern(self, src_char: str, tgt_char: str):
        name = f"c_{src_char}->{tgt_char}"
        for p in self.patterns:
            if p.name == name: return p
        
        src = self.char_cells.get(src_char)
        tgt = self.char_cells.get(tgt_char)
        if not src or not tgt: return None

        query_embedding = tgt.as_tensor() - src.as_tensor()
        restored = self.restore_pattern_from_archive(name, query_embedding=query_embedding)
        if restored is not None:
            self._refresh_learner()
            return restored
        
        new_p = Cell(name=name, dim=1, embedding=tgt.embedding - src.embedding, 
                     source=src, target=tgt)
        self.patterns.append(new_p)
        # We need to refresh the MPR/Learner when patterns are added
        self._refresh_learner()
        return new_p

    def _paging_lookup(self):
        return {cell.name: cell for cell in self.char_cells.values()}

    def _refresh_learner(self):
        # Logic to re-initialize internal MPR with updated pattern list
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

    def process_text(self, text: str):
        text = text.lower()
        if self.char_cells:
            sample_dim = next(iter(self.char_cells.values())).as_numpy().shape[0]
            self.drop_incompatible_patterns(sample_dim)
        seq = []
        for i in range(len(text)-1):
            c1, c2 = text[i], text[i+1]
            if c1 in self.alphabet and c2 in self.alphabet:
                self._ensure_pattern(c1, c2)
                seq.append(self.char_cells[c1])
        if len(text) > 0 and text[-1] in self.alphabet:
            seq.append(self.char_cells[text[-1]])
            
        if len(seq) > 1:
            self.perceive(seq, list(self.char_cells.values()), context={})
            
    def get_word_boundaries(self) -> List[str]:
        """Returns patterns that likely represent word boundaries (e.g., ends with space)."""
        boundaries = []
        weights = self.get_weights()
        for i, p in enumerate(self.patterns):
            if p.target and p.target.name == "char_ ":
                if weights[i] > 0.05: # threshold
                    boundaries.append(p.name)
        return boundaries
