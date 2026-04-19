"""SyntaxMixin: adds POS tagging and syntactic parsing capabilities to ReaderAgent."""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from hfn.hfn import HFN


class SyntaxMixin:
    """
    Mixin for ReaderAgent that adds syntactic awareness.
    Learns POS tagging rules (macros) from minimal examples.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_rules: Dict[str, str] = {}  # token/suffix/pattern -> tag
        self.pos_macro: Optional[List[HFN]] = None

    def learn_pos_tagger(self, inputs: List[List[str]], outputs: List[List[str]]) -> bool:
        """
        Induces a POS tagging macro from example sentences.
        Example: inputs=[["the", "cat", "chases"]], outputs=[["DET", "NOUN", "VERB"]]
        """
        # 1. Build a simple dictionary-based and rule-based rule set
        rules = {}
        # Count occurrences of token -> tag
        token_tag_counts: Dict[str, Dict[str, int]] = {}
        suffix_tag_counts: Dict[str, Dict[str, int]] = {}

        for sent, tags in zip(inputs, outputs):
            for token, tag in zip(sent, tags):
                token = token.lower()
                if token not in token_tag_counts: token_tag_counts[token] = {}
                token_tag_counts[token][tag] = token_tag_counts[token].get(tag, 0) + 1
                
                # Suffixes (for ADV, VERB, etc.)
                for l in [2, 3]:
                    if len(token) > l:
                        suffix = token[-l:]
                        if suffix not in suffix_tag_counts: suffix_tag_counts[suffix] = {}
                        suffix_tag_counts[suffix][tag] = suffix_tag_counts[suffix].get(tag, 0) + 1

        # Finalise rules (take most frequent)
        for token, tag_counts in token_tag_counts.items():
            rules[token] = max(tag_counts, key=tag_counts.get)
            
        for suffix, tag_counts in suffix_tag_counts.items():
            # Only use suffixes that are predictive (>0.6)
            best_tag = max(tag_counts, key=tag_counts.get)
            if tag_counts[best_tag] / sum(tag_counts.values()) >= 0.6:
                rules[f"suffix_{suffix}"] = best_tag

        self.pos_rules = rules
        
        # 2. Represent this as an HFN macro (simplified for now)
        # We create a node representing the POS tagger function
        mu = np.zeros(self.m_dim)
        prim = "POS_NOUN"
        if hasattr(self, 'config') and prim in self.config.concepts:
            mu[self.config.S_DIM + self.config.concepts.index(prim)] = 1.0
        else:
            mu[0] = 1.0
        node = HFN(mu=mu, sigma=np.ones(self.m_dim), id="pos_tagger_macro", use_diag=True)
        node.metadata = {"rules": rules, "type": "syntax_macro"}
        self.observer.register(node, protected=True)
        self.patterns["pos_tagger_macro"] = node
        self.pos_macro = [node]
        
        return True

    def _tag_sentence(self, sentence: List[str] | HFN) -> List[str]:
        """Apply the induced rules to tag a new sentence (token list or HFN node)."""
        if isinstance(sentence, HFN):
            sentence = [c.metadata.get("char", c.id.replace("CHAR_", "")) if getattr(c, "relation_type", None) == "character" else c.metadata.get("word", c.id.replace("word_spelling_", "")) for c in sentence.children()]
            
        tags = []
        for token in sentence:
            token = token.lower()
            tag = "NOUN" # Default fallback
            
            # Punctuation check
            if not token.isalnum():
                tag = "PUNCT"
            # 1. Token match (high priority)
            elif token in self.pos_rules:
                tag = self.pos_rules[token]
            # 2. Suffix match
            else:
                for l in [3, 2]:
                    suffix = token[-l:]
                    if f"suffix_{suffix}" in self.pos_rules:
                        tag = self.pos_rules[f"suffix_{suffix}"]
                        break
            tags.append(tag)
        return tags

    def extract_noun_phrases(self, sentence_raw: str) -> List[str]:
        """Extract noun phrases using the learned POS tagger."""
        from hpm_ai_v2.domains.text_domain import tokenise_raw
        tokens = tokenise_raw(sentence_raw)
        tags = self._tag_sentence(tokens)
        
        nps = []
        current_chunk = []
        
        # NP structure: (DET) (ADJ)* (NOUN)+
        for i, (token, tag) in enumerate(zip(tokens, tags)):
            if tag in ["DET", "ADJ", "NOUN"]:
                current_chunk.append(token)
                # If this is the last token or next token is not part of NP, flush it
                is_last = (i == len(tokens) - 1)
                next_tag = tags[i+1] if not is_last else None
                if is_last or next_tag not in ["DET", "ADJ", "NOUN"]:
                    # Ensure it contains at least one noun to be an NP (unless it's just a DET/ADJ which we'll ignore for now)
                    chunk_tags = tags[max(0, i-len(current_chunk)+1) : i+1]
                    if "NOUN" in chunk_tags:
                        nps.append(" ".join(current_chunk))
                    current_chunk = []
            else:
                current_chunk = []
                
        return nps

    def get_sentence_structure(self, sentence: str | HFN) -> List[Tuple[str, str]]:
        """Returns list of (token, tag) for a sentence (string or HFN node)."""
        if isinstance(sentence, HFN):
            tokens = [c.metadata.get("char", c.id.replace("CHAR_", "")) if getattr(c, "relation_type", None) == "character" else c.metadata.get("word", c.id.replace("word_spelling_", "")) for c in sentence.children()]
            tags = self._tag_sentence(sentence)
        else:
            from hpm_ai_v2.domains.text_domain import tokenise_raw
            tokens = tokenise_raw(sentence)
            tags = self._tag_sentence(tokens)
        return list(zip(tokens, tags))
