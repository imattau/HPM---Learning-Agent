"""DictionaryAgent: specialised HFN agent for external lexical knowledge."""
from __future__ import annotations
import uuid
import numpy as np
from typing import Optional, Dict, Any
from hfn.hfn import HFN
from hpm_ai_v2.agents.learning_agent import LearningAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig

class DictionaryAgent(LearningAgent):
    """
    HFN-native agent for interfacing with an external dictionary.
    Converts definitions, POS tags, and examples into structural HFN nodes
    without requiring learning, acting as a deterministic semantic grounding source.
    """
    def __init__(
        self,
        config: TextDomainConfig,
        reader_agent: "ReaderAgent",
        forest=None,
        dict_source: str = "wordnet",
        **kwargs
    ) -> None:
        super().__init__(config, forest=forest, **kwargs)
        self.reader_agent = reader_agent
        self.dict_source = dict_source
        self.mock_dict: Dict[str, Any] = {}
        self._in_lookup: Set[str] = set() # Recursion protection
        
        if dict_source == "wordnet":
            try:
                import nltk
                from nltk.corpus import wordnet
                # Ensure wordnet is downloaded
                try:
                    nltk.data.find('corpora/wordnet')
                except LookupError:
                    print("      [DICTIONARY] Downloading WordNet...")
                    nltk.download('wordnet', quiet=True)
                self.wordnet = wordnet
            except ImportError:
                print("      [DICTIONARY] NLTK not found, falling back to mock.")
                self.dict_source = "mock"

        # Foundational mock entries for system bootstrap if needed
        self.mock_dict.update({
            "neural": {
                "pos": "adjective",
                "definition": "Relating to a nerve or the nervous system.",
                "example": "Neural networks are inspired by biological neurons."
            },
            "nerve": {
                "pos": "noun",
                "definition": "A whitish fiber or bundle of fibers that transmits impulses of sensation to the brain or spinal cord, and impulses from these to the muscles and organs.",
                "example": "The optic nerve transmits visual information."
            },
            "fractal": {
                "pos": "noun",
                "definition": "A curve or geometric figure, each part of which has the same statistical character as the whole.",
                "example": "A snowflake exhibits a fractal structure."
            }
        })

    def lookup(self, word: str, pos_hint: Optional[str] = None) -> Optional[HFN]:
        """
        Lookup a word in the dictionary and return its definition HFN node.
        If the word has not been looked up before, it fetches the definition,
        builds the HFN structure, and registers it in the shared forest.
        """
        word_lower = word.lower()
        if word_lower in self._in_lookup:
            return None
        
        def_id = f"definition_{word_lower}"
        
        # 1. Check if we already have this definition in the forest
        def_node = self.forest.get(def_id)
        
        # 2. Fetch from external source
        entry = self._fetch_entry(word_lower, pos_hint=pos_hint)
        if not entry:
            return def_node # Return old one if lookup failed

        # NATIVE CORRECTION: Check for POS contradictions
        if def_node:
            old_pos = def_node.metadata.get("pos")
            new_pos = entry["pos"]
            if old_pos != new_pos:
                # This old definition is likely incorrect (e.g. rare noun sense vs common adj)
                print(f"      [DICTIONARY] Correction: '{word_lower}' {old_pos} -> {new_pos}. Penalizing old definition.")
                self.observer.penalize_id(def_id, penalty=0.8) # Heavy penalty
                # We rename the old definition to clear the way for the new one
                # In HPM, it's better to let them compete, but for this experiment 
                # we want the new one to take the primary def_id.
                new_def_id = f"definition_{word_lower}_{old_pos}"
                def_node.id = new_def_id
                self.forest.register(def_node)
                self.forest.deregister(def_id)
                def_node = None # Force creation of new one with primary def_id
            else:
                return def_node # Identical POS, keep existing

        self._in_lookup.add(word_lower)
        try:
            # 3. Ensure the word macro exists in the forest
            word_node = self.reader_agent._ensure_word_macro(word_lower)

            # 4. Ingest the definition text to create structural nodes
            # We disable proactive lookup here to prevent infinite recursion
            def_doc_node = self.reader_agent.ingest_text(entry["definition"], title=f"def_text_{word_lower}", proactive_lookup=False)
            
            # 5. Ingest the example sentence (if any)
            example_doc_node = None
            examples = entry.get("examples") or ([entry["example"]] if entry.get("example") else [])
            if examples:
                example_doc_node = self.reader_agent.ingest_text(examples[0], title=f"ex_text_{word_lower}", proactive_lookup=False)

            # 6. Ensure POS node exists
            pos_node = self.reader_agent._ensure_word_macro(entry["pos"])

            # 7. Create the composite definition node
            inputs = [word_node, pos_node, def_doc_node]
            if example_doc_node:
                inputs.append(example_doc_node)
                
            for n in inputs:
                self._fit_node_to_forest(n)
                
            mu = np.mean([n.mu for n in inputs], axis=0)
            
            def_node = HFN(mu=mu, sigma=np.ones(self.m_dim) * 0.1, id=def_id, use_diag=True)
            def_node.relation_type = "definition"
            def_node.metadata = {
                "word": word_lower,
                "pos": entry["pos"],
                "definition": entry["definition"],
                "example": examples[0] if examples else "",
                "sentiment": entry.get("sentiment", 0.0)
            }

            # Add structural children
            for child in inputs:
                def_node.add_child(child)

            # Add relational edge from the word macro to its definition
            word_node.add_edge(word_node, def_node, "defined_as")
            
            # 8. HPM-Native: Add hypernym links if available (is-a relationship)
            if "hypernyms" in entry:
                for hn_name in entry["hypernyms"]:
                    hn_node = self.reader_agent._ensure_word_macro(hn_name)
                    def_node.add_edge(def_node, hn_node, "is_a")

            # Register the new knowledge in the forest
            self.observer.register(def_node, protected=False)
            self.observer.observe(def_node.mu) # [DYNAMICS] Reinforce the new pattern
            self.agent_pattern_ids.add(def_id)
            
            print(f"      [DICTIONARY] Learned definition for '{word_lower}' ({entry['pos']}).")
        finally:
            self._in_lookup.remove(word_lower)
            
        return def_node

    def _fetch_entry(self, word: str, pos_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch dictionary entry from the configured source."""
        # 1. Always check mock_dict first (allows manual seeding/overrides)
        if word in self.mock_dict:
            return self.mock_dict[word]
            
        # 2. Try WordNet
        if self.dict_source == "wordnet":
            return self._fetch_wordnet_entry(word, pos_hint=pos_hint)
            
        return None

    def _fetch_wordnet_entry(self, word: str, pos_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Extract a structured entry from NLTK WordNet."""
        from nltk.corpus import wordnet
        
        # Map HPM POS to WordNet POS
        hpm_to_wn = {
            'noun': wordnet.NOUN,
            'verb': wordnet.VERB,
            'adjective': wordnet.ADJ,
            'adverb': wordnet.ADV
        }
        
        synsets = []
        if pos_hint and pos_hint in hpm_to_wn:
            synsets = wordnet.synsets(word, pos=hpm_to_wn[pos_hint])
            
        if not synsets:
            synsets = wordnet.synsets(word)
            
        if not synsets:
            return None
        
        # Heuristic: If multiple POS exist, prefer Verb or Adjective over Noun 
        # for common words that are often mis-tagged as rare nouns.
        # This addresses 'high' and 'led' appearing as nouns.
        s = synsets[0]
        if len(synsets) > 1 and s.pos() == 'n':
            # Check if there's a verb or adjective sense in the top 10
            for candidate in synsets[1:10]:
                if candidate.pos() in ['v', 'a', 's']:
                    s = candidate
                    break
        
        # Map WordNet POS to HPM-friendly names
        pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 's': 'adjective', 'r': 'adverb'}
        pos_str = pos_map.get(s.pos(), 'unknown')
        
        entry = {
            "pos": pos_str,
            "definition": s.definition(),
            "examples": s.examples(),
            "hypernyms": [h.name().split('.')[0] for h in s.hypernyms()]
        }
        return entry
