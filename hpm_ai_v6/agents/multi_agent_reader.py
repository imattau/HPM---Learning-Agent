import os
import sys
import string
from typing import List, Dict, Optional, Any
import numpy as np
import torch

# Ensure the project root is in the path
sys.path.append(os.getcwd())

from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField
from hpm_ai_v6.hpm_model.fields.institution import ReplicationInstitution
from hpm_ai_v6.agents.character_agent import CharacterAgent
from hpm_ai_v6.agents.word_agent import WordAgent
from hpm_ai_v6.agents.phrase_agent import PhraseAgent
from hpm_ai_v6.agents.semantic_agent import SemanticAgent
from hpm_ai_v6.agents.causal_agent import CausalAgent

class MultiAgentReader:
    """
    Orchestrator for the Multi-Agent HPM Reading System.
    Coordinates Character, Word, Phrase, Semantic, and Causal agents.
    """
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.shared_field = DynamicPatternField(influence_rate=0.1)
        self.institution = ReplicationInstitution(prune_ratio=0.1)
        
        # Initialize lower-level agents
        self.char_agent = CharacterAgent(shared_field=self.shared_field)
        self.word_agent = WordAgent(shared_field=self.shared_field)
        self.phrase_agent = PhraseAgent(shared_field=self.shared_field)
        self.semantic_agent = SemanticAgent(shared_field=self.shared_field)
        
        # Initialize Causal Agent, providing access to other agents
        self.causal_agent = CausalAgent(
            other_agents={
                "char": self.char_agent,
                "word": self.word_agent,
                "phrase": self.phrase_agent,
                "semantic": self.semantic_agent
            },
            shared_field=self.shared_field
        )
        
        # Simple POS lookup for demonstration
        self.pos_map = {
            "alice": "NOUN", "rabbit": "NOUN", "sister": "NOUN", "book": "NOUN",
            "was": "VERB", "get": "VERB", "sitting": "VERB", "having": "VERB", "reading": "VERB",
            "tired": "ADJ", "very": "ADV", "sleepy": "ADJ", "natural": "ADJ",
            "the": "DET", "a": "DET", "her": "PRON", "she": "PRON", "it": "PRON",
            "and": "CONJ", "but": "CONJ", "of": "PREP", "on": "PREP", "by": "PREP"
        }

    @staticmethod
    def _safe_best_pattern(agent: Any):
        if not getattr(agent, "patterns", None):
            return None
        return agent.get_best_pattern()

    def _get_tags(self, words: List[str]) -> List[str]:
        return [self.pos_map.get(w.lower(), "NOUN") for w in words]

    def _segment_text(self, text: str) -> List[str]:
        return [token for token in text.split() if token]

    def train(
        self,
        episodes: int = 5,
        max_chunks: Optional[int] = None,
        max_words_per_chunk: Optional[int] = None,
        enable_causal: bool = True,
        enable_pruning: bool = True,
    ):
        if not os.path.exists(self.corpus_path):
            print(f"Error: Corpus not found at {self.corpus_path}")
            return

        with open(self.corpus_path, 'r') as f:
            full_text = f.read()

        chunks = [c.strip() for c in full_text.split('.') if len(c.strip()) > 10]
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        if max_words_per_chunk is not None:
            chunks = [
                " ".join(chunk.split()[:max_words_per_chunk])
                for chunk in chunks
            ]

        print(f"Starting Hierarchical Multi-Agent Training on {len(chunks)} chunks...")

        for ep in range(episodes):
            print(f"\n--- Episode {ep} ---")
            
            # 1. Semantic agent processes thematic flow
            self.semantic_agent.process_sentences(chunks)
            
            # 2. Causal Agent: Perform active interventions
            if enable_causal:
                self.causal_agent.perform_interventions(chunks)
            
            for chunk in chunks:
                # 3. Character Agent: Process raw text
                self.char_agent.process_text(chunk)
                
                # 4. Word Agent: Process segmented words
                words = self._segment_text(chunk)
                clean_words = [w.strip(string.punctuation).lower() for w in words]
                self.word_agent.process_words(clean_words)
                
                # 5. Phrase Agent: Process POS tags
                tags = self._get_tags(clean_words)
                self.phrase_agent.process_tags(tags)

            # 6. Global Field Update
            agents = [self.char_agent, self.word_agent, self.phrase_agent, self.semantic_agent, self.causal_agent]
            all_patterns = []
            all_weights_list = []
            for agent in agents:
                all_patterns.extend(agent.patterns)
                weights = agent.get_weights()
                if len(weights) > 0:
                    all_weights_list.append(weights)

            if all_patterns and all_weights_list:
                all_weights = torch.cat([torch.as_tensor(weights, dtype=torch.float32) for weights in all_weights_list])
                self.shared_field.update_tensor(
                    all_patterns,
                    all_weights,
                    torch.ones_like(all_weights) * 0.5
                )

            # 7. Institutional Pruning
            if enable_pruning and ep % 2 == 0:
                print("Running Institutional Pruning...")
                for agent in agents:
                    val_pop = list(self.char_agent.char_cells.values())
                    if agent.patterns:
                        agent.meta_rule.set_weights_tensor(self.institution.filter_population_tensor(
                            agent.patterns, agent.get_weights_tensor(), val_pop
                        )
                        )

            print(f"  Patterns: Char({len(self.char_agent.patterns)}), Word({len(self.word_agent.patterns)}), Phrase({len(self.phrase_agent.patterns)}), Semantic({len(self.semantic_agent.patterns)}), Causal({len(self.causal_agent.patterns)})")

    def query(self, text: str):
        print(f"\nQuerying system with: '{text}'")
        
        next_char = self.char_agent.get_word_boundaries()
        best_word_pattern = self._safe_best_pattern(self.word_agent)
        best_phrase_pattern = self._safe_best_pattern(self.phrase_agent)
        best_semantic_pattern = self._safe_best_pattern(self.semantic_agent)
        causal_insights = self.causal_agent.get_causal_insights()
        
        print(f"  Char Agent Boundary Patterns: {next_char[:3]}")
        print(f"  Word Agent Top Transition: {best_word_pattern.name if best_word_pattern else 'None'}")
        print(f"  Phrase Agent Top Rule: {best_phrase_pattern.name if best_phrase_pattern else 'None'}")
        print(f"  Semantic Agent Top Theme: {best_semantic_pattern.name if best_semantic_pattern else 'None'}")
        print(f"  Causal Agent Top Insights: {causal_insights}")

if __name__ == "__main__":
    reader = MultiAgentReader("hpm_ai_v6/data/corpus/alice_mini.txt")
    reader.train(
        episodes=1,
        max_chunks=1,
        max_words_per_chunk=32,
        enable_pruning=False,
    )
    reader.query("Alice was beginning to get very tired")
