import os
import sys
import string
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import torch

# Ensure the project root is in the path
sys.path.append(os.getcwd())

from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField
from hpm_ai_v6.hpm_model.fields.institution import ReplicationInstitution
from hpm_ai_v6.agents.character_agent import CharacterAgent
from hpm_ai_v6.agents.word_agent import WordAgent
from hpm_ai_v6.agents.contextual_agent import ContextualAgent
from hpm_ai_v6.agents.phrase_agent import PhraseAgent
from hpm_ai_v6.agents.semantic_agent import SemanticAgent
from hpm_ai_v6.agents.causal_agent import CausalAgent
from hpm_ai_v6.agents.active_learning_agent import ActiveLearningAgent, ActiveCorpus
from hpm_ai_v6.agents.utility_agent import UtilityAgent
from hpm_ai_v6.hpm_model.core.cell import Cell

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
        self.contextual_agent = ContextualAgent(shared_field=self.shared_field)
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
        self.active_learning_agent = ActiveLearningAgent(
            agents={
                "char": self.char_agent,
                "word": self.word_agent,
                "phrase": self.phrase_agent,
                "semantic": self.semantic_agent,
            },
            tag_fn=self._get_tags,
        )
        self.utility_agent = UtilityAgent(
            word_agent=self.word_agent,
            phrase_agent=self.phrase_agent,
            semantic_agent=self.semantic_agent,
            tag_fn=self._get_tags,
            contextual_agent=self.contextual_agent,
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

    def _clean_words(self, text: str) -> List[str]:
        return [w.strip(string.punctuation).lower() for w in self._segment_text(text) if w.strip(string.punctuation)]

    def _split_sentences(self, text: str) -> List[str]:
        return [c.strip() for c in text.split(".") if len(c.strip()) > 10]

    def _load_documents(
        self,
        max_chunks: Optional[int] = None,
        max_words_per_chunk: Optional[int] = None,
    ) -> List[List[str]]:
        if os.path.isdir(self.corpus_path):
            file_paths = sorted(
                os.path.join(self.corpus_path, name)
                for name in os.listdir(self.corpus_path)
                if name.endswith(".txt")
            )
        else:
            file_paths = [self.corpus_path]

        documents: List[List[str]] = []
        total_chunks = 0
        for path in file_paths:
            with open(path, "r") as f:
                chunks = self._split_sentences(f.read())

            if max_words_per_chunk is not None:
                chunks = [" ".join(chunk.split()[:max_words_per_chunk]) for chunk in chunks]

            if max_chunks is not None:
                remaining = max_chunks - total_chunks
                if remaining <= 0:
                    break
                chunks = chunks[:remaining]

            if chunks:
                documents.append(chunks)
                total_chunks += len(chunks)

            if max_chunks is not None and total_chunks >= max_chunks:
                break

        return documents

    def _agent_sequence(self, agent_name: str, sentence: str) -> Tuple[List[Cell], List[Cell]]:
        if agent_name == "char":
            chars = [c for c in sentence.lower() if c in self.char_agent.char_cells]
            return [self.char_agent.char_cells[c] for c in chars], list(self.char_agent.char_cells.values())

        if agent_name == "word":
            words = self._clean_words(sentence)
            population = list(self.word_agent.word_cells.values())
            seq: List[Cell] = []
            extra: List[Cell] = []
            for word in words:
                if word in self.word_agent.word_cells:
                    seq.append(self.word_agent.word_cells[word])
                else:
                    if population:
                        proto = self.word_agent.word_cells.get("alice") or population[0]
                        emb = proto.as_tensor() * 0.0
                    else:
                        emb = [0.0] * 16
                    cell = Cell(name=f"word_eval_{word}", dim=0, embedding=emb)
                    seq.append(cell)
                    extra.append(cell)
            return seq, population + extra

        if agent_name == "contextual":
            words = self._clean_words(sentence)
            if len(words) <= self.contextual_agent.context_length:
                return [], []
            seq = []
            for end_idx in range(self.contextual_agent.context_length, len(words)):
                key = tuple(words[end_idx - self.contextual_agent.context_length : end_idx])
                ctx_cell = self.contextual_agent.context_cells.get(key)
                if ctx_cell is not None:
                    seq.append(ctx_cell)
            return seq, list(self.contextual_agent.context_cells.values())

        if agent_name == "phrase":
            tags = self._get_tags(self._clean_words(sentence))
            return [self.phrase_agent.pos_cells[t] for t in tags if t in self.phrase_agent.pos_cells], list(self.phrase_agent.pos_cells.values())

        if agent_name == "semantic":
            cell = self.semantic_agent._get_or_create_sent_cell(sentence)
            seq = [cell] if cell is not None else []
            return seq, list(self.semantic_agent.sent_cells.values())

        return [], []

    def evaluate_sentences(self, sentences: List[str]) -> Dict[str, float]:
        agent_map = {
            "char": self.char_agent,
            "word": self.word_agent,
            "contextual": self.contextual_agent,
            "phrase": self.phrase_agent,
            "semantic": self.semantic_agent,
        }

        metrics: Dict[str, float] = {}
        total_nll = 0.0
        active_agents = 0
        for name, agent in agent_map.items():
            if not agent.patterns:
                metrics[f"{name}_nll"] = 0.0
                continue

            evaluator = agent.learner.epistemic
            weights = agent.get_weights()
            sentence_scores: List[float] = []
            for sentence in sentences:
                observation_seq, population = self._agent_sequence(name, sentence)
                if len(observation_seq) < 2 or not population:
                    continue

                weighted_nll = 0.0
                total_weight = 0.0
                field_amplifications = {}
                if getattr(agent, "shared_field", None) is not None:
                    field_amplifications = agent.shared_field.get_amplifications(agent.patterns)

                for idx, pattern in enumerate(agent.patterns):
                    result = evaluator.evaluate(
                        pattern,
                        {
                            "observation_seq": observation_seq,
                            "population": population,
                            "field_amplification": field_amplifications.get(pattern.name, 0.0),
                        },
                    )
                    match_count = int(result.metadata.get("count", 0))
                    if match_count <= 0:
                        continue
                    weight = float(weights[idx]) if idx < len(weights) else 0.0
                    weight = max(weight, 1e-6)
                    weighted_nll += float(result.metadata.get("nll", 0.0)) * weight
                    total_weight += weight

                if total_weight > 0.0:
                    sentence_scores.append(weighted_nll / total_weight)

            mean_nll = float(sum(sentence_scores) / len(sentence_scores)) if sentence_scores else 0.0
            metrics[f"{name}_nll"] = mean_nll
            if sentence_scores:
                total_nll += mean_nll
                active_agents += 1

        metrics["mean_nll"] = total_nll / active_agents if active_agents else 0.0
        return metrics

    def train_sequence(self, sentences: List[str], enable_causal: bool = False):
        if len(sentences) > 1:
            self.semantic_agent.process_sentences(sentences)
        if enable_causal:
            self.causal_agent.perform_interventions(sentences)

        for sentence in sentences:
            self.char_agent.process_text(sentence)
            clean_words = self._clean_words(sentence)
            self.word_agent.process_words(clean_words)
            self.contextual_agent.process_words(clean_words)
            self.phrase_agent.process_tags(self._get_tags(clean_words))

    def train_sequence_active(self, sentences: List[str], enable_causal: bool = False):
        if not sentences:
            return

        self.active_learning_agent.update_pool(ActiveCorpus.from_documents([sentences]))
        semantic_seen: List[str] = []
        while True:
            sentence = self.active_learning_agent.select_next_sentence()
            if sentence is None:
                break
            semantic_seen.append(sentence)
            if len(semantic_seen) > 1:
                self.semantic_agent.process_sentences(semantic_seen[-2:])
            if enable_causal:
                self.causal_agent.perform_interventions([sentence])
            clean_words = self._clean_words(sentence)
            self.char_agent.process_text(sentence)
            self.word_agent.process_words(clean_words)
            self.contextual_agent.process_words(clean_words)
            self.phrase_agent.process_tags(self._get_tags(clean_words))

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

        documents = self._load_documents(max_chunks=max_chunks, max_words_per_chunk=max_words_per_chunk)
        chunks = [chunk for document in documents for chunk in document]

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
                self.contextual_agent.process_words(clean_words)
                
                # 5. Phrase Agent: Process POS tags
                tags = self._get_tags(clean_words)
                self.phrase_agent.process_tags(tags)

            # 6. Global Field Update
            agents = [self.char_agent, self.word_agent, self.contextual_agent, self.phrase_agent, self.semantic_agent, self.causal_agent]
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

            print(f"  Patterns: Char({len(self.char_agent.patterns)}), Word({len(self.word_agent.patterns)}), Contextual({len(self.contextual_agent.patterns)}), Phrase({len(self.phrase_agent.patterns)}), Semantic({len(self.semantic_agent.patterns)}), Causal({len(self.causal_agent.patterns)})")

    def train_active(
        self,
        episodes: int = 1,
        max_chunks: Optional[int] = None,
        max_words_per_chunk: Optional[int] = None,
        enable_causal: bool = False,
        enable_pruning: bool = False,
    ):
        if not os.path.exists(self.corpus_path):
            print(f"Error: Corpus not found at {self.corpus_path}")
            return

        documents = self._load_documents(max_chunks=max_chunks, max_words_per_chunk=max_words_per_chunk)
        chunks = [chunk for document in documents for chunk in document]

        print(f"Starting Active Hierarchical Multi-Agent Training on {len(chunks)} chunks...")

        for ep in range(episodes):
            print(f"\n--- Active Episode {ep} ---")
            self.active_learning_agent.update_pool(ActiveCorpus.from_documents(documents))

            semantic_seen: List[str] = []
            selected_chunks: List[str] = []
            while True:
                chunk = self.active_learning_agent.select_next_sentence()
                if chunk is None:
                    break

                selected_chunks.append(chunk)
                semantic_seen.append(chunk)
                if len(semantic_seen) > 1:
                    self.semantic_agent.process_sentences(semantic_seen[-2:])

                if enable_causal:
                    self.causal_agent.perform_interventions([chunk])

                self.char_agent.process_text(chunk)
                words = self._segment_text(chunk)
                clean_words = [w.strip(string.punctuation).lower() for w in words]
                self.word_agent.process_words(clean_words)
                self.contextual_agent.process_words(clean_words)
                self.phrase_agent.process_tags(self._get_tags(clean_words))

            agents = [self.char_agent, self.word_agent, self.contextual_agent, self.phrase_agent, self.semantic_agent, self.causal_agent]
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

            if enable_pruning and ep % 2 == 0:
                print("Running Institutional Pruning...")
                for agent in agents:
                    val_pop = list(self.char_agent.char_cells.values())
                    if agent.patterns:
                        agent.meta_rule.set_weights_tensor(
                            self.institution.filter_population_tensor(
                                agent.patterns,
                                agent.get_weights_tensor(),
                                val_pop,
                            )
                        )

            print(f"  Selected Chunks: {len(selected_chunks)}")
            print(f"  Patterns: Char({len(self.char_agent.patterns)}), Word({len(self.word_agent.patterns)}), Contextual({len(self.contextual_agent.patterns)}), Phrase({len(self.phrase_agent.patterns)}), Semantic({len(self.semantic_agent.patterns)}), Causal({len(self.causal_agent.patterns)})")

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
        print(f"  Utility Next-Word Guess: {self.utility_agent.predict_next_word(text)}")

if __name__ == "__main__":
    reader = MultiAgentReader("hpm_ai_v6/data/corpus/alice_mini.txt")
    reader.train(
        episodes=1,
        max_chunks=1,
        max_words_per_chunk=32,
        enable_pruning=False,
    )
    reader.query("Alice was beginning to get very tired")
