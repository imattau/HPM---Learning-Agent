import os
import sys
import re
import string
from typing import List, Dict, Optional, Any, Tuple, Sequence
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
from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
from hpm_ai_v6.agents.utility_agent import UtilityAgent
from hpm_ai_v6.agents.response_generation_agent import ResponseGenerationAgent
from hpm_ai_v6.hpm_model.core.cell import Cell

class MultiAgentReader:
    """
    Orchestrator for the Multi-Agent HPM Reading System.
    Coordinates Character, Word, Phrase, Semantic, and Causal agents.
    """
    def __init__(
        self,
        corpus_path: str,
        pattern_cache_dir: Optional[str] = None,
        max_active_patterns: int = 1000,
        warm_start: bool = True,
        warm_start_limit: Optional[int] = None,
    ):
        self.corpus_path = corpus_path
        self.shared_field = DynamicPatternField(influence_rate=0.1)
        self.institution = ReplicationInstitution(prune_ratio=0.1)
        self.max_words_per_chunk = 32
        self._known_corpus_sentences = 0
        self._corpus_offset = os.path.getsize(corpus_path) if os.path.exists(corpus_path) else 0
        self.pattern_cache_dir = pattern_cache_dir or self._default_pattern_cache_dir()
        self.max_active_patterns = max_active_patterns
        
        # Initialize lower-level agents
        self.char_agent = CharacterAgent(
            shared_field=self.shared_field,
            pattern_cache_dir=self.pattern_cache_dir,
            max_active_patterns=self.max_active_patterns,
        )
        self.word_agent = WordAgent(
            shared_field=self.shared_field,
            pattern_cache_dir=self.pattern_cache_dir,
            max_active_patterns=self.max_active_patterns,
        )
        self.contextual_agent = ContextualAgent(
            shared_field=self.shared_field,
            pattern_cache_dir=self.pattern_cache_dir,
            max_active_patterns=self.max_active_patterns,
        )
        self.phrase_agent = PhraseAgent(
            shared_field=self.shared_field,
            pattern_cache_dir=self.pattern_cache_dir,
            max_active_patterns=self.max_active_patterns,
        )
        self.semantic_agent = SemanticAgent(
            shared_field=self.shared_field,
            pattern_cache_dir=self.pattern_cache_dir,
            max_active_patterns=self.max_active_patterns,
        )
        
        # Initialize Causal Agent, providing access to other agents
        self.causal_agent = CausalAgent(
            other_agents={
                "char": self.char_agent,
                "word": self.word_agent,
                "phrase": self.phrase_agent,
                "semantic": self.semantic_agent
            },
            shared_field=self.shared_field,
            pattern_cache_dir=self.pattern_cache_dir,
            max_active_patterns=self.max_active_patterns,
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
        self.response_agent = ResponseGenerationAgent(
            contextual_agent=self.contextual_agent,
            word_agent=self.word_agent,
            phrase_agent=self.phrase_agent,
            semantic_agent=self.semantic_agent,
            tag_fn=self._get_tags,
        )
        self.reasoning_agent = ReasoningAgent(self)
        self.agents = {
            "char": self.char_agent,
            "word": self.word_agent,
            "contextual": self.contextual_agent,
            "phrase": self.phrase_agent,
            "semantic": self.semantic_agent,
            "causal": self.causal_agent,
            "active_learning": self.active_learning_agent,
            "utility": self.utility_agent,
            "response": self.response_agent,
            "reasoning": self.reasoning_agent,
        }
        self.warm_start = warm_start
        self.warm_start_limit = warm_start_limit
        if self.warm_start:
            self.warm_start_from_cache(limit=self.warm_start_limit)
        
        # Simple POS lookup for demonstration
        self.pos_map = {
            "alice": "NOUN", "rabbit": "NOUN", "sister": "NOUN", "book": "NOUN",
            "was": "VERB", "get": "VERB", "sitting": "VERB", "having": "VERB", "reading": "VERB",
            "tired": "ADJ", "very": "ADV", "sleepy": "ADJ", "natural": "ADJ",
            "the": "DET", "a": "DET", "her": "PRON", "she": "PRON", "it": "PRON",
            "and": "CONJ", "but": "CONJ", "of": "PREP", "on": "PREP", "by": "PREP"
        }

    def __del__(self):
        for agent in getattr(self, "agents", {}).values():
            if hasattr(agent, "close_pager"):
                try:
                    agent.close_pager()
                except Exception:
                    pass

    def close_pagers(self) -> None:
        for agent in getattr(self, "agents", {}).values():
            if hasattr(agent, "close_pager"):
                try:
                    agent.close_pager()
                except Exception:
                    pass

    def _default_pattern_cache_dir(self) -> str:
        corpus_dir = os.path.dirname(os.path.abspath(self.corpus_path)) or os.getcwd()
        corpus_name = os.path.splitext(os.path.basename(self.corpus_path))[0]
        return os.path.join(corpus_dir, ".hpm_pattern_cache", corpus_name)

    @staticmethod
    def _safe_best_pattern(agent: Any):
        if not getattr(agent, "patterns", None):
            return None
        return agent.get_best_pattern()

    def warm_start_from_cache(self, limit: Optional[int] = None) -> int:
        loaded = 0
        for name in ("char", "word", "contextual", "phrase", "semantic", "causal"):
            agent = self.agents.get(name)
            if agent is None or not hasattr(agent, "hydrate_patterns_from_archive"):
                continue

            remaining = None if limit is None else max(limit - loaded, 0)
            if remaining == 0:
                break

            loaded += agent.hydrate_patterns_from_archive(limit=remaining)
        return loaded

    @staticmethod
    def _agent_learning_snapshot(agent: Any) -> Dict[str, Any]:
        patterns = list(getattr(agent, "patterns", []) or [])
        weights = []
        if hasattr(agent, "get_weights"):
            try:
                weights = list(agent.get_weights())
            except Exception:
                weights = []

        best_pattern = None
        if hasattr(agent, "get_best_pattern"):
            try:
                best_pattern = agent.get_best_pattern()
            except Exception:
                best_pattern = None

        best_weight = max((float(weight) for weight in weights), default=0.0)
        return {
            "patterns": len(patterns),
            "weights": len(weights),
            "best_pattern": getattr(best_pattern, "name", None),
            "best_weight": best_weight,
        }

    def maintenance_cycle(
        self,
        sentences: Sequence[str],
        *,
        hydrate_limit: Optional[int] = None,
        retrain_epochs: int = 1,
        enable_causal: bool = False,
        agent_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Hydrate archived patterns, run another learning pass, and report per-agent changes.
        """
        sentences_list = list(sentences)
        tracked_names = list(agent_names or ("char", "word", "contextual", "phrase", "semantic", "causal"))
        before = {
            name: self._agent_learning_snapshot(self.agents[name])
            for name in tracked_names
            if self.agents.get(name) is not None
        }

        loaded = self.warm_start_from_cache(limit=hydrate_limit)

        after_hydrate = {
            name: self._agent_learning_snapshot(self.agents[name])
            for name in tracked_names
            if self.agents.get(name) is not None
        }

        for _ in range(max(retrain_epochs, 1)):
            self.train_sequence(sentences_list, enable_causal=enable_causal)

        after_train = {
            name: self._agent_learning_snapshot(self.agents[name])
            for name in tracked_names
            if self.agents.get(name) is not None
        }

        report: Dict[str, Dict[str, Any]] = {}
        for name in tracked_names:
            if self.agents.get(name) is None:
                continue
            before_snapshot = before.get(name, {})
            hydrated_snapshot = after_hydrate.get(name, {})
            after_snapshot = after_train.get(name, {})
            pattern_delta = int(after_snapshot.get("patterns", 0)) - int(before_snapshot.get("patterns", 0))
            best_weight_before = float(before_snapshot.get("best_weight", 0.0))
            best_weight_after = float(after_snapshot.get("best_weight", 0.0))
            report[name] = {
                "before": before_snapshot,
                "after_hydrate": hydrated_snapshot,
                "after_train": after_snapshot,
                "pattern_delta": pattern_delta,
                "best_weight_delta": best_weight_after - best_weight_before,
                "improved": pattern_delta > 0 or best_weight_after > best_weight_before,
            }

        report["_summary"] = {
            "loaded": loaded,
            "sentences": len(sentences_list),
            "retrain_epochs": retrain_epochs,
            "improved_agents": [name for name, data in report.items() if name != "_summary" and data.get("improved")],
        }
        return report

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

    def load_corpus(self, max_chunks: Optional[int] = None, max_words_per_chunk: Optional[int] = None) -> List[List[str]]:
        if max_words_per_chunk is None:
            max_words_per_chunk = self.max_words_per_chunk
        documents = self._load_documents(max_chunks=max_chunks, max_words_per_chunk=max_words_per_chunk)
        self._known_corpus_sentences = sum(len(document) for document in documents)
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

        syn_agent = self.agents.get("syntactic")
        if syn_agent is not None and hasattr(syn_agent, "learn_from_corpus"):
            syn_agent.learn_from_corpus(sentences)

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

        if max_words_per_chunk is not None:
            self.max_words_per_chunk = max_words_per_chunk
        documents = self._load_documents(max_chunks=max_chunks, max_words_per_chunk=max_words_per_chunk)
        self._known_corpus_sentences = sum(len(document) for document in documents)
        self._corpus_offset = os.path.getsize(self.corpus_path) if os.path.exists(self.corpus_path) else 0
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

        if max_words_per_chunk is not None:
            self.max_words_per_chunk = max_words_per_chunk
        documents = self._load_documents(max_chunks=max_chunks, max_words_per_chunk=max_words_per_chunk)
        self._known_corpus_sentences = sum(len(document) for document in documents)
        self._corpus_offset = os.path.getsize(self.corpus_path) if os.path.exists(self.corpus_path) else 0
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
        print(f"  Response Generation: {self.response_agent.generate(text, max_length=12)}")

    def generate(self, seed_text: str, max_length: int = 50, temperature: float = 0.0) -> str:
        return self.response_agent.generate(seed_text, max_length=max_length, temperature=temperature)

    def reason(self, question: str) -> str:
        return self.reasoning_agent.reason(question)

    def retrain_on_new_data(self, epochs: int = 1):
        """Incrementally train on sentences appended to the corpus since the last load/train."""
        if not os.path.exists(self.corpus_path):
            print(f"Error: Corpus not found at {self.corpus_path}")
            return 0

        current_size = os.path.getsize(self.corpus_path)
        if current_size <= self._corpus_offset:
            print("No new corpus sentences to retrain on.")
            return 0

        with open(self.corpus_path, "r", encoding="utf-8") as handle:
            handle.seek(self._corpus_offset)
            tail_text = handle.read()

        raw_blocks = [block.strip() for block in re.split(r"\n\s*\n|\n", tail_text) if block.strip()]
        new_sentences = [block for block in raw_blocks if block]
        if not new_sentences:
            self._corpus_offset = current_size
            print("No new corpus sentences to retrain on.")
            return 0

        print(f"Retraining on {len(new_sentences)} new sentences...")
        for _ in range(epochs):
            self.train_sequence(new_sentences, enable_causal=False)
        self._known_corpus_sentences += len(new_sentences)
        self._corpus_offset = current_size
        return len(new_sentences)

if __name__ == "__main__":
    reader = MultiAgentReader("hpm_ai_v6/data/corpus/alice_mini.txt")
    reader.train(
        episodes=1,
        max_chunks=1,
        max_words_per_chunk=32,
        enable_pruning=False,
    )
    reader.query("Alice was beginning to get very tired")
