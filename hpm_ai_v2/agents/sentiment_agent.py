"""SentimentAgent: HFN-native sentiment and emotion analysis agent."""
from __future__ import annotations
import numpy as np
from typing import List, Optional, Any, Dict, Tuple
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.domains.sentiment_domain import SentimentDomainConfig

class SentimentAgent(BaseHFNAgent):
    """
    Agent specializing in sentiment analysis and emotional classification.
    Learns to compose macros from lexicon and structural primitives.
    """
    def __init__(self, config: SentimentDomainConfig, forest=None, dictionary_agent=None, **kwargs) -> None:
        # Enable affective evaluator in observer for native HPM sentiment grounding
        kwargs["use_affective_evaluator"] = True
        # If persistence is required, ensure cold_dir for affective state is set
        from pathlib import Path
        if "affective_cold_dir" not in kwargs and "cold_dir" in kwargs:
            kwargs["affective_cold_dir"] = Path(kwargs["cold_dir"]) / "affective"
            
        super().__init__(config, forest=forest, **kwargs)
        self.dictionary_agent = dictionary_agent
        self._register_sentiment_primitives()

    def _register_sentiment_primitives(self) -> None:
        """Register L1 primitives for sentiment logic."""
        self.add_strategy("lexicon_score", self.primitive_lexicon_score)
        self.add_strategy("is_negation", self.primitive_is_negation)
        self.add_strategy("is_intensifier", self.primitive_is_intensifier)
        self.add_strategy("classify_sentiment", self.primitive_classify)

    # --- Primitives (L1) ---

    def primitive_lexicon_score(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[float]]:
        """
        Lookup sentiment score for a word.
        HPM-Native: Queries the AffectiveEvaluator for the pattern's affect score.
        """
        if not inputs: return [0.0]
        word = inputs[0]
        word_id = word.id if isinstance(word, HFN) else f"word_spelling_{str(word).lower()}"
        
        # HPM-Native: Get affect score (valence) from the internal evaluator
        # 0.5 is neutral. Mapping [0, 1] -> [-1, 1]
        if hasattr(self.observer, "evaluator") and hasattr(self.observer.evaluator, "get_affect_score"):
            raw_score = self.observer.evaluator.get_affect_score(word_id)
            # Center around 0.5 (neutral)
            return [(raw_score - 0.5) * 2.0]
        
        return [0.0]

    def primitive_is_negation(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[bool]]:
        """Check if word is a negation."""
        if not inputs: return [False]
        word = inputs[0]
        word_str = (word.metadata.get("word", str(word)) if isinstance(word, HFN) else str(word)).lower()
        negations = {"not", "never", "no", "neither", "nor", "none", "hardly", "scarcely", "barely"}
        return [word_str in negations]

    def primitive_is_intensifier(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[bool]]:
        """Check if word is an intensifier."""
        if not inputs: return [False]
        word = inputs[0]
        word_str = (word.metadata.get("word", str(word)) if isinstance(word, HFN) else str(word)).lower()
        intensifiers = {"very", "extremely", "really", "so", "totally", "absolutely", "completely"}
        return [word_str in intensifiers]

    def primitive_classify(self, inputs: List[Any], outputs: List[Any]) -> Optional[List[str]]:
        """Map score to pos/neg/neutral."""
        if not inputs: return ["neutral"]
        score = float(inputs[0])
        if score > 0.2: return ["positive"]
        if score < -0.2: return ["negative"]
        return ["neutral"]

    # --- Sentiment Analysis ---

    def analyze_sentence(self, sent_node: HFN) -> HFN:
        """
        Heuristic-based sentiment analysis (L2 macro equivalent).
        In a full L5 implementation, this would use a discovered macro node.
        """
        # 1. Extract words from sentence children
        word_nodes = [c for c in sent_node.children() if c.relation_type in ("spelling", "word")]
        
        total_score = 0.0
        negate = False
        intensify = 1.0
        
        for wn in word_nodes:
            word_str = wn.metadata.get("word", wn.id.split("_")[-1]).lower()
            # Check negation
            if self.primitive_is_negation([wn], [])[0]:
                negate = True
                continue
            
            # Check intensifier
            if self.primitive_is_intensifier([wn], [])[0]:
                intensify = 1.5
                continue
            
            # Get lexicon score
            s = self.primitive_lexicon_score([wn], [])[0]
            print(f"      [DEBUG] Word: {word_str}, Score: {s}")
            
            # Apply logic
            if s != 0:
                current_s = s * intensify
                if negate:
                    current_s = -current_s
                    negate = False 
                total_score += current_s
                intensify = 1.0 
        
        # 2. Classify
        label = self.primitive_classify([total_score], [])[0]
        print(f"      [DEBUG] Total Score: {total_score}, Label: {label}")
        
        # 3. Create sentiment node
        res_id = f"sentiment_{sent_node.id}"
        mu = np.zeros(self.m_dim)
        # Use manifold for labeling
        if label == "positive": mu[self.s_dim + self.config.concepts.index("SENTIMENT_POS")] = 1.0
        elif label == "negative": mu[self.s_dim + self.config.concepts.index("SENTIMENT_NEG")] = 1.0
        else: mu[self.s_dim + self.config.concepts.index("SENTIMENT_NEUTRAL")] = 1.0
        
        res_node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=res_id, use_diag=True)
        res_node.relation_type = "sentiment_result"
        res_node.metadata = {
            "label": label, 
            "score": total_score, 
            "type": "sentiment",
            "text": f"Sentiment is {label} (score: {total_score:.2f})"
        }
        
        self.observer.register(res_node)
        # Use add_child instead of add_edge since Sentiment result is a child of the sentence
        sent_node.add_child(res_node, relation="has_sentiment")
        return res_node

    def seed_lexicon(self) -> None:
        """Seed DictionaryAgent and AffectiveEvaluator with foundational sentiment scores."""
        lexicon = {
            "love": 0.9, "great": 0.8, "excellent": 0.9, "good": 0.6, "happy": 0.7,
            "hate": -0.9, "terrible": -0.9, "bad": -0.7, "awful": -0.8, "worst": -0.9,
            "okay": 0.1, "fine": 0.1, "average": 0.0, "normal": 0.0
        }
        
        print(f"      [SENTIMENT] Seeding native affective state for {len(lexicon)} words...")
        for word, score in lexicon.items():
            word_id = f"word_spelling_{word.lower()}"
            # Map [-1, 1] -> [0, 1] for AffectiveEvaluator
            native_score = (score / 2.0) + 0.5
            
            if hasattr(self.observer, "evaluator") and hasattr(self.observer.evaluator, "set_affect_score"):
                self.observer.evaluator.set_affect_score(word_id, native_score)
                
            # Still update dictionary for metadata completeness if available
            if self.dictionary_agent:
                if hasattr(self.dictionary_agent, "mock_dict"):
                    if word not in self.dictionary_agent.mock_dict:
                        self.dictionary_agent.mock_dict[word] = {
                            "pos": "sentiment_word",
                            "definition": f"A word with {word} sentiment.",
                            "sentiment": score
                        }
                    else:
                        self.dictionary_agent.mock_dict[word]["sentiment"] = score
                self.dictionary_agent.lookup(word)
