from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import copy
import string
import torch
from pydantic import Field
from hpm_ai_v6.hpm_model.core.cell import Cell
from hpm_ai_v6.hpm_model.agents.social_agent import SocialAgent
from hpm_ai_v6.hpm_model.fields.pattern_field import DynamicPatternField

class CausalRule(Cell):
    """
    A 2-cell representing a causal dependency discovered via intervention.
    Source: The intervention (e.g., 'replace X with Y').
    Target: The observed effect (e.g., 'Surprise in Phrase Agent').
    """
    intervention: str = ""
    effect_magnitude: float = 0.0
    agent_impacted: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        name: str,
        intervention: str,
        effect_magnitude: float,
        agent_impacted: str,
        source: Optional[Cell] = None,
        target: Optional[Cell] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        emb = np.array([effect_magnitude])
        super().__init__(
            name=name,
            dim=2,
            embedding=emb,
            source=source,
            target=target,
            intervention=intervention,
            effect_magnitude=effect_magnitude,
            agent_impacted=agent_impacted,
            metadata=metadata or {},
            **kwargs,
        )

class CausalAgent(SocialAgent):
    """
    Agent that performs active interventions on the corpus to discover causal dependencies.
    It probes other agents to see how their predictions break under counterfactuals.
    """
    def __init__(self, other_agents: Dict[str, SocialAgent], shared_field: Optional[DynamicPatternField] = None, **kwargs):
        self.other_agents = other_agents
        self.causal_patterns = []
        self.cause_cells: Dict[str, Cell] = {}
        self.effect_cells: Dict[str, Cell] = {}
        super().__init__(patterns=[], shared_field=shared_field, **kwargs)

    @staticmethod
    def _clean_words(text: str) -> List[str]:
        return [w.strip(string.punctuation).lower() for w in text.split() if w.strip(string.punctuation)]

    def _sequence_for_agent(self, agent_name: str, chunk: str) -> Tuple[List[Cell], List[Cell]]:
        agent = self.other_agents.get(agent_name)
        if agent is None:
            return [], []

        if agent_name == "char":
            chars = [c for c in chunk.lower() if c in agent.char_cells]
            seq = [agent.char_cells[c] for c in chars]
            population = list(agent.char_cells.values())
            return seq, population

        if agent_name == "word":
            words = self._clean_words(chunk)
            population = list(agent.word_cells.values())
            temp_cells = []
            seq = []
            for word in words:
                if word in agent.word_cells:
                    seq.append(agent.word_cells[word])
                    continue

                if population:
                    proto = agent.word_cells.get("alice") or population[0]
                    temp_embedding = proto.as_tensor() * 0.0
                else:
                    temp_embedding = np.zeros(16, dtype=float)
                temp_cell = Cell(name=f"word_cf_{word}", dim=0, embedding=temp_embedding)
                temp_cells.append(temp_cell)
                seq.append(temp_cell)
            population = population + temp_cells
            return seq, population

        if agent_name == "phrase":
            words = self._clean_words(chunk)
            tags = []
            for word in words:
                if word in {"alice", "rabbit", "sister", "book"}:
                    tags.append("NOUN")
                elif word in {"was", "get", "sitting", "having", "reading"}:
                    tags.append("VERB")
                elif word in {"tired", "sleepy", "natural"}:
                    tags.append("ADJ")
                elif word in {"very"}:
                    tags.append("ADV")
                elif word in {"the", "a"}:
                    tags.append("DET")
                elif word in {"of", "on", "by"}:
                    tags.append("PREP")
                elif word in {"and", "but"}:
                    tags.append("CONJ")
                elif word in {"her", "she", "it"}:
                    tags.append("PRON")
                else:
                    tags.append("NOUN")
            seq = [agent.pos_cells[t] for t in tags if t in agent.pos_cells]
            population = list(agent.pos_cells.values())
            return seq, population

        if agent_name == "semantic":
            cell = agent._get_or_create_sent_cell(chunk)
            seq = [cell] if cell is not None else []
            population = list(agent.sent_cells.values())
            return seq, population

        return [], []

    def _measure_surprise(self, agent_name: str, sequence: List[Any]) -> float:
        """
        Measures the average Negative Log-Likelihood (Surprise) of an agent on a sequence.
        """
        agent = self.other_agents.get(agent_name)
        if not agent or not sequence or not agent.patterns:
            return 0.0

        chunk = sequence[0]
        observation_seq, population = self._sequence_for_agent(agent_name, chunk)
        if len(observation_seq) < 2 or not population:
            return 0.0

        field_amplifications = {}
        if agent.shared_field is not None:
            field_amplifications = agent.shared_field.get_amplifications(agent.patterns)

        total_nll = 0.0
        total_weight = 0.0
        evaluator = agent.learner.epistemic
        weights = agent.get_weights()
        for idx, pattern in enumerate(agent.patterns):
            context = {
                "observation_seq": observation_seq,
                "population": population,
                "field_amplification": field_amplifications.get(pattern.name, 0.0),
            }
            result = evaluator.evaluate(pattern, context)
            match_count = int(result.metadata.get("count", 0))
            if match_count <= 0:
                continue
            weight = float(weights[idx]) if idx < len(weights) else 0.0
            total_nll += float(result.metadata.get("nll", 0.0)) * max(weight, 1e-6)
            total_weight += max(weight, 1e-6)

        if total_weight == 0.0:
            return 0.0
        return total_nll / total_weight

    def _get_or_create_cause_cell(self, original_word: str, counterfactual_word: str, position: int) -> Cell:
        key = f"{original_word.lower()}->{counterfactual_word.lower()}@{position}"
        if key not in self.cause_cells:
            self.cause_cells[key] = Cell(
                name=f"cause_{key}",
                dim=0,
                embedding=np.array([float(len(original_word)), float(position), 1.0], dtype=float),
            )
        return self.cause_cells[key]

    def _get_or_create_effect_cell(self, original_word: str, agent_name: str) -> Cell:
        key = f"{original_word.lower()}->{agent_name}"
        if key not in self.effect_cells:
            self.effect_cells[key] = Cell(
                name=f"effect_{key}",
                dim=0,
                embedding=np.array([float(len(original_word)), float(len(agent_name)), 1.0], dtype=float),
            )
        return self.effect_cells[key]

    def perform_interventions(self, original_text_chunks: List[str]):
        """
        Main loop: Propose, Intervene, Measure, Learn.
        """
        print(f"Causal Agent performing interventions on {len(original_text_chunks)} chunks...")
        
        for chunk in original_text_chunks:
            words = chunk.split()
            if len(words) < 5: continue
            
            # 1. Propose Intervention: Swap a word
            idx_to_swap = np.random.randint(0, len(words))
            original_word = words[idx_to_swap]
            # Simple counterfactual: replace with a generic noun
            counterfactual_word = "suddenly"
            
            intervened_words = copy.copy(words)
            intervened_words[idx_to_swap] = counterfactual_word
            intervened_chunk = " ".join(intervened_words)
            
            # 2. Measure Causal Effect across agents
            effects = {}
            for name in self.other_agents:
                # Surprise on original vs intervened
                # Note: In a real run, we'd feed the actual cells/tokens to the agents
                orig_surprise = self._measure_surprise(name, [chunk])
                new_surprise = self._measure_surprise(name, [intervened_chunk])
                
                effect = abs(new_surprise - orig_surprise)
                effects[name] = effect
                
                # 3. If effect is significant, create a Causal Rule (2-cell)
                if effect > 0.005:
                    rule_name = f"causal_{original_word}_in_{name}"
                    cause_cell = self._get_or_create_cause_cell(original_word, counterfactual_word, idx_to_swap)
                    effect_cell = self._get_or_create_effect_cell(original_word, name)
                    rule = CausalRule(
                        name=rule_name,
                        intervention=f"replace '{original_word}' at pos {idx_to_swap}",
                        effect_magnitude=effect,
                        agent_impacted=name,
                        source=cause_cell,
                        target=effect_cell,
                        metadata={
                            "original_word": original_word.lower(),
                            "counterfactual_word": counterfactual_word.lower(),
                            "position": idx_to_swap,
                            "agent_impacted": name,
                            "effect_label": f"surprise in {name}",
                        },
                    )
                    self.patterns.append(rule)
                    print(f"  [Causal Discovery] {rule_name}: Impact {effect:.4f}")

        self._refresh_learner()

    def _refresh_learner(self):
        from hpm_ai_v6.hpm_model.dynamics.meta_rule import MetaPatternRule
        from hpm_ai_v6.hpm_model.dynamics.learning import HPMLearner
        
        old_weights = self.get_weights_dict() if hasattr(self, "meta_rule") else {}
        self.meta_rule = MetaPatternRule(patterns=self.patterns, learning_rate=0.1)
        if old_weights:
            weights = torch.ones(len(self.patterns), dtype=torch.float32) / (len(self.patterns) + 1e-9)
            for i, pattern in enumerate(self.patterns):
                if pattern.name in old_weights:
                    weights[i] = float(old_weights[pattern.name])
            self.meta_rule.set_weights_tensor(weights / (weights.sum() + 1e-9))
        self.learner = HPMLearner(meta_rule=self.meta_rule)

    def _paging_lookup(self):
        lookup = {cell.name: cell for cell in self.cause_cells.values()}
        lookup.update({cell.name: cell for cell in self.effect_cells.values()})
        return lookup

    def get_causal_insights(self) -> List[str]:
        """Returns the most robust causal rules discovered."""
        weights = self.get_weights()
        top_indices = np.argsort(weights)[-3:][::-1]
        return [self.patterns[i].name for i in top_indices if weights[i] > 0]
