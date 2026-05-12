from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from hpm_ai_v6.hpm_model.core.cell import Cell


@dataclass(frozen=True)
class EdgeRecord:
    pattern: Cell
    source: Cell
    target: Cell
    score: float
    raw_weight: float
    agent_name: str
    source_key: str
    target_key: str
    relation: str


@dataclass(frozen=True)
class PathStep:
    source: Cell
    pattern: Cell
    target: Cell
    score: float
    raw_weight: float
    agent_name: str
    source_key: str
    target_key: str
    relation: str


class ReasoningAgent:
    """
    Lightweight reasoning layer over the learned HPM pattern graph.

    This version uses the actually trained structures that exist today:
    - 0-cells for concepts/entities
    - 1-cells for directed weighted transitions
    - 2-cell "analogies" approximated by embedding similarity over 1-cells
    """

    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "because", "been", "but",
        "by", "did", "do", "does", "for", "from", "had", "has", "have", "how",
        "i", "in", "is", "it", "like", "me", "of", "on", "or", "our", "out",
        "so", "that", "the", "their", "there", "this", "to", "was", "were",
        "what", "when", "where", "why", "with", "you",
    }

    def __init__(self, reader, beam_width: int = 5, max_depth: int = 4, analogy_threshold: float = 0.85):
        self.reader = reader
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.analogy_threshold = analogy_threshold
        self._edge_index: Dict[str, List[EdgeRecord]] = {}
        self._node_index: Dict[str, Cell] = {}
        self._alias_index: Dict[str, List[str]] = {}
        self._all_patterns: List[Cell] = []
        self._sentence_labels: Dict[str, str] = {}
        self._analogy_cache: Dict[str, List[Tuple[float, Cell]]] = {}
        self._explicit_analogy_index: Dict[str, List[Tuple[float, Cell]]] = {}
        self._explicit_rule_index: Dict[str, List[Tuple[float, Cell]]] = {}
        self._transient_rule_edge_index: Dict[str, List[EdgeRecord]] = {}
        self._forward_rule_patterns: List[Tuple[float, Cell]] = []
        self._dirty: bool = True
        self._relation_registry = getattr(reader, "relation_registry", None)
        self._relation_cell_index: Dict[str, Cell] = {}

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        tokens = []
        for raw in re.findall(r"[a-zA-Z']+", text.lower()):
            token = raw.strip("'")
            if token and token not in cls.STOPWORDS:
                tokens.append(token)
        return tokens

    @staticmethod
    def _strip_prefix(name: str) -> str:
        for prefix in ("word_", "char_", "pos_", "sent_", "ctx_"):
            if name.startswith(prefix):
                return name[len(prefix):]
        return name

    @staticmethod
    def _safe_label(text: str, max_len: int = 48) -> str:
        clean = re.sub(r"\s+", " ", text.strip())
        if len(clean) <= max_len:
            return clean
        return clean[: max_len - 3] + "..."

    def _cell_key(self, cell: Cell) -> str:
        name = cell.name
        if name.startswith("word_"):
            return f"word:{name.removeprefix('word_')}"
        if name.startswith("char_"):
            return f"char:{name.removeprefix('char_')}"
        if name.startswith("pos_"):
            return f"pos:{name.removeprefix('pos_')}"
        if name.startswith("ctx_"):
            return f"context:{name.removeprefix('ctx_')}"
        if name.startswith("sent_"):
            return f"sentence:{name.removeprefix('sent_')}"
        if name.startswith("cause_"):
            return f"cause:{name.removeprefix('cause_')}"
        if name.startswith("effect_"):
            return f"effect:{name.removeprefix('effect_')}"
        return f"cell:{name}"

    def _cell_display(self, cell: Cell) -> str:
        key = self._cell_key(cell)
        if key.startswith("sentence:"):
            text = self._sentence_labels.get(key)
            if text:
                return self._safe_label(text)
        if key.startswith("context:"):
            return key.split(":", 1)[1].replace("|", " ")
        if key.startswith("cause:"):
            cause = key.split(":", 1)[1]
            return cause.replace("->", " to ").replace("@", " at ")
        if key.startswith("effect:"):
            effect = key.split(":", 1)[1]
            original, agent = effect.split("->", 1) if "->" in effect else (effect, "agent")
            return f"surprise in {agent} after {original}"
        return self._strip_prefix(cell.name)

    @staticmethod
    def _agent_relation(agent_name: str) -> str:
        return {
            "word": "lexical_transition",
            "contextual": "contextual_prediction",
            "semantic": "semantic_transition",
            "phrase": "syntactic_transition",
            "char": "character_transition",
            "causal": "causal_relation",
        }.get(agent_name, "transition")

    @staticmethod
    def _calibrate_scores(weights: Sequence[float]) -> List[float]:
        if not weights:
            return []
        if len(weights) == 1:
            return [1.0]
        ordered = sorted(float(weight) for weight in weights)
        calibrated: List[float] = []
        denom = max(len(ordered) - 1, 1)
        for weight in weights:
            rank = 0
            while rank < len(ordered) and ordered[rank] <= float(weight):
                rank += 1
            percentile = (rank - 1) / denom
            calibrated.append(0.05 + 0.95 * percentile)
        return calibrated

    @staticmethod
    def _cell_priority(cell: Cell) -> Tuple[int, int]:
        if cell.name.startswith("word_"):
            return (0, -len(cell.name))
        if cell.name.startswith("sent_"):
            return (1, -len(cell.name))
        if cell.name.startswith("ctx_"):
            return (2, -len(cell.name))
        if cell.name.startswith("cause_"):
            return (3, -len(cell.name))
        if cell.name.startswith("effect_"):
            return (4, -len(cell.name))
        if cell.name.startswith("pos_"):
            return (5, -len(cell.name))
        if cell.name.startswith("char_"):
            return (6, -len(cell.name))
        return (7, -len(cell.name))

    def _iter_reasoning_agents(self) -> Iterable[Tuple[str, object]]:
        for name in ("word", "contextual", "semantic", "phrase", "char", "causal", "syntactic"):
            agent = self.reader.agents.get(name)
            if agent is not None:
                yield name, agent

    @staticmethod
    def _bridge_pattern_name(source_key: str, target_key: str, relation: str) -> str:
        safe_source = source_key.replace(":", "_")
        safe_target = target_key.replace(":", "_")
        return f"bridge_{relation}_{safe_source}_to_{safe_target}"

    def _make_bridge_pattern(self, source: Cell, target: Cell, relation: str) -> Cell:
        return Cell(
            name=self._bridge_pattern_name(self._cell_key(source), self._cell_key(target), relation),
            dim=1,
            embedding=target.as_numpy() - source.as_numpy(),
            source=source,
            target=target,
        )

    def _add_edge_record(
        self,
        edge_index: Dict[str, List[EdgeRecord]],
        node_index: Dict[str, Cell],
        pattern: Cell,
        score: float,
        raw_weight: float,
        agent_name: str,
        relation: str,
    ) -> None:
        if pattern.source is None or pattern.target is None:
            return
        source_key = self._cell_key(pattern.source)
        target_key = self._cell_key(pattern.target)
        node_index[source_key] = pattern.source
        node_index[target_key] = pattern.target
        edge_index.setdefault(source_key, []).append(
            EdgeRecord(
                pattern=pattern,
                source=pattern.source,
                target=pattern.target,
                score=max(float(score), 1e-6),
                raw_weight=float(raw_weight),
                agent_name=agent_name,
                source_key=source_key,
                target_key=target_key,
                relation=relation,
            )
        )

    def _add_causal_bridge_edges(
        self,
        edge_index: Dict[str, List[EdgeRecord]],
        node_index: Dict[str, Cell],
    ) -> None:
        causal_agent = self.reader.agents.get("causal")
        if causal_agent is None:
            return

        word_lookup = {}
        word_agent = self.reader.agents.get("word")
        if word_agent is not None:
            word_lookup = getattr(word_agent, "word_cells", {}) or {}
            if not word_lookup and hasattr(word_agent, "_paging_lookup"):
                for cell in word_agent._paging_lookup().values():
                    if cell.name.startswith("word_"):
                        word_lookup[cell.name.removeprefix("word_")] = cell

        semantic_agent = self.reader.agents.get("semantic")
        sent_lookup = getattr(semantic_agent, "sent_cells", {}) if semantic_agent is not None else {}
        if semantic_agent is not None and not sent_lookup and hasattr(semantic_agent, "_paging_lookup"):
            for cell in semantic_agent._paging_lookup().values():
                if cell.name.startswith("sent_"):
                    sent_lookup[cell.name] = cell

        for rule in getattr(causal_agent, "patterns", []) or []:
            if getattr(rule, "source", None) is None or getattr(rule, "target", None) is None:
                continue
            metadata = getattr(rule, "metadata", {}) or {}
            original_word = metadata.get("original_word")
            if not original_word:
                continue

            word_cell = word_lookup.get(original_word)
            if word_cell is not None:
                self._add_edge_record(
                    edge_index=edge_index,
                    node_index=node_index,
                    pattern=self._make_bridge_pattern(word_cell, rule.source, "causal_anchor"),
                    score=0.98,
                    raw_weight=max(getattr(rule, "effect_magnitude", 0.0), 1e-6),
                    agent_name="causal",
                    relation="causal_anchor",
                )
                self._add_edge_record(
                    edge_index=edge_index,
                    node_index=node_index,
                    pattern=self._make_bridge_pattern(rule.target, word_cell, "causal_reentry"),
                    score=0.78,
                    raw_weight=max(getattr(rule, "effect_magnitude", 0.0), 1e-6),
                    agent_name="causal",
                    relation="causal_reentry",
                )

            if metadata.get("agent_impacted") == "semantic" and sent_lookup:
                for sentence_text, sentence_cell in sent_lookup.items():
                    if original_word in self._tokenize(sentence_text):
                        self._add_edge_record(
                            edge_index=edge_index,
                            node_index=node_index,
                            pattern=self._make_bridge_pattern(rule.target, sentence_cell, "causal_semantic_reentry"),
                            score=0.72,
                            raw_weight=max(getattr(rule, "effect_magnitude", 0.0), 1e-6),
                            agent_name="causal",
                            relation="causal_semantic_reentry",
                        )

    def _add_sentence_word_bridge_edges(
        self,
        edge_index: Dict[str, List[EdgeRecord]],
        node_index: Dict[str, Cell],
        sentence_labels: Dict[str, str],
    ) -> None:
        """Add bidirectional bridge edges between word cells and sentence cells.

        For each sentence, any word token that resolves to a known word cell gets:
          word_cell → sentence_cell  (word appears in sentence)
          sentence_cell → word_cell  (sentence mentions this word)

        This allows multi-hop paths like: word_alice → sentence → word_rabbit.
        """
        if not sentence_labels:
            return

        # Build word key → Cell lookup from node_index
        word_cells: Dict[str, Cell] = {
            key: cell
            for key, cell in node_index.items()
            if key.startswith("word:")
        }
        if not word_cells:
            return

        bridge_score = 0.45  # moderate confidence — structural, not learned

        for sent_key, sent_text in sentence_labels.items():
            sent_cell = node_index.get(sent_key)
            if sent_cell is None:
                continue

            tokens = self._tokenize(sent_text)
            for token in tokens:
                word_key = f"word:{token}"
                word_cell = word_cells.get(word_key)
                if word_cell is None:
                    continue

                # Use a zero embedding for cross-dim bridge patterns
                import numpy as _np
                bridge_emb = _np.zeros(max(
                    len(word_cell.as_numpy()), len(sent_cell.as_numpy())
                ), dtype=float).tolist()

                bridge_word_sent = Cell(
                    name=self._bridge_pattern_name(word_key, sent_key, "word_in_sentence"),
                    dim=1,
                    embedding=bridge_emb,
                    source=word_cell,
                    target=sent_cell,
                )
                bridge_sent_word = Cell(
                    name=self._bridge_pattern_name(sent_key, word_key, "sentence_mentions_word"),
                    dim=1,
                    embedding=bridge_emb,
                    source=sent_cell,
                    target=word_cell,
                )

                # word → sentence
                self._add_edge_record(
                    edge_index=edge_index,
                    node_index=node_index,
                    pattern=bridge_word_sent,
                    score=bridge_score,
                    raw_weight=bridge_score,
                    agent_name="semantic",
                    relation="word_in_sentence",
                )
                # sentence → word
                self._add_edge_record(
                    edge_index=edge_index,
                    node_index=node_index,
                    pattern=bridge_sent_word,
                    score=bridge_score,
                    raw_weight=bridge_score,
                    agent_name="semantic",
                    relation="sentence_mentions_word",
                )

    def _build_analogy_cache(self, patterns: Sequence[Cell], top_k: int = 8) -> Dict[str, List[Tuple[float, Cell]]]:
        one_cells = [pattern for pattern in patterns if getattr(pattern, "dim", 0) == 1]
        cache: Dict[str, List[Tuple[float, Cell]]] = {}
        for pattern in one_cells:
            scored: List[Tuple[float, Cell]] = []
            for other in one_cells:
                if other.name == pattern.name:
                    continue
                try:
                    score = pattern.similarity(other)
                except Exception:
                    continue
                scored.append((score, other))
            scored.sort(key=lambda item: item[0], reverse=True)
            cache[pattern.name] = scored[:top_k]
        return cache

    def _build_explicit_analogy_index(self, patterns: Sequence[Cell], weights: Dict[str, float]) -> Dict[str, List[Tuple[float, Cell]]]:
        index: Dict[str, List[Tuple[float, Cell]]] = {}
        for pattern in patterns:
            if getattr(pattern, "dim", 0) != 2:
                continue
            source = getattr(pattern, "source", None)
            target = getattr(pattern, "target", None)
            if source is None or target is None:
                continue
            if getattr(source, "dim", 0) != 1 or getattr(target, "dim", 0) != 1:
                continue
            score = float(weights.get(pattern.name, getattr(pattern, "weight", 0.0) or 0.0))
            index.setdefault(source.name, []).append((score, pattern))

        for source_name, entries in index.items():
            entries.sort(key=lambda item: item[0], reverse=True)
            index[source_name] = entries
        return index

    def _build_explicit_rule_index(self, patterns: Sequence[Cell], weights: Dict[str, float]) -> Dict[str, List[Tuple[float, Cell]]]:
        index: Dict[str, List[Tuple[float, Cell]]] = {}
        for pattern in patterns:
            if getattr(pattern, "dim", 0) != 3:
                continue
            source = getattr(pattern, "source", None)
            target = getattr(pattern, "target", None)
            if source is None or target is None:
                continue
            if getattr(source, "dim", 0) != 2 or getattr(target, "dim", 0) != 2:
                continue
            score = float(weights.get(pattern.name, getattr(pattern, "weight", 0.0) or 0.0))
            index.setdefault(source.name, []).append((score, pattern))

        for source_name, entries in index.items():
            entries.sort(key=lambda item: item[0], reverse=True)
            index[source_name] = entries
        return index

    def _build_forward_rule_patterns(self, patterns: Sequence[Cell], weights: Dict[str, float]) -> List[Tuple[float, Cell]]:
        rules: List[Tuple[float, Cell]] = []
        for pattern in patterns:
            if getattr(pattern, "dim", 0) != 3:
                continue
            name = getattr(pattern, "name", "").lower()
            metadata = getattr(pattern, "metadata", {}) or {}
            rule_type = str(metadata.get("rule_type", "")).lower()
            if (
                "transitivity" not in name
                and rule_type not in {"transitivity", "edge_derivation", "subgraph_derivation"}
                and not metadata.get("derive_source")
                and not metadata.get("derive_target")
                and not metadata.get("antecedent_edges")
            ):
                continue
            score = float(weights.get(pattern.name, getattr(pattern, "weight", 0.0) or 0.0))
            rules.append((score, pattern))
        rules.sort(key=lambda item: item[0], reverse=True)
        return rules

    @staticmethod
    def _resolve_rule_endpoint(spec: str, first_edge: EdgeRecord, second_edge: EdgeRecord) -> Optional[Cell]:
        if spec == "source.source":
            return first_edge.source
        if spec == "source.target":
            return first_edge.target
        if spec == "target.source":
            return second_edge.source
        if spec == "target.target":
            return second_edge.target
        return None

    def _build_transient_rule_edge_index(self, rule_index: Dict[str, List[Tuple[float, Cell]]]) -> Dict[str, List[EdgeRecord]]:
        transient: Dict[str, List[EdgeRecord]] = {}
        for rule_entries in rule_index.values():
            for score, rule in rule_entries:
                source_analogy = getattr(rule, "source", None)
                target_analogy = getattr(rule, "target", None)
                if source_analogy is None or target_analogy is None:
                    continue
                source_edge = getattr(source_analogy, "source", None)
                refined_edge = getattr(target_analogy, "target", None)
                if source_edge is None or refined_edge is None:
                    continue
                start_cell = getattr(source_edge, "source", None)
                end_cell = getattr(refined_edge, "target", None)
                if start_cell is None or end_cell is None:
                    continue
                if getattr(source_edge, "dim", 0) != 1 or getattr(refined_edge, "dim", 0) != 1:
                    continue
                source_key = self._cell_key(start_cell)
                target_key = self._cell_key(end_cell)
                pattern = Cell(
                    name=f"transient_{rule.name}_{source_edge.name}_to_{refined_edge.name}",
                    dim=1,
                    embedding=end_cell.as_numpy() - start_cell.as_numpy(),
                    source=start_cell,
                    target=end_cell,
                )
                transient.setdefault(source_key, []).append(
                    EdgeRecord(
                        pattern=pattern,
                        source=start_cell,
                        target=end_cell,
                        score=max(float(score), 1e-6),
                        raw_weight=max(float(score), 1e-6),
                        agent_name="reasoning",
                        source_key=source_key,
                        target_key=target_key,
                        relation="rule_application",
                    )
                )
        for source_key, records in transient.items():
            records.sort(key=lambda item: item.score, reverse=True)
            transient[source_key] = records
        return transient

    @staticmethod
    def _resolve_edge_ref(ref: str, matched_edges: Sequence[EdgeRecord]) -> Optional[Cell]:
        if "." not in ref:
            return None
        edge_part, endpoint = ref.split(".", 1)
        if not edge_part.startswith("edge") or not edge_part[4:].isdigit():
            return None
        edge_idx = int(edge_part[4:])
        if edge_idx >= len(matched_edges):
            return None
        edge = matched_edges[edge_idx]
        if endpoint == "source":
            return edge.source
        if endpoint == "target":
            return edge.target
        return None

    def _resolve_subgraph_consequent_cell(
        self,
        spec: object,
        bindings: Dict[str, Cell],
        matched_edges: Sequence[EdgeRecord],
    ) -> Optional[Cell]:
        if not isinstance(spec, str) or not spec:
            return None
        if spec in bindings:
            return bindings[spec]
        return self._resolve_edge_ref(spec, matched_edges)

    def _edge_matches_template(
        self,
        edge: EdgeRecord,
        template: Dict[str, Any],
        bindings: Dict[str, Cell],
    ) -> Optional[Dict[str, Cell]]:
        next_bindings = dict(bindings)
        penalty = float(next_bindings.get("_analogy_penalty", 1.0))

        for endpoint_name, cell in (("source", edge.source), ("target", edge.target)):
            exact_key = template.get(f"{endpoint_name}_key")
            if exact_key and self._cell_key(cell) != str(exact_key):
                return None
            exact_name = template.get(f"{endpoint_name}_name")
            if exact_name and cell.name != str(exact_name):
                return None
            var_name = template.get(f"{endpoint_name}_var")
            if isinstance(var_name, str) and var_name:
                bound = next_bindings.get(var_name)
                if bound is not None:
                    if self._cell_key(bound) != self._cell_key(cell):
                        # Attempt analogical binding
                        sim = bound.similarity(cell)
                        if sim < self.analogy_threshold:
                            return None
                        # Apply penalty based on similarity (Phase 1, Step 3)
                        penalty *= max(sim, 0.5)
                else:
                    next_bindings[var_name] = cell

        relation = template.get("relation")
        if relation and edge.relation != str(relation):
            return None
        agent_name = template.get("agent")
        if agent_name and edge.agent_name != str(agent_name):
            return None

        next_bindings["_analogy_penalty"] = penalty
        return next_bindings

    def _match_subgraph_templates(
        self,
        templates: Sequence[Dict[str, Any]],
        candidate_edges: Sequence[EdgeRecord],
    ) -> List[Tuple[List[EdgeRecord], Dict[str, Cell]]]:
        matches: List[Tuple[List[EdgeRecord], Dict[str, Cell]]] = []
        if not templates:
            return matches

        def backtrack(
            template_idx: int,
            bindings: Dict[str, Cell],
            chosen_edges: List[EdgeRecord],
            used_indices: set[int],
        ) -> None:
            if template_idx >= len(templates):
                matches.append((list(chosen_edges), dict(bindings)))
                return
            template = templates[template_idx]
            for edge_idx, edge in enumerate(candidate_edges):
                if edge_idx in used_indices:
                    continue
                next_bindings = self._edge_matches_template(edge, template, bindings)
                if next_bindings is None:
                    continue
                chosen_edges.append(edge)
                used_indices.add(edge_idx)
                backtrack(template_idx + 1, next_bindings, chosen_edges, used_indices)
                used_indices.remove(edge_idx)
                chosen_edges.pop()

        backtrack(0, {}, [], set())
        return matches

    def _best_sentence_match(self, question: str) -> Optional[Cell]:
        semantic_agent = self.reader.agents.get("semantic")
        if semantic_agent is None or not hasattr(semantic_agent, "_get_or_create_sent_cell"):
            return None

        sentences: List[Cell] = []
        sent_cells = getattr(semantic_agent, "sent_cells", {}) or {}
        if sent_cells:
            sentences = list(sent_cells.values())
        elif hasattr(semantic_agent, "_paging_lookup"):
            sentences = [cell for cell in semantic_agent._paging_lookup().values() if cell.name.startswith("sent_")]
        if not sentences:
            return None

        try:
            query_cell = semantic_agent._get_or_create_sent_cell(question)
        except Exception:
            return None
        if query_cell is None:
            return None

        best_cell = None
        best_score = -1.0
        for cell in sentences:
            try:
                score = query_cell.similarity(cell)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_cell = cell
        return best_cell if best_score > 0.2 else None

    def invalidate(self) -> None:
        self._dirty = True

    def _ensure_fresh(self) -> None:
        if self._dirty:
            self.refresh()

    def refresh(self) -> None:
        edge_index: Dict[str, List[EdgeRecord]] = {}
        node_index: Dict[str, Cell] = {}
        alias_index: Dict[str, List[str]] = {}
        all_patterns: List[Cell] = []
        sentence_labels: Dict[str, str] = {}

        for agent_name, agent in self._iter_reasoning_agents():
            patterns = list(getattr(agent, "patterns", []) or [])
            weights = list(agent.get_weights()) if hasattr(agent, "get_weights") else []
            lookup = getattr(agent, "_paging_lookup", lambda: {})()
            calibrated = self._calibrate_scores(weights)

            for idx, pattern in enumerate(patterns):
                all_patterns.append(pattern)
                if pattern.source is None or pattern.target is None:
                    continue

                raw_weight = float(weights[idx]) if idx < len(weights) else float(getattr(pattern, "weight", 0.0))
                score = calibrated[idx] if idx < len(calibrated) else max(raw_weight, 1e-6)
                score = max(float(score), 1e-6)
                self._add_edge_record(
                    edge_index=edge_index,
                    node_index=node_index,
                    pattern=pattern,
                    score=score,
                    raw_weight=raw_weight,
                    agent_name=agent_name,
                    relation=self._agent_relation(agent_name),
                )

            for cell in lookup.values():
                key = self._cell_key(cell)
                node_index[key] = cell
                alias = self._normalize(self._strip_prefix(cell.name))
                if alias:
                    alias_index.setdefault(alias, []).append(key)
                    alias_index.setdefault(self._normalize(cell.name), []).append(key)
                if key.startswith("cause:"):
                    for token in self._tokenize(alias.replace("->", " ").replace("@", " ")):
                        alias_index.setdefault(token, []).append(key)
                if key.startswith("effect:"):
                    effect_alias = alias.replace("->", " ")
                    for token in self._tokenize(effect_alias):
                        alias_index.setdefault(token, []).append(key)
                if key.startswith("sentence:") and hasattr(agent, "sent_text_by_name"):
                    text = getattr(agent, "sent_text_by_name", {}).get(cell.name)
                    if text:
                        sentence_labels[key] = text
                        for token in self._tokenize(text):
                            alias_index.setdefault(token, []).append(key)
                if key.startswith("context:"):
                    for token in self._tokenize(alias.replace("|", " ")):
                        alias_index.setdefault(token, []).append(key)

        self._add_causal_bridge_edges(edge_index=edge_index, node_index=node_index)
        self._add_sentence_word_bridge_edges(
            edge_index=edge_index,
            node_index=node_index,
            sentence_labels=sentence_labels,
        )

        # POS-tag enrichment: re-tag word agent edges with POS roles if syntactic agent available
        _syn_agent = self.reader.agents.get("syntactic") if hasattr(self.reader, "agents") else None
        if _syn_agent is not None and hasattr(_syn_agent, "get_pos"):
            tagged_index: Dict[str, List[EdgeRecord]] = {}
            for source_key, records in edge_index.items():
                new_records = []
                for rec in records:
                    if rec.agent_name == "word" and rec.source.name.startswith("word_"):
                        src_word = rec.source.name[len("word_"):]
                        pos = _syn_agent.get_pos(src_word)
                        if pos is not None:
                            rec = EdgeRecord(
                                pattern=rec.pattern,
                                source=rec.source,
                                target=rec.target,
                                score=rec.score,
                                raw_weight=rec.raw_weight,
                                agent_name=rec.agent_name,
                                source_key=rec.source_key,
                                target_key=rec.target_key,
                                relation=f"pos_{pos}",
                            )
                    new_records.append(rec)
                tagged_index[source_key] = new_records
            edge_index = tagged_index

        for records in edge_index.values():
            records.sort(key=lambda item: item.score, reverse=True)

        self._edge_index = edge_index
        self._node_index = node_index
        self._alias_index = alias_index
        self._all_patterns = all_patterns
        self._sentence_labels = sentence_labels
        self._analogy_cache = self._build_analogy_cache(all_patterns)
        pattern_weights = {
            pattern.name: float(getattr(pattern, "weight", 0.0))
            for pattern in all_patterns
        }
        for agent_name, agent in self._iter_reasoning_agents():
            patterns = list(getattr(agent, "patterns", []) or [])
            weights = list(agent.get_weights()) if hasattr(agent, "get_weights") else []
            for idx, pattern in enumerate(patterns):
                if idx < len(weights):
                    pattern_weights[pattern.name] = float(weights[idx])
        self._explicit_analogy_index = self._build_explicit_analogy_index(all_patterns, pattern_weights)
        self._explicit_rule_index = self._build_explicit_rule_index(all_patterns, pattern_weights)
        self._transient_rule_edge_index = self._build_transient_rule_edge_index(self._explicit_rule_index)
        self._forward_rule_patterns = self._build_forward_rule_patterns(all_patterns, pattern_weights)
        self._relation_cell_index = {
            pattern.name.removeprefix("rel_"): pattern
            for pattern in all_patterns
            if getattr(pattern, "dim", 0) == 2
            and getattr(pattern, "name", "").startswith("rel_")
        }
        self._dirty = False

    def _choose_cell(self, candidates: Sequence[str]) -> Optional[Cell]:
        if not candidates:
            return None
        unique_cells = []
        seen = set()
        for key in candidates:
            if key in seen:
                continue
            seen.add(key)
            cell = self._node_index.get(key)
            if cell is not None:
                unique_cells.append(cell)
        if not unique_cells:
            return None
        ordered = sorted(unique_cells, key=self._cell_priority)
        return ordered[0]

    def _resolve_cell(self, token_or_phrase: str) -> Optional[Cell]:
        self._ensure_fresh()

        key = self._normalize(token_or_phrase)
        exact = self._choose_cell(self._alias_index.get(key, []))
        if exact is not None:
            return exact

        if key.endswith("ed") and key[:-2] in self._alias_index:
            resolved = self._choose_cell(self._alias_index.get(key[:-2], []))
            if resolved is not None:
                return resolved
        if key.endswith("ing") and key[:-3] in self._alias_index:
            resolved = self._choose_cell(self._alias_index.get(key[:-3], []))
            if resolved is not None:
                return resolved
        if key.endswith("s") and key[:-1] in self._alias_index:
            resolved = self._choose_cell(self._alias_index.get(key[:-1], []))
            if resolved is not None:
                return resolved

        for alias, cells in self._alias_index.items():
            if len(alias) < 3:
                continue
            if key in alias or alias in key:
                resolved = self._choose_cell(cells)
                if resolved is not None:
                    return resolved
        return None

    def _resolve_cells(self, token_or_phrase: str) -> List[Cell]:
        self._ensure_fresh()
        key = self._normalize(token_or_phrase)
        candidates = self._alias_index.get(key, [])
        resolved: List[Cell] = []
        seen = set()
        for candidate_key in candidates:
            cell = self._node_index.get(candidate_key)
            if cell is None or cell.name in seen:
                continue
            seen.add(cell.name)
            resolved.append(cell)
        return sorted(resolved, key=self._cell_priority)

    def _extract_query_terms(self, question: str) -> List[str]:
        return self._tokenize(question)

    _EXPLANATION_TOKENS = frozenset({"why", "because", "reason", "explain", "cause", "due"})
    _ANALOGY_TOKENS = frozenset({"analog", "analogous", "similar", "compare", "alike", "correspond"})
    _CONNECTION_TOKENS = frozenset({"how", "connect", "lead", "path", "relate", "link", "between", "reach"})

    def _parse_question(self, question: str) -> Dict[str, object]:
        tokens = set(re.findall(r"[a-z]+", self._normalize(question)))
        terms = self._extract_query_terms(question)

        if tokens & self._EXPLANATION_TOKENS:
            return {"type": "path", "terms": terms, "mode": "explanation"}
        if tokens & self._ANALOGY_TOKENS:
            return {"type": "analogy", "terms": terms}
        if tokens & self._CONNECTION_TOKENS:
            return {"type": "path", "terms": terms, "mode": "connection"}
        return {"type": "path", "terms": terms, "mode": "default"}

    def _pick_path_endpoints(self, terms: Sequence[str]) -> Tuple[Optional[Cell], Optional[Cell]]:
        resolved: List[Cell] = []
        for term in terms:
            for cell in self._resolve_cells(term)[:3]:
                if cell not in resolved:
                    resolved.append(cell)

        if not resolved:
            return None, None
        if len(resolved) == 1:
            return resolved[0], None
        word_cells = [cell for cell in resolved if cell.name.startswith("word_")]
        if len(word_cells) >= 2:
            return word_cells[0], word_cells[-1]
        return resolved[0], resolved[-1]

    def _pick_path_endpoints_with_fallback(self, question: str, terms: Sequence[str]) -> Tuple[Optional[Cell], Optional[Cell]]:
        start, goal = self._pick_path_endpoints(terms)
        if start is not None:
            return start, goal

        sentence_cell = self._best_sentence_match(question)
        if sentence_cell is None:
            return None, None

        sentence_key = self._cell_key(sentence_cell)
        outgoing = self._edge_index.get(sentence_key, [])
        if outgoing:
            return sentence_cell, outgoing[0].target
        return sentence_cell, None

    def _pick_analogy_endpoints(self, terms: Sequence[str]) -> Tuple[Optional[Cell], Optional[Cell]]:
        resolved: List[Cell] = []
        for term in terms:
            cell = self._resolve_cell(term)
            if cell is not None and cell not in resolved:
                resolved.append(cell)
        if not resolved:
            return None, None
        return resolved[0], resolved[1] if len(resolved) > 1 else None

    def _pick_explanation_endpoint(self, terms: Sequence[str]) -> Optional[Cell]:
        original_terms = [self._normalize(term) for term in terms]
        for term in original_terms:
            effect_candidates = self._alias_index.get(term, [])
            for candidate_key in effect_candidates:
                if candidate_key.startswith("effect:"):
                    cell = self._node_index.get(candidate_key)
                    if cell is not None:
                        return cell

        for term in original_terms:
            for candidate_key, cell in self._node_index.items():
                if not candidate_key.startswith("effect:"):
                    continue
                if term in candidate_key:
                    return cell
        return None

    def _pick_explanation_goal(self, terms: Sequence[str], effect: Cell) -> Optional[Cell]:
        effect_key = self._cell_key(effect)
        goals: List[Cell] = []
        for term in terms:
            for cell in self._resolve_cells(term):
                cell_key = self._cell_key(cell)
                if cell_key == effect_key or cell_key.startswith("cause:") or cell_key.startswith("effect:"):
                    continue
                if cell not in goals:
                    goals.append(cell)
        if not goals:
            return None
        word_goals = [cell for cell in goals if cell.name.startswith("word_")]
        if word_goals:
            return word_goals[-1]
        return goals[-1]

    def _best_causal_edge_for_effect(self, effect: Cell) -> Optional[EdgeRecord]:
        effect_key = self._cell_key(effect)
        best: Optional[EdgeRecord] = None
        for records in self._edge_index.values():
            for edge in records:
                if edge.agent_name != "causal" or edge.target_key != effect_key:
                    continue
                if best is None or edge.score > best.score:
                    best = edge
        return best

    def _forward_chain_edges(self, start_key: str, max_rounds: int = 3) -> Dict[str, List[EdgeRecord]]:
        if not self._forward_rule_patterns:
            return {}

        base_edges: Dict[str, List[EdgeRecord]] = {}
        for source_key, records in self._edge_index.items():
            base_edges[source_key] = list(records)
        for source_key, records in self._transient_rule_edge_index.items():
            base_edges.setdefault(source_key, []).extend(records)

        reachable = {start_key}
        frontier = {start_key}
        for _ in range(self.max_depth):
            next_frontier = set()
            for node_key in frontier:
                for edge in base_edges.get(node_key, []):
                    if edge.target_key not in reachable:
                        reachable.add(edge.target_key)
                        next_frontier.add(edge.target_key)
            if not next_frontier:
                break
            frontier = next_frontier

        derived: Dict[str, List[EdgeRecord]] = {}
        best_scores: Dict[Tuple[str, str], float] = {}
        current_edges = {source_key: list(records) for source_key, records in base_edges.items()}

        transitivity_rules = list(self._forward_rule_patterns)
        if not transitivity_rules:
            return {}

        for round_idx in range(max_rounds):
            added = False
            snapshot = {source_key: list(records) for source_key, records in current_edges.items()}
            all_snapshot_edges = [edge for records in snapshot.values() for edge in records]
            reachable_edges = [
                edge
                for edge in all_snapshot_edges
                if edge.source_key in reachable or edge.target_key in reachable
            ]
            for rule_score, rule in transitivity_rules:
                metadata = getattr(rule, "metadata", {}) or {}
                rule_type = str(metadata.get("rule_type", "")).lower()
                pair_mode = str(metadata.get("pair_mode", "chain")).lower()
                if rule_type != "subgraph_derivation" and not metadata.get("antecedent_edges"):
                    continue
                antecedent_edges = metadata.get("antecedent_edges", [])
                consequent = metadata.get("consequent", {}) or {}
                if not isinstance(antecedent_edges, list) or not isinstance(consequent, dict):
                    continue
                candidate_edges = all_snapshot_edges if pair_mode == "any_reachable" else reachable_edges
                for matched_edges, bindings in self._match_subgraph_templates(antecedent_edges, candidate_edges):
                    derived_source = self._resolve_subgraph_consequent_cell(
                        consequent.get("source_var") or consequent.get("source_ref"),
                        bindings,
                        matched_edges,
                    )
                    derived_target = self._resolve_subgraph_consequent_cell(
                        consequent.get("target_var") or consequent.get("target_ref"),
                        bindings,
                        matched_edges,
                    )
                    if derived_source is None or derived_target is None:
                        continue
                    source_key = self._cell_key(derived_source)
                    target_key = self._cell_key(derived_target)
                    if source_key == target_key:
                        continue
                    support = min(edge.score for edge in matched_edges)
                    penalty = float(bindings.get("_analogy_penalty", 1.0))
                    derived_score = max(rule_score * support * penalty, 1e-6)
                    previous = best_scores.get((source_key, target_key))
                    if previous is not None and derived_score <= previous:
                        continue
                    best_scores[(source_key, target_key)] = derived_score
                    pattern = Cell(
                        name=f"subgraph_{rule.name}_r{round_idx}_{'_'.join(edge.pattern.name for edge in matched_edges)}",
                        dim=1,
                        embedding=derived_target.as_numpy() - derived_source.as_numpy(),
                        source=derived_source,
                        target=derived_target,
                    )
                    record = EdgeRecord(
                        pattern=pattern,
                        source=derived_source,
                        target=derived_target,
                        score=derived_score,
                        raw_weight=derived_score,
                        agent_name="reasoning",
                        source_key=source_key,
                        target_key=target_key,
                        relation="subgraph_rule",
                    )
                    current_edges.setdefault(source_key, []).append(record)
                    derived.setdefault(source_key, []).append(record)
                    if source_key in reachable and target_key not in reachable:
                        reachable.add(target_key)
                    added = True
            for source_key in list(reachable):
                first_hops = snapshot.get(source_key, [])
                for first_edge in first_hops:
                    for rule_score, rule in transitivity_rules:
                        metadata = getattr(rule, "metadata", {}) or {}
                        rule_type = str(metadata.get("rule_type", "")).lower()
                        pair_mode = str(metadata.get("pair_mode", "chain")).lower()
                        if rule_type == "subgraph_derivation" or metadata.get("antecedent_edges"):
                            continue
                        if rule_type not in {"transitivity", "edge_derivation"} and "transitivity" not in getattr(rule, "name", "").lower():
                            continue
                        if pair_mode == "any_reachable":
                            second_hops = all_snapshot_edges
                        else:
                            middle_key = first_edge.target_key
                            second_hops = snapshot.get(middle_key, [])

                        for second_edge in second_hops:
                            target_key = second_edge.target_key
                            if target_key == source_key and pair_mode != "any_reachable":
                                continue
                            if (source_key, target_key) in best_scores and best_scores[(source_key, target_key)] >= first_edge.score * second_edge.score:
                                continue
                            derive_source_spec = str(metadata.get("derive_source", "source.source"))
                            derive_target_spec = str(metadata.get("derive_target", "target.target"))
                            derived_source = self._resolve_rule_endpoint(derive_source_spec, first_edge, second_edge)
                            derived_target = self._resolve_rule_endpoint(derive_target_spec, first_edge, second_edge)
                            if derived_source is None or derived_target is None:
                                continue
                            source_key = self._cell_key(derived_source)
                            target_key = self._cell_key(derived_target)
                            if source_key == target_key:
                                continue
                            antecedent_support = min(first_edge.score, second_edge.score)
                            derived_score = max(rule_score * antecedent_support, 1e-6)
                            previous = best_scores.get((source_key, target_key))
                            if previous is not None and derived_score <= previous:
                                continue
                            best_scores[(source_key, target_key)] = derived_score
                            pattern = Cell(
                                name=f"forward_{rule.name}_r{round_idx}_{first_edge.pattern.name}_{second_edge.pattern.name}",
                                dim=1,
                                embedding=derived_target.as_numpy() - derived_source.as_numpy(),
                                source=derived_source,
                                target=derived_target,
                            )
                            record = EdgeRecord(
                                pattern=pattern,
                                source=derived_source,
                                target=derived_target,
                                score=derived_score,
                                raw_weight=derived_score,
                                agent_name="reasoning",
                                source_key=source_key,
                                target_key=target_key,
                                relation="forward_chain",
                            )
                            current_edges.setdefault(source_key, []).append(record)
                            derived.setdefault(source_key, []).append(record)
                            if target_key not in reachable:
                                reachable.add(target_key)
                            added = True
            if not added:
                break

        for source_key, records in derived.items():
            records.sort(key=lambda item: item.score, reverse=True)
        return derived

    def _beam_search_path(self, start: Cell, goal: Cell) -> Optional[List[PathStep]]:
        self._ensure_fresh()

        start_key = self._cell_key(start)
        goal_key = self._cell_key(goal)
        forward_edges = self._forward_chain_edges(start_key)
        beams: List[Tuple[float, List[PathStep], str]] = [(0.0, [], start_key)]
        best_path: Optional[List[PathStep]] = None
        best_cost = math.inf
        best_seen_cost: Dict[str, float] = {start_key: 0.0}

        for _depth in range(self.max_depth):
            next_beams: List[Tuple[float, List[PathStep], str]] = []
            for cost, path, node_name in beams:
                if node_name == goal_key:
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
                    continue

                combined_edges = (
                    list(self._edge_index.get(node_name, []))
                    + list(self._transient_rule_edge_index.get(node_name, []))
                    + list(forward_edges.get(node_name, []))
                )
                combined_edges.sort(key=lambda item: item.score, reverse=True)
                for edge in combined_edges[: self.beam_width]:
                    if any(step.target_key == edge.target_key for step in path):
                        continue
                    edge_cost = -math.log(max(edge.score, 1e-9))
                    rel_cell = self._relation_cell_index.get(edge.relation)
                    if rel_cell is not None:
                        try:
                            import numpy as np
                            rel_emb = rel_cell.as_numpy()
                            src_emb = edge.source.as_numpy()
                            tgt_emb = edge.target.as_numpy()
                            predicted = src_emb + rel_emb
                            np_p = np.linalg.norm(predicted)
                            nt = np.linalg.norm(tgt_emb)
                            if np_p > 1e-9 and nt > 1e-9:
                                coherence = float(np.dot(predicted, tgt_emb) / (np_p * nt))
                                bonus = 0.5 + 0.5 * max(coherence, 0.0)
                                edge_cost /= bonus
                        except Exception:
                            pass
                    elif self._relation_registry is not None:
                        try:
                            coherence = self._relation_registry.coherence_score(
                                edge.source.as_numpy(), edge.relation, edge.target.as_numpy()
                            )
                            bonus = 0.5 + 0.5 * max(coherence, 0.0)
                            edge_cost /= bonus
                        except Exception:
                            pass
                    new_cost = cost + edge_cost
                    previous_best = best_seen_cost.get(edge.target_key)
                    if previous_best is not None and new_cost >= previous_best:
                        continue
                    best_seen_cost[edge.target_key] = new_cost
                    step = PathStep(
                        source=edge.source,
                        pattern=edge.pattern,
                        target=edge.target,
                        score=edge.score,
                        raw_weight=edge.raw_weight,
                        agent_name=edge.agent_name,
                        source_key=edge.source_key,
                        target_key=edge.target_key,
                        relation=edge.relation,
                    )
                    new_path = path + [step]
                    next_beams.append((new_cost, new_path, edge.target_key))

            if not next_beams:
                break

            next_beams.sort(key=lambda item: item[0])
            beams = next_beams[: self.beam_width]

            for cost, path, node_name in beams:
                if node_name == goal_key and cost < best_cost:
                    best_cost = cost
                    best_path = path

        return best_path

    def _step_to_trace(self, step: PathStep) -> Dict[str, Any]:
        return {
            "source_key": step.source_key,
            "source_label": self._cell_display(step.source),
            "target_key": step.target_key,
            "target_label": self._cell_display(step.target),
            "relation": step.relation,
            "agent": step.agent_name,
            "score": step.score,
            "raw_weight": step.raw_weight,
            "pattern": step.pattern.name,
        }

    def _path_to_trace(self, path: Sequence[PathStep]) -> Dict[str, Any]:
        steps = [self._step_to_trace(step) for step in path]
        nodes: List[Dict[str, str]] = []
        if path:
            nodes.append({"key": path[0].source_key, "label": self._cell_display(path[0].source)})
            for step in path:
                nodes.append({"key": step.target_key, "label": self._cell_display(step.target)})
        combined_score = 1.0
        for step in path:
            combined_score *= max(step.score, 1e-9)
        return {"nodes": nodes, "steps": steps, "combined_score": combined_score}

    @staticmethod
    def _is_var(spec: object) -> bool:
        return isinstance(spec, str) and spec.startswith("{") and spec.endswith("}") and len(spec) > 2

    @staticmethod
    def _var_name(spec: str) -> str:
        return spec[1:-1]

    def _default_rule_templates(self, rule: Cell) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        metadata = getattr(rule, "metadata", {}) or {}
        antecedents = metadata.get("antecedents")
        if isinstance(antecedents, list) and antecedents:
            consequent = metadata.get("consequent")
            if isinstance(consequent, dict):
                return antecedents, consequent
        return (
            [
                {"source": "{X}", "target": "{Y}"},
                {"source": "{Y}", "target": "{Z}"},
            ],
            {"source": "{X}", "target": "{Z}"},
        )

    def _unify_goal_with_consequent(
        self,
        consequent: Dict[str, str],
        source: Cell,
        target: Cell,
    ) -> Optional[Dict[str, Cell]]:
        bindings: Dict[str, Cell] = {}
        for spec, concrete in (
            (consequent.get("source"), source),
            (consequent.get("target"), target),
        ):
            if not isinstance(spec, str):
                return None
            if self._is_var(spec):
                bindings[self._var_name(spec)] = concrete
                continue
            if spec != self._cell_key(concrete) and spec != concrete.name:
                return None
        return bindings

    def _candidate_edges_for_goal(
        self,
        facts: Dict[str, List[EdgeRecord]],
        source: Optional[Cell] = None,
        target: Optional[Cell] = None,
    ) -> List[EdgeRecord]:
        if source is not None:
            edges = list(facts.get(self._cell_key(source), []))
        else:
            edges = [edge for records in facts.values() for edge in records]
        if target is not None:
            target_key = self._cell_key(target)
            edges = [edge for edge in edges if edge.target_key == target_key]
        edges.sort(key=lambda item: item.score, reverse=True)
        return edges

    def _prove_subgraph_antecedents(
        self,
        antecedents: Sequence[Dict[str, Any]],
        facts: Dict[str, List[EdgeRecord]],
        bindings: Dict[str, Cell],
        depth: int,
        max_depth: int,
        visiting: set[Tuple[str, str]],
        memo: Dict[Tuple[str, str, int], Optional[List[PathStep]]],
        max_branching: int = 8,
    ) -> Optional[Tuple[List[PathStep], Dict[str, Cell]]]:
        best_path: Optional[List[PathStep]] = None
        best_bindings: Optional[Dict[str, Cell]] = None
        best_score = -1.0

        def backtrack(
            idx: int,
            current_bindings: Dict[str, Cell],
            current_path: List[PathStep],
        ) -> None:
            nonlocal best_path, best_bindings, best_score
            if idx >= len(antecedents):
                score = 1.0
                for step in current_path:
                    score *= max(step.score, 1e-9)
                penalty = float(current_bindings.get("_analogy_penalty", 1.0))
                score *= penalty
                if score > best_score:
                    best_score = score
                    best_path = list(current_path)
                    best_bindings = dict(current_bindings)
                return

            template = antecedents[idx]
            source_var = template.get("source_var")
            target_var = template.get("target_var")
            source_cell = current_bindings.get(source_var) if isinstance(source_var, str) else None
            target_cell = current_bindings.get(target_var) if isinstance(target_var, str) else None

            if source_cell is not None and target_cell is not None:
                subpath = self._prove_edge_backward(
                    source_cell,
                    target_cell,
                    facts,
                    depth + 1,
                    max_depth,
                    visiting,
                    memo,
                )
                if not subpath:
                    return
                current_path.extend(subpath)
                backtrack(idx + 1, current_bindings, current_path)
                del current_path[-len(subpath):]
                return

            candidate_edges = self._candidate_edges_for_goal(
                facts,
                source=source_cell,
                target=target_cell,
            )[:max_branching]
            for edge in candidate_edges:
                next_bindings = self._edge_matches_template(edge, template, current_bindings)
                if next_bindings is None:
                    continue
                subpath = self._prove_edge_backward(
                    edge.source,
                    edge.target,
                    facts,
                    depth + 1,
                    max_depth,
                    visiting,
                    memo,
                )
                if not subpath:
                    continue
                current_path.extend(subpath)
                backtrack(idx + 1, next_bindings, current_path)
                del current_path[-len(subpath):]

        backtrack(0, dict(bindings), [])
        if best_path is None or best_bindings is None:
            return None
        return best_path, best_bindings

    def _prove_edge_backward(
        self,
        source: Cell,
        target: Cell,
        facts: Dict[str, List[EdgeRecord]],
        depth: int,
        max_depth: int,
        visiting: set[Tuple[str, str]],
        memo: Dict[Tuple[str, str, int], Optional[List[PathStep]]],
    ) -> Optional[List[PathStep]]:
        source_key = self._cell_key(source)
        target_key = self._cell_key(target)
        memo_key = (source_key, target_key, depth)
        if memo_key in memo:
            return memo[memo_key]

        goal_key = (source_key, target_key)
        if goal_key in visiting:
            memo[memo_key] = None
            return None

        direct_edges = self._candidate_edges_for_goal(facts, source=source, target=target)
        if direct_edges:
            edge = direct_edges[0]
            memo[memo_key] = [
                PathStep(
                    source=edge.source,
                    pattern=edge.pattern,
                    target=edge.target,
                    score=edge.score,
                    raw_weight=edge.raw_weight,
                    agent_name=edge.agent_name,
                    source_key=edge.source_key,
                    target_key=edge.target_key,
                    relation=edge.relation,
                )
            ]
            return memo[memo_key]

        if depth >= max_depth:
            memo[memo_key] = None
            return None

        visiting.add(goal_key)
        best_path: Optional[List[PathStep]] = None
        best_score = -1.0

        for rule_score, rule in self._forward_rule_patterns:
            metadata = getattr(rule, "metadata", {}) or {}
            rule_type = str(metadata.get("rule_type", "")).lower()
            if rule_type == "subgraph_derivation" or metadata.get("antecedent_edges"):
                antecedents = metadata.get("antecedent_edges", [])
                consequent = metadata.get("consequent", {}) or {}
                if not isinstance(antecedents, list) or not isinstance(consequent, dict):
                    continue
                bindings: Dict[str, Cell] = {}
                consequent_source = consequent.get("source_var") or consequent.get("source_ref")
                consequent_target = consequent.get("target_var") or consequent.get("target_ref")
                derived_source = self._resolve_subgraph_consequent_cell(consequent_source, bindings, [])
                if isinstance(consequent_source, str) and consequent_source and consequent_source not in bindings:
                    if consequent.get("source_var"):
                        bindings[str(consequent["source_var"])] = source
                    elif consequent.get("source_ref") and consequent["source_ref"] not in {self._cell_key(source), source.name}:
                        continue
                if isinstance(consequent_target, str) and consequent_target and consequent_target not in bindings:
                    if consequent.get("target_var"):
                        bindings[str(consequent["target_var"])] = target
                    elif consequent.get("target_ref") and consequent["target_ref"] not in {self._cell_key(target), target.name}:
                        continue
                proved = self._prove_subgraph_antecedents(
                    antecedents,
                    facts,
                    bindings,
                    depth,
                    max_depth,
                    visiting,
                    memo,
                )
                if not proved:
                    continue
                antecedent_path, final_bindings = proved
                resolved_source = self._resolve_subgraph_consequent_cell(consequent_source, final_bindings, [])
                resolved_target = self._resolve_subgraph_consequent_cell(consequent_target, final_bindings, [])
                if resolved_source is None or resolved_target is None:
                    continue
                if self._cell_key(resolved_source) != source_key or self._cell_key(resolved_target) != target_key:
                    continue
                antecedent_score = 1.0
                for step in antecedent_path:
                    antecedent_score *= max(step.score, 1e-9)
                derived_score = max(float(rule_score) * antecedent_score, 1e-6)
                derived_pattern = Cell(
                    name=f"backward_{rule.name}_{resolved_source.name}_to_{resolved_target.name}",
                    dim=1,
                    embedding=resolved_target.as_numpy() - resolved_source.as_numpy(),
                    source=resolved_source,
                    target=resolved_target,
                )
                candidate_path = [
                    PathStep(
                        source=resolved_source,
                        pattern=derived_pattern,
                        target=resolved_target,
                        score=derived_score,
                        raw_weight=derived_score,
                        agent_name="reasoning",
                        source_key=source_key,
                        target_key=target_key,
                        relation="subgraph_rule",
                    )
                ]
                candidate_score = float(rule_score)
                for step in antecedent_path:
                    candidate_score *= max(step.score, 1e-9)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_path = candidate_path
                continue
            if rule_type not in {"transitivity", "edge_derivation"} and "transitivity" not in getattr(rule, "name", "").lower():
                continue
            antecedents, consequent = self._default_rule_templates(rule)
            bindings = self._unify_goal_with_consequent(consequent, source, target)
            if bindings is None:
                continue
            if len(antecedents) != 2:
                continue

            first = antecedents[0]
            second = antecedents[1]
            first_source = bindings.get(self._var_name(first["source"])) if self._is_var(first["source"]) else None
            first_target = bindings.get(self._var_name(first["target"])) if self._is_var(first["target"]) else None
            candidate_first_edges = self._candidate_edges_for_goal(facts, source=first_source, target=first_target)
            for first_edge in candidate_first_edges[: self.beam_width]:
                local_bindings = dict(bindings)
                for spec, concrete in ((first["source"], first_edge.source), (first["target"], first_edge.target)):
                    if self._is_var(spec):
                        name = self._var_name(spec)
                        bound = local_bindings.get(name)
                        if bound is not None and self._cell_key(bound) != self._cell_key(concrete):
                            local_bindings = {}
                            break
                        local_bindings[name] = concrete
                    elif spec not in {self._cell_key(concrete), concrete.name}:
                        local_bindings = {}
                        break
                if not local_bindings:
                    continue

                second_source = local_bindings.get(self._var_name(second["source"])) if self._is_var(second["source"]) else None
                second_target = local_bindings.get(self._var_name(second["target"])) if self._is_var(second["target"]) else None
                if second_source is not None and second_target is not None:
                    first_path = self._prove_edge_backward(
                        first_edge.source,
                        first_edge.target,
                        facts,
                        depth + 1,
                        max_depth,
                        visiting,
                        memo,
                    )
                    if not first_path:
                        continue
                    second_path = self._prove_edge_backward(
                        second_source,
                        second_target,
                        facts,
                        depth + 1,
                        max_depth,
                        visiting,
                        memo,
                    )
                    if not second_path:
                        continue
                    candidate_path = list(first_path) + list(second_path)
                    candidate_score = float(rule_score)
                    for step in candidate_path:
                        candidate_score *= max(step.score, 1e-9)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_path = candidate_path
                    continue
                candidate_second_edges = self._candidate_edges_for_goal(facts, source=second_source, target=second_target)
                for second_edge in candidate_second_edges[: self.beam_width]:
                    final_bindings = dict(local_bindings)
                    valid = True
                    for spec, concrete in ((second["source"], second_edge.source), (second["target"], second_edge.target)):
                        if self._is_var(spec):
                            name = self._var_name(spec)
                            bound = final_bindings.get(name)
                            if bound is not None and self._cell_key(bound) != self._cell_key(concrete):
                                valid = False
                                break
                            final_bindings[name] = concrete
                        elif spec not in {self._cell_key(concrete), concrete.name}:
                            valid = False
                            break
                    if not valid:
                        continue

                    first_path = self._prove_edge_backward(
                        first_edge.source,
                        first_edge.target,
                        facts,
                        depth + 1,
                        max_depth,
                        visiting,
                        memo,
                    )
                    if not first_path:
                        continue
                    second_path = self._prove_edge_backward(
                        second_edge.source,
                        second_edge.target,
                        facts,
                        depth + 1,
                        max_depth,
                        visiting,
                        memo,
                    )
                    if not second_path:
                        continue
                    candidate_path = list(first_path) + list(second_path)
                    candidate_score = float(rule_score)
                    for step in candidate_path:
                        candidate_score *= max(step.score, 1e-9)
                    if candidate_score > best_score:
                        best_score = candidate_score
                        best_path = candidate_path

        visiting.remove(goal_key)
        memo[memo_key] = best_path
        return best_path

    def _backward_chain_path(self, start: Cell, goal: Cell, max_depth: Optional[int] = None) -> Optional[List[PathStep]]:
        self._ensure_fresh()
        facts: Dict[str, List[EdgeRecord]] = {
            source_key: list(records)
            for source_key, records in self._edge_index.items()
        }
        for source_key, records in self._transient_rule_edge_index.items():
            facts.setdefault(source_key, []).extend(records)
        for records in facts.values():
            records.sort(key=lambda item: item.score, reverse=True)
        return self._prove_edge_backward(
            start,
            goal,
            facts,
            depth=0,
            max_depth=max_depth or self.max_depth,
            visiting=set(),
            memo={},
        )

    def _cell_ref(self, cell: Optional[Cell]) -> Optional[Dict[str, str]]:
        if cell is None:
            return None
        return {"key": self._cell_key(cell), "label": self._cell_display(cell), "name": cell.name}

    def _top_analogies(self, source_pattern: Cell, top_k: int = 3) -> List[Tuple[float, Cell]]:
        if source_pattern is None:
            return []
        self._ensure_fresh()
        cached = self._analogy_cache.get(source_pattern.name)
        if cached is not None:
            return cached[:top_k]

        scored: List[Tuple[float, Cell]] = []
        for pattern in self._all_patterns:
            if pattern.dim != 1 or pattern.name == source_pattern.name:
                continue
            try:
                score = source_pattern.similarity(pattern)
            except Exception:
                continue
            scored.append((score, pattern))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]

    def _explicit_analogies(self, source_pattern: Optional[Cell], top_k: int = 5) -> List[Tuple[float, Cell]]:
        if source_pattern is None:
            return []
        self._ensure_fresh()
        return self._explicit_analogy_index.get(source_pattern.name, [])[:top_k]

    def _explicit_rules(self, source_pattern: Optional[Cell], top_k: int = 5) -> List[Tuple[float, Cell]]:
        if source_pattern is None:
            return []
        self._ensure_fresh()
        return self._explicit_rule_index.get(source_pattern.name, [])[:top_k]

    @staticmethod
    def _render_path(path: Sequence[PathStep]) -> str:
        if not path:
            return "No path found."
        agent = None
        nodes = [path[0].source.name]
        for step in path:
            agent = step
            nodes.append(step.target.name)
        return " -> ".join(nodes)

    def _render_path_labels(self, path: Sequence[PathStep]) -> str:
        if not path:
            return "No path found."
        nodes = [self._cell_display(path[0].source)]
        for step in path:
            nodes.append(self._cell_display(step.target))
        return " -> ".join(nodes)

    def _path_answer(self, question: str, start: Cell, goal: Optional[Cell], method: str = "auto") -> str:
        start_key = self._cell_key(start)
        if goal is None:
            outgoing = self._edge_index.get(start_key, [])
            if not outgoing:
                return f"I found {self._cell_display(start)}, but no strong outgoing transitions to reason from yet."
            top = outgoing[: self.beam_width]
            items = ", ".join(
                f"{self._cell_display(edge.target)} [{edge.agent_name}, score={edge.score:.2f}]"
                for edge in top
            )
            return f"From {self._cell_display(start)}, the strongest learned transitions are: {items}."

        path = self._select_path(start, goal, method=method)
        if not path:
            analogies = self._top_analogies(
                self._edge_index.get(start_key, [])[0].pattern
                if self._edge_index.get(start_key)
                else next((p for p in self._all_patterns if p.dim == 1), None)
                if self._all_patterns
                else None,
                top_k=3,
            )
            if analogies:
                similar = ", ".join(f"{pattern.name} (sim={score:.2f})" for score, pattern in analogies)
                return (
                    f"I could not find a direct path from {self._cell_display(start)} to {self._cell_display(goal)}, "
                    f"but similar learned transitions are: {similar}."
                )
            return f"I could not find a learned path from {self._cell_display(start)} to {self._cell_display(goal)}."

        rendered = self._render_path_labels(path)
        path_score = 1.0
        for step in path:
            path_score *= max(step.score, 1e-9)
        evidence = "; ".join(
            f"{self._cell_display(step.source)} -[{step.relation}/{step.agent_name}:{step.score:.2f}]-> {self._cell_display(step.target)}"
            for step in path
        )

        return (
            f"Because the reasoning graph connects {self._cell_display(start)} to {self._cell_display(goal)} via {rendered}, "
            f"the combined calibrated score is {path_score:.4f}. Evidence: {evidence}."
        )

    def _explanation_answer(self, terms: Sequence[str]) -> str:
        effect = self._pick_explanation_endpoint(terms)
        if effect is None:
            start, goal = self._pick_path_endpoints_with_fallback(" ".join(terms), terms)
            if start is None:
                return "I could not match enough learned concepts in that question."
            if goal is None or goal.name == start.name:
                return self._path_answer("", start, None)
            return self._path_answer("", start, goal)

        best_edge = self._best_causal_edge_for_effect(effect)
        if best_edge is None:
            return f"I found {self._cell_display(effect)}, but no causal evidence chain for it yet."

        rule = best_edge.pattern
        intervention = getattr(rule, "intervention", self._cell_display(best_edge.source))
        agent_impacted = getattr(rule, "agent_impacted", best_edge.target.name)
        effect_mag = getattr(rule, "effect_magnitude", best_edge.raw_weight)
        original_word = getattr(rule, "metadata", {}).get("original_word", self._cell_display(best_edge.source))
        goal = self._pick_explanation_goal(terms, effect)
        downstream = ""
        if goal is not None:
            downstream_path = self._beam_search_path(effect, goal)
            if downstream_path:
                rendered = self._render_path_labels(downstream_path)
                downstream_score = 1.0
                for step in downstream_path:
                    downstream_score *= max(step.score, 1e-9)
                downstream = (
                    f" Downstream chain: {rendered} "
                    f"(score={downstream_score:.4f})."
                )
        return (
            f"The strongest causal explanation is that changing {original_word} triggers {self._cell_display(effect)}. "
            f"Evidence: {intervention}; impacted agent={agent_impacted}; causal score={best_edge.score:.2f}; effect={effect_mag:.4f}."
            f"{downstream}"
        )

    def _analogy_answer(self, question: str, terms: Sequence[str]) -> str:
        source, target = self._pick_analogy_endpoints(terms)
        if source is None:
            return "I do not have enough learned structure to compare those relations yet."

        source_pattern = None
        source_key = self._cell_key(source)
        target_key = self._cell_key(target) if target is not None else None
        if target_key is not None:
            for edge in self._edge_index.get(source_key, []):
                if edge.target_key == target_key:
                    source_pattern = edge.pattern
                    break
        if source_pattern is None:
            outgoing = self._edge_index.get(source_key, [])
            if outgoing:
                source_pattern = outgoing[0].pattern
        if source_pattern is None:
            source_pattern = next((p for p in self._all_patterns if p.dim == 1), None)
        if source_pattern is None:
            return f"I found {self._cell_display(source)}, but no strong analogical matches yet."

        explicit_analogies = self._explicit_analogies(source_pattern, top_k=5)
        if explicit_analogies:
            top_explicit_pattern = explicit_analogies[0][1]
            explicit_rules = self._explicit_rules(top_explicit_pattern, top_k=5)
            if explicit_rules:
                items = ", ".join(
                    f"{entry.target.name} via {entry.name} (score={score:.2f})"
                    for score, entry in explicit_rules
                )
                return (
                    f"The strongest compositional learned rules from {self._cell_display(source)} are: {items}. "
                    f"They refine the base analogy {top_explicit_pattern.name}."
                )
        if explicit_analogies:
            items = ", ".join(
                f"{entry.target.name} via {entry.name} (score={score:.2f})"
                for score, entry in explicit_analogies
            )
            return f"The strongest explicit learned analogies from {self._cell_display(source)} are: {items}."

        analogies = self._top_analogies(
            source_pattern,
            top_k=5,
        )
        if not analogies:
            return f"I found {self._cell_display(source)}, but no strong analogical matches yet."

        items = ", ".join(f"{pattern.name} (sim={score:.2f})" for score, pattern in analogies)
        return f"The closest learned analogies to {self._cell_display(source)} are: {items}."

    @staticmethod
    def _noisy_or_score(path_scores: List[float]) -> float:
        """Aggregate independent path scores via noisy-OR: 1 - product(1 - s_i)."""
        result = 1.0
        for s in path_scores:
            result *= 1.0 - max(s, 0.0)
        return 1.0 - result

    def _beam_search_all_paths(self, start: Cell, goal: Cell) -> List[List[PathStep]]:
        """Like _beam_search_path but collects every path that reaches goal."""
        self._ensure_fresh()

        start_key = self._cell_key(start)
        goal_key = self._cell_key(goal)
        forward_edges = self._forward_chain_edges(start_key)
        beams: List[Tuple[float, List[PathStep], str]] = [(0.0, [], start_key)]
        found: List[List[PathStep]] = []
        best_seen_cost: Dict[str, float] = {start_key: 0.0}

        for _depth in range(self.max_depth):
            next_beams: List[Tuple[float, List[PathStep], str]] = []
            for cost, path, node_name in beams:
                if node_name == goal_key:
                    found.append(path)
                    continue

                combined_edges = (
                    list(self._edge_index.get(node_name, []))
                    + list(self._transient_rule_edge_index.get(node_name, []))
                    + list(forward_edges.get(node_name, []))
                )
                combined_edges.sort(key=lambda item: item.score, reverse=True)
                for edge in combined_edges[: self.beam_width]:
                    if any(step.target_key == edge.target_key for step in path):
                        continue
                    edge_cost = -math.log(max(edge.score, 1e-9))
                    rel_cell = self._relation_cell_index.get(edge.relation)
                    if rel_cell is not None:
                        try:
                            import numpy as np
                            rel_emb = rel_cell.as_numpy()
                            src_emb = edge.source.as_numpy()
                            tgt_emb = edge.target.as_numpy()
                            predicted = src_emb + rel_emb
                            np_p = np.linalg.norm(predicted)
                            nt = np.linalg.norm(tgt_emb)
                            if np_p > 1e-9 and nt > 1e-9:
                                coherence = float(np.dot(predicted, tgt_emb) / (np_p * nt))
                                bonus = 0.5 + 0.5 * max(coherence, 0.0)
                                edge_cost /= bonus
                        except Exception:
                            pass
                    elif self._relation_registry is not None:
                        try:
                            coherence = self._relation_registry.coherence_score(
                                edge.source.as_numpy(), edge.relation, edge.target.as_numpy()
                            )
                            bonus = 0.5 + 0.5 * max(coherence, 0.0)
                            edge_cost /= bonus
                        except Exception:
                            pass
                    new_cost = cost + edge_cost
                    previous_best = best_seen_cost.get(edge.target_key)
                    # Allow re-visiting goal via different routes
                    if edge.target_key != goal_key:
                        if previous_best is not None and new_cost >= previous_best:
                            continue
                        best_seen_cost[edge.target_key] = new_cost
                    step = PathStep(
                        source=edge.source,
                        pattern=edge.pattern,
                        target=edge.target,
                        score=edge.score,
                        raw_weight=edge.raw_weight,
                        agent_name=edge.agent_name,
                        source_key=edge.source_key,
                        target_key=edge.target_key,
                        relation=edge.relation,
                    )
                    next_beams.append((new_cost, path + [step], edge.target_key))

            if not next_beams:
                break
            next_beams.sort(key=lambda item: item[0])
            beams = next_beams[: self.beam_width]
            for cost, path, node_name in beams:
                if node_name == goal_key:
                    found.append(path)

        # Deduplicate by path signature
        seen: set = set()
        unique: List[List[PathStep]] = []
        for path in found:
            sig = tuple((s.source_key, s.target_key, s.agent_name) for s in path)
            if sig not in seen:
                seen.add(sig)
                unique.append(path)
        return unique

    def _predict_missing_edge(
        self, source: "Cell", relation_name: str, top_k: int = 5
    ) -> "List[Tuple[float, Cell]]":
        """Use TransE prediction to find likely targets for (source, relation)."""
        import numpy as np
        self._ensure_fresh()
        try:
            src_vec = source.as_numpy()
            rel_cell = self._relation_cell_index.get(relation_name)
            if rel_cell is not None:
                predicted = src_vec + np.asarray(rel_cell.as_numpy(), dtype=np.float32)
            elif self._relation_registry is not None:
                predicted = self._relation_registry.predict_target(src_vec, relation_name)
            else:
                return []
        except Exception:
            return []
        scored = []
        for key, cell in self._node_index.items():
            try:
                cell_vec = cell.as_numpy()
                norm_p = np.linalg.norm(predicted)
                norm_c = np.linalg.norm(cell_vec)
                if norm_p < 1e-9 or norm_c < 1e-9:
                    continue
                sim = float(np.dot(predicted, cell_vec) / (norm_p * norm_c))
                scored.append((sim, cell))
            except Exception:
                continue
        scored.sort(reverse=True)
        return scored[:top_k]

    def _select_path(self, start: Cell, goal: Cell, method: str = "auto") -> Optional[List[PathStep]]:
        if method == "backward":
            return self._backward_chain_path(start, goal)
        if method == "beam":
            return self._beam_search_path(start, goal)
        path = self._beam_search_path(start, goal)
        if path:
            return path
        return self._backward_chain_path(start, goal)

    def reason_with_trace(self, question: str, method: str = "auto") -> Dict[str, Any]:
        if not question.strip():
            return {
                "question": question,
                "intent": "invalid",
                "mode": "invalid",
                "method": method,
                "terms": [],
                "anchors": {},
                "candidate_paths": [],
                "chosen_path": None,
                "evidence": [],
                "answer": "Ask a non-empty question.",
            }

        self._ensure_fresh()
        parsed = self._parse_question(question)
        terms = parsed.get("terms", [])
        trace: Dict[str, Any] = {
            "question": question,
            "intent": parsed["type"],
            "mode": parsed.get("mode", "default"),
            "method": method,
            "terms": list(terms),
            "anchors": {},
            "candidate_paths": [],
            "chosen_path": None,
            "evidence": [],
            "answer": "",
        }

        if parsed["type"] == "analogy":
            source, target = self._pick_analogy_endpoints(terms)
            trace["anchors"] = {"source": self._cell_ref(source), "target": self._cell_ref(target)}
            if source is None:
                trace["answer"] = "I do not have enough learned structure to compare those relations yet."
                return trace
            source_pattern = None
            source_key = self._cell_key(source)
            target_key = self._cell_key(target) if target is not None else None
            if target_key is not None:
                for edge in self._edge_index.get(source_key, []):
                    if edge.target_key == target_key:
                        source_pattern = edge.pattern
                        break
            if source_pattern is None:
                outgoing = self._edge_index.get(source_key, [])
                if outgoing:
                    source_pattern = outgoing[0].pattern
            if source_pattern is None:
                source_pattern = next((p for p in self._all_patterns if p.dim == 1), None)
            explicit_analogies = self._explicit_analogies(source_pattern, top_k=5)
            if explicit_analogies:
                top_explicit_pattern = explicit_analogies[0][1]
                explicit_rules = self._explicit_rules(top_explicit_pattern, top_k=5)
                if explicit_rules:
                    trace["evidence"] = [
                        {
                            "pattern": entry.name,
                            "score": score,
                            "target_pattern": getattr(entry.target, "name", ""),
                            "source_pattern": getattr(entry.source, "name", ""),
                            "kind": "dim3_rule",
                        }
                        for score, entry in explicit_rules
                    ]
                    items = ", ".join(
                        f"{entry.target.name} via {entry.name} (score={score:.2f})"
                        for score, entry in explicit_rules
                    )
                    trace["answer"] = (
                        f"The strongest compositional learned rules from {self._cell_display(source)} are: {items}. "
                        f"They refine the base analogy {top_explicit_pattern.name}."
                    )
                    return trace
                trace["evidence"] = [
                    {
                        "pattern": entry.name,
                        "score": score,
                        "target_pattern": getattr(entry.target, "name", ""),
                        "kind": "dim2_analogy",
                    }
                    for score, entry in explicit_analogies
                ]
                items = ", ".join(
                    f"{entry.target.name} via {entry.name} (score={score:.2f})"
                    for score, entry in explicit_analogies
                )
                trace["answer"] = f"The strongest explicit learned analogies from {self._cell_display(source)} are: {items}."
                return trace
            analogies = self._top_analogies(source_pattern, top_k=5)
            trace["evidence"] = [{"pattern": pattern.name, "score": score} for score, pattern in analogies]
            if not analogies:
                trace["answer"] = f"I found {self._cell_display(source)}, but no strong analogical matches yet."
                return trace
            items = ", ".join(f"{pattern.name} (sim={score:.2f})" for score, pattern in analogies)
            trace["answer"] = f"The closest learned analogies to {self._cell_display(source)} are: {items}."
            return trace

        if parsed.get("mode") == "explanation":
            effect = self._pick_explanation_endpoint(terms)
            trace["anchors"]["effect"] = self._cell_ref(effect)
            if effect is None:
                start, goal = self._pick_path_endpoints_with_fallback(question, terms)
                trace["anchors"]["start"] = self._cell_ref(start)
                trace["anchors"]["goal"] = self._cell_ref(goal)
                if start is None:
                    trace["answer"] = "I could not match enough learned concepts in that question."
                    return trace
                if goal is None or goal.name == start.name:
                    trace["answer"] = self._path_answer(question, start, None, method=method)
                    return trace
                path = self._select_path(start, goal, method=method)
                if path:
                    path_trace = self._path_to_trace(path)
                    trace["candidate_paths"].append(path_trace)
                    trace["chosen_path"] = path_trace
                    trace["evidence"] = list(path_trace["steps"])
                trace["answer"] = self._path_answer(question, start, goal, method=method)
                return trace

            best_edge = self._best_causal_edge_for_effect(effect)
            if best_edge is None:
                trace["answer"] = f"I found {self._cell_display(effect)}, but no causal evidence chain for it yet."
                return trace

            cause_step = PathStep(
                source=best_edge.source,
                pattern=best_edge.pattern,
                target=best_edge.target,
                score=best_edge.score,
                raw_weight=best_edge.raw_weight,
                agent_name=best_edge.agent_name,
                source_key=best_edge.source_key,
                target_key=best_edge.target_key,
                relation=best_edge.relation,
            )
            cause_trace = self._path_to_trace([cause_step])
            trace["candidate_paths"].append(cause_trace)
            goal = self._pick_explanation_goal(terms, effect)
            trace["anchors"]["goal"] = self._cell_ref(goal)
            downstream_path = self._select_path(effect, goal, method=method) if goal is not None else None
            if downstream_path:
                downstream_trace = self._path_to_trace(downstream_path)
                trace["candidate_paths"].append(downstream_trace)
                trace["chosen_path"] = self._path_to_trace([cause_step] + list(downstream_path))
            else:
                trace["chosen_path"] = cause_trace
            rule = best_edge.pattern
            trace["evidence"] = [{
                "intervention": getattr(rule, "intervention", self._cell_display(best_edge.source)),
                "agent_impacted": getattr(rule, "agent_impacted", best_edge.target.name),
                "effect_magnitude": getattr(rule, "effect_magnitude", best_edge.raw_weight),
                "original_word": getattr(rule, "metadata", {}).get("original_word", self._cell_display(best_edge.source)),
            }]
            trace["answer"] = self._explanation_answer(terms)
            return trace

        start, goal = self._pick_path_endpoints_with_fallback(question, terms)
        trace["anchors"] = {"start": self._cell_ref(start), "goal": self._cell_ref(goal)}
        if start is None:
            trace["answer"] = "I could not match enough learned concepts in that question."
            return trace
        if goal is None or goal.name == start.name:
            outgoing = self._edge_index.get(self._cell_key(start), [])
            trace["evidence"] = [
                {"target": self._cell_display(edge.target), "agent": edge.agent_name, "relation": edge.relation, "score": edge.score}
                for edge in outgoing[: self.beam_width]
            ]
            trace["answer"] = self._path_answer(question, start, None, method=method)
            return trace

        if method == "backward":
            single = self._backward_chain_path(start, goal)
            all_paths = [single] if single else []
        else:
            all_paths = self._beam_search_all_paths(start, goal)
            if not all_paths and method != "beam":
                backward = self._backward_chain_path(start, goal)
                if backward:
                    all_paths = [backward]

        if all_paths:
            for p in all_paths:
                trace["candidate_paths"].append(self._path_to_trace(p))

            if len(all_paths) == 1:
                best = all_paths[0]
                chosen_trace = self._path_to_trace(best)
            else:
                path_scores = [self._path_to_trace(p)["combined_score"] for p in all_paths]
                synthesized = self._noisy_or_score(path_scores)
                best = max(all_paths, key=lambda p: self._path_to_trace(p)["combined_score"])
                chosen_trace = dict(self._path_to_trace(best))
                chosen_trace["combined_score"] = synthesized

            trace["chosen_path"] = chosen_trace
            trace["evidence"] = list(self._path_to_trace(best)["steps"])

        trace["answer"] = self._path_answer(question, start, goal, method=method)
        return trace

    def reason(self, question: str, method: str = "auto") -> str:
        return str(self.reason_with_trace(question, method=method).get("answer", "I could not reason about that question."))
