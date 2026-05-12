from __future__ import annotations
import json
import os
from typing import Any, Dict, List
import numpy as np
from hpm_ai_v6.hpm_model.core.cell import Cell

_EMB_DIM = 16

def _word_embedding(word: str) -> np.ndarray:
    emb = np.zeros(_EMB_DIM, dtype=float)
    for i, ch in enumerate(word.lower()[:_EMB_DIM]):
        emb[i % _EMB_DIM] = ord(ch) / 128.0
    return emb


class DependencyRelationAgent:
    """
    Extracts SVO + prepositional dependency triples from sentences using spaCy
    and emits dim-1 Cell edges (subject_of, object_of, prep_X) between word cells.

    Interface matches StubAgent: .patterns, .get_weights(), ._paging_lookup()
    """

    _SUBJECT_DEPS = {"nsubj", "nsubjpass"}
    _OBJECT_DEPS = {"dobj", "pobj", "attr"}

    def __init__(self, nlp: Any = None, base_score: float = 0.85, prep_score: float = 0.75):
        self._nlp = nlp
        self.base_score = base_score
        self.prep_score = prep_score
        self.patterns: List[Cell] = []
        self._weights: List[float] = []
        self._word_cells: Dict[str, Cell] = {}

    def get_weights(self) -> List[float]:
        return list(self._weights)

    def _paging_lookup(self) -> Dict[str, Cell]:
        return {cell.name: cell for cell in self._word_cells.values()}

    def _get_or_create_word_cell(self, word: str) -> Cell:
        key = word.lower()
        if key not in self._word_cells:
            self._word_cells[key] = Cell(
                name=f"word_{key}",
                dim=0,
                embedding=_word_embedding(key).tolist(),
            )
        return self._word_cells[key]

    def _make_edge(self, source: Cell, target: Cell, relation: str, score: float) -> Cell:
        emb = (target.as_numpy() - source.as_numpy()).tolist()
        return Cell(
            name=f"dep_{relation}_{source.name}_{target.name}",
            dim=1,
            embedding=emb,
            source=source,
            target=target,
            weight=score,
        )

    def learn_from_corpus(self, sentences: List[str]) -> None:
        self.patterns = []
        self._weights = []
        if not sentences:
            return
        nlp = self._get_nlp()
        for doc in nlp.pipe(sentences, batch_size=64):
            for token in doc:
                dep = token.dep_.lower() if hasattr(token, 'dep_') else ""
                head_text = token.head.text if hasattr(token, 'head') else ""

                if dep in self._SUBJECT_DEPS:
                    subj_cell = self._get_or_create_word_cell(token.lower_)
                    verb_cell = self._get_or_create_word_cell(head_text.lower())
                    edge = self._make_edge(subj_cell, verb_cell, "subject_of", self.base_score)
                    self.patterns.append(edge)
                    self._weights.append(self.base_score)

                elif dep in self._OBJECT_DEPS:
                    verb_cell = self._get_or_create_word_cell(head_text.lower())
                    obj_cell = self._get_or_create_word_cell(token.lower_)
                    
                    # Distinguish between direct object and prepositional object
                    # if it's pobj, it usually attaches to a preposition, not the verb directly
                    # but _OBJECT_DEPS includes pobj. 
                    # Actually, for pobj we usually want the verb -> obj edge with prep_ relation.
                    # Let's check the design spec.
                    # Spec says: pobj + prep -> prep_{preposition} (verb_word -> pobj_word)
                    
                    if dep == "pobj":
                        prep_text = head_text.lower()
                        verb_node = getattr(token.head, 'head', None)
                        if verb_node is not None:
                            verb_cell = self._get_or_create_word_cell(verb_node.text.lower())
                            obj_cell = self._get_or_create_word_cell(token.lower_)
                            relation = f"prep_{prep_text}"
                            edge = self._make_edge(verb_cell, obj_cell, relation, self.prep_score)
                            self.patterns.append(edge)
                            self._weights.append(self.prep_score)
                    else:
                        # dobj or attr
                        edge = self._make_edge(verb_cell, obj_cell, "object_of", self.base_score)
                        self.patterns.append(edge)
                        self._weights.append(self.base_score)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        triples = []
        for p, w in zip(self.patterns, self._weights):
            triples.append({
                "source": p.source.name if p.source else "",
                "target": p.target.name if p.target else "",
                "name": p.name,
                "weight": w,
            })
        with open(path, "w") as f:
            json.dump({"triples": triples}, f)

    def load(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.patterns = []
        self._weights = []
        for t in data.get("triples", []):
            # The source/target names are "word_{word}"
            src_name = t["source"].removeprefix("word_")
            tgt_name = t["target"].removeprefix("word_")
            src_cell = self._get_or_create_word_cell(src_name)
            tgt_cell = self._get_or_create_word_cell(tgt_name)
            w = float(t.get("weight", self.base_score))
            emb = (tgt_cell.as_numpy() - src_cell.as_numpy()).tolist()
            edge = Cell(name=t["name"], dim=1, embedding=emb, source=src_cell, target=tgt_cell, weight=w)
            self.patterns.append(edge)
            self._weights.append(w)

    def _get_nlp(self) -> Any:
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("Run: python -m spacy download en_core_web_sm")
        return self._nlp
