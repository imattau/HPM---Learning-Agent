"""ReaderAgent: observes text passages and retrieves relevant ones for queries."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.syntax import SyntaxMixin
from hpm_ai_v2.agents.mixins.srl import SemanticRoleMixin
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.text_renderer import TextRenderer
from hpm_ai_v2.utils.oracle.text_oracle import TextOracle
from hpm_ai_v2.utils.text_fetcher import fetch_passages


class ReaderAgent(BaseHFNAgent, SyntaxMixin, SemanticRoleMixin):
    """
    HFN-native agent that reads text/webpages and retrieves relevant passages.
    Extended with structural hierarchy (L2-L5), recursive summarization, 
    predictive curiosity, structural analogy, syntax, and semantics (SP-Reader 7).
    """

    def __init__(self, config: TextDomainConfig, **kwargs) -> None:
        renderer = TextRenderer(config)
        super().__init__(config, renderer=renderer, **kwargs)
        self.oracle = TextOracle(config)
        self.counting_oracle.wrapped = self.oracle
        self._documents: List[List[int]] = []
        self._last_topic_mu: Optional[np.ndarray] = None
        self._analogy_map: Dict[str, str] = {} # target_node_id -> source_node_id
        
        # Override retriever slice to point to Concept/Text manifold
        from hfn.retriever import GoalConditionedRetriever
        curr = self.retriever
        while hasattr(curr, "base_retriever"): curr = curr.base_retriever
        if isinstance(curr, GoalConditionedRetriever):
            curr.target_slice = slice(self.config.S_DIM, self.config.S_DIM + self.config.DIM)
            curr.target_weight = 100.0

    def reindex_knowledge_base(self) -> None:
        """Atomic reindexing of all forest nodes to match new vocabulary dimension."""
        new_dim = self.config.m_dim
        s_dim = self.config.S_DIM
        self.dim = self.config.DIM
        self.m_dim = self.config.m_dim
        self.forest._D = new_dim
        
        # 1. Update Mu Index and Hot Nodes
        if hasattr(self.forest, "_mu_index"):
            idx = self.forest._mu_index
            for nid in list(idx.keys()):
                old_mu = idx[nid]
                if old_mu.shape[0] != new_dim:
                    new_mu = np.zeros(new_dim)
                    # Copy State part
                    new_mu[:s_dim] = old_mu[:s_dim]
                    # Copy Concept part (at middle)
                    old_v_dim = max(0, old_mu.shape[0] - 2*s_dim)
                    copy_len = min(old_v_dim, new_dim - 2*s_dim)
                    new_mu[s_dim : s_dim + copy_len] = old_mu[s_dim : s_dim + copy_len]
                    # Copy Delta part (at end)
                    new_mu[-s_dim:] = old_mu[-s_dim:]
                    
                    idx[nid] = new_mu
                    node = self.forest.get(nid)
                    if node:
                        node.mu = new_mu
                        node.sigma = np.ones(new_dim) * 0.2
        
        # 2. Rebuild Forest Hierarchy Cache (Crucial for TieredForest)
        if hasattr(self.forest, "rebuild_hierarchy_cache"):
            self.forest.rebuild_hierarchy_cache()
        
        # 3. Sync Passage Vectors in Config
        self.config._passage_vecs = [self.config.encode_passage(p) for p in self.config._passages]
        for i, p_mu in enumerate(self.config._passage_vecs):
            node_id = f"passage_{i}"
            if hasattr(self.forest, "_mu_index"): self.forest._mu_index[node_id] = p_mu
            if node_id in self.patterns: self.patterns[node_id].mu = p_mu
            
        # 4. Update Retriever Target Slice
        from hfn.retriever import GoalConditionedRetriever
        curr = self.retriever
        while hasattr(curr, "base_retriever"): curr = curr.base_retriever
        if isinstance(curr, GoalConditionedRetriever):
            curr.target_slice = slice(s_dim, s_dim + self.config.DIM)

    def expand_vocabulary(self, texts: List[str], max_new: int = 10) -> int:
        added = self.config.expand_vocab(texts, max_new=max_new)
        if added > 0: self.reindex_knowledge_base()
        return added

    def observe_passage(self, text: str) -> int:
        idx = self.config.register_passage(text)
        mu = self.config.encode_passage(text)
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=f"passage_{idx}", use_diag=True)
        node.metadata = {"passage_idx": idx, "text": text, "type": "passage"}
        self.observer.register(node, protected=False, initial_weight=1.0)
        self.patterns[f"passage_{idx}"] = node
        
        # Update last topic mu for predictive curiosity
        topics = [n for k,n in self.patterns.items() if k.startswith("topic_")]
        if topics:
            best_t = min(topics, key=lambda t: np.linalg.norm(t.mu - mu))
            self._last_topic_mu = best_t.mu.copy()
        return idx

    def observe_document(self, text: str, min_length: int = 40) -> List[int]:
        passages = fetch_passages(text=text, min_length=min_length)
        indices = [self.observe_passage(p) for p in passages]
        self._documents.append(indices)
        return indices

    def query(self, question: str, top_k: int = 1, node_type: Optional[str] = "passage") -> Optional[str]:
        if not self.config._passages: return None
        query_mu = self.config.encode_passage(question)
        query_node = HFN(mu=query_mu, sigma=np.ones(self.m_dim), id="__query__", use_diag=True)
        
        # We might want to over-fetch if we're filtering
        pool_size = top_k * 10 if node_type else top_k
        candidates = self.retriever.retrieve(query_node, k=pool_size)
        
        if node_type:
            candidates = [n for n in candidates if getattr(n, "metadata", {}).get("type") == node_type]
        
        if not candidates: return None
        return self.renderer.render(candidates[0])

    def query_scored(self, question: str, k: int = 10) -> List[tuple]:
        if not self.config._passage_vecs: return []
        q_mu = self.config.encode_passage(question)
        s_dim, dim = self.config.S_DIM, self.config.DIM
        q_v = q_mu[s_dim : s_dim + dim]
        res = []
        for i, pvec in enumerate(self.config._passage_vecs):
            pv = pvec[s_dim : s_dim + dim]
            denom = (np.linalg.norm(q_v)*np.linalg.norm(pv) + 1e-9)
            score = float(np.dot(q_v, pv) / denom)
            res.append((self.config.get_passage(i), score))
        return sorted(res, key=lambda x: x[1], reverse=True)[:k]

    def save_agent(self, directory: str) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        super().save_state(str(path / "agent_state.pkl"))
        meta = {
            "concepts": self.config.concepts, 
            "idf": self.config.idf, 
            "passages": self.config._passages, 
            "docs": self._documents,
            "pos_rules": getattr(self, "pos_rules", {}),
            "role_knowledge": getattr(self, "role_knowledge", [])
        }
        with open(path / "reader_meta.json", "w") as f: json.dump(meta, f)

    @classmethod
    def load_agent(cls, directory: str) -> "ReaderAgent":
        path = Path(directory)
        with open(path / "reader_meta.json") as f: meta = json.load(f)
        config = TextDomainConfig(meta["concepts"], meta["idf"])
        config._passages = meta["passages"]
        config._passage_vecs = [config.encode_passage(p) for p in meta["passages"]]
        agent = cls(config, cold_dir=str(path))
        agent._documents = meta.get("docs", [])
        agent.pos_rules = meta.get("pos_rules", {})
        agent.role_knowledge = meta.get("role_knowledge", [])
        agent.load_state(str(path / "agent_state.pkl"))
        # FIX: Explicitly reindex all nodes (hot and cold) after load
        agent.reindex_knowledge_base()
        return agent

    def curiosity_score(self, text: str) -> float:
        if not self.config._passage_vecs: return 1.0
        s_dim, dim = self.config.S_DIM, self.config.DIM
        q_v = self.config.encode_passage(text)[s_dim : s_dim + dim]
        sims = [float(np.dot(q_v, p[s_dim : s_dim + dim]) / (np.linalg.norm(q_v)*np.linalg.norm(p[s_dim : s_dim + dim]) + 1e-9)) for p in self.config._passage_vecs]
        return 1.0 - max(sims)

    def predictive_curiosity_score(self, text: str) -> float:
        """Measure curiosity as prediction error against L4 narrative models."""
        if self._last_topic_mu is None: return self.curiosity_score(text)
        current_mu = self.config.encode_passage(text)
        prediction = self.predict_next_topic(self._last_topic_mu)
        dist = float(np.linalg.norm(current_mu - prediction))
        return dist

    def observe_if_curious(self, text: str, threshold: float = 0.3, use_predictive: bool = True) -> bool:
        score = self.predictive_curiosity_score(text) if use_predictive else self.curiosity_score(text)
        if score >= threshold:
            self.observe_passage(text)
            return True
        return False

    def build_topic_clusters(self, n_clusters: int = 5) -> None:
        """K-means clustering with deep structural wiring (L3 -> L2)."""
        if not self.config._passage_vecs: return
        s_dim, dim = self.config.S_DIM, self.config.DIM
        vecs = np.array([v[s_dim : s_dim + dim] for v in self.config._passage_vecs])
        k = min(n_clusters, len(vecs))
        # K-means++ style initialization
        centroids = [vecs[np.random.choice(len(vecs))]]
        for _ in range(1, k):
            dists = np.array([min([np.linalg.norm(v-c)**2 for c in centroids]) for v in vecs])
            d_sum = dists.sum()
            if d_sum == 0:
                # Fallback to random if all remaining points are identical to centroids
                remaining = [i for i in range(len(vecs)) if not any(np.allclose(vecs[i], c) for c in centroids)]
                if not remaining: remaining = list(range(len(vecs)))
                centroids.append(vecs[np.random.choice(remaining)])
            else:
                probs = dists / d_sum
                centroids.append(vecs[np.random.choice(len(vecs), p=probs)])
        centroids = np.array(centroids)
        for _ in range(10):
            dists = np.linalg.norm(vecs[:,None]-centroids[None], axis=2)
            labels = dists.argmin(axis=1)
            centroids = np.array([vecs[labels==i].mean(axis=0) if (labels==i).any() else centroids[i] for i in range(k)])
        
        for i, c in enumerate(centroids):
            mu = np.zeros(self.m_dim); mu[s_dim : s_dim + dim] = c
            node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.2, id=f"topic_{i}", use_diag=True)
            node.metadata = {"cluster_id": i, "type": "topic"}; node.relation_type = "topic"
            indices = np.where(labels == i)[0]
            for idx in indices:
                p_id = f"passage_{idx}"
                if p_id in self.patterns: node.add_child(self.patterns[p_id])
            self.observer.register(node, protected=True, initial_weight=2.0)
            self.patterns[f"topic_{i}"] = node

    def stabilize_universal_concepts(self, n_concepts: int = 3) -> int:
        """Higher-order clustering with deep structural wiring (L5 -> L3)."""
        topics = [n for k,n in self.patterns.items() if k.startswith("topic_")]
        if not topics: return 0
        s_dim, dim = self.config.S_DIM, self.config.DIM
        vecs = np.array([n.mu[s_dim : s_dim + dim] for n in topics])
        k = min(n_concepts, len(vecs))
        # K-means++ style initialization
        centroids = [vecs[np.random.choice(len(vecs))]]
        for _ in range(1, k):
            dists = np.array([min([np.linalg.norm(v-c)**2 for c in centroids]) for v in vecs])
            d_sum = dists.sum()
            if d_sum == 0:
                # Fallback to random if all remaining points are identical to centroids
                remaining = [i for i in range(len(vecs)) if not any(np.allclose(vecs[i], c) for c in centroids)]
                if not remaining: remaining = list(range(len(vecs)))
                centroids.append(vecs[np.random.choice(remaining)])
            else:
                probs = dists / d_sum
                centroids.append(vecs[np.random.choice(len(vecs), p=probs)])
        centroids = np.array(centroids)
        for _ in range(10):
            dists = np.linalg.norm(vecs[:,None]-centroids[None], axis=2)
            labels = dists.argmin(axis=1)
            centroids = np.array([vecs[labels==i].mean(axis=0) if (labels==i).any() else centroids[i] for i in range(k)])
        for i, c in enumerate(centroids):
            mu = np.zeros(self.m_dim); mu[s_dim : s_dim + dim] = c
            node_id = f"concept_{i}"
            
            # Preserve metadata if node already exists
            old_metadata = {}
            if node_id in self.patterns:
                old_metadata = getattr(self.patterns[node_id], "metadata", {})
            
            node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.15, id=node_id, use_diag=True)
            node.relation_type = "concept"
            node.metadata = {"concept_id": i, "type": "concept"}
            node.metadata.update(old_metadata)
            
            indices = np.where(labels == i)[0]
            for idx in indices: node.add_child(topics[idx])
            self.observer.register(node, protected=True, initial_weight=5.0)
            self.patterns[node_id] = node
        return k

    def summarize_node(self, node_id: str, top_n: int = 5) -> str:
        """Extract top-N keywords from an HFN mu vector."""
        if node_id not in self.patterns: return "Unknown Node"
        node = self.patterns[node_id]
        s_dim, dim = self.config.S_DIM, self.config.DIM
        mu_vec = node.mu[s_dim : s_dim + dim]
        indices = np.argsort(mu_vec)[::-1][:top_n]
        words = [self.config.concepts[i] for i in indices if mu_vec[i] > 1e-3]
        prefix = getattr(node, "metadata", {}).get("type", "Pattern").capitalize()
        return f"{prefix} [{', '.join(words)}]"

    def query_hierarchical(self, question: str) -> Optional[str]:
        """Perform top-down hierarchical search with analogical fallback."""
        concepts = [n for k,n in self.patterns.items() if k.startswith("concept_")]
        if not concepts: return self.query(question)
        q_mu = self.config.encode_passage(question)
        s_dim, dim = self.config.S_DIM, self.config.DIM
        q_v = q_mu[s_dim : s_dim + dim]
        
        target_concept = max(concepts, key=lambda n: float(np.dot(q_v, n.mu[s_dim : s_dim + dim])))
        
        def best_child(parent_node):
            children = parent_node.children()
            if not children: return None
            return max(children, key=lambda n: float(np.dot(q_v, n.mu[s_dim : s_dim + dim])))

        target_topic = best_child(target_concept)
        if not target_topic: return self.renderer.render(target_concept)
        
        # Analogical traversal: if this topic is mapped to another domain, 
        # we might want to return results from BOTH.
        source_topic_id = self._analogy_map.get(target_topic.id)
        if source_topic_id and source_topic_id in self.patterns:
            source_topic = self.patterns[source_topic_id]
            # [Optional] We could return a joint summary, but for now we find best passage
            target_passage = best_child(target_topic)
            source_passage = best_child(source_topic)
            if target_passage and source_passage:
                return f"[Target] {self.renderer.render(target_passage)}\n[Analogy] {self.renderer.render(source_passage)}"
        
        target_passage = best_child(target_topic)
        if not target_passage: return self.renderer.render(target_topic)
        return self.renderer.render(target_passage)

    def learn_thematic_transitions(self, doc_idx: int) -> int:
        """Learns sequential transitions and updates concept transition matrix."""
        indices = self._documents[doc_idx] if doc_idx < len(self._documents) else []
        if len(indices) < 2: return 0
        all_topics = [n for k,n in self.patterns.items() if k.startswith("topic_")]
        if not all_topics:
            print("      [DEBUG] learn_thematic_transitions: No topics found in forest.")
            return 0
        
        # Find which concept this document mostly belongs to
        doc_passage_vecs = [self.config._passage_vecs[idx] for idx in indices]
        doc_topics = [min(all_topics, key=lambda t: np.linalg.norm(t.mu-v)) for v in doc_passage_vecs]
        concepts = [n for k,n in self.patterns.items() if k.startswith("concept_")]
        if not concepts:
            print("      [DEBUG] learn_thematic_transitions: No concepts found.")
            return 0
        
        # Find best concept using cosine similarity between document mean and concept mean
        doc_mu_mean = np.mean([self.config._passage_vecs[idx] for idx in indices], axis=0)
        s_dim, dim = self.config.S_DIM, self.config.DIM
        doc_v = doc_mu_mean[s_dim : s_dim + dim]
        doc_v = doc_v / (np.linalg.norm(doc_v) + 1e-9)
        
        best_c = None
        best_sim = -1.0
        for c in concepts:
            c_v = c.mu[s_dim : s_dim + dim]
            c_v = c_v / (np.linalg.norm(c_v) + 1e-9)
            sim = float(np.dot(doc_v, c_v))
            if sim > best_sim:
                best_sim = sim
                best_c = c
        
        print(f"      [DEBUG] learn_thematic_transitions: assigned Doc {doc_idx} to {best_c.id} (sim={best_sim:.3f})")
        
        # Use children of this concept to build local transition matrix
        concept_topics = best_c.children()
        # Filter children to only include those still in forest (if we rebuilt)
        concept_topics = [t for t in concept_topics if t.id in self.patterns]
        
        if not concept_topics:
            print(f"      [DEBUG] learn_thematic_transitions: {best_c.id} has no valid children.")
            return 0
        
        # Track transitions for concept-local matrix
        topic_idx = {t.id: i for i, t in enumerate(concept_topics)}
        n_topics = len(concept_topics)
        trans_counts = np.zeros((n_topics, n_topics))
        
        count = 0
        for i in range(len(indices)-1):
            p1_mu, p2_mu = self.config._passage_vecs[indices[i]], self.config._passage_vecs[indices[i+1]]
            t1 = min(concept_topics, key=lambda t: np.linalg.norm(t.mu-p1_mu))
            t2 = min(concept_topics, key=lambda t: np.linalg.norm(t.mu-p2_mu))
            
            trans_counts[topic_idx[t1.id], topic_idx[t2.id]] += 1
            
            delta = t2.mu - t1.mu
            node_id = f"theme_{doc_idx}_{i}"
            node = HFN(mu=delta, sigma=np.ones(delta.size)*0.1, id=node_id, use_diag=True)
            node.relation_type = "theme_transition"
            self.observer.register(node)
            self.patterns[node_id] = node; count += 1
            
        # Normalize and store in concept metadata
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        safe_sums = np.where(row_sums == 0, 1, row_sums)
        m_norm = trans_counts / safe_sums
        
        if "transition_matrix" not in best_c.metadata or best_c.metadata["transition_matrix"].shape != m_norm.shape:
            best_c.metadata["transition_matrix"] = m_norm
            best_c.metadata["topic_ids"] = [t.id for t in concept_topics]
        else:
            best_c.metadata["transition_matrix"] = 0.7 * best_c.metadata["transition_matrix"] + 0.3 * m_norm
            best_c.metadata["topic_ids"] = [t.id for t in concept_topics] 
                
        return count

    def get_concept_transition_matrix(self, concept_id: str) -> Optional[np.ndarray]:
        if concept_id not in self.patterns: return None
        return getattr(self.patterns[concept_id], "metadata", {}).get("transition_matrix")

    def find_structural_analogy(self, target_concept_id: str) -> Optional[str]:
        """Finds most similar Source concept based on transition matrix isomorphism."""
        target_matrix = self.get_concept_transition_matrix(target_concept_id)
        if target_matrix is None:
            print(f"      [DEBUG] find_structural_analogy: No target matrix for {target_concept_id}")
            return None
        
        concepts = [n for k,n in self.patterns.items() if k.startswith("concept_") and k != target_concept_id]
        print(f"      [DEBUG] find_structural_analogy: comparing {target_concept_id} against {len(concepts)} concepts")
        
        best_source = None
        best_score = float('inf')
        
        for source in concepts:
            source_matrix = source.metadata.get("transition_matrix")
            if source_matrix is None:
                print(f"      [DEBUG] find_structural_analogy: No matrix for source {source.id}")
                continue
            
            # Use Frobenius norm of matrices. 
            s1, s2 = target_matrix.shape, source_matrix.shape
            print(f"      [DEBUG] find_structural_analogy: target shape {s1}, source {source.id} shape {s2}")
            # ...
            if s1 == s2:
                score = np.linalg.norm(target_matrix - source_matrix)
            else:
                # Pad to max size
                max_d = max(s1[0], s2[0])
                m1 = np.zeros((max_d, max_d))
                m2 = np.zeros((max_d, max_d))
                m1[:s1[0], :s1[1]] = target_matrix
                m2[:s2[0], :s2[1]] = source_matrix
                score = np.linalg.norm(m1 - m2)

            if score < best_score:
                best_score = score
                best_source = source.id
        
        return best_source

    def transfer_strategy(self, source_concept_id: str, target_concept_id: str) -> Dict[str, str]:
        """Maps functional roles between source and target domains."""
        source_concept = self.patterns[source_concept_id]
        target_concept = self.patterns[target_concept_id]
        
        source_matrix = source_concept.metadata["transition_matrix"]
        target_matrix = target_concept.metadata["transition_matrix"]
        
        source_topics = source_concept.metadata["topic_ids"]
        target_topics = target_concept.metadata["topic_ids"]
        
        mapping = {}
        # Ensure we have enough topics in source to map to
        for i, t_id in enumerate(target_topics):
            target_profile = target_matrix[i]
            # find best matching row in source (after padding if needed)
            scores = []
            for j in range(len(source_topics)):
                s_profile = source_matrix[j]
                # Pad profiles to match lengths
                max_l = max(len(target_profile), len(s_profile))
                p1 = np.pad(target_profile, (0, max_l - len(target_profile)))
                p2 = np.pad(s_profile, (0, max_l - len(s_profile)))
                scores.append(np.linalg.norm(p1 - p2))
                
            best_j = np.argmin(scores)
            mapping[t_id] = source_topics[best_j]
            self._analogy_map[t_id] = source_topics[best_j]
            
        return mapping

    def predict_next_topic(self, current_topic_mu: np.ndarray) -> np.ndarray:
        trans = [n for k,n in self.patterns.items() if n.relation_type == "theme_transition"]
        if not trans: return current_topic_mu
        return current_topic_mu + trans[0].mu

    def query_via_clusters(self, question: str) -> Optional[str]: return self.query(question)
    def query_via_concepts(self, question: str) -> Optional[str]: return self.query_hierarchical(question)

    def find_cross_doc_patterns(self, threshold: float = 0.5) -> List[tuple]:
        res = []
        s_dim, dim = self.config.S_DIM, self.config.DIM
        vecs = [v[s_dim : s_dim + dim] for v in self.config._passage_vecs]
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                sim = float(np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i])*np.linalg.norm(vecs[j])+1e-9))
                if sim >= threshold: res.append((self.config.get_passage(i), self.config.get_passage(j), sim))
        return sorted(res, key=lambda x: x[2], reverse=True)
