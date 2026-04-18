"""ReaderAgent: observes text passages and retrieves relevant ones for queries."""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
from hfn.hfn import HFN
from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.syntax import SyntaxMixin
from hpm_ai_v2.agents.mixins.srl import SemanticRoleMixin
from hpm_ai_v2.agents.mixins.spelling import SpellingMixin
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.domains.text_renderer import TextRenderer
from hpm_ai_v2.utils.oracle.text_oracle import TextOracle
from hpm_ai_v2.utils.text_fetcher import fetch_passages
from hpm_ai_v2.utils.sentence_splitter import SentenceSplitter


class ReaderAgent(BaseHFNAgent, SyntaxMixin, SemanticRoleMixin, SpellingMixin):
    """
    HFN-native agent that reads text/webpages and retrieves relevant passages.
    Extended with structural hierarchy (L2-L5), recursive summarization, 
    predictive curiosity, structural analogy, syntax, semantics, and spelling (SP-Reader 8).
    Upgraded for Wikipedia ingestion and dynamic vocabulary (SP-Reader 9).
    """

    def __init__(self, config: TextDomainConfig, dynamic_vocab: bool = True, max_vocab: Optional[int] = None, **kwargs) -> None:
        renderer = TextRenderer(config)
        super().__init__(config, renderer=renderer, **kwargs)
        self.oracle = TextOracle(config)
        self.counting_oracle.wrapped = self.oracle
        self._documents: List[List[int]] = []
        self._last_topic_mu: Optional[np.ndarray] = None
        self._analogy_map: Dict[str, str] = {} # target_node_id -> source_node_id
        self.dynamic_vocab = dynamic_vocab
        self.max_vocab = max_vocab
        self.sentence_splitter = SentenceSplitter()
        
        # Ensure mixin attributes are initialized if super() chain was interrupted
        if not hasattr(self, "pos_rules"): self.pos_rules = {}
        if not hasattr(self, "role_knowledge"): self.role_knowledge = []
        if not hasattr(self, "word_spellings"): self.word_spellings = {}
        
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

    def build_sentence_node(self, tokens: List[str]) -> HFN:
        """Create a sentence node with inputs = list of word nodes."""
        word_nodes = [self._ensure_word_macro(t) for t in tokens]
        # Use existing encoding for mu
        sentence_mu = self.config.encode_passage(" ".join(tokens))
        
        sentence_id = f"sentence_{uuid.uuid4().hex[:8]}"
        sentence_node = HFN(
            mu=sentence_mu,
            sigma=np.ones(self.m_dim)*0.05,
            id=sentence_id,
            use_diag=True,
        )
        for wn in word_nodes:
            sentence_node.add_child(wn)
            
        sentence_node.metadata = {"tokens": tokens, "type": "sentence"}
        sentence_node.relation_type = "sentence"
        self.observer.register(sentence_node, protected=False)
        self.patterns[sentence_id] = sentence_node
        return sentence_node

    def build_paragraph_node(self, sentence_nodes: List[HFN]) -> HFN:
        """Create a paragraph node with inputs = sentence nodes."""
        if not sentence_nodes: return None
        para_mu = np.mean([n.mu for n in sentence_nodes], axis=0)
        
        para_id = f"paragraph_{uuid.uuid4().hex[:8]}"
        para_node = HFN(
            mu=para_mu,
            sigma=np.ones(self.m_dim)*0.08,
            id=para_id,
            use_diag=True,
        )
        for sn in sentence_nodes:
            para_node.add_child(sn)
            
        para_node.metadata = {"type": "paragraph"}
        para_node.relation_type = "paragraph"
        self.observer.register(para_node, protected=False)
        self.patterns[para_id] = para_node
        return para_node

    def build_document_node(self, para_nodes: List[HFN], title: str = "document") -> HFN:
        """Build document node from list of paragraph nodes."""
        if not para_nodes: return None
        doc_mu = np.mean([n.mu for n in para_nodes], axis=0)
        
        doc_id = f"document_{uuid.uuid4().hex[:8]}"
        doc_node = HFN(
            mu=doc_mu,
            sigma=np.ones(self.m_dim)*0.1,
            id=doc_id,
            use_diag=True,
        )
        for pn in para_nodes:
            doc_node.add_child(pn)
            
        doc_node.metadata = {"type": "document", "title": title}
        doc_node.relation_type = "document"
        self.observer.register(doc_node, protected=False)
        self.patterns[doc_id] = doc_node
        return doc_node

    def expand_vocabulary(self, texts: List[str], max_new: int = 10) -> int:
        added = self.config.expand_vocab(texts, max_new=max_new)
        if added > 0: self.reindex_knowledge_base()
        return added

    def _ensure_word_macro(self, word: str) -> HFN:
        """Ensure a character-level macro exists for the word, adding to vocab if needed."""
        word_id = f"word_spelling_{word.lower()}"
        if word_id in self.patterns:
            return self.patterns[word_id]
        
        # Add to config vocabulary (dynamic expansion)
        if self.dynamic_vocab:
            self.config.add_word(word)
            # Reindex if dimension changed
            if self.config.m_dim != self.forest._D:
                self.reindex_knowledge_base()
        
        # Create character-level macro
        return self.learn_word_spelling(word, case_sensitive=False)

    def observe_passage(self, text: str) -> int:
        if self.dynamic_vocab:
            from hpm_ai_v2.domains.text_domain import tokenise
            tokens = tokenise(text)
            for t in tokens:
                self._ensure_word_macro(t)
        
        # Build structural hierarchy (Fractal Uniformity)
        sentences = self.sentence_splitter.split(text)
        sentence_nodes = []
        from hpm_ai_v2.domains.text_domain import tokenise_raw
        for s in sentences:
            tokens = tokenise_raw(s)
            if tokens:
                s_node = self.build_sentence_node(tokens)
                sentence_nodes.append(s_node)
        
        # Build paragraph node (treat passage as one paragraph for now)
        para_node = self.build_paragraph_node(sentence_nodes)
        
        idx = self.config.register_passage(text)
        mu = self.config.encode_passage(text)
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=f"passage_{idx}", use_diag=True)
        node.metadata = {"passage_idx": idx, "text": text, "type": "passage"}
        
        # Link paragraph node to passage node
        if para_node: node.add_child(para_node)
        
        self.observer.register(node, protected=False, initial_weight=1.0)
        self.patterns[f"passage_{idx}"] = node
        
        # Update last topic mu for predictive curiosity
        topics = [n for k,n in self.patterns.items() if k.startswith("topic_")]
        if topics:
            best_t = min(topics, key=lambda t: np.linalg.norm(t.mu - mu))
            self._last_topic_mu = best_t.mu.copy()
        return idx

    def ingest_wikipedia_page(self, title: str, chunk_size: int = 5, overlap: int = 2):
        """Fetch and ingest a Wikipedia page."""
        try:
            import wikipedia
        except ImportError:
            print("      [ERROR] ingest_wikipedia_page: 'wikipedia' library not found. Run 'pip install wikipedia'.")
            return
            
        print(f"      [INFO] Ingesting Wikipedia page: {title}")
        try:
            page = wikipedia.page(title)
            text = page.content
        except Exception as e:
            print(f"      [ERROR] Failed to fetch Wikipedia page '{title}': {e}")
            return
            
        sentences = self.sentence_splitter.split(text)
        from hpm_ai_v2.utils.text_chunker import chunk_passages
        passages = chunk_passages(sentences, window_size=chunk_size, overlap=overlap)
        
        indices = []
        passage_nodes = []
        for p in passages:
            idx = self.observe_passage(p)
            indices.append(idx)
            passage_nodes.append(self.patterns[f"passage_{idx}"])
        
        # Build Document hierarchy
        # A document node whose children are paragraph nodes (here passage nodes are paragraphs)
        para_nodes = []
        for pn in passage_nodes:
            # Each passage node contains one paragraph node
            children = pn.children()
            para = next((c for c in children if getattr(c, "relation_type", None) == "paragraph"), None)
            if para: para_nodes.append(para)
            
        self.build_document_node(para_nodes, title=title)
        
        self._documents.append(indices)
        
        # After large ingestion, it's good to rebuild hierarchy
        if len(passages) > 10:
            print(f"      [INFO] Building topic clusters for {title}...")
            self.build_topic_clusters()
            self.stabilize_universal_concepts()
        
        return indices

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
            "role_knowledge": getattr(self, "role_knowledge", []),
            "word_spellings": getattr(self, "word_spellings", {}),
            "include_char_primitives": getattr(self.config, "include_char_primitives", False)
        }
        with open(path / "reader_meta.json", "w") as f: json.dump(meta, f)

    @classmethod
    def load_agent(cls, directory: str) -> "ReaderAgent":
        path = Path(directory)
        with open(path / "reader_meta.json") as f: meta = json.load(f)
        config = TextDomainConfig(
            meta["concepts"], 
            meta["idf"], 
            include_char_primitives=meta.get("include_char_primitives", False)
        )
        config._passages = meta["passages"]
        config._passage_vecs = [config.encode_passage(p) for p in meta["passages"]]
        agent = cls(config, cold_dir=str(path))
        agent._documents = meta.get("docs", [])
        agent.pos_rules = meta.get("pos_rules", {})
        agent.role_knowledge = meta.get("role_knowledge", [])
        agent.word_spellings = meta.get("word_spellings", {})
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
        """K-means clustering of structural nodes (sentences/paragraphs)."""
        # Find all sentence nodes
        struct_nodes = [n for k,n in self.patterns.items() if getattr(n, "relation_type", None) in ["sentence", "paragraph"]]
        if not struct_nodes:
            # Fallback to passage vectors if no structural nodes found
            if not self.config._passage_vecs: return
            vecs = np.array([v[self.config.S_DIM : self.config.S_DIM + self.config.DIM] for v in self.config._passage_vecs])
            node_ids = [f"passage_{i}" for i in range(len(self.config._passages))]
        else:
            vecs = np.array([n.mu[self.config.S_DIM : self.config.S_DIM + self.config.DIM] for n in struct_nodes])
            node_ids = [n.id for n in struct_nodes]
            
        k = min(n_clusters, len(vecs))
        if k == 0: return
        
        # K-means++ style initialization
        centroids = [vecs[np.random.choice(len(vecs))]]
        for _ in range(1, k):
            dists = np.array([min([np.linalg.norm(v-c)**2 for c in centroids]) for v in vecs])
            d_sum = dists.sum()
            if d_sum == 0:
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
            mu = np.zeros(self.m_dim); mu[self.config.S_DIM : self.config.S_DIM + self.config.DIM] = c
            node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.2, id=f"topic_{uuid.uuid4().hex[:8]}", use_diag=True)
            node.metadata = {"cluster_id": i, "type": "topic"}; node.relation_type = "topic"
            indices = np.where(labels == i)[0]
            for idx in indices:
                child_id = node_ids[idx]
                if child_id in self.patterns: node.add_child(self.patterns[child_id])
            self.observer.register(node, protected=True, initial_weight=2.0)
            self.patterns[node.id] = node

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

    def retrieve_by_predicate(self, predicate_word: str) -> List[HFN]:
        """Find sentence nodes whose predicate matches predicate_word."""
        candidates = []
        p_lower = predicate_word.lower()[:4] # stem
        for k, n in self.patterns.items():
            if getattr(n, "relation_type", None) == "sentence":
                roles = self.extract_roles_from_node(n)
                c_pred = roles.get("PREDICATE", "").lower()
                if c_pred and c_pred[:4] == p_lower:
                    candidates.append(n)
        return candidates

    def answer_question_hierarchical(self, question: str) -> Optional[str]:
        """Answer a question by searching the HFN hierarchy for structured evidence."""
        q_roles = self.extract_roles(question)
        print(f"      [DEBUG] Question Roles: {q_roles}")
        if not q_roles: return None
        
        # 1. Determine target role
        q_lower = question.lower()
        target_role = "AGENT" if any(w in q_lower for w in ["who", "which animal"]) else "PATIENT"
        
        # 2. Retrieve candidates by predicate
        predicate = q_roles.get("PREDICATE")
        if not predicate: return None
        
        candidates = self.retrieve_by_predicate(predicate)
        if not candidates:
            # Fallback to general hierarchical retrieval
            topic_res = self.query_hierarchical(question)
            return topic_res
            
        # 3. Score candidates by role alignment
        best_sent = None
        best_score = -1.0
        
        for sent_node in candidates:
            s_roles = self.extract_roles_from_node(sent_node)
            score = self.score_role_match(q_roles, s_roles)
            if score > best_score:
                best_score = score
                best_sent = sent_node
                
        # 4. Extract answer role
        if best_sent and best_score > 0.5:
            s_roles = self.extract_roles_from_node(best_sent)
            ans = s_roles.get(target_role)
            if ans:
                # Reconstruct full NP if needed, for now return the word
                return ans
            else:
                # Fallback to rendering the whole sentence as evidence
                return self.renderer.render(best_sent)
                
        return None

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
