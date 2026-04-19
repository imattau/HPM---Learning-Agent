"""LibrarianAgent: specialized HFN agent for high-level concept discovery, topic mapping, and cross-document synthesis."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Set
from collections import Counter
from hfn.hfn import HFN

# New dependency for better topic discovery
try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False

from hpm_ai_v2.agents.base_agent import BaseHFNAgent
from hpm_ai_v2.agents.mixins.l3_relational import L3RelationalMixin
from hpm_ai_v2.agents.mixins.l3_analogy import L3AnalogyMixin
from hpm_ai_v2.domains.text_domain import TextDomainConfig

class LibrarianAgent(BaseHFNAgent, L3RelationalMixin, L3AnalogyMixin):
    """
    HFN-native agent for high-level conceptual mapping.
    Monitors the forest for new documents and discovers cross-cutting themes, 
    meta-schemas, and analogies.
    """
    def __init__(self, config: TextDomainConfig, forest=None, reader_agent=None, **kwargs):
        if "renderer" not in kwargs:
            from hpm_ai_v2.domains.text_renderer import TextRenderer
            kwargs["renderer"] = TextRenderer(config)
        super().__init__(config, forest=forest, **kwargs)
        self.reader_agent = reader_agent
        self.indexed_docs: Set[str] = set()
        self.knowledge_gaps: List[Dict[str, Any]] = []

    def monitor_gaps(self, x: np.ndarray, result: Optional[Any] = None) -> float:
        """Analyze observer residual surprise to detect knowledge gaps."""
        # If no result provided, run a quick expansion
        if result is None:
            result = self.observer.expand(x)
            
        surprise = result.residual_surprise
        if surprise > self.observer.tau * 1.5:
            # We have a significant gap
            gap_id = f"gap_{int(time.time()*1000)}"
            self.knowledge_gaps.append({
                "id": gap_id,
                "mu": x,
                "surprise": surprise,
                "timestamp": time.time()
            })
            print(f"      [LIBRARIAN] Significant knowledge gap detected! (Surprise: {surprise:.4f})")
            
        return surprise

    def discover_topics(self, doc_node: HFN) -> List[HFN]:
        """Discover and link topics/themes for a given document."""
        if doc_node.id in self.indexed_docs:
            return []
            
        # 1. Monitor gaps in document semantics
        self.monitor_gaps(doc_node.mu)

        # 2. Aggregate text from paragraphs
        text_parts = []
        for para in doc_node.children():
            if para.relation_type == "paragraph":
                t = para.metadata.get("text", "")
                if t: text_parts.append(t)
        
        full_text = " ".join(text_parts)
        if not full_text: return []

        # --- Keyword extraction ---
        keywords = []
        if YAKE_AVAILABLE:
            kw_extractor = yake.KeywordExtractor(lan="en", top=10)
            keywords = [kw for kw, _ in kw_extractor.extract_keywords(full_text)]
        else:
            # Fallback to simple token counting
            from hpm_ai_v2.domains.text_domain import tokenise
            tokens = tokenise(full_text)
            keywords = [w for w, _ in Counter(tokens).most_common(5) if len(w) > 3]

        if not keywords: return []
        
        discovered = []
        for word in keywords[:5]: # limit to top 5
            topic_id = f"topic_{word.lower()}"
            topic_node = self.forest.get(topic_id)
            if not topic_node:
                mu = self.config.encode_passage(word)
                topic_node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.2, id=topic_id, use_diag=True)
                topic_node.relation_type = "topic"
                topic_node.metadata = {"word": word}
                self.observer.register(topic_node, protected=False)
                self.agent_pattern_ids.add(topic_id)
            
            doc_node.add_edge(doc_node, topic_node, "about")
            discovered.append(topic_node)
            
        self.indexed_docs.add(doc_node.id)
        return discovered

    def search_knowledge(self, query: str) -> List[HFN]:
        """Semantic search across topics and documents using accelerated Forest retrieval."""
        mu = self.config.encode_passage(query)
        # Accelerated retrieval for candidates
        candidates = self.forest.retrieve(mu, k=30)
        
        # [UPGRADE] Return documents, passages, and topics to enable deep research
        # Sort by distance in semantic subspace for better relevance
        def _sem_dist(n: HFN) -> float:
            s_slice = slice(self.config.S_DIM, self.config.S_DIM + self.config.DIM)
            n_v = n.mu[s_slice]
            q_v = mu[s_slice]
            norm_n = np.linalg.norm(n_v)
            norm_q = np.linalg.norm(q_v)
            if norm_n > 0 and norm_q > 0:
                return float(1.0 - np.dot(n_v, q_v) / (norm_n * norm_q))
            return float(np.sum((n_v - q_v)**2))

        valid_types = ("topic", "document", "html_document", "passage", "paragraph", "sentence")
        results = [n for n in candidates if n.relation_type in valid_types]
        results.sort(key=_sem_dist)
        
        return results[:10]

    def find_cross_domain_analogies(self, source_topic: HFN, target_domain_mu: np.ndarray) -> List[HFN]:
        """Use L3AnalogyMixin to find structural similarities across domains."""
        return self.find_analogies(source_topic, target_domain_mu)
        
    def answer_question_hierarchical(self, question: str) -> Optional[str]:
        """BFS search over document hierarchy to find relevant concept, accelerated by FAISS."""
        mu = self.config.encode_passage(question)
        
        # [UPGRADE] Normalized distance for better matching in unified representational space
        def _dist(n_mu: np.ndarray, q_mu: np.ndarray) -> float:
            # Mask out structural flags (10+ or as appropriate) if needed, 
            # but here we focus on the DIM slice
            s_slice = slice(self.config.S_DIM, self.config.S_DIM + self.config.DIM)
            n_v = n_mu[s_slice]
            q_v = q_mu[s_slice]
            # Normalizing helps when comparing query (1.0 at index) with sentence (1/N at index)
            norm_n = np.linalg.norm(n_v)
            norm_q = np.linalg.norm(q_v)
            if norm_n > 0 and norm_q > 0:
                # Return cosine distance (1 - cosine similarity)
                return float(1.0 - np.dot(n_v, q_v) / (norm_n * norm_q))
            return float(np.sum((n_v - q_v)**2))

        # Accelerated retrieval for top-level documents
        candidates = self.forest.retrieve(mu, k=20)
        doc_nodes = [
            n for n in candidates
            if n.relation_type in ("document", "html_document")
        ]
        
        if not doc_nodes: 
            # Fallback to any node with children if no docs found
            doc_nodes = [n for n in candidates if n.children()]
            
        if not doc_nodes: return None
        
        print(f"      [DEBUG] Librarian: Searching for '{question}'")
        final_candidates = []
        
        for doc in doc_nodes[:5]:
            dist_doc = _dist(doc.mu, mu)
            title = doc.metadata.get('title') or doc.id
            print(f"        - Candidate: {title} (dist: {dist_doc:.4f})")
            
            # Search paragraphs in this doc
            paras = [c for c in doc.children() if c.relation_type == "paragraph"]
            if not paras: continue
            
            paras.sort(key=lambda n: _dist(n.mu, mu))
            best_para = paras[0]
            
            # Search sentences in this paragraph
            sents = [c for c in best_para.children() if c.relation_type == "sentence"]
            if not sents: continue
            
            sents.sort(key=lambda n: _dist(n.mu, mu))
            best_sent = sents[0]
            dist = _dist(best_sent.mu, mu)
            final_candidates.append((dist, best_sent))
            
        if not final_candidates: return None
        
        # Return the sentence with the absolute minimum distance
        final_candidates.sort(key=lambda x: x[0])
        best_sent = final_candidates[0][1]
        print(f"      [DEBUG] Librarian: Selected best sentence (dist: {final_candidates[0][0]:.4f})")
        
        sent_text = getattr(best_sent, "metadata", {}).get("text")
        if sent_text:
            return sent_text
        return self.renderer.render(best_sent)

    def summarize_theme(self, topic_node: HFN) -> str:
        """Find key documents associated with a theme and summarize."""
        # Find docs pointing to this topic via 'about'
        docs = []
        for node in self.forest.active_nodes():
            if node.relation_type == "document":
                # Check edges
                for edge in node.edges():
                    if edge.target.id == topic_node.id and edge.relation == "about":
                        docs.append(node)
                        break
        
        if not docs:
            return f"Topic: {topic_node.metadata.get('word', topic_node.id)}"
            
        titles = [d.metadata.get("title", d.id) for d in docs[:3]]
        return f"Theme '{topic_node.metadata.get('word')}' is discussed in: {', '.join(titles)}"
