"""HtmlReaderAgent: Specialized HFN agent for parsing HTML into fractal hierarchies."""
from __future__ import annotations
import uuid
import numpy as np
from html.parser import HTMLParser
from typing import List, Dict, Optional, Tuple, Any
from hfn.hfn import HFN
from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.domains.text_domain import TextDomainConfig, tokenise_raw

class _HtmlToHFNParser(HTMLParser):
    """Internal DOM-like builder using Python's standard HTMLParser."""
    def __init__(self, agent: HtmlReaderAgent):
        super().__init__()
        self.agent = agent
        self.stack: List[HFN] = []
        self.root_nodes: List[HFN] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]):
        # 1. Create tag word macro
        tag_word_node = self.agent._ensure_word_macro(tag)
        tag_word_node.relation_type = "html_tag"

        # 2. Build attribute nodes
        attr_nodes = []
        for name, value in attrs:
            if value is not None:
                attr_node = self.agent._build_attribute_node(name, value)
                attr_nodes.append(attr_node)

        # 3. Create element node
        elem_id = f"html_element_{uuid.uuid4().hex[:8]}"
        mu = np.zeros(self.agent.m_dim)
        # We can mix in tag and attrs into mu for better similarity search
        mu += tag_word_node.mu
        for an in attr_nodes:
            mu += an.mu
        if len(attr_nodes) + 1 > 1: mu /= (len(attr_nodes) + 1)
        
        elem_node = HFN(mu=mu, sigma=np.ones(self.agent.m_dim)*0.1, id=elem_id, use_diag=True)
        elem_node.relation_type = "html_element"
        elem_node.metadata = {"type": "html_element", "tag": tag}
        
        # Structure: children[0] = tag, then attributes, then nested content
        elem_node.add_child(tag_word_node)
        for an in attr_nodes:
            elem_node.add_child(an)
            
        if self.stack:
            self.stack[-1].add_child(elem_node)
        else:
            self.root_nodes.append(elem_node)
            
        self.stack.append(elem_node)

    def handle_endtag(self, tag: str):
        if self.stack:
            # We don't strictly enforce tag matching for speed, 
            # but we pop the stack.
            node = self.stack.pop()
            self.agent.observer.register(node, protected=False)
            self.agent.patterns[node.id] = node

    def handle_data(self, data: str):
        text = data.strip()
        if not text: return
        
        # 4. Ingest text content as a sentence or paragraph
        text_node = self.agent._ingest_text_fragment(text)
        if text_node:
            if self.stack:
                self.stack[-1].add_child(text_node)
            else:
                self.root_nodes.append(text_node)

class HtmlReaderAgent(ReaderAgent):
    """
    Agent for fractal HTML parsing.
    Extends ReaderAgent to build deep hierarchies from raw HTML.
    """
    def __init__(self, config: TextDomainConfig, forest=None, web_agent=None, **kwargs):
        super().__init__(config, forest=forest, web_agent=web_agent, **kwargs)
        self._init_html_primitives()

    def _init_html_primitives(self):
        """Seed basic HTML concepts if needed."""
        pass

    def ingest_html(self, html: str, url: str, webpage_node: Optional[HFN] = None) -> HFN:
        """Parse raw HTML and return the root HFN document node."""
        print(f"      [HTML] Ingesting: {url}")
        parser = _HtmlToHFNParser(self)
        parser.feed(html)
        
        # Create Document Node
        doc_id = f"html_document_{uuid.uuid4().hex[:8]}"
        mu = np.zeros(self.m_dim)
        if parser.root_nodes:
            mu = np.mean([n.mu for n in parser.root_nodes], axis=0)
            
        doc_node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=doc_id, use_diag=True)
        doc_node.relation_type = "html_document"
        doc_node.metadata = {"type": "html_document", "url": url}
        
        for rn in parser.root_nodes:
            doc_node.add_child(rn)
            
        # Link to webpage source
        if webpage_node:
            doc_node.add_edge(doc_node, webpage_node, "derived_from")
            
        self.observer.register(doc_node, protected=False)
        self.patterns[doc_id] = doc_node
        return doc_node

    def _build_attribute_node(self, name: str, value: str) -> HFN:
        """Create an HFN node for an attribute-value pair."""
        name_node = self._ensure_word_macro(name)
        value_node = self._ensure_word_macro(value)
        
        attr_id = f"html_attr_{uuid.uuid4().hex[:8]}"
        mu = (name_node.mu + value_node.mu) / 2
        
        attr_node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=attr_id, use_diag=True)
        attr_node.relation_type = "html_attribute"
        attr_node.metadata = {"type": "html_attribute", "name": name, "value": value}
        
        attr_node.add_child(name_node)
        attr_node.add_child(value_node)
        
        self.observer.register(attr_node, protected=False)
        self.patterns[attr_id] = attr_node
        return attr_node

    def _ingest_text_fragment(self, text: str) -> Optional[HFN]:
        """Convert a text string into a sentence-level HFN node."""
        tokens = tokenise_raw(text)
        if not tokens: return None
        
        # Reuse ReaderAgent's sentence builder
        # Note: we don't call observe_passage here to avoid duplicated passage nodes,
        # we just want the sentence hierarchy.
        for t in tokens:
            self._ensure_word_macro(t)
            
        sent_node = self.build_sentence_node(tokens)
        return sent_node
