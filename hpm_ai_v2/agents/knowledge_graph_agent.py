"""KnowledgeGraphAgent: specialized HFN agent for structured factual reasoning via external KGs."""
from __future__ import annotations
import time
import numpy as np
from typing import List, Dict, Optional, Any, Set
from hfn.hfn import HFN, Edge
from hpm_ai_v2.agents.learning_agent import LearningAgent
from hpm_ai_v2.domains.knowledge_graph_domain import KnowledgeGraphDomainConfig

class KnowledgeGraphAgent(LearningAgent):
    """
    HFN-native agent for interfacing with external Knowledge Graphs (Wikidata).
    Translates structured facts into forest nodes and edges.
    """
    def __init__(self, config: Optional[KnowledgeGraphDomainConfig] = None, forest=None, **kwargs):
        if config is None:
            config = KnowledgeGraphDomainConfig()
        super().__init__(config, forest=forest, **kwargs)
        self.endpoint = "https://query.wikidata.org/sparql"
        self._user_agent = "HPM-Learning-Agent/1.0 (https://github.com/google/gemini-cli)"
        
        # Internal cache of entity label -> HFN node
        self.entity_cache: Dict[str, HFN] = {}

    def query_wikidata(self, sparql_query: str) -> List[HFN]:
        """Execute a SPARQL query on Wikidata and import results into the forest."""
        import requests
        
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": self._user_agent
        }
        
        try:
            response = requests.get(self.endpoint, params={'query': sparql_query}, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"      [KG] Error: Wikidata returned status {response.status_code}")
                return []
                
            data = response.json()
            results = data.get("results", {}).get("bindings", [])
            print(f"      [KG] Found {len(results)} results from Wikidata.")
            
            imported_nodes = []
            for item in results:
                # Exhaustive extraction: find all URIs
                for var, binding in item.items():
                    if binding.get("type") == "uri":
                        uri = binding["value"]
                        
                        # Try to find a label in the same binding set
                        # Prefer varLabel, then label, then uri-split
                        label = None
                        for label_var in [f"{var}Label", "label", "itemLabel"]:
                            if label_var in item:
                                label = item[label_var]["value"]
                                break
                        
                        if not label:
                            label = uri.split("/")[-1].replace("_", " ")
                        
                        node = self._ensure_entity(uri, label)
                        imported_nodes.append(node)
            
            return list(set(imported_nodes))
            
        except Exception as e:
            print(f"      [KG] Query failed: {str(e)}")
            return []

    def _ensure_entity(self, uri: str, label: str) -> HFN:
        """Get or create an HFN node for a KG entity."""
        entity_id = f"kg_entity_{uri.split('/')[-1]}"
        node = self.forest.get(entity_id)
        if node:
            return node
            
        mu = self.config.encode_entity(label)
        self._fit_mu_to_forest(mu)
        
        node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=entity_id, use_diag=True)
        node.relation_type = "kg_entity"
        node.metadata = {"uri": uri, "label": label}
        
        self.observer.register(node, protected=False)
        self.observer.observe(node.mu) # [DYNAMICS] Reinforce KG entity
        self.agent_pattern_ids.add(entity_id)
        self.entity_cache[label] = node
        return node

    def find_facts(self, entity_label: str) -> List[HFN]:
        """Search for facts about an entity and link them in the forest."""
        query = f"""
        SELECT ?prop ?propLabel ?obj ?objLabel WHERE {{
          ?subj rdfs:label "{entity_label}"@en .
          ?subj ?prop ?obj .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }} LIMIT 10
        """
        # Simplified query for demonstration; in reality we need more robust SPARQL
        return self.query_wikidata(query)

    def enrich_forest_with_fact(self, subj_label: str, rel_type: str, obj_label: str) -> HFN:
        """Manually create a fact node linking two entities."""
        subj_node = self._ensure_entity(f"http://example.org/{subj_label}", subj_label)
        obj_node = self._ensure_entity(f"http://example.org/{obj_label}", obj_label)
        
        fact_id = f"kg_fact_{subj_label}_{rel_type}_{obj_label}".replace(" ", "_")
        fact_node = self.forest.get(fact_id)
        if fact_node:
            return fact_node
            
        mu = (subj_node.mu + obj_node.mu) / 2
        fact_node = HFN(mu=mu, sigma=np.ones(self.m_dim)*0.1, id=fact_id, use_diag=True)
        fact_node.relation_type = "kg_statement"
        fact_node.metadata = {"subject": subj_label, "relation": rel_type, "object": obj_label}
        
        fact_node.add_child(subj_node)
        fact_node.add_child(obj_node)
        subj_node.add_edge(subj_node, obj_node, rel_type)
        
        self.observer.register(fact_node, protected=False)
        self.observer.observe(fact_node.mu) # [DYNAMICS] Reinforce KG fact
        self.agent_pattern_ids.add(fact_id)
        return fact_node
