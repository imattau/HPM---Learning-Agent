import numpy as np
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.retriever import StructuralRetriever, HybridRetriever, GeometricRetriever

def test_structural_retriever():
    D = 4
    forest = Forest(D=D)
    
    # Node A: Leaf (0 children, 0 inputs)
    node_a = HFN(mu=np.zeros(D), sigma=np.ones(D), id="leaf_a", use_diag=True)
    forest.register(node_a)
    
    # Node B: Macro (2 inputs)
    node_b = HFN(mu=np.ones(D), sigma=np.ones(D), id="macro_b", use_diag=True, relation_type="macro")
    node_b.inputs = [node_a, node_a]
    forest.register(node_b)
    
    retriever = StructuralRetriever(forest)
    
    # Query for something like a leaf
    query_leaf = HFN(mu=np.zeros(D), sigma=np.ones(D), id="query_leaf")
    results = retriever.retrieve(query_leaf, k=1)
    assert results[0].id == "leaf_a"
    
    # Query for something like a macro
    query_macro = HFN(mu=np.zeros(D), sigma=np.ones(D), id="query_macro", relation_type="macro")
    query_macro.inputs = [node_a]
    results = retriever.retrieve(query_macro, k=1)
    assert results[0].id == "macro_b"

def test_hybrid_retriever():
    D = 4
    forest = Forest(D=D)
    
    # Node A: Geometrically close, structurally different
    node_a = HFN(mu=np.array([0.1, 0, 0, 0]), sigma=np.ones(D), id="geo_close", use_diag=True)
    forest.register(node_a)
    
    # Node B: Geometrically far, structurally identical
    node_b = HFN(mu=np.array([10.0, 0, 0, 0]), sigma=np.ones(D), id="struct_close", use_diag=True, relation_type="macro")
    node_b.inputs = [node_a]
    forest.register(node_b)
    
    # Query structurally identical to B
    query = HFN(mu=np.array([0.0, 0, 0, 0]), sigma=np.ones(D), id="query", relation_type="macro")
    query.inputs = [node_a]
    
    # Struct-heavy hybrid
    retriever = HybridRetriever(forest, geometric_weight=0.1, structural_weight=0.9)
    results = retriever.retrieve(query, k=1)
    assert results[0].id == "struct_close"
    
    # Geo-heavy hybrid
    retriever = HybridRetriever(forest, geometric_weight=0.9, structural_weight=0.1)
    results = retriever.retrieve(query, k=1)
    assert results[0].id == "geo_close"
