"""
Forest Diagnostic Script.
Checks the structural integrity and growth of the HFN forest.
"""
from hfn.tiered_forest import TieredForest
import os

def check_forest(path: str):
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        return

    # Use D=None for autodetection
    forest = TieredForest(D=None, cold_dir=path)
    
    nodes = forest.active_nodes()
    print(f"================================================================================")
    print(f"Forest Diagnostic: {path}")
    print(f"================================================================================")
    print(f"Total Active Nodes: {len(nodes)}")
    print(f"Current Dimension (D): {forest._D}")
    
    # Type breakdown
    types = {}
    for n in nodes:
        rtype = getattr(n, "relation_type", "unknown")
        types[rtype] = types.get(rtype, 0) + 1
    
    print("\nNode Type Breakdown:")
    for rtype, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {rtype}: {count}")
        
    # Edge density
    total_edges = sum(len(list(n.edges())) for n in nodes)
    total_children = sum(len(list(n.children())) for n in nodes)
    print(f"\nConnectivity:")
    print(f"  - Total Edges: {total_edges}")
    print(f"  - Total Children: {total_children}")
    print(f"  - Avg Edges/Node: {total_edges / len(nodes):.2f}")
    
    # Sample check
    print("\nSample Relationships & Metadata:")
    # Find nodes with edges
    edged_nodes = [n for n in nodes if n.edges()]
    for n in edged_nodes[:5]:
        for e in n.edges():
            print(f"  - {n.id} --[{e.relation}]--> {e.target.id}")

    # Find definition nodes
    defs = [n for n in nodes if getattr(n, "relation_type", "") == "definition"]
    for d in defs[:2]:
        print(f"  - Definition for '{d.metadata.get('word')}': {d.metadata.get('definition')[:50]}...")
        is_a = [e.target.id for e in d.edges() if e.relation == "is_a"]
        print(f"    Hypernyms: {is_a}")

    # Hierarchy depth
    passages = [n for n in nodes if getattr(n, "relation_type", "") == "passage"]
    if passages:
        p = passages[0]
        children = list(p.children())
        print(f"\nStructure Sample:")
        print(f"  - Passage '{p.id}' children types: {[getattr(c, 'relation_type', 'none') for c in children]}")

if __name__ == "__main__":
    check_forest("data/scientific_curiosity")
