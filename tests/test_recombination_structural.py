import numpy as np
from hfn.hfn import HFN
from hfn.forest import Forest
from hfn.recombination import Recombination

def test_recombine_structural():
    D = 4
    forest = Forest(D=D)
    recombiner = Recombination()
    
    # Macro A: x=inp, res=[]
    node_inp = HFN(mu=np.zeros(D), sigma=np.ones(D), id="VAR_INP")
    node_init = HFN(mu=np.zeros(D), sigma=np.ones(D), id="LIST_INIT")
    macro_a = HFN(mu=np.ones(D), sigma=np.ones(D), id="macro_a")
    macro_a.inputs = [node_inp, node_init]
    forest.register(node_inp)
    forest.register(node_init)
    forest.register(macro_a)
    
    # Macro B: val=item, append(val)
    node_item = HFN(mu=np.ones(D), sigma=np.ones(D), id="ITEM_ACCESS")
    node_app = HFN(mu=np.ones(D), sigma=np.ones(D), id="LIST_APPEND")
    macro_b = HFN(mu=np.zeros(D), sigma=np.ones(D), id="macro_b")
    macro_b.inputs = [node_item, node_app]
    forest.register(node_item)
    forest.register(node_app)
    forest.register(macro_b)
    
    recomb = recombiner.recombine_structural(macro_a, macro_b, forest)
    
    assert recomb.relation_type == "recombined"
    assert len(recomb.inputs) == 4
    assert [n.id for n in recomb.inputs] == ["VAR_INP", "LIST_INIT", "ITEM_ACCESS", "LIST_APPEND"]
    
    # Check "then" edge
    edges = recomb.edges()
    found_then = False
    for u, v, rel in edges:
        if u.id == "LIST_INIT" and v.id == "ITEM_ACCESS" and rel == "then":
            found_then = True
            break
    assert found_then
