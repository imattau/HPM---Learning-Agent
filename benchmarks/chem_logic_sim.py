"""Molecular Simulator for Chem-Logic Benchmark (SP12).
Mocks RDKit behavior for functional group identification and valence checking.
"""

import numpy as np
from dataclasses import dataclass

# Functional Group Registry (Index mapping for L2 vectors)
GROUPS = {
    "hydroxyl": 0,    # -OH
    "aldehyde": 1,    # -CHO
    "ketone": 2,      # >C=O
    "carboxyl": 3,    # -COOH
    "ester": 4,       # -COOR
    "ether": 5,       # -C-O-C-
    "alkane": 6       # -C-C-
}

@dataclass
class Molecule:
    smiles: str
    features: np.ndarray # Multi-hot encoding of groups
    is_valid: bool = True

def get_molecule(smiles: str) -> Molecule:
    """Mock SMILES parser."""
    feats = np.zeros(len(GROUPS))
    valid = True
    
    # Simple rule-based feature extraction
    if "OH" in smiles: feats[GROUPS["hydroxyl"]] = 1
    if "CHO" in smiles: feats[GROUPS["aldehyde"]] = 1
    if "C(=O)" in smiles and not "OH" in smiles: feats[GROUPS["ketone"]] = 1
    if "COOH" in smiles: feats[GROUPS["carboxyl"]] = 1
    if "COOC" in smiles: feats[GROUPS["ester"]] = 1
    if "COC" in smiles: feats[GROUPS["ether"]] = 1
    
    # Mock valence check: molecules with multiple conflicting oxygens on same carbon are 'unstable'
    if smiles.count("O") > 3:
        valid = False
        
    return Molecule(smiles, feats, valid)

def generate_chem_tasks(n_tasks: int = 20, seed: int = 42) -> list[dict]:
    """Generate reactant-product pairs for hidden law inference."""
    rng = np.random.default_rng(seed)
    tasks = []
    
    reactions = ["oxidation", "reduction", "esterification"]
    
    for i in range(n_tasks):
        rxn = rng.choice(reactions)
        
        if rxn == "oxidation":
            # Primary Alcohol -> Aldehyde
            reactant = get_molecule("CH3CH2OH")
            product = get_molecule("CH3CHO")
        elif rxn == "reduction":
            # Ketone -> Secondary Alcohol
            reactant = get_molecule("CH3C(=O)CH3")
            product = get_molecule("CH3CH(OH)CH3")
        elif rxn == "esterification":
            # Alcohol + Acid -> Ester
            reactant = get_molecule("CH3OH.CH3COOH")
            product = get_molecule("CH3COOCH3")
        else:
            reactant = get_molecule("CH3CH3")
            product = get_molecule("CH3CH3")
            
        # Create candidates
        candidates = [product]
        for _ in range(4):
            # Wrong candidate: random functional group flip
            wrong_feats = product.features.copy()
            idx = rng.integers(0, len(GROUPS))
            wrong_feats[idx] = 1 - wrong_feats[idx]
            candidates.append(Molecule("MOCK_SMILES", wrong_feats, rng.choice([True, False])))
            
        rng.shuffle(candidates)
        correct_idx = next(i for i, c in enumerate(candidates) if np.array_equal(c.features, product.features))
        
        tasks.append({
            "task_id": f"chem_{i}",
            "reaction": rxn,
            "reactant": reactant,
            "product": product,
            "candidates": candidates,
            "correct_idx": correct_idx
        })
        
    return tasks
