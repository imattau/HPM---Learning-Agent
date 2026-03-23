"""Molecular Simulator for Chem-Logic Benchmark (SP12).
Uses RDKit for functional group identification and valence checking.
"""

import numpy as np
from dataclasses import dataclass
from rdkit import Chem

# Functional Group Registry (SMARTS patterns for L2 vectors)
# We use standard SMARTS for common organic functional groups
SMARTS = {
    "hydroxyl": "[OX2H]",                # Alcohol
    "aldehyde": "[CX3H1](=O)",           # Aldehyde
    "ketone": "[#6][CX3](=O)[#6]",       # Ketone
    "carboxyl": "[CX3](=O)[OX2H1]",      # Carboxylic acid
    "ester": "[CX3](=O)[OX2H0][#6]",     # Ester
    "ether": "[#6][OD2][#6]",            # Ether
    "alkane": "[CX4]"                    # Saturated carbon
}

GROUPS = {name: i for i, name in enumerate(SMARTS.keys())}

@dataclass
class Molecule:
    smiles: str
    features: np.ndarray # Multi-hot encoding of groups
    is_valid: bool = True

def get_molecule(smiles: str) -> Molecule:
    """RDKit-based molecular parser and feature extractor."""
    mol = Chem.MolFromSmiles(smiles)
    feats = np.zeros(len(GROUPS))
    
    if mol is None:
        # Invalid SMILES or valence error
        return Molecule(smiles, feats, is_valid=False)
    
    # Check validity using RDKit's built-in sanitizer
    try:
        Chem.SanitizeMol(mol)
        valid = True
    except:
        valid = False
        
    # Extract functional groups using substructure matching
    for name, smarts in SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            feats[GROUPS[name]] = 1
            
    return Molecule(smiles, feats, valid)

def generate_chem_tasks(n_tasks: int = 20, seed: int = 42) -> list[dict]:
    """Generate reactant-product pairs for hidden law inference."""
    rng = np.random.default_rng(seed)
    tasks = []
    
    reactions = ["oxidation", "reduction", "esterification"]
    
    for i in range(n_tasks):
        rxn = rng.choice(reactions)
        
        if rxn == "oxidation":
            # Ethanol -> Acetaldehyde
            reactant = get_molecule("CCO")
            product = get_molecule("CC=O")
        elif rxn == "reduction":
            # Acetone -> Isopropanol
            reactant = get_molecule("CC(=O)C")
            product = get_molecule("CC(O)C")
        elif rxn == "esterification":
            # Methanol + Acetic Acid -> Methyl Acetate
            reactant = get_molecule("CO.CC(=O)O")
            product = get_molecule("CC(=O)OC")
        else:
            reactant = get_molecule("CC")
            product = get_molecule("CC")
            
        # Create candidates
        candidates = [product]
        for _ in range(4):
            # Wrong candidate: random functional group flip
            wrong_feats = product.features.copy()
            idx = rng.integers(0, len(GROUPS))
            wrong_feats[idx] = 1 - wrong_feats[idx]
            # Use a mock SMILES for wrong candidates as they are feature-based discriminators
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
