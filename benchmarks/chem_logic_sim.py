"""Molecular Simulator for Chem-Logic Benchmark (SP12/13).
Uses RDKit for functional group identification and valence checking.
Introduces Ambiguity and Competitive Logic for SP13.
"""

import numpy as np
from dataclasses import dataclass
from rdkit import Chem

# Functional Group Registry (SMARTS patterns for L2 vectors)
SMARTS = {
    "amine": "[NX3H2,NX3H1,NX3H0]",      # Amine (Highest priority)
    "hydroxyl": "[OX2H]",                # Alcohol
    "aldehyde": "[CX3H1](=O)",           # Aldehyde
    "ketone": "[#6][CX3](=O)[#6]",       # Ketone
    "carboxyl": "[CX3](=O)[OX2H1]",      # Carboxylic acid
    "ester": "[CX3](=O)[OX2H0][#6]",     # Ester
    "ether": "[#6][OD2][#6]",            # Ether
}

# Reactivity Priority (High = Reacts first)
PRIORITY = {
    "amine": 10,
    "hydroxyl": 5,
    "ketone": 3,
    "carboxyl": 2,
    "aldehyde": 1,
}

GROUPS = {name: i for i, name in enumerate(SMARTS.keys())}

@dataclass
class Molecule:
    smiles: str
    features: np.ndarray # Multi-hot encoding of groups
    is_valid: bool = True
    ph_sensitive: bool = False # Whether it protonates

def get_molecule(smiles: str) -> Molecule:
    """RDKit-based molecular parser and feature extractor."""
    mol = Chem.MolFromSmiles(smiles)
    feats = np.zeros(len(GROUPS))
    
    if mol is None:
        return Molecule(smiles, feats, is_valid=False)
    
    try:
        Chem.SanitizeMol(mol)
        valid = True
    except:
        valid = False
        
    for name, smarts in SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            feats[GROUPS[name]] = 1
            
    return Molecule(smiles, feats, valid, ph_sensitive=("N" in smiles))

def generate_chem_tasks(n_tasks: int = 20, seed: int = 42) -> list[dict]:
    """Standard SP12 tasks (Deterministic)."""
    rng = np.random.default_rng(seed)
    tasks = []
    reactions = ["oxidation", "reduction", "esterification"]
    for i in range(n_tasks):
        rxn = rng.choice(reactions)
        if rxn == "oxidation":
            reactant, product = get_molecule("CCO"), get_molecule("CC=O")
        elif rxn == "reduction":
            reactant, product = get_molecule("CC(=O)C"), get_molecule("CC(O)C")
        elif rxn == "esterification":
            reactant, product = get_molecule("CO.CC(=O)O"), get_molecule("CC(=O)OC")
        else:
            reactant, product = get_molecule("CC"), get_molecule("CC")
        
        candidates = [product]
        for _ in range(4):
            wrong_feats = product.features.copy()
            idx = rng.integers(0, len(GROUPS))
            wrong_feats[idx] = 1 - wrong_feats[idx]
            candidates.append(Molecule("MOCK_SMILES", wrong_feats, rng.choice([True, False])))
        rng.shuffle(candidates)
        correct_idx = next(i for i, c in enumerate(candidates) if np.array_equal(c.features, product.features))
        tasks.append({"task_id": f"chem_sp12_{i}", "reaction": rxn, "reactant": reactant, "product": product, "candidates": candidates, "correct_idx": correct_idx})
    return tasks

def generate_ambiguous_chem_tasks(n_tasks: int = 20, seed: int = 42) -> list[dict]:
    """SP13 tasks: Competitive Inhibition and Latent pH."""
    rng = np.random.default_rng(seed)
    tasks = []
    
    for i in range(n_tasks):
        # 50% chance of Competition, 50% chance of Latent pH
        scenario = rng.choice(["competition", "latent_ph"])
        
        if scenario == "competition":
            # Molecule with both Amine and Hydroxyl. Reagent: Acyl Chloride (Acetylation)
            # Amine has priority 10, Hydroxyl has priority 5.
            reactant = get_molecule("NCCCO") # 3-amino-1-propanol
            # Product should be N-acetylated (Amine wins)
            product = get_molecule("CC(=O)NCCCO")
            latent = {"type": "competition", "priority": "amine > hydroxyl"}
        else:
            # Latent pH: Amine protonation
            # If pH is low (Latent=1), Amine is protonated and cannot react.
            # If pH is high (Latent=0), Amine reacts normally.
            ph_is_low = rng.choice([True, False])
            reactant = get_molecule("N") # Ammonia
            if ph_is_low:
                # Protonated: No reaction or different outcome
                product = get_molecule("[NH4+]") 
            else:
                # Normal reaction (e.g. to Methylamine if reacting with MeI)
                product = get_molecule("CN")
            latent = {"type": "ph", "ph_is_low": ph_is_low}
            
        # Create candidates
        candidates = [product]
        
        # Add a "Wrong Intuition" lure
        if scenario == "competition":
            # Lure: Acetylated Hydroxyl (wrong priority)
            lure = get_molecule("NCCCOC(=O)C")
            candidates.append(lure)
        else:
            # Latent pH lure: If correct is protonated, lure is normal reaction. If correct is normal, lure is protonated.
            if ph_is_low: lure = get_molecule("CN") # Normal reaction lure
            else: lure = get_molecule("[NH4+]") # Protonated lure
            candidates.append(lure)

        for _ in range(3): # Fill rest with randoms
            wrong_feats = product.features.copy()
            idx = rng.integers(0, len(GROUPS))
            wrong_feats[idx] = 1 - wrong_feats[idx]
            candidates.append(Molecule("MOCK_SMILES", wrong_feats, True))
        rng.shuffle(candidates)
        correct_idx = next(idx for idx, c in enumerate(candidates) if np.array_equal(c.features, product.features))
        
        tasks.append({
            "task_id": f"chem_sp13_{i}",
            "scenario": scenario,
            "reactant": reactant,
            "product": product,
            "candidates": candidates,
            "correct_idx": correct_idx,
            "latent": latent # Hidden from the agent's sensory encoders
        })
    return tasks
