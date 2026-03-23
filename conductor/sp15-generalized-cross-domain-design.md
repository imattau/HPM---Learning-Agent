# Sub-project 15: Generalized Cross-Domain Alignment — Bridging the Symbolic Gap

## Goal
Extend the cross-domain transfer capabilities of the HPM framework. SP10 established that relational delta alignment works for continuous domains (PhyRE, ARC) but fails on symbolic ones (Math). SP15 aims to bridge this gap by incorporating the new "Boss Fight" domains (DS-1000, Chem-Logic, Linguistic) and implementing a multi-domain alignment strategy.

## Hypotheses
1.  **Symbolic Coherence**: Aligning Math with other symbolic domains (DS-1000, Chem-Logic) will yield higher transfer accuracy than aligning it with perceptual domains (PhyRE, ARC).
2.  **Multi-Domain Reference**: Using a shared "Reference Space" (Generalized Procrustes Analysis) to align 3+ domains simultaneously will produce a more robust relational operator than pairwise alignment.
3.  **Register Portability**: The "Register Shift" logic from Linguistic SP14 can be aligned with the "pH Shift" logic in Chem-Logic SP13, revealing a domain-agnostic "Latent Shift" pattern.

## New Architecture

### 1. Generalized Alignment Script (`benchmarks/multi_domain_alignment.py`)
- **Padded Dimension**: Increased to **32** to accommodate higher-dimensional symbolic encoders.
- **Domains**: `math`, `phyre`, `arc`, `ds1000`, `chem`, `linguistic`.
- **Algorithm**:
    - Compute $M_d$ for each domain.
    - Select a "Reference Domain" (e.g., ARC) and align all others to it via Procrustes.
    - Compute a **Global Relational Operator** $M_{global} = \text{mean}(R_d^T M_d)$.

### 2. Cross-Domain "Boss Fight"
- **Transfer Scenarios**:
    - `[DS-1000 + Chem] -> Math`: Testing the symbolic-to-symbolic transfer.
    - `[PhyRE + ARC + DS-1000] -> Chem`: Testing if diverse grounding improves chemical reasoning.
    - `[Linguistic] -> [Chem (pH)]`: Testing the transfer of latent-variable surprise detection.

## Success Criteria
- **Math Recovery**: `delta_alignment` accuracy on Math should increase by >10pp compared to SP10 by using symbolic source domains.
- **Global Stability**: The Global Relational Operator should maintain >80% of within-domain baseline performance across all continuous domains.
- **Surprise Transfer**: L5 correctly identifies surprise in a new domain using a threshold learned in a different domain.
