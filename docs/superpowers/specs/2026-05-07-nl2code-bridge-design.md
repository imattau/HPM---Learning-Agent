# Natural Language to Code Bridge (NL2Code) Design (SP54+)

## Overview
The **NL-to-Code Bridge** demonstrates HPM v5's ability to perform **Cross-Modal Structural Transfer**. It allows the engine to learn structural logic in one domain (e.g., Python source code) and recognize that same logic when expressed in another domain (e.g., Natural Language), without requiring paired translation data.

## Lessons Learnt from Development

### 1. The Universal Structural Latent Space
Transfer between language and code is achieved not by translation, but by alignment to a shared **UnifiedVocabulary**.
- **Design Resolution:** By mapping linguistic concepts (e.g., "if", "check") and code constructs (e.g., `if_statement`) to the same Universal IDs (e.g., `U_IF`), the HPM core treats them as identical structural events. This enables zero-shot recognition of intent as logic.

### 2. Polygraphs as Semantic Hypothesis Testers
Polygraphs are used to resolve ambiguity in natural language by testing multiple "hypotheses" of a word's meaning.
- **Design Resolution:** The **Semantic Polygraph View** leverages an external Knowledge Base (KB) to expand novel words into candidate synonyms. The engine selects the view that maximizes the match confidence against known structural patterns, effectively "snapping" a new word into a known functional slot.

### 3. Sequential Delta Learning vs. Static Snapshots
HPM is natively a delta-based sequence engine. Structural idioms must be learned as transitions.
- **Design Resolution:** To learn a code idiom like `if -> call`, the engine must observe sequential deltas. Feeding structural skeletons as sequential `observe()` calls allows the engine to build a transition matrix of idioms, whereas feeding them as single tuples makes them harder to recognize in variable-length natural language.

### 4. Functional Skeletons as Minimal Meaning Units
Effective bridging requires filtering out "surface noise" from both code and language.
- **Design Resolution:** The **Functional Skeleton View** filters for high-signal constructs only (`U_IF`, `U_WHILE`, `U_CALL`, `U_ASSIGN`). By stripping away variable names, punctuation, and specific syntax, the engine focuses on the "DNA" of the intent, reaching near 100% recognition accuracy across paraphrases.

### 5. Grounding Meaning via Utility
The meaning of a linguistic structure is "grounded" by the utility of the code it invokes.
- **Design Resolution:** HPM **Rewards** are applied to structural patterns that lead to successful tool/code execution. This turns language acquisition into a reinforcement learning problem, where the engine learns that the structure of "Validate X" is synonymous with "If X is valid" because they both trigger the same useful functional path.

## Architectural Components

### NL2CodeBridgeAdapter
- **Function:** Maps functional NL tokens to `UnifiedVocabulary` structural IDs.
- **Role:** Converts a linguistic "intent skeleton" into a numeric sequence compatible with patterns learned from code ASTs.

### KnowledgeBaseLookup
- **Function:** Simulates external dictionary/synonym expansion.
- **Role:** Provides the "Semantic Candidates" required for the Polygraph-based synonym resolution.

### UnifiedVocabulary (CLT)
- **Function:** Maintains stable numeric IDs for all universal structural constructs.
- **Role:** Ensures that "If" in English and `if` in Python occupy the same coordinate in the pattern store.
