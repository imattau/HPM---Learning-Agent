# SP21: Toddler Language Acquisition — Design Specification

## 1. Overview and Rationale

The previous experiment (SP20: Thematic Anchors) attempted to force the system to extract complex narrative motifs from an unstructured text stream. If we view the current HPM AI through the lens of developmental psychology (a 2-year-old child), SP20 was asking a toddler to write a book report.

A 2-year-old doesn't extract "themes." A 2-year-old performs **Categorical Grounding** and **Basic Syntax Discovery**. They learn that "dog," "cat," and "bird" are all *Animals*, and that *Animals* usually *do things* (Actions).

**SP21** steps back to an age-appropriate milestone. It tests the system's ability to discover basic syntactic categories (Noun, Verb) and simple relational pairs (Agent-Action) from a highly structured, repetitive "child-directed" corpus.

## 2. The Multi-Process Architecture

We will use a 3-process architecture: two "perceptual" baselines and one "synthesizer."

### 2.1 Worker 1: The Lexical Observer (Degree 1.0)
*   **Focus**: Word recognition.
*   **Priors**: The 107-token vocabulary.
*   **Manifold**: 107D one-hot space.
*   **Role**: Identifies the specific word (e.g., "dog").

### 2.2 Worker 2: The Ontological Observer (Degree 0.5)
*   **Focus**: Category recognition.
*   **Priors**: Seeded with a few core semantic categories (e.g., `[Animal]`, `[Action]`, `[Food]`).
*   **Manifold**: A smaller "Semantic Space" (e.g., 20D) where related words are manually clustered.
*   **Role**: Identifies the *kind* of thing (e.g., "This is an Animal").

### 2.3 Worker 3: The Toddler Synthesizer (L2, Degree 0.0)
*   **Focus**: Basic grammar discovery.
*   **Input**: The concatenated winners from Worker 1 and Worker 2.
*   **Role**: Discovers simple 2-word rules like `[Animal] -> [Action]` or `[Person] -> ate -> [Food]`.
*   **Dynamics**: Uses high plasticity to absorb recurring pairs into stable "Grammar Nodes."

## 3. The "Child-Directed" Dataset

Instead of the complex prose of *Peter Rabbit*, we will use the highly structured, repetitive `generate_sentences()` function from `nlp_loader.py`.

This dataset contains sentences like:
*   "The dog barked."
*   "The cat meowed."
*   "Mum ate the apple."
*   "Dad ate the bread."

This is the exact type of data a toddler uses to infer the rule: `[Family Member] -> ate -> [Food]`.

## 4. Evaluation Metrics

1.  **Category Discovery**: Does the Ontological Observer successfully cluster new, unseen words into the correct category (e.g., placing "rabbit" near "dog" based on context)?
2.  **Grammar Stabilization**: Does the Toddler Synthesizer create distinct nodes for the core grammatical templates (e.g., a node for the `[Animal] [Action]` pattern)?
3.  **Generalization**: If the system learns `[Animal] [Action]` using "dog barked" and "cat meowed," does the L2 node fire correctly when it sees the novel sentence "bird chirped"?

## 5. Implementation Roadmap

1.  **Ontological Loader**: Create a function to map words to a denser, 20D semantic space based on their category in `_WORD_TO_CATEGORY`.
2.  **Worker Configs**: Define the three specialized workers.
3.  **Experiment Loop**: Implement `experiment_toddler_language.py` using the `generate_sentences` corpus, routing data to L1s and synthesizing in L2.
