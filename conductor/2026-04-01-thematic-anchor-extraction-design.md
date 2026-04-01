# SP20: Thematic Anchor Extraction — Design Specification

## 1. Overview and Rationale

The **Thematic Anchor Extraction** experiment (SP20) tests the HPM AI's capacity for "Structural Summarization." While the system cannot yet generate natural language prose, it can perform deep information bottlenecking. 

By passing a continuous stream of text (e.g., a chapter of *Peter Rabbit*) through the multi-process hierarchy, we can test if the system naturally "compresses" the narrative into a handful of dominant, high-level HFN nodes (the **"Anchors"**).

## 2. The Theoretical Premise: "Temporal Co-occurrence"

In previous experiments, "Stereo Vision" occurred instantaneously across domains. In SP20, we test **Temporal Stereo Vision**: the binding of concepts across time. 
If the words "Peter", "ran", and "McGregor" activate in close sequence repeatedly across a page, the Level 2 (L2) Relational Specialist should compress them into a single "Narrative Event" node. The summary of the page is simply the list of L2 nodes with the highest activation weights at the end of the run.

## 3. The Multi-Tier Architecture

The system uses a two-tier Governor-Worker model, optimized for continuous sequence processing.

### 3.1 Level 1: The Perceptual Specialists
*   **Worker 1: Lexical Specialist** (Degree 1.0)
    *   **Focus**: Pure vocabulary mapping.
    *   **Priors**: Pre-populated with the Peter Rabbit vocabulary (from `build_nlp_world_model`).
    *   **Output**: The $\mu$ of the specific word node.
*   **Worker 2: Syntactic/POS Specialist** (Degree 0.5)
    *   **Focus**: Parts of speech and transition probabilities.
    *   **Priors**: Basic POS embeddings (Noun, Verb, Adjective).
    *   **Output**: The $\mu$ of the POS node.

### 3.2 Level 2: The Relational "Summarizer"
*   **Worker 3: Thematic Synthesizer** (Degree 0.0)
    *   **Input**: A rolling window (e.g., $N=5$ tokens) of L1 winners.
    *   **Role**: Discovers N-grams and relational motifs.
    *   **Dynamics**: Uses high plasticity to absorb recurring sequences into single "Phrase Nodes" or "Event Nodes".

## 4. The Saliency Engine (The Governor)

Instead of just routing data, the Governor now acts as a **Saliency Engine**. 

1.  **Continuous Ingestion**: The Governor reads the `peter_rabbit.txt` corpus sentence by sentence.
2.  **Tiered Routing**: It passes tokens to L1, collects the winners, and passes the rolling window to L2.
3.  **Global Weight Decay**: Crucially, the Governor applies a slow time-decay to all node weights. This ensures that "Anchors" are not just common words ("the", "and") but structurally significant concepts that are reinforced continuously throughout the specific page.

## 5. The "Summary" Extraction Protocol

At the end of the text stream, the Governor extracts the summary by querying the L2 Synthesizer.

**The Filter Criteria for Anchors:**
1.  **High Weight**: Must be in the top 95th percentile of activation weights.
2.  **High Complexity**: Must have a high `description_length` (i.e., it is a compressed combination of multiple L1 concepts, not just a single word).
3.  **Cross-Domain Binding**: Ideally, it should link a Lexical identity to a Syntactic structure.

**The Output Format:**
The system will print the top $K$ Anchor Nodes. Since we don't have a natural language decoder, the Governor will "Reverse Engineer" the nodes by querying the L1 specialists to find which words are closest to the components of the L2 Anchor's $\mu$.

*Example Output:*
`Anchor 1 (Weight 0.94): [Subject: Peter] + [Action: Run] + [Location: Garden]`

## 6. Evaluation Metrics

1.  **Compression Ratio**: Ratio of total tokens processed to the number of final Anchor nodes holding >50% of the total system weight.
2.  **Stop-Word Rejection**: Does the system successfully demote high-frequency, low-meaning words (the, and) in favor of high-meaning entities (Peter, McGregor)?
3.  **Narrative Coherence**: Do the extracted L2 anchors map conceptually to the actual plot of the text?

## 7. Implementation Roadmap

1.  **Data Streamer**: Create a streaming loader for `data/corpus/peter_rabbit.txt` that yields rolling token windows.
2.  **Rolling Window Vectorizer**: Map a sequence of 5 words into a single fixed-width spatial vector for L2 observation.
3.  **Saliency Decay**: Implement the slow weight decay mechanism in the `Observer` or the Governor loop.
4.  **Reverse-Mapping Logic**: Write the function that takes an L2 node's $\mu$ and prints the human-readable words it represents.
5.  **Experiment Script**: Assemble `experiment_thematic_anchor.py`.
