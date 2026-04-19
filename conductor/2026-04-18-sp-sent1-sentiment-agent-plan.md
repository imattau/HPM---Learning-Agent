# Plan: SP-Sent1 – SentimentAgent Implementation

## Objective
Implement a specialized `SentimentAgent` for HFN-native emotion and opinion analysis. This agent will learn to classify sentiment (positive, negative, neutral) and emotional intensity from a few examples by composing macros from L1 primitives.

## Background & Motivation
Sentiment analysis in HPM should be interpretable and data-efficient. Instead of opaque embeddings, `SentimentAgent` will use explicit, composable rules (macros) grounded in a lexical dictionary. This aligns with the HPM principles of hierarchical abstraction and structural uniformity.

## Scope & Impact
- **Manifold Design**: Create `SentimentDomainConfig` to define the semantic space for sentiment concepts.
- **Agent Implementation**: Implement `SentimentAgent` with L1 primitives for lexicon scoring, negation/intensifier detection, and classification.
- **Integration**:
    - **DictionaryAgent**: Seed with sentiment scores for foundational words.
    - **ReaderAgent**: Reuse sentence and word macros for sentiment targets.
    - **AgentOrchestrator**: Register for dimension synchronization.
    - **WriterAgent**: Enable natural language sentiment reporting and explanations.
- **Persistence**: Ensure all learned macros and sentiment nodes are persisted in `data/sentiment_forest` without deletion between runs.

## Proposed Solution
1. **SentimentDomainConfig**: Define concepts like `LEXICON_SCORE`, `IS_NEGATION`, `IS_INTENSIFIER`, `SENTENCE_SCORE`, `NORMALIZE`, and `CLASSIFY`.
2. **SentimentAgent**:
    - `primitive_lexicon_score`: Query `DictionaryAgent` for a word's sentiment metadata.
    - `primitive_is_negation`: Identify "not", "never", etc.
    - `primitive_classify`: Map numerical scores to "positive", "negative", or "neutral".
    - `detect_sentiment(sentence)`: Apply a learned macro to a sentence HFN node.
    - Link results to the target node via `"has_sentiment"` edges.
3. **Macro Learning**: Use a few examples (k=2) to discover a macro that correctly predicts sentiment by combining primitives.
4. **Writer Enrichment**: Update `WriterAgent`'s `answer_natural` to check for `"has_sentiment"` nodes when asked about opinions or feelings.

## Implementation Steps
- [ ] Create `hpm_ai_v2/domains/sentiment_domain.py`.
- [ ] Create `hpm_ai_v2/agents/sentiment_agent.py`.
- [ ] Update `hpm_ai_v2/agents/writer_agent.py` and `WriterMixin` to support sentiment reporting.
- [ ] Create `hpm_ai_v2/experiments/experiment_sp_sent1_few_shot.py` with persistent storage.
- [ ] Verify the agent correctly handles negations ("not good") and intensifiers ("really love").

## Verification & Testing
- Run `experiment_sp_sent1_few_shot.py`.
- **Test 1: Simple Sentiment**: "I love this product!" → positive.
- **Test 2: Negation**: "This is not good." → negative.
- **Test 3: Intensification**: "I really hate it." → negative (high intensity).
- **Test 4: Neutral**: "The box is blue." → neutral.
- **Test 5: Persistence**: Run the experiment twice and ensure the second run reuses the learned macro from the first run.
- **Test 6: Explanation**: `WriterAgent` should explain: "The sentiment is positive because of the word 'love'."
