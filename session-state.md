# Session State Checkpoint
Generated: 2026-05-08
Branch: hpm-ai-v5 (24 commits ahead of origin)

## Current Task
ATIS benchmark full-corpus run in progress (background task b2o4626dt).
Expected output: /tmp/claude-1000/-home-mattthomson-workspace-HPM---Learning-Agent/ad7394c5-0eeb-4983-aefd-b34849f43863/tasks/b2o4626dt.output

## What Was Done This Session

### SNLP Benchmark
- Fixed T2/T3/T4/T5 validity (was measuring wrong things)
- Added SkeletonNgramAdapter + skeleton_bigram_view
- Switched to en_core_web_sm, mapped AUX/PART in skeleton
- Fixed T4 to use match status (not decaying confidence)
- Task isolation via PatternStore reset between tasks
- All passes: T1~83%, T2~100%, T3=100%, T4~80% AUC, T5=100%

### PatternVariant Consolidation
- PatternVariant dataclass + make_variant (core/variant.py)
- CoreConfig: consolidation_threshold (0.8), consolidation_distance (new, separate from near_threshold)
- PatternStore: variants dict, register_variant(), variant fallback in match() with status="variant"
- PatternManager._consolidate_variants() uses consolidation_distance at episode end
- All committed and tested

### ATIS Benchmark
- load_atis() reads from data/atis/atis_train.csv + atis_test.csv (local)
- Pipeline: NLPTokenizer → StartOfEpisodeAdapter → CanonicalPhraser → NamedEntityCanonicaliser → SkeletonExtractor → SkeletonNgramAdapter → KnowledgeBaseLookup → IntentLabelAdapter
- NamedEntityCanonicaliser: replaces GPE/ORG/DATE/LOC entities with type tag
- KnowledgeBaseLookup: WordNet-backed (replaces static dict)
- Intent prediction: flat dict pattern_name→intent, vote across all view engines
- B2 now uses NER entity novelty criterion (not raw word-set)
- Adapter reset between utterances (SNLP lesson applied)
- near_threshold=1.5, consolidation_distance=0.8
- Training cap removed (now trains on all 4978)

### Last benchmark result (1000-item run)
- B1: 69.1% (target >60%) ✓
- B2: 65.5% (target >70%) close
- B3: 1.6% reduction, 7 variants (target >30%) still gap
- B4: 0% (target >0%) still gap

### Documentation
- Design notes consolidated → hpm_ai_v5/design.md (1750 lines)
- API reference split: API_REFERENCE.md, API_ADAPTERS.md, API_AGENTS.md, API_POLYGRAPHS.md
- SCB spec + plan committed

## Pending: When background task completes
Check: cat /tmp/claude-1000/-home-mattthomson-workspace-HPM---Learning-Agent/ad7394c5-0eeb-4983-aefd-b34849f43863/tasks/b2o4626dt.output | tail -20

Expected improvements with full corpus:
- B1 should rise toward 80%
- B3 should show more variants (more near-duplicates within class)
- B4 may activate if variants are created and matched

If B3/B4 still flat, the issue is that ATIS skeleton space is too diverse even within classes — next step would be intent-group-aware clustering (only cluster patterns with same predicted intent).
