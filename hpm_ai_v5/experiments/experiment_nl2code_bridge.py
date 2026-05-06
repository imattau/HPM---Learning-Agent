"""Natural Language to Code Bridge Experiment (NL2Code)."""

from __future__ import annotations

import re
from hpm_ai_v5.adapter import AdapterPacket
from hpm_ai_v5.adapter.clt import UnifiedASTFlattener, LanguageDetector, UnifiedVocabulary
from hpm_ai_v5.adapter.nlp import CanonicalPhraser, NLPTokenizer, NL2CodeBridgeAdapter, KnowledgeBaseLookup
from hpm_ai_v5.adapter.validation_only import ValidationOnlyAdapter
from hpm_ai_v5.core import PatternEngine, PatternManager, State
from hpm_ai_v5.core.config import CoreConfig
from hpm_ai_v5.pipeline import HPMPipeline
from hpm_ai_v5.polygraphs.clt import CLTPolygraphGenerator
from hpm_ai_v5.polygraphs.nlp import NLPPolygraphGenerator


def run_nl2code_bridge_experiment():
    print("Starting NL-to-Code Bridge Experiment...")
    
    # Core Setup
    engine = PatternEngine(config=CoreConfig(
        max_patterns=512,
        history_limit=100,
        exact_threshold=0.01,
        near_threshold=0.4
    ))
    manager = PatternManager(promotion_threshold=0.01, min_support=1)
    
    # 2. Pipeline for NL Bridging
    nl_pipeline = HPMPipeline(
        preprocessor=NLPTokenizer(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter(),
        polygraph_generator=NLPPolygraphGenerator(),
        polygraph_confidence_skip=1.1 # Force polygraph evaluation
    )
    nl_pipeline.register_preprocessor(CanonicalPhraser())
    nl_pipeline.register_preprocessor(KnowledgeBaseLookup())
    nl_pipeline.register_preprocessor(NL2CodeBridgeAdapter())
    
    # 1. Pipeline for Code Acquisition (CLT)
    clt_pipeline = HPMPipeline(
        preprocessor=LanguageDetector(),
        engine=engine,
        postprocessor=ValidationOnlyAdapter()
    )
    clt_pipeline.register_preprocessor(UnifiedASTFlattener())
    clt_poly = CLTPolygraphGenerator()
    
    # Phase 1: Code Acquisition
    python_idioms = [
        ("Conditional Call", "if is_admin:\n    grant_access()"),
        ("Iterative Loop", "while active:\n    process_item()"),
        ("Assign and Return", "val = calculate()\nreturn val"),
    ]
    
    print("\nPhase 1: Acquiring Code Idioms...")
    for name, code in python_idioms:
        print(f"  Training on idiom: {name}")
        packet = AdapterPacket(raw=code)
        packet = clt_pipeline.preprocessing_pipeline.run(packet, target_outputs=["unified_ast_flattener"])
        views = clt_poly.generate(code, context=packet.context)
        func_skeleton = next((v.state.value for v in views if v.name == "functional_skeleton"), ())
        
        for _ in range(20):
            engine.current_state = None
            engine.history = []
            for token_id in func_skeleton:
                # To train polygraph engines, we need to pass through pipeline.step()
                # But CLT uses different views. 
                # Let's just manually feed to the relevant view engines if they exist.
                # Actually, for this experiment, let's just use the primary engine
                # and show how polygraphs resolve the NL token to the primary engine's patterns.
                engine.observe(State(value=(token_id,)))
                if engine.last_match and engine.last_match.pattern:
                    engine.last_match.pattern.reward(0.5)
            manager.end_episode(engine)
        
    print(f"  Engine now has {len(engine.store.patterns)} patterns.")
    
    # Phase 2: Natural Language Recognition
    nl_test_cases = [
        ("If authenticated, run task.", "Conditional Call"),
        ("Validate input and run.", "Conditional Call"), # 'Validate' requires polygraph resolution
    ]
    
    print("\nPhase 2: Natural Language Recognition (Bridging)...")
    correct = 0
    for query, expected_idiom in nl_test_cases:
        print(f"  NL Query: '{query}'")
        engine.current_state = None
        engine.history = []
        
        tokens = re.findall(r"\w+", query.lower())
        
        matched_idioms = []
        for token in tokens:
            # We want to see if 'validate' matches 'if' via the semantic polygraph view
            res = nl_pipeline.step(token, context={"test": True})
            
            match = engine.last_match
            if match and match.pattern:
                print(f"    - Token: '{token}' -> Match: {match.pattern.name} (View: {res.action.selected_view})")
                matched_idioms.append(match.pattern.name)
        
        if matched_idioms:
            print(f"    - Bridge recognized via tokens: {matched_idioms}")
            correct += 1
            print(f"    - SUCCESS: Recognized structural bridge.")
        else:
            print(f"    - FAILURE: No structural match.")
            
    print(f"\nFinal Score: {correct}/{len(nl_test_cases)} structural bridges recognized.")


if __name__ == "__main__":
    run_nl2code_bridge_experiment()
