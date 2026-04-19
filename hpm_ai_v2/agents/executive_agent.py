"""
ExecutiveAgent: orchestrates multi-step learning workflows (Study -> Exam -> Research -> Resit).
Provides the 'planning' and 'attention' logic for the agent society.
"""
from __future__ import annotations
import time
from typing import List, Dict, Optional, Any, TYPE_CHECKING
import numpy as np
from hfn.hfn import HFN
from hpm_ai_v2.agents.coordinator_agent import CoordinatorAgent

if TYPE_CHECKING:
    from hpm_ai_v2.agents.reader_agent import ReaderAgent
    from hpm_ai_v2.agents.writer_agent import WriterAgent
    from hpm_ai_v2.agents.librarian_agent import LibrarianAgent

class ExecutiveAgent(CoordinatorAgent):
    """
    The 'CEO' of the agent society.
    Decomposes high-level goals into sub-tasks for specialist agents.
    """
    def __init__(self, config, reader_agent: ReaderAgent, writer_agent: WriterAgent, librarian_agent: LibrarianAgent, **kwargs):
        super().__init__(config, **kwargs)
        self.reader = reader_agent
        self.writer = writer_agent
        self.librarian = librarian_agent
        self.gaps: List[Dict[str, Any]] = []
        
        # Register specialists for Coordinator methods
        self.specialists["reader"] = reader_agent
        self.specialists["writer"] = writer_agent
        self.specialists["librarian"] = librarian_agent

    def run_exam_workflow(self, corpus: Dict[str, str], questions: List[Dict[str, Any]], grader_func) -> Dict[str, Any]:
        """
        Full autonomous learning cycle:
        1. Read initial corpus.
        2. Take exam.
        3. Identify knowledge gaps from failures.
        4. Targeted research (Librarian + Web).
        5. Resit exam.
        """
        print("\n[EXECUTIVE] Starting Autonomous Learning Workflow...")
        
        # 1. Study phase
        print("\nPhase 1: Initial Study Phase...")
        for title, content in corpus.items():
            print(f"  [READING] {title}")
            self.reader.ingest_text(content, title=title)
            
        # 2. Initial Exam
        print("\nPhase 2: Initial Exam Assessment...")
        results = []
        score = 0
        for q in questions:
            print(f"  Q: {q['question']}")
            answer = self.writer.answer_natural(q['question'])
            passed = grader_func(answer, q)
            results.append({"q": q, "a": answer, "passed": passed})
            if passed: 
                score += 1
                # [HPM CORE] Reinforce specialists for success
                self.writer.reinforce(f"search_query_{q['id']}", reward=0.2)
                self.reader.reinforce(f"topic_{q.get('chapter', 'general').lower()}", reward=0.1)
            else:
                self.gaps.append(q)
            print(f"    - A: {answer} [{'PASS' if passed else 'FAIL'}]")
            
        print(f"\n[EXECUTIVE] Initial Score: {score}/{len(questions)}")
        
        if score == len(questions):
            print("[EXECUTIVE] All questions passed. No remedial action needed.")
            return {"final_score": score, "initial_score": score, "gaps_filled": 0}

        # 3. Remedial Research
        print(f"\nPhase 3: Remedial Research ({len(self.gaps)} gaps identified)...")
        gaps_filled = 0
        for gap in self.gaps:
            print(f"  [GAP] Analyzing: '{gap['question']}'")
            
            # A: Librarian search for context
            internal_sources = self.librarian.search_knowledge(gap['question'])
            remedial_doc = None
            for res in internal_sources:
                if res.relation_type == "document":
                    remedial_doc = res
                    break
                # If it's a topic, we could search for docs about it, 
                # but Librarian.search_knowledge now returns passages too.
            
            if remedial_doc:
                title = remedial_doc.metadata.get("title")
                print(f"    - Librarian found internal context: {title}")
                # Re-reading might trigger better hierarchical patterns
                self.reader.ingest_text(corpus.get(title, ""), title=title)
                
            # B: Web search for missing facts
            print(f"    - Triggering Web Research...")
            # Here we use Writer's request_knowledge which uses WebAgent
            success = self.writer.request_knowledge(gap['question'])
            if success:
                gaps_filled += 1
                print(f"    - Research successful for gap.")
            else:
                print(f"    - Research failed to find new info.")

        # 4. Resit Phase
        print("\nPhase 4: Exam Resit...")
        final_score = 0
        for q in questions:
            print(f"  Q: {q['question']}")
            answer = self.writer.answer_natural(q['question'])
            passed = grader_func(answer, q)
            if passed: final_score += 1
            print(f"    - A: {answer} [{'PASS' if passed else 'FAIL'}]")
            
        print(f"\n[EXECUTIVE] Final Score: {final_score}/{len(questions)}")
        improvement = final_score - score
        print(f"[EXECUTIVE] Workflow complete. Improvement: +{improvement}")
        
        return {
            "initial_score": score,
            "final_score": final_score,
            "improvement": improvement,
            "gaps_filled": gaps_filled
        }
