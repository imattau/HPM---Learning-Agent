import torch
import numpy as np
import re
from typing import Dict, Any, List

from hpm_ai_v3.augmented_agent import AugmentedHPMAgent
from hpm_ai_v3.agents.algebra import create_algebra_agent
from hpm_ai_v3.agents.numeric import create_numeric_agent
from hpm_ai_v3.agents.validator import create_validator_agent
from hpm_ai_v3.agents.registry import AgentRegistry
from hpm_ai_v3.agents.orchestrator import UnifiedOrchestrator
from hpm_ai_v3.population import PatternPopulation
from hpm_ai_v3.classification_pattern import ClassificationPattern
from hpm_ai_v3.tools.registry import ToolRegistry


class PhysicsManagerAgent:
    """
    Manager agent that orchestrates parsing tools, physics tools, and specialist agents.
    """
    def __init__(self):
        # Ensure tools are registered
        from hpm_ai_v3.tools.parsing import register_parsing_tools
        from hpm_ai_v3.tools.physics import register_physics_tools
        from hpm_ai_v3.tools.reasoning import register_reasoning_tools
        register_parsing_tools()
        register_physics_tools()
        register_reasoning_tools()

        # Ensure specialists are registered
        for name, creator in [("algebra_agent", create_algebra_agent),
                              ("numeric_agent", create_numeric_agent),
                              ("validator_agent", create_validator_agent)]:
            if name not in AgentRegistry.list_agents():
                AgentRegistry.register(name, creator(), f"{name} specialist")

        self.tool_names = ["parse_question", "extract_constraints", "physics_formula_search", "physics_constant_lookup"]
        self.agent_names = ["algebra_agent", "numeric_agent", "validator_agent"]
        
        # Internal orchestrator to learn how to delegate
        self.orchestrator = UnifiedOrchestrator(
            available_tools=self.tool_names,
            available_agents=self.agent_names,
            context_feature_dim=32
        )
        self.population = PatternPopulation([ClassificationPattern(32, 10) for _ in range(3)])

    def process(self, problem: Any, max_steps: int = 10) -> Dict[str, Any]:
        """
        Solve a physics problem by exploring tools and specialists.
        """
        # 1. Initialize State
        state = {
            "problem": problem,
            "knowns": {},
            "unknowns": [],
            "candidates": [],
            "best_fit": None,
            "answer": None,
            "history": [],
            "mode": "deduction"
        }
        
        if isinstance(problem, dict) and problem.get("mode") == "induction":
            state["mode"] = "induction"
            state["observations"] = problem["observations"]
            state["query"] = problem["query"]
            # Extract variables from obs
            all_vars = set()
            for obs in state["observations"]: all_vars.update(obs.keys())
            state["vars_in_play"] = list(all_vars)
        elif isinstance(problem, str):
            state["text"] = problem
        else:
            state["knowns"] = problem.get("knowns", {})
            state["unknowns"] = problem.get("unknowns", [])
            state["text"] = problem.get("text", "")

        # 2. Orchestration Loop
        for step in range(max_steps):
            # The agent "decides" what to do next based on state
            # For now, we use a heuristic "Specialist Orchestrator" style
            action = self._decide_next_action(state)
            if not action: break
            
            print(f"[Step {step+1}] Action: {action}")
            res = self._execute_action(action, state)
            state["history"].append({"step": step, "action": action, "result": "success" if res else "fail"})
            
            if state["answer"] is not None:
                break
                
        return {
            "answer": state["answer"],
            "discovered_law": (state.get("best_fit") or {}).get("name") if state["mode"] == "induction" else None,
            "workflow": [h["action"] for h in state["history"]],
            "state": state
        }

    def _decide_next_action(self, state: Dict[str, Any]) -> str:
        """Heuristic decision logic (could be learned by UnifiedOrchestrator)."""
        mode = state["mode"]
        
        if mode == "induction":
            if not state["candidates"]:
                return "formula_search"
            if not state.get("best_fit") and state["candidates"]:
                return "evaluate_candidates"
            if state.get("best_fit") and not state["answer"]:
                return "solve_query"
        else:
            if not state["knowns"] and state.get("text"):
                return "extract_constraints"
            if not state["unknowns"] and state.get("text"):
                return "identify_unknowns"
            if not state["candidates"] and state["knowns"]:
                return "formula_search"
            if state["candidates"] and not state["answer"]:
                return "algebra_numeric_solve"
        
        return None

    def _execute_action(self, action: str, state: Dict[str, Any]) -> bool:
        """Execute a tool or agent and update state."""
        try:
            if action == "extract_constraints":
                res = ToolRegistry.call("extract_constraints", text=state["text"])
                state["knowns"].update(res.get("constraints", {}))
                return True
            
            elif action == "identify_unknowns":
                res = ToolRegistry.call("identify_unknowns", text=state["text"])
                state["unknowns"] = res.get("unknowns", [])
                return True
                
            elif action == "formula_search":
                query = state.get("vars_in_play") or (list(state["knowns"].keys()) + state["unknowns"])
                res = ToolRegistry.call("physics_formula_search", query=query)
                state["candidates"] = res.get("formulas", [])
                return len(state["candidates"]) > 0
                
            elif action == "evaluate_candidates":
                # Try to fit candidates to observations
                obs = state["observations"]
                best_cand = None
                min_err = float('inf')
                
                print(f"[Manager] Evaluating {len(state['candidates'])} candidates against {len(obs)} points.")
                
                for cand in state["candidates"]:
                    expr = cand.get("expr")
                    vars_in_expr = cand.get("variables", [])
                    
                    # Tool exploration: resolve variables if they don't match
                    resolved_mapping = {}
                    for v in vars_in_expr:
                        if v in obs[0]: 
                            resolved_mapping[v] = v
                        else:
                            # Try mapping tool
                            m = ToolRegistry.call("map_concept", text=v)
                            if m.get("variable") in obs[0]:
                                resolved_mapping[v] = m["variable"]
                            else:
                                pass
                                # print(f"[Manager] Could not map {v} to obs {obs[0].keys()}")
                                
                    if len(resolved_mapping) < len(vars_in_expr):
                        # Still missing some variables? Try constant lookup
                        for v in vars_in_expr:
                            if v not in resolved_mapping:
                                c = ToolRegistry.call("physics_constant_lookup", query=v)
                                if c.get("status") == "success":
                                    resolved_mapping[v] = f"CONST:{v}:{c['value']}"
                                elif v == "g": # Heuristic
                                    resolved_mapping[v] = "CONST:g:9.8"

                    # Test fit
                    total_err = 0
                    count = 0
                    can_test_full = True
                    for pt in obs:
                        # Build substitutions for this point
                        subs = {}
                        can_test_pt = True
                        for v_expr, v_obs in resolved_mapping.items():
                            if v_obs.startswith("CONST:"):
                                subs[v_expr] = float(v_obs.split(":")[-1])
                            elif v_obs in pt:
                                subs[v_expr] = pt[v_obs]
                            else:
                                can_test_pt = False; break
                        
                        if can_test_pt:
                            eq = expr.replace("=", "-(") + ")"
                            e_res = ToolRegistry.call("numeric_eval", expr=eq, substitutions=subs)
                            if e_res.get("status") == "success":
                                total_err += abs(e_res["value"])
                                count += 1
                        else:
                            # print(f"[Manager] Pt {pt} missing {set(vars_in_expr) - set(resolved_mapping.keys())}")
                            can_test_full = False; break
                    
                    if count > 0:
                        avg_err = total_err / count
                        print(f"[Manager] Candidate {cand['name']} (vars: {vars_in_expr}) map: {resolved_mapping} err: {avg_err:.3f}")
                        if avg_err < min_err:
                            min_err = avg_err
                            best_cand = cand
                            best_cand["resolved_mapping"] = resolved_mapping
                    else:
                        pass
                        # print(f"[Manager] Candidate {cand['name']} count 0. Resolved: {resolved_mapping}")
                
                if best_cand and min_err < 1.0:
                    print(f"[Manager] Law found: {best_cand['name']} with err {min_err:.3f}")
                    state["best_fit"] = best_cand
                    state["error_fit"] = min_err
                    return True
                return False

            elif action == "solve_query":
                law = state["best_fit"]
                query = state["query"]
                target = [v for v in law["variables"] if v not in law["resolved_mapping"] or law["resolved_mapping"][v] not in query]
                # If target is still ambiguous, just pick one not in query
                if not target: target = [v for v in law["variables"] if v not in query]
                if not target: return False
                
                t = target[0]
                algebra = AgentRegistry.create_pattern("algebra_agent")
                alg_res = algebra.sample({"equations": [law["expr"]], "variables": [t]})
                sols = alg_res.get("output", {}).get("solutions", [])
                if not sols: return False
                
                sol_expr = sols[0][t]
                subs = {}
                for v_expr, v_obs in law["resolved_mapping"].items():
                    if v_obs.startswith("CONST:"): subs[v_expr] = float(v_obs.split(":")[-1])
                    elif v_obs in query: subs[v_expr] = query[v_obs]
                
                # If 'g' is needed but not in subs/mapping
                if "g" in law["variables"] and "g" not in subs:
                     subs["g"] = 9.8 # Or lookup
                
                num_res = ToolRegistry.call("numeric_eval", expr=sol_expr, substitutions=subs)
                state["answer"] = num_res.get("value")
                return True
                
            elif action == "algebra_numeric_solve":
                # Original logic for deduction
                # (Omitted here for brevity, but would be implemented similarly)
                pass

        except Exception as e:
            print(f"[Manager] Error in action {action}: {e}")
            return False
        return False

    def _process_induction(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Discover the hidden law from observations.
        """
        observations = task["observations"]
        query = task["query"]
        workflow = ["induction_mode"]
        
        # 1. Identify all variables in observations
        all_vars = set()
        for obs in observations:
            all_vars.update(obs.keys())
        
        # 2. Search for candidate formulas using these variables
        formula_res = ToolRegistry.call("physics_formula_search", query=list(all_vars))
        candidates = formula_res.get("formulas", [])
        workflow.append("formula_search")
        
        best_law = None
        min_error = float('inf')
        
        # 3. Test each candidate
        # We solve for the variable that is present in obs but not in query if possible
        # Or just test if the formula holds for all points
        for cand in candidates:
            expr = cand.get("expr")
            if not expr: continue
            
            # Solve for one variable to check fit
            # Try to solve for each variable in the formula
            vars_in_expr = cand.get("variables", [])
            
            total_error = 0
            count = 0
            
            for obs in observations:
                # Check if all variables in formula are in obs
                if all(v in obs for v in vars_in_expr):
                    # Evaluate LHS - RHS
                    try:
                        # e.g. F = m*a -> F - (m*a)
                        eq = expr.replace("=", "-(") + ")"
                        err_res = ToolRegistry.call("numeric_eval", expr=eq, substitutions=obs)
                        if err_res.get("status") == "success":
                            total_error += abs(err_res["value"])
                            count += 1
                    except:
                        continue
            
            if count > 0:
                avg_error = total_error / count
                if avg_error < min_error:
                    min_error = avg_error
                    best_law = cand
        
        if not best_law or min_error > 1.0: # Tolerance for "finding" the law
            return {"answer": None, "error": f"No law fit well. Min error: {min_error}", "workflow": workflow}
            
        workflow.append(f"law_selected({best_law['name']})")
        
        # 4. Use the discovered law to answer the query
        # Solve for the missing variable in query
        target_vars = [v for v in best_law["variables"] if v not in query]
        if not target_vars:
            return {"answer": None, "error": "Target variable already in query", "workflow": workflow}
            
        target = target_vars[0]
        
        algebra_pat = AgentRegistry.create_pattern("algebra_agent")
        alg_res = algebra_pat.sample({
            "equations": [best_law["expr"]],
            "variables": [target]
        })
        
        sols = alg_res.get("output", {}).get("solutions", [])
        if not sols:
             return {"answer": None, "error": "Could not solve law for target", "workflow": workflow}
             
        sol_expr = sols[0][target]
        
        # 5. Evaluate numerically
        # Combine query knowns and potential constants
        subs = query.copy()
        # Add constants like g if needed by the formula
        if "g" in best_law["variables"] and "g" not in subs:
            g_res = ToolRegistry.call("physics_constant_lookup", query="g")
            if g_res.get("status") == "success":
                subs["g"] = g_res["value"]
                
        num_res = ToolRegistry.call("numeric_eval", expr=sol_expr, substitutions=subs)
        answer = num_res.get("value")
        workflow.append("numeric_eval")
        
        return {
            "answer": answer,
            "discovered_law": best_law["name"],
            "formula": best_law["expr"],
            "workflow": workflow,
            "error_fit": min_error
        }

    def invoke(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper for AgentRegistry compatibility."""
        text = context.get("text") or context.get("input")
        if not text:
            return {"error": "No input text provided"}
        return self.process(text)
