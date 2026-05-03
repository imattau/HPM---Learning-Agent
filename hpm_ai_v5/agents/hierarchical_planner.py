"""Hierarchical Planning Agent for multi-step prerequisite chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .base import AgentInput, AgentOutput, BaseAgent
from ..core import Action, PatternEngine
from ..pipeline import PipelineResult


@dataclass
class HierarchicalPlanningAgent(BaseAgent):
    """Agent that manages a goal stack and decomposes goals into subgoals."""

    goal_stack: list[str] = field(default_factory=list)
    prerequisite_map: dict[str, list[str]] = field(default_factory=dict)
    engines: dict[str, PatternEngine] = field(default_factory=dict)
    completion_checker: Callable[[str, Any], bool] | None = None

    def push_goal(self, goal: str) -> None:
        """Add a goal to the stack, decomposing if prerequisites are known."""
        # For a simple stack-based planner, we might want to check if the goal 
        # is already met, but usually the outer loop handles that.
        prereqs = self.prerequisite_map.get(goal, [])
        # To handle LIFO properly, we push the main goal first, then its prereqs 
        # so that the first prereq is at the top.
        self.goal_stack.append(goal)
        for prereq in reversed(prereqs):
            self.push_goal(prereq)

    def _check_top_goal(self, last_output: Any) -> bool:
        """Check if the top goal on the stack is completed."""
        if not self.goal_stack:
            return True
        
        top_goal = self.goal_stack[-1]
        if self.completion_checker:
            if self.completion_checker(top_goal, last_output):
                self.goal_stack.pop()
                return True
        return False

    def decide(self) -> dict[str, Any]:
        """Override decide to manage the goal stack before calling pipeline."""
        last_input = self.state.get("last_input")
        if last_input is None:
            self.last_decision = {"decision": None, "packet": None}
            return self.last_decision

        # 1. Check if the top goal was met by the current observation
        current_raw = last_input.raw
        while self.goal_stack and self._check_top_goal(current_raw):
            pass

        if not self.goal_stack:
            self.last_decision = {"decision": None, "packet": None, "status": "all_goals_completed"}
            return self.last_decision

        # 2. Set active goal from stack
        active_goal_name = self.goal_stack[-1]
        self.state["active_goal"] = {"name": active_goal_name}

        # Swap engine if specific one exists for this subgoal, otherwise use default core
        if active_goal_name in self.engines:
            self.pipeline.engine = self.engines[active_goal_name]
        else:
            self.pipeline.engine = self.core
        
        # 3. Proceed with standard pipeline decision for the current subgoal
        decision = self.pipeline.step(
            last_input.raw, 
            goal=self.state["active_goal"], 
            context=last_input.context
        )
        
        self.state["last_packet"] = decision.input.packet
        self.state["last_action"] = decision.action
        self.state["last_pattern"] = decision.action.selected_pattern.name if decision.action.selected_pattern else None
        
        self.last_decision = {
            "decision": decision, 
            "packet": decision.input.packet,
            "goal_stack": list(self.goal_stack)
        }
        return self.last_decision

    def act(self) -> AgentOutput:
        if self.last_decision and self.last_decision.get("status") == "all_goals_completed":
            return AgentOutput(
                content=None, 
                action_type="idle", 
                confidence=1.0, 
                valid=True, 
                trace={"agent": self.name, "status": "completed"}
            )
        
        return super().act()
