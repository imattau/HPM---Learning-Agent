# hpm_ai_v6/agents/quiz_agent.py
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

from hpm_ai_v6.agents.reasoning_agent import ReasoningAgent
from hpm_ai_v6.agents.multi_agent_reader import MultiAgentReader


@dataclass
class QuizQuestion:
    id: str
    question: str
    options: List[str]      # always exactly 4
    correct_index: int       # 0-3
    topic: str
    explanation: str
    difficulty: str          # "easy" | "medium" | "hard"
    source: str              # "model" | "bank"


_BANK_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "quiz_banks")


class QuizAgent:
    """Generates and self-evaluates multi-choice quiz questions using the learned corpus."""

    def __init__(self, reader: MultiAgentReader, reasoning_agent: ReasoningAgent) -> None:
        self._reader = reader
        self._reasoner = reasoning_agent

    def generate_quiz(self, n: int, difficulty: str, source: str) -> List[QuizQuestion]:
        """Generate n questions from model (corpus) or bank (static JSON)."""
        if source == "bank":
            return self._load_bank(difficulty, n)
        questions = []
        for _ in range(n):
            q = self.generate_question(topic=None, difficulty=difficulty)
            if q:
                questions.append(q)
        return questions

    def generate_question(self, topic: Optional[str], difficulty: str) -> Optional[QuizQuestion]:
        """Generate a single question with reasoning verification. Returns None if generation fails."""
        raise NotImplementedError

    def _load_bank(self, difficulty: str, n: int) -> List[QuizQuestion]:
        """Load up to n questions from the static JSON bank for the given difficulty."""
        path = os.path.join(_BANK_DIR, f"{difficulty}.json")
        with open(path) as f:
            raw = json.load(f)
        questions = []
        for item in raw[:n]:
            questions.append(QuizQuestion(
                id=item.get("id", str(uuid.uuid4())),
                question=item["question"],
                options=item["options"],
                correct_index=item["correct_index"],
                topic=item["topic"],
                explanation=item["explanation"],
                difficulty=difficulty,
                source="bank",
            ))
        return questions
