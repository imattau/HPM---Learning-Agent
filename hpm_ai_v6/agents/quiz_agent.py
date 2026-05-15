# hpm_ai_v6/agents/quiz_agent.py
from __future__ import annotations

import json
import os
import uuid
import re as _re
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

_DIFFICULTY_PROMPT = {
    "easy": "Focus on a single well-known fact. The question should test direct recall.",
    "medium": "Test application of a concept or moderate inference from known information.",
    "hard": "Test inference across multiple related concepts or relational reasoning.",
}


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
        if topic is None:
            topics = getattr(self._reader, "top_concepts", lambda n: [])(10)
            topic = topics[0] if topics else "general knowledge"

        difficulty_hint = _DIFFICULTY_PROMPT.get(difficulty, _DIFFICULTY_PROMPT["medium"])

        prompt = (
            f"You are a quiz question author. Generate a multiple-choice question about: {topic}\n"
            f"Difficulty: {difficulty}. {difficulty_hint}\n"
            "Return ONLY valid JSON with this exact structure:\n"
            '{"question": "...", "options": ["A", "B", "C", "D"], "correct_index": N, "explanation": "..."}\n'
            "Rules:\n"
            "- options must have exactly 4 items\n"
            "- correct_index is 0-3\n"
            "- distractors must be plausible, not obviously wrong\n"
            "- explanation must say WHY the correct answer is right\n"
        )

        raw = self._reasoner.reason(prompt)
        parsed = self._parse_question_json(raw)
        if parsed is None:
            return None

        # Verification: ask the reasoner to confirm the correct answer
        verify_prompt = (
            f"Question: {parsed['question']}\n"
            f"Options: {parsed['options']}\n"
            f"Claimed correct answer index: {parsed['correct_index']} = "
            f"'{parsed['options'][parsed['correct_index']]}'\n"
            "Is this correct? Reply with only YES or NO."
        )
        verdict = self._reasoner.reason(verify_prompt).strip().upper()
        if "NO" in verdict:
            return None

        return QuizQuestion(
            id=str(uuid.uuid4()),
            question=parsed["question"],
            options=parsed["options"],
            correct_index=parsed["correct_index"],
            topic=topic,
            explanation=parsed["explanation"],
            difficulty=difficulty,
            source="model",
        )

    def _parse_question_json(self, raw: str) -> Optional[dict]:
        """Extract and validate JSON from reasoner output."""
        match = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None
        if not all(k in data for k in ("question", "options", "correct_index", "explanation")):
            return None
        if len(data["options"]) != 4:
            return None
        if not isinstance(data["correct_index"], int) or not (0 <= data["correct_index"] <= 3):
            return None
        return data

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
