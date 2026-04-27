"""Basic chat wrapper around the stacked HPM text system."""
import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from hpm_ai_v4.simulations.full_simulation import WikipediaStream
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.tools.text_signals import TextSignalExtractor, TextSignalPack

CHAT_SEED_CORPUS = os.path.join(os.path.dirname(__file__), "data", "chat_seed.txt")
CHAT_LIBRARY_ENV = "HPM_CHAT_LIBRARY"
CHAT_LIBRARY_CANDIDATES = [
    os.environ.get(CHAT_LIBRARY_ENV, ""),
    os.path.join(os.getcwd(), "library_bootstrap", "chat_ultra_bundle", "chat_ultra_bundle"),
    "/tmp/hpm_chat_ultra_bundle/chat_ultra_bundle",
    os.path.join(os.getcwd(), "library_bootstrap", "chat_ultra", "chat_ultra_library"),
    os.path.join(os.getcwd(), "library_bootstrap", "chat_ultra", "chat_ultra_library.pkl"),
    os.path.join(os.getcwd(), "library_bootstrap", "chat_super", "chat_super_library"),
    os.path.join(os.getcwd(), "library_bootstrap", "chat_super", "chat_super_library.pkl"),
    os.path.join(os.getcwd(), "library_bootstrap", "chat_mixed", "conversational_chat_library"),
    os.path.join(os.getcwd(), "library_bootstrap", "chat_mixed", "conversational_chat_library.pkl"),
    os.path.join(os.getcwd(), "library_bootstrap", "chat_dailydialog", "daily_dialog_chat_library"),
    os.path.join(os.getcwd(), "library_bootstrap", "chat_dailydialog", "daily_dialog_chat_library.pkl"),
    "/tmp/hpm_chat_super/chat_super_library",
    "/tmp/hpm_chat_super/chat_super_library.pkl",
    "/tmp/hpm_chat_ultra/chat_ultra_library",
    "/tmp/hpm_chat_ultra/chat_ultra_library.pkl",
    "/tmp/hpm_conversational_chat/conversational_chat_library",
    "/tmp/hpm_conversational_chat/conversational_chat_library.pkl",
    "/tmp/hpm_dailydialog_chat/daily_dialog_chat_library",
]


def _resolve_chat_library_path(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    for candidate in CHAT_LIBRARY_CANDIDATES:
        if not candidate:
            continue
        base = candidate[:-4] if candidate.endswith(".pkl") else candidate
        if os.path.exists(base + ".l1.pkl"):
            return base
    return None


def _load_chat_library(layered: LayeredAgent, resolved_library_path: str) -> int:
    if os.path.exists(resolved_library_path + ".l1.pkl"):
        return layered.load_bundle(resolved_library_path)
    return 0


@dataclass
class ChatTurn:
    role: str
    text: str


@dataclass
class ChatResult:
    user_text: str
    response_text: str
    prompt_text: str
    learn_stats: Dict[str, Any]
    response_stats: Dict[str, Any]


class BasicChatSession:
    """Minimal dialogue wrapper over LayeredAgent."""

    def __init__(
        self,
        agent: LayeredAgent,
        history_window: int = 6,
        response_steps: int = 48,
        system_prompt: str = "You are a concise assistant.",
        use_constraints: Optional[bool] = None,
        learn_from_user: bool = True,
        learn_from_reply: bool = False,
        reply_feedback_weight: float = 0.02,
        text_signals: Optional[TextSignalExtractor] = None,
    ):
        self.agent = agent
        self.history_window = max(1, int(history_window))
        self.response_steps = max(1, int(response_steps))
        self.system_prompt = system_prompt.strip()
        self.use_constraints = use_constraints
        self.learn_from_user = bool(learn_from_user)
        self.learn_from_reply = bool(learn_from_reply)
        self.reply_feedback_weight = float(reply_feedback_weight)
        self.text_signals = text_signals or TextSignalExtractor()
        self.history: List[ChatTurn] = []

    def _dialogue_act(self, user_text: str) -> str:
        text = user_text.strip().lower()
        if not text:
            return "default"
        if any(text.startswith(prefix) for prefix in ("hi", "hello", "hey", "good morning", "good evening")):
            return "greeting"
        if any(phrase in text for phrase in ("what do you mean", "clarify", "can you explain", "explain that", "what does that mean")):
            return "clarification"
        if any(text.startswith(prefix) for prefix in ("bye", "goodbye", "thanks", "thank you")):
            return "closing"
        if text.endswith("?"):
            return "question"
        if any(text.startswith(prefix) for prefix in ("tell me", "give me", "show me", "explain", "help me")):
            return "request"
        return "default"

    def reset(self) -> None:
        self.history.clear()

    def chat(self, user_text: str, target_reply: str | None = None) -> str:
        return self.chat_turn(user_text, target_reply=target_reply).response_text

    def chat_turn(self, user_text: str, target_reply: str | None = None) -> ChatResult:
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("user_text must not be empty")

        prior_texts = [turn.text for turn in self._recent_turns() if turn.text.strip()]
        dialogue_act = self._dialogue_act(user_text)
        self._append("user", user_text)
        learn_stats: Dict[str, Any] = {"target_chars": 0, "self_chars": 0}
        if self.learn_from_user:
            learn_stats = self.agent.observe_text(user_text, feedback_mode="target")

        prompt_text = self._build_prompt()
        response_text = self._generate_response(
            prompt_text,
            user_text=user_text,
            target_reply=target_reply,
            dialogue_act=dialogue_act,
        )

        response_signal = self._response_signal_pack(
            response_text,
            user_text=user_text,
            target_reply=target_reply,
            context_texts=[user_text, *prior_texts],
        )
        self._append("assistant", response_text)
        response_stats: Dict[str, Any] = {}
        response_stats.update(response_signal.to_dict())
        response_stats["text_signal_score"] = response_signal.combined_score()
        if target_reply:
            response_stats = self.agent.evaluate_generated_text(response_text, target_reply)
            response_stats.update(response_signal.to_dict())
            response_stats["text_signal_score"] = response_signal.combined_score()
            if self.learn_from_reply:
                response_stats.update(
                    self.agent.observe_text(
                        target_reply,
                        feedback_mode="hybrid",
                        generated_text=response_text,
                        self_feedback_weight=self.reply_feedback_weight,
                        feedback_signal=response_signal.to_dict(),
                    )
                )
        elif self.learn_from_reply and response_text:
            # Optional self-feedback for unconstrained chat, kept off by default.
            response_stats = self.agent.observe_text(
                response_text,
                feedback_mode="self",
                generated_text=response_text,
                self_feedback_weight=self.reply_feedback_weight,
                feedback_signal=response_signal.to_dict(),
            )

        return ChatResult(
            user_text=user_text,
            response_text=response_text,
            prompt_text=prompt_text,
            learn_stats=learn_stats,
            response_stats=response_stats,
        )

    def transcript(self) -> str:
        return "\n".join(f"{turn.role.title()}: {turn.text}" for turn in self.history)

    def _append(self, role: str, text: str) -> None:
        self.history.append(ChatTurn(role=role, text=text.strip()))

    def _recent_turns(self) -> List[ChatTurn]:
        limit = self.history_window * 2
        return self.history[-limit:]

    def _build_prompt(self) -> str:
        lines: List[str] = []
        if self.system_prompt:
            lines.append(self.system_prompt)
        for turn in self._recent_turns():
            lines.append(f"{turn.role.title()}: {turn.text}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _should_use_constraints(self) -> bool:
        if self.use_constraints is not None:
            return self.use_constraints
        return bool(self.agent.dictionary or self.agent.grammar)

    def _generate_response(
        self,
        prompt_text: str,
        user_text: str,
        target_reply: str | None = None,
        dialogue_act: str = "default",
    ) -> str:
        use_constraints = self._should_use_constraints()
        mode = "target" if target_reply else "decode"
        candidates: List[str] = []
        seed_text = self._response_seed_text(user_text)
        short_steps = max(8, self.response_steps // 2)
        long_steps = max(self.response_steps, int(self.response_steps * 1.25))
        context_features = {
            "dialogue_act": dialogue_act,
            "reply_budget": "short" if len(user_text) < 40 or dialogue_act in {"greeting", "closing"} else "medium",
            "reply_style": "concise",
        }

        def add(candidate: str) -> None:
            cleaned = self._sanitize_response(candidate)
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

        if use_constraints:
            add(
                self.agent.generate_constrained_text(
                    steps=self.response_steps,
                    seed_text=seed_text,
                    target_text=target_reply,
                    mode=mode,
                    include_seed=False,
                    update_policy=bool(target_reply),
                    context_features=context_features,
                )
            )
            add(
                self.agent.generate_constrained_text(
                    steps=short_steps,
                    seed_text=seed_text,
                    target_text=target_reply,
                    mode="hybrid" if mode != "target" else mode,
                    include_seed=True,
                    update_policy=False,
                    context_features=context_features,
                )
            )
            add(
                self.agent.generate_text(
                    steps=long_steps,
                    seed_text=seed_text,
                    target_text=target_reply,
                    mode=mode,
                    include_seed=False,
                    update_policy=False,
                    context_features=context_features,
                )
            )
            if target_reply:
                add(
                    self.agent.generate_text(
                        steps=short_steps,
                        seed_text=seed_text,
                        target_text=target_reply,
                        mode="target",
                        include_seed=False,
                        update_policy=False,
                        context_features=context_features,
                    )
                )
        else:
            add(
                self.agent.generate_text(
                    steps=self.response_steps,
                    seed_text=seed_text,
                    target_text=target_reply,
                    mode=mode,
                    include_seed=False,
                    update_policy=False,
                    context_features=context_features,
                )
            )
            add(
                self.agent.generate_text(
                    steps=short_steps,
                    seed_text=seed_text,
                    target_text=target_reply,
                    mode="hybrid" if mode != "target" else mode,
                    include_seed=True,
                    update_policy=False,
                    context_features=context_features,
                )
            )
            add(
                self.agent.generate_chars(
                    steps=self.response_steps,
                    seed_text=seed_text,
                    target_text=target_reply,
                    mode=mode,
                    include_seed=False,
                    update_policy=False,
                    context_features=context_features,
                )
            )
            add(
                self.agent.generate_chars(
                    steps=short_steps,
                    seed_text=seed_text,
                    target_text=target_reply,
                    mode="hybrid" if mode != "target" else mode,
                    include_seed=True,
                    update_policy=False,
                    context_features=context_features,
                )
            )
            if target_reply:
                add(
                    self.agent.generate_text(
                        steps=short_steps,
                        seed_text=seed_text,
                        target_text=target_reply,
                        mode="target",
                        include_seed=False,
                        update_policy=False,
                        context_features=context_features,
                    )
                )

        for fallback in self._response_bank(dialogue_act):
            add(fallback)

        if not candidates:
            candidates = [self._sanitize_response(seed_text)]

        response = self._choose_chat_candidate(candidates, user_text=user_text, dialogue_act=dialogue_act)
        return self._sanitize_response(response)

    def _response_seed_text(self, user_text: str) -> str:
        recent_texts = [turn.text for turn in self._recent_turns() if turn.text.strip()]
        seed_parts: List[str] = []
        if self.system_prompt:
            seed_parts.append(self.system_prompt)
        if recent_texts:
            seed_parts.extend(recent_texts[-2:])
        else:
            seed_parts.append(user_text.strip())
        seed = " ".join(part for part in seed_parts if part)
        return seed.strip() or user_text.strip()

    def _choose_chat_candidate(self, candidates: List[str], user_text: str, dialogue_act: str = "default") -> str:
        best_text = ""
        best_score = -1e9
        for text in candidates:
            score = self._chat_response_score(text, user_text, dialogue_act=dialogue_act)
            if score > best_score:
                best_score = score
                best_text = text
        return best_text

    def _chat_response_score(self, text: str, user_text: str, dialogue_act: str = "default") -> float:
        if not text:
            return -1e9
        tokens = self.agent._tokenize_words(text)
        if not tokens:
            return -1e9

        recent_assistant = [turn.text for turn in self._recent_turns() if turn.role == "assistant" and turn.text.strip()]
        recent_context = recent_assistant[-4:] if recent_assistant else []
        plausibility = self.agent._text_plausibility(text)
        uniq_ratio = len({tok.lower() for tok in tokens}) / max(1, len(tokens))
        lower_text = text.lower()
        repetition_penalty = 0.0
        if len(tokens) >= 4:
            bigrams = [tuple(tok.lower() for tok in tokens[i:i + 2]) for i in range(len(tokens) - 1)]
            if bigrams:
                repetition_penalty = 1.0 - (len(set(bigrams)) / len(bigrams))

        marker_penalty = 0.0
        if "l3:" in lower_text or "user:" in lower_text or "assistant:" in lower_text:
            marker_penalty = 0.4

        if text.strip() in {user_text.strip(), user_text.strip().lower()}:
            marker_penalty += 0.35

        if recent_context and text.strip() in {ctx.strip() for ctx in recent_context}:
            marker_penalty += 0.65

        punctuation_bonus = 0.08 if text.rstrip().endswith((".", "!", "?", ":")) else 0.0
        length_penalty = min(0.2, abs(len(text) - 40) / 200.0)
        signal_pack = self._response_signal_pack(text, user_text=user_text, context_texts=recent_context)
        duplicate_penalty = signal_pack.repeat_score
        if recent_context:
            duplicate_penalty = min(1.0, duplicate_penalty + max(
                self.text_signals.analyze(ctx, context_texts=[user_text], dictionary=self.agent.dictionary, grammar=self.agent.grammar).repeat_score
                for ctx in recent_context
            ) * 0.35)

        act_bonus = self._dialogue_act_bonus(dialogue_act, text)

        return (
            0.48 * plausibility
            + 0.24 * signal_pack.combined_score()
            + 0.18 * uniq_ratio
            + punctuation_bonus
            + act_bonus
            - 0.30 * repetition_penalty
            - 0.22 * duplicate_penalty
            - marker_penalty
            - length_penalty
        )

    def _dialogue_act_bonus(self, dialogue_act: str, text: str) -> float:
        lower = text.strip().lower()
        if not lower:
            return 0.0
        if dialogue_act == "greeting":
            return 0.16 if any(phrase in lower for phrase in ("how can i help", "hello", "hi", "what can i do")) else 0.0
        if dialogue_act == "question":
            if lower.endswith("?"):
                return 0.04
            if any(phrase in lower for phrase in ("i can", "it depends", "here is", "the answer", "you can", "because")):
                return 0.08
        if dialogue_act == "request":
            if any(phrase in lower for phrase in ("here is", "i can", "let me", "the main idea", "short answer")):
                return 0.08
        if dialogue_act == "clarification":
            if lower.endswith("?") or any(phrase in lower for phrase in ("what do you mean", "can you clarify", "which part")):
                return 0.12
        if dialogue_act == "closing":
            if any(phrase in lower for phrase in ("thanks", "understood", "okay", "bye")):
                return 0.08
        return 0.0

    def _response_signal_pack(
        self,
        text: str,
        user_text: str,
        target_reply: str | None = None,
        context_texts: Optional[List[str]] = None,
    ) -> TextSignalPack:
        recent_texts = context_texts if context_texts is not None else [turn.text for turn in self._recent_turns() if turn.text.strip()]
        return self.text_signals.analyze(
            text,
            context_texts=[user_text, *recent_texts],
            target_text=target_reply,
            dictionary=self.agent.dictionary,
            grammar=self.agent.grammar,
        )

    def _sanitize_response(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return "I can help with that."
        for marker in ("\nUser:", "\nAssistant:", "User:", "Assistant:"):
            if marker in cleaned:
                cleaned = cleaned.split(marker, 1)[0].strip()
        if cleaned.lower().startswith("assistant:"):
            cleaned = cleaned[len("assistant:"):].strip()
        cleaned = self._truncate_response(cleaned)
        if not cleaned:
            return "I can help with that."
        return cleaned

    def _response_bank(self, dialogue_act: str) -> List[str]:
        bank = {
            "greeting": [
                "Hello.",
                "Hi there.",
                "How can I help?",
                "What can I do for you?",
            ],
            "question": [
                "I can help with that.",
                "It depends on what you need.",
                "Here is a short answer.",
                "The main idea is simple.",
            ],
            "request": [
                "Here is the short answer.",
                "I can do that.",
                "Let me keep it brief.",
                "The main idea is this.",
            ],
            "clarification": [
                "What do you mean?",
                "Can you clarify that?",
                "Which part should I focus on?",
                "Do you want a brief explanation?",
            ],
            "closing": [
                "Thanks.",
                "Understood.",
                "Okay.",
                "Goodbye.",
            ],
            "default": [
                "I can help with that.",
                "Here is a short answer.",
                "What would you like next?",
                "Let me know what you need.",
            ],
        }
        return bank.get(dialogue_act, bank["default"])

    def _truncate_response(self, text: str) -> str:
        if not text:
            return text

        sentence_end = re.search(r"[.!?]", text)
        if sentence_end and sentence_end.end() >= 12:
            return text[:sentence_end.end()].strip()

        words = text.split()
        if len(words) > 20:
            return " ".join(words[:20]).strip()
        return text


class ReverseChatSession(BasicChatSession):
    """Dialogue wrapper that asks questions and learns from answers."""

    def __init__(
        self,
        agent: LayeredAgent,
        history_window: int = 6,
        response_steps: int = 48,
        system_prompt: str = "You are a concise questioner. Ask one short question at a time.",
        use_constraints: Optional[bool] = None,
        learn_from_user: bool = True,
        learn_from_reply: bool = False,
        reply_feedback_weight: float = 0.02,
        text_signals: Optional[TextSignalExtractor] = None,
    ):
        super().__init__(
            agent=agent,
            history_window=history_window,
            response_steps=response_steps,
            system_prompt=system_prompt,
            use_constraints=use_constraints,
            learn_from_user=learn_from_user,
            learn_from_reply=learn_from_reply,
            reply_feedback_weight=reply_feedback_weight,
            text_signals=text_signals,
        )
        self._question_history: List[str] = []

    def ask(self, seed_text: str | None = None) -> str:
        question = self._generate_question(seed_text=seed_text)
        self._append("assistant", question)
        self._question_history.append(question)
        return question

    def answer_turn(self, answer_text: str) -> ChatResult:
        answer_text = answer_text.strip()
        if not answer_text:
            raise ValueError("answer_text must not be empty")

        prior_questions = [turn.text for turn in self._recent_turns() if turn.role == "assistant" and turn.text.strip()]
        dialogue_act = self._dialogue_act(answer_text)
        self._append("user", answer_text)

        learn_stats: Dict[str, Any] = {"target_chars": 0, "self_chars": 0}
        if self.learn_from_user:
            learn_stats = self.agent.observe_text(answer_text, feedback_mode="target")

        prompt_text = self._build_prompt()
        question_text = self._generate_question(
            seed_text=answer_text,
            dialogue_act=dialogue_act,
            prior_questions=prior_questions,
            prompt_text=prompt_text,
        )

        question_signal = self._response_signal_pack(
            question_text,
            user_text=answer_text,
            context_texts=[answer_text, *prior_questions],
        )
        self._append("assistant", question_text)
        response_stats = dict(question_signal.to_dict())
        response_stats["text_signal_score"] = question_signal.combined_score()
        if self.learn_from_reply and question_text:
            response_stats.update(
                self.agent.observe_text(
                    question_text,
                    feedback_mode="self",
                    generated_text=question_text,
                    self_feedback_weight=self.reply_feedback_weight,
                    feedback_signal=question_signal.to_dict(),
                )
            )

        return ChatResult(
            user_text=answer_text,
            response_text=question_text,
            prompt_text=prompt_text,
            learn_stats=learn_stats,
            response_stats=response_stats,
        )

    def chat_turn(self, user_text: str, target_reply: str | None = None) -> ChatResult:
        return self.answer_turn(user_text)

    def _generate_question(
        self,
        seed_text: str | None = None,
        dialogue_act: str = "default",
        prior_questions: Optional[List[str]] = None,
        prompt_text: str | None = None,
    ) -> str:
        use_constraints = self._should_use_constraints()
        candidates: List[str] = []
        seed_text = self._response_seed_text(seed_text or "")
        short_steps = max(8, self.response_steps // 2)
        context_features = {
            "dialogue_act": dialogue_act,
            "reply_budget": "short",
            "reply_style": "question",
            "conversation_mode": "reverse",
            "desired_question": True,
        }

        def add(candidate: str) -> None:
            cleaned = self._sanitize_question(candidate)
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

        if use_constraints:
            add(
                self.agent.generate_constrained_text(
                    steps=self.response_steps,
                    seed_text=seed_text,
                    target_text=None,
                    mode="decode",
                    include_seed=False,
                    update_policy=False,
                    context_features=context_features,
                )
            )
            add(
                self.agent.generate_constrained_text(
                    steps=short_steps,
                    seed_text=seed_text,
                    target_text=None,
                    mode="hybrid",
                    include_seed=True,
                    update_policy=False,
                    context_features=context_features,
                )
            )
        else:
            add(
                self.agent.generate_text(
                    steps=self.response_steps,
                    seed_text=seed_text,
                    target_text=None,
                    mode="decode",
                    include_seed=False,
                    update_policy=False,
                    context_features=context_features,
                )
            )
            add(
                self.agent.generate_text(
                    steps=short_steps,
                    seed_text=seed_text,
                    target_text=None,
                    mode="hybrid",
                    include_seed=True,
                    update_policy=False,
                    context_features=context_features,
                )
            )
            add(
                self.agent.generate_chars(
                    steps=short_steps,
                    seed_text=seed_text,
                    target_text=None,
                    mode="decode",
                    include_seed=False,
                    update_policy=False,
                    context_features=context_features,
                )
            )

        for fallback in self._question_bank(dialogue_act):
            add(fallback)

        if prior_questions:
            for q in prior_questions[-3:]:
                add(q)

        if not candidates:
            candidates = [self._sanitize_question(seed_text)]

        question = self._choose_question_candidate(candidates, seed_text=seed_text, dialogue_act=dialogue_act)
        return self._sanitize_question(question)

    def _choose_question_candidate(self, candidates: List[str], seed_text: str, dialogue_act: str = "default") -> str:
        best_text = ""
        best_score = -1e9
        for text in candidates:
            score = self._question_score(text, seed_text, dialogue_act=dialogue_act)
            if score > best_score:
                best_score = score
                best_text = text
        return best_text

    def _question_score(self, text: str, seed_text: str, dialogue_act: str = "default") -> float:
        if not text:
            return -1e9
        base = self._chat_response_score(text, seed_text, dialogue_act="question")
        lower = text.strip().lower()
        question_bonus = 0.0
        if lower.endswith("?"):
            question_bonus += 0.20
        if any(lower.startswith(prefix) for prefix in ("what", "why", "how", "when", "where", "who", "which", "can you", "could you", "would you", "do you", "is it", "are you")):
            question_bonus += 0.12
        if any(phrase in lower for phrase in ("tell me", "explain", "help me understand", "what do you mean", "what happened")):
            question_bonus += 0.05
        if dialogue_act in {"greeting", "question", "request", "clarification"}:
            question_bonus += 0.05
        if "assistant:" in lower or "user:" in lower:
            question_bonus -= 0.35
        return base + question_bonus

    def _sanitize_question(self, text: str) -> str:
        cleaned = self._sanitize_response(text)
        if not cleaned:
            return "What do you mean?"
        if not cleaned.endswith("?"):
            cleaned = cleaned.rstrip(".!") + "?"
        return cleaned

    def _question_bank(self, dialogue_act: str) -> List[str]:
        bank = {
            "greeting": [
                "How can I help?",
                "What would you like to know?",
                "What can I do for you?",
            ],
            "question": [
                "What do you mean?",
                "Can you say more?",
                "What is the main point?",
                "Why do you say that?",
            ],
            "request": [
                "What should I focus on?",
                "Can you give one example?",
                "What is the most important part?",
            ],
            "clarification": [
                "Which part should I clarify?",
                "Can you be more specific?",
                "What does that refer to?",
            ],
            "closing": [
                "Would you like to continue?",
                "Is there anything else you want?",
            ],
            "default": [
                "What do you think?",
                "Can you tell me more?",
                "Why is that important?",
                "What should I ask next?",
            ],
        }
        return bank.get(dialogue_act, bank["default"])


def run_basic_chat_simulation(
    corpus_path: str,
    prompts: Optional[List[str]] = None,
    warmup_chars: int = 800,
    response_steps: int = 48,
    history_window: int = 6,
    num_workers: int = 1,
    use_dict: bool = True,
    library_path: Optional[str] = None,
    checkpoint_dir: str = ".",
    seed_corpus_path: Optional[str] = CHAT_SEED_CORPUS,
) -> List[Dict[str, Any]]:
    """Run a small scripted chat benchmark."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)

    resolved_library_path = _resolve_chat_library_path(library_path)
    if resolved_library_path:
        loaded = _load_chat_library(layered, resolved_library_path)
        print(f"Loaded library from {resolved_library_path} ({loaded} bundle parts)")

    seed_source = seed_corpus_path if seed_corpus_path and os.path.exists(seed_corpus_path) else corpus_path

    stream = WikipediaStream(seed_source)
    stream_iter = iter(stream)
    raw_ids = [next(stream_iter) for _ in range(max(warmup_chars, 1))]
    warmup_text = "".join(chr(v + 32) for v in raw_ids if 0 <= v <= 94)
    if warmup_text:
        layered.observe_text(warmup_text, feedback_mode="target")

    session = BasicChatSession(
        layered,
        history_window=history_window,
        response_steps=response_steps,
        use_constraints=use_dict,
    )
    prompts = prompts or [
        "Hello.",
        "What can you do?",
        "Give me a short answer.",
    ]

    history: List[Dict[str, Any]] = []
    print(
        f"Starting basic chat simulation: prompts={len(prompts)} warmup={warmup_chars} "
        f"workers={num_workers} dict={use_dict}"
    )
    for idx, prompt in enumerate(prompts, start=1):
        result = session.chat_turn(prompt)
        record = {
            "turn": idx,
            "user_text": result.user_text,
            "response_text": result.response_text,
            "prompt_text": result.prompt_text,
            "learn_stats": result.learn_stats,
            "response_stats": result.response_stats,
            "transcript": session.transcript(),
        }
        history.append(record)
        print(f"[turn {idx}] user={prompt!r}")
        print(f"         reply={result.response_text!r}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_base = os.path.join(checkpoint_dir, "final_chat_library")
    layered.save_bundle(final_base)
    print(f"Final chat library saved: {final_base}.l1.pkl + .l2.pkl + .l3.pkl + .l4.pkl + .l5.pkl")

    return history


def run_reverse_chat_simulation(
    corpus_path: str,
    answers: Optional[List[str]] = None,
    warmup_chars: int = 800,
    response_steps: int = 48,
    history_window: int = 6,
    num_workers: int = 1,
    use_dict: bool = True,
    library_path: Optional[str] = None,
    checkpoint_dir: str = ".",
    seed_corpus_path: Optional[str] = CHAT_SEED_CORPUS,
) -> List[Dict[str, Any]]:
    """Run a small scripted reverse-chat benchmark where the model asks and the user answers."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar)

    resolved_library_path = _resolve_chat_library_path(library_path)
    if resolved_library_path:
        loaded = _load_chat_library(layered, resolved_library_path)
        print(f"Loaded library from {resolved_library_path} ({loaded} bundle parts)")

    seed_source = seed_corpus_path if seed_corpus_path and os.path.exists(seed_corpus_path) else corpus_path

    stream = WikipediaStream(seed_source)
    stream_iter = iter(stream)
    raw_ids = [next(stream_iter) for _ in range(max(warmup_chars, 1))]
    warmup_text = "".join(chr(v + 32) for v in raw_ids if 0 <= v <= 94)
    if warmup_text:
        layered.observe_text(warmup_text, feedback_mode="target")

    session = ReverseChatSession(
        layered,
        history_window=history_window,
        response_steps=response_steps,
        use_constraints=use_dict,
    )
    answers = answers or [
        "I need a plan.",
        "I want something brief.",
        "The main issue is clarity.",
    ]

    history: List[Dict[str, Any]] = []
    opening = session.ask()
    history.append(
        {
            "turn": 0,
            "user_text": "",
            "response_text": opening,
            "prompt_text": "",
            "learn_stats": {},
            "response_stats": {},
            "transcript": session.transcript(),
        }
    )
    print(
        f"Starting reverse chat simulation: answers={len(answers)} warmup={warmup_chars} "
        f"workers={num_workers} dict={use_dict}"
    )
    print(f"[turn 0] ask={opening!r}")
    for idx, answer in enumerate(answers, start=1):
        result = session.answer_turn(answer)
        record = {
            "turn": idx,
            "user_text": result.user_text,
            "response_text": result.response_text,
            "prompt_text": result.prompt_text,
            "learn_stats": result.learn_stats,
            "response_stats": result.response_stats,
            "transcript": session.transcript(),
        }
        history.append(record)
        print(f"[turn {idx}] answer={answer!r}")
        print(f"         question={result.response_text!r}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    final_base = os.path.join(checkpoint_dir, "final_reverse_chat_library")
    layered.save_bundle(final_base)
    print(f"Final reverse chat library saved: {final_base}.l1.pkl + .l2.pkl + .l3.pkl + .l4.pkl + .l5.pkl")

    return history


def _parse_args():
    p = argparse.ArgumentParser(description="Basic chat simulation")
    p.add_argument("--corpus", required=True, help="Path to plain-text corpus file")
    p.add_argument("--prompt", action="append", default=[], help="User prompt; may be repeated")
    p.add_argument("--answer", action="append", default=[], help="Reverse-chat answer; may be repeated")
    p.add_argument("--warmup-chars", type=int, default=800)
    p.add_argument("--response-steps", type=int, default=48)
    p.add_argument("--history-window", type=int, default=6)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--library", default=None, help="Base path to pre-built pattern library (no .pkl suffix)")
    p.add_argument("--checkpoint-dir", default=".", help="Directory for checkpoint files")
    p.add_argument("--seed-corpus", default=CHAT_SEED_CORPUS, help="Optional dialogue seed corpus for warmup")
    p.add_argument("--reverse", action="store_true", help="Run the question-first reverse chat mode")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.reverse:
        run_reverse_chat_simulation(
            corpus_path=args.corpus,
            answers=args.answer or None,
            warmup_chars=args.warmup_chars,
            response_steps=args.response_steps,
            history_window=args.history_window,
            num_workers=args.workers,
            use_dict=args.dict,
            library_path=args.library,
            checkpoint_dir=args.checkpoint_dir,
            seed_corpus_path=args.seed_corpus,
        )
    else:
        run_basic_chat_simulation(
            corpus_path=args.corpus,
            prompts=args.prompt or None,
            warmup_chars=args.warmup_chars,
            response_steps=args.response_steps,
            history_window=args.history_window,
            num_workers=args.workers,
            use_dict=args.dict,
            library_path=args.library,
            checkpoint_dir=args.checkpoint_dir,
            seed_corpus_path=args.seed_corpus,
        )
