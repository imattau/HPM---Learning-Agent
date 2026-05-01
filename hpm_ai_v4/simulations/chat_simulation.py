"""Basic chat wrapper around the stacked HPM text system."""
import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

from hpm_ai_v4.simulations.full_simulation import WikipediaStream
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.dictionary import DictionaryValidator
from hpm_ai_v4.tools.grammar import GrammarValidator, HeuristicGrammarLibrary
from hpm_ai_v4.tools.library_registry import LibraryRegistry
from hpm_ai_v4.io.adapters import SentenceAdapter
from hpm_ai_v4.tools.text_signals import TextSignalExtractor, TextSignalPack

CHAT_SEED_CORPUS = os.path.join(os.path.dirname(__file__), "data", "chat_seed.txt")
CHAT_LIBRARY_ENV = "HPM_CHAT_LIBRARY"
CHAT_REGISTRY_ENV = "HPM_LIBRARY_REGISTRY"
CHAT_REGISTRY_CANDIDATES = [
    os.environ.get(CHAT_REGISTRY_ENV, ""),
    os.path.join(os.getcwd(), "library_bootstrap", "registry.json"),
    "/tmp/hpm_library_registry.json",
    "/tmp/hpm_chat_registry.json",
    "/tmp/hpm_dailydialog_registry.json",
]
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
    os.path.join(os.getcwd(), "library_bootstrap", "nltk_large", "nltk_large_nlp_2000"),
    os.path.join(os.getcwd(), "library_bootstrap", "nltk_large", "nltk_large_nlp_2000.pkl"),
    "/tmp/hpm_chat_super/chat_super_library",
    "/tmp/hpm_chat_super/chat_super_library.pkl",
    "/tmp/hpm_chat_ultra/chat_ultra_library",
    "/tmp/hpm_chat_ultra/chat_ultra_library.pkl",
    "/tmp/hpm_conversational_chat/conversational_chat_library",
    "/tmp/hpm_conversational_chat/conversational_chat_library.pkl",
    "/tmp/hpm_dailydialog_chat/daily_dialog_chat_library",
]

_DISCOURSE_PRONOUNS = {
    "it",
    "they",
    "them",
    "their",
    "theirs",
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "this",
    "that",
    "these",
    "those",
    "its",
    "we",
    "us",
    "our",
    "ours",
    "you",
    "your",
    "yours",
}

_DISCOURSE_INTERROGATIVES = {
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "which",
    "whom",
    "whose",
}

_DISCOURSE_AUXILIARIES = {
    "did",
    "do",
    "does",
    "was",
    "were",
    "is",
    "are",
    "am",
    "will",
    "would",
    "could",
    "should",
    "can",
    "has",
    "have",
    "had",
}

_DISCOURSE_NEGATIONS = {
    "not",
    "never",
    "no",
    "none",
    "nothing",
    "nowhere",
    "n't",
}

_DISCOURSE_CLAUSE_MARKERS = {
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "because",
    "although",
    "when",
    "if",
}

_DISCOURSE_PREPOSITIONS = {
    "on",
    "in",
    "at",
    "with",
    "to",
    "from",
    "by",
    "under",
    "over",
    "of",
    "for",
    "about",
    "into",
    "onto",
    "inside",
    "outside",
    "near",
    "behind",
    "beneath",
    "between",
}

_DISCOURSE_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "so",
    "because",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "at",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
    "must",
    "have",
    "has",
    "had",
    "i",
    "me",
    "my",
    "mine",
    "we",
    "us",
    "our",
    "you",
    "your",
}

ACTION_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "give": {
        "category": "transfer",
        "effects": {
            "recipient_has_object": True,
            "agent_has_object": False,
        },
    },
    "buy": {
        "category": "transfer",
        "effects": {
            "agent_has_object": True,
        },
    },
    "receive": {
        "category": "transfer",
        "effects": {
            "agent_has_object": True,
        },
    },
    "get": {
        "category": "transfer",
        "effects": {
            "agent_has_object": True,
        },
    },
    "have": {
        "category": "possession",
        "effects": {
            "agent_has_object": True,
        },
    },
    "own": {
        "category": "possession",
        "effects": {
            "agent_has_object": True,
        },
    },
    "possess": {
        "category": "possession",
        "effects": {
            "agent_has_object": True,
        },
    },
    "break": {
        "category": "state",
        "effects": {
            "object_state": "broken",
        },
    },
}


def _normalize_verb_lemma(token: str) -> str:
    tok = token.strip().lower()
    if tok in ACTION_SCHEMAS:
        return tok
    irregular = {
        "gave": "give",
        "given": "give",
        "gives": "give",
        "bought": "buy",
        "buys": "buy",
        "broke": "break",
        "broken": "break",
        "receives": "receive",
        "received": "receive",
        "gets": "get",
        "got": "get",
        "having": "have",
        "has": "have",
        "had": "have",
        "owns": "own",
        "owned": "own",
        "possesses": "possess",
        "possessed": "possess",
    }
    if tok in irregular:
        return irregular[tok]
    if tok.endswith("ied") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("ed") and len(tok) > 4:
        base = tok[:-2]
        if base.endswith(base[-1:]) and len(base) > 2:
            base = base[:-1]
        return base
    if tok.endswith("ing") and len(tok) > 5:
        base = tok[:-3]
        if base.endswith(base[-1:]) and len(base) > 2:
            base = base[:-1]
        return base
    if tok.endswith("es") and len(tok) > 3:
        return tok[:-2]
    if tok.endswith("s") and len(tok) > 3:
        return tok[:-1]
    return tok


def _resolve_registered_chat_library() -> Optional[str]:
    for registry_path in CHAT_REGISTRY_CANDIDATES:
        if not registry_path or not os.path.exists(registry_path):
            continue
        registry = LibraryRegistry(registry_path)
        resolution = registry.resolve_bundle(view="chat")
        if resolution is not None:
            return resolution.path
    return None


def _resolve_chat_library_path(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    resolved = _resolve_registered_chat_library()
    if resolved:
        return resolved
    for candidate in CHAT_LIBRARY_CANDIDATES:
        if not candidate:
            continue
        base = candidate[:-4] if candidate.endswith(".pkl") else candidate
        if os.path.exists(base + ".l1.pkl"):
            return base
        if os.path.exists(candidate):
            return candidate
        if os.path.exists(base + ".pkl"):
            return base + ".pkl"
    return None


def _load_chat_library(layered: LayeredAgent, resolved_library_path: str) -> int:
    if os.path.exists(resolved_library_path + ".l1.pkl"):
        return layered.load_bundle(resolved_library_path)
    if os.path.exists(resolved_library_path):
        from hpm_ai_v4.tools.serializer import PatternSerializer

        layered.l1.patterns = PatternSerializer.load(resolved_library_path)
        return 1 if layered.l1.patterns else 0
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


@dataclass
class DiscourseState:
    """Lightweight persistent discourse memory for a chat session."""

    turn_index: int = 0
    topic: str = "unknown"
    topic_confidence: float = 0.0
    active_entities: List[str] = field(default_factory=list)
    entity_salience: Dict[str, float] = field(default_factory=dict)
    pronoun_candidates: List[str] = field(default_factory=list)
    last_dialogue_act: str = "default"
    last_sentence_type: str = "fragment"
    focus_stack: List[str] = field(default_factory=list)
    discourse_summary: str = ""
    archived_entities: List[str] = field(default_factory=list)
    archived_summary: str = ""
    entity_registry: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "topic": self.topic,
            "topic_confidence": self.topic_confidence,
            "active_entities": list(self.active_entities),
            "entity_salience": dict(self.entity_salience),
            "pronoun_candidates": list(self.pronoun_candidates),
            "last_dialogue_act": self.last_dialogue_act,
            "last_sentence_type": self.last_sentence_type,
            "focus_stack": list(self.focus_stack),
            "discourse_summary": self.discourse_summary,
            "archived_entities": list(self.archived_entities),
            "archived_summary": self.archived_summary,
            "entity_registry_size": len(self.entity_registry),
            "entity_registry_top": list(self.entity_registry.keys())[:4],
        }

    def reset(self) -> None:
        self.turn_index = 0
        self.topic = "unknown"
        self.topic_confidence = 0.0
        self.active_entities.clear()
        self.entity_salience.clear()
        self.pronoun_candidates.clear()
        self.last_dialogue_act = "default"
        self.last_sentence_type = "fragment"
        self.focus_stack.clear()
        self.discourse_summary = ""
        self.archived_entities.clear()
        self.archived_summary = ""
        self.entity_registry.clear()


@dataclass
class RelationalState:
    """Persistent relational frame for lightweight binding and slot-filling."""

    subject: str = "unknown"
    predicate: str = "unknown"
    object: str = "unknown"
    voice: str = "active"
    agent: str = "unknown"
    proposition: str = ""
    confidence: float = 0.0
    role_bindings: Dict[str, str] = field(default_factory=dict)
    proposition_history: List[str] = field(default_factory=list)
    binding_stack: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "voice": self.voice,
            "agent": self.agent,
            "proposition": self.proposition,
            "confidence": self.confidence,
            "role_bindings": dict(self.role_bindings),
            "proposition_history": list(self.proposition_history[-5:]),
            "binding_stack": list(self.binding_stack[-4:]),
        }

    def reset(self) -> None:
        self.subject = "unknown"
        self.predicate = "unknown"
        self.object = "unknown"
        self.voice = "active"
        self.agent = "unknown"
        self.proposition = ""
        self.confidence = 0.0
        self.role_bindings.clear()
        self.proposition_history.clear()
        self.binding_stack.clear()


@dataclass
class SituationState:
    """Unified chat situation state spanning discourse, relation, and history."""

    discourse: DiscourseState = field(default_factory=DiscourseState)
    relational: RelationalState = field(default_factory=RelationalState)
    history: List[ChatTurn] = field(default_factory=list)
    response_mode: str = "answer"
    dialogue_act: str = "default"
    last_update_role: str = "unknown"
    last_update_turn: int = 0

    def reset(self) -> None:
        self.discourse.reset()
        self.relational.reset()
        self.history.clear()
        self.response_mode = "answer"
        self.dialogue_act = "default"
        self.last_update_role = "unknown"
        self.last_update_turn = 0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "discourse": self.discourse.to_dict(),
            "relational": self.relational.to_dict(),
            "history_size": len(self.history),
            "response_mode": self.response_mode,
            "dialogue_act": self.dialogue_act,
            "last_update_role": self.last_update_role,
            "last_update_turn": self.last_update_turn,
        }


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
        use_sentence_features: bool = True,
        use_relational_heuristics: bool = True,
        use_passive_normalization: Optional[bool] = None,
        use_clause_stack: Optional[bool] = None,
        use_registry_scoring: Optional[bool] = None,
        use_coreference_scoring: Optional[bool] = None,
        use_property_scoring: Optional[bool] = None,
        dictionary: Optional[DictionaryValidator] = None,
        grammar: Optional[GrammarValidator] = None,
    ):
        self.agent = agent
        self.dictionary = dictionary if dictionary is not None else getattr(agent, "dictionary", None)
        self.grammar = grammar if grammar is not None else getattr(agent, "grammar", None)
        self.history_window = max(1, int(history_window))
        self.response_steps = max(1, int(response_steps))
        self.system_prompt = system_prompt.strip()
        self.use_constraints = use_constraints
        self.learn_from_user = bool(learn_from_user)
        self.learn_from_reply = bool(learn_from_reply)
        self.reply_feedback_weight = float(reply_feedback_weight)
        self.text_signals = text_signals or TextSignalExtractor()
        self.sentence_adapter = SentenceAdapter()
        self.use_sentence_features = bool(use_sentence_features)
        self.use_relational_heuristics = bool(use_relational_heuristics)
        self.use_passive_normalization = bool(use_relational_heuristics if use_passive_normalization is None else use_passive_normalization)
        self.use_clause_stack = bool(use_relational_heuristics if use_clause_stack is None else use_clause_stack)
        self.use_registry_scoring = bool(use_relational_heuristics if use_registry_scoring is None else use_registry_scoring)
        self.use_coreference_scoring = bool(use_relational_heuristics if use_coreference_scoring is None else use_coreference_scoring)
        self.use_property_scoring = bool(use_relational_heuristics if use_property_scoring is None else use_property_scoring)
        self.situation_state = SituationState()
        self.history = self.situation_state.history
        self.history_cap = max(4, self.history_window * 4)
        self.discourse_state = self.situation_state.discourse
        self.relational_state = self.situation_state.relational

    def _grammar_pos(self, word: str) -> str:
        token = word.strip().lower()
        if not token:
            return ""
        grammar = self.grammar
        if grammar is not None:
            try:
                pos = str(grammar.get_pos(token)).upper()
                if pos and pos != "NN":
                    return pos
            except Exception:
                pass
        if token in _DISCOURSE_PRONOUNS:
            return "PR"
        if token in _DISCOURSE_PREPOSITIONS:
            return "IN"
        if token in _DISCOURSE_AUXILIARIES:
            return "VB"
        return "NN"

    def _grammar_lemma(self, word: str) -> str:
        token = word.strip().lower()
        if not token:
            return ""
        grammar = self.grammar
        if grammar is not None:
            lemma_fn = getattr(grammar, "normalize_lemma", None)
            if callable(lemma_fn):
                try:
                    lemma = str(lemma_fn(token)).strip().lower()
                    if lemma:
                        return lemma
                except Exception:
                    pass
        return _normalize_verb_lemma(token)

    def _is_pronoun_token(self, word: str) -> bool:
        return self._grammar_pos(word) == "PR"

    def _is_preposition_token(self, word: str) -> bool:
        return self._grammar_pos(word) == "IN"

    def _is_auxiliary_token(self, word: str) -> bool:
        token = word.strip().lower()
        return self._grammar_pos(token) == "VB" and token in _DISCOURSE_AUXILIARIES

    def _is_interrogative_token(self, word: str) -> bool:
        return word.strip().lower() in _DISCOURSE_INTERROGATIVES

    def _is_stopword_token(self, word: str) -> bool:
        token = word.strip().lower()
        return token in _DISCOURSE_STOPWORDS or self._is_pronoun_token(token)

    def _is_possession_verb(self, word: str) -> bool:
        token = self._grammar_lemma(word)
        return token in {"have", "own", "possess", "contain", "hold", "carry"}

    def _grammar_sentence_parse(self, text: str) -> Any:
        grammar = self.grammar
        if grammar is None:
            return None
        parse_fn = getattr(grammar, "parse_sentence", None)
        if not callable(parse_fn):
            return None
        try:
            return parse_fn(text)
        except Exception:
            return None

    def _relation_heuristic_scale(self) -> float:
        return 1.0 if self.use_relational_heuristics else 0.0

    def _component_enabled(self, component: str) -> bool:
        component = str(component).strip().lower()
        return {
            "passive": self.use_passive_normalization,
            "clause": self.use_clause_stack,
            "registry": self.use_registry_scoring,
            "coreference": self.use_coreference_scoring,
            "property": self.use_property_scoring,
        }.get(component, self.use_relational_heuristics)

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

    def _select_response_mode(self, user_text: str, dialogue_act: str) -> str:
        lower = user_text.strip().lower()
        relation_conf = float(self.relational_state.confidence)
        topic_conf = float(self.discourse_state.topic_confidence)
        if dialogue_act in {"greeting", "closing"}:
            return "acknowledge"
        if dialogue_act == "clarification":
            return "clarify"
        if dialogue_act == "question":
            if relation_conf < 0.2 and (not lower or lower.endswith("?")):
                return "clarify"
            if topic_conf < 0.15 and not self.discourse_state.active_entities:
                return "defer"
            return "answer"
        if dialogue_act == "request":
            return "answer"
        if any(self._is_pronoun_token(tok) for tok in re.findall(r"[A-Za-z']+", lower)):
            return "answer" if self.discourse_state.topic != "unknown" else "clarify"
        return "answer" if relation_conf >= 0.15 or topic_conf >= 0.15 else "acknowledge"

    def reset(self) -> None:
        self.situation_state.reset()

    def chat(self, user_text: str, target_reply: str | None = None) -> str:
        return self.chat_turn(user_text, target_reply=target_reply).response_text

    def observe_text(
        self,
        text: str,
        role: str = "user",
        dialogue_act: str = "default",
        feedback_mode: str = "target",
        generated_text: str | None = None,
        self_feedback_weight: float = 0.05,
        feedback_signal: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train on external text with sentence-level binding evaluator bracketing."""
        sentences = self.sentence_adapter.split(text)
        stats = {
            "target_chars": 0,
            "self_chars": 0,
            "binding_predictions": 0,
            "binding_prediction_hits": 0,
            "binding_prediction_misses": 0,
        }

        for sentence in sentences:
            if not sentence.strip():
                continue

            # 1. Predict (prior)
            predicted = self._predict_next_entity()

            # 2. Process (symbolic L4)
            # This triggers _update_entity_registry which records l3_state
            sentence_features = self._sentence_features(sentence)
            self._update_discourse_state(
                sentence,
                role=role,
                dialogue_act=dialogue_act,
                sentence_features=sentence_features
            )

            # 3. Observe & Learn (probabilistic L1-L3)
            current_feedback = dict(feedback_signal or {})
            discourse_signal = self._discourse_feedback_signal(sentence, role=role, dialogue_act=dialogue_act)
            current_feedback.update(discourse_signal)

            learn_stats = self.agent.observe_text(
                sentence,
                feedback_mode=feedback_mode,
                generated_text=generated_text if sentence in (generated_text or "") else None,
                self_feedback_weight=self_feedback_weight,
                feedback_signal=current_feedback,
            )
            for k, v in learn_stats.items():
                if isinstance(v, (int, float)):
                    stats[k] = stats.get(k, 0) + v

            # 4. Evaluate (posterior)
            content_tokens = self._content_tokens(sentence)
            stats["binding_predictions"] += 1
            self._tick_binding_evaluator(predicted, content_tokens)

            # 5. Close loop back to L3
            matched = any(t.lower() == predicted.lower() for t in content_tokens)
            if matched:
                stats["binding_prediction_hits"] += 1
            else:
                stats["binding_prediction_misses"] += 1
            self._feed_binding_prediction_to_l3(predicted, matched)

        return stats

    def chat_turn(self, user_text: str, target_reply: str | None = None) -> ChatResult:
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("user_text must not be empty")

        prior_texts = [turn.text for turn in self._recent_turns() if turn.text.strip()]
        dialogue_act = self._dialogue_act(user_text)
        response_mode = self._select_response_mode(user_text, dialogue_act)
        self.situation_state.dialogue_act = dialogue_act
        self.situation_state.response_mode = response_mode

        self._append("user", user_text)
        # Process user text (includes prediction, discourse update, learning, evaluation)
        learn_stats = self.observe_text(
            user_text,
            role="user",
            dialogue_act=dialogue_act,
            feedback_mode="target" if self.learn_from_user else "none"
        )

        prompt_text = self._build_prompt()
        discourse_features = self._discourse_context_features()
        discourse_features["response_mode"] = response_mode
        response_text = self._generate_response(
            prompt_text,
            user_text=user_text,
            target_reply=target_reply,
            dialogue_act=dialogue_act,
            target_sentence_features=self._sentence_features(target_reply) if (self.use_sentence_features and target_reply) else {},
            discourse_features=discourse_features,
        )

        response_signal = self._response_signal_pack(
            response_text,
            user_text=user_text,
            target_reply=target_reply,
            context_texts=[user_text, self.discourse_state.discourse_summary, *prior_texts],
        )
        self._append("assistant", response_text)
        # Process assistant response (includes discourse update and learning)
        response_stats = self.observe_text(
            target_reply if target_reply else response_text,
            role="assistant",
            dialogue_act=dialogue_act,
            feedback_mode="target" if self.learn_from_reply else "none",
            generated_text=response_text,
            self_feedback_weight=self.reply_feedback_weight,
            feedback_signal=response_signal.to_dict() if response_signal else None,
        )

        # Add additional metadata to stats
        if response_signal:
            response_stats.update(response_signal.to_dict())
            response_stats["text_signal_score"] = response_signal.combined_score()
        response_stats.update(self._sentence_features(response_text))
        response_stats.update(self._discourse_context_features())

        if target_reply:
            eval_stats = self.agent.evaluate_generated_text(response_text, target_reply)
            response_stats.update(eval_stats)
        response_stats["response_mode"] = response_mode

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
        if len(self.history) > self.history_cap:
            overflow = len(self.history) - self.history_cap
            pruned = self.history[:overflow]
            self.history = self.history[overflow:]
            self._fold_pruned_turns(pruned)

    def _recent_turns(self) -> List[ChatTurn]:
        limit = self.history_window * 2
        return self.history[-limit:]

    def _fold_pruned_turns(self, turns: List[ChatTurn]) -> None:
        if not turns:
            return
        pruned_tokens: List[str] = []
        for turn in turns:
            pruned_tokens.extend(self._content_tokens(turn.text))
        if pruned_tokens:
            ranked = self._rank_tokens(pruned_tokens)[:3]
            for tok in ranked:
                if tok not in self.discourse_state.archived_entities:
                    self.discourse_state.archived_entities.append(tok)
            archive = ", ".join(ranked)
            if archive:
                self.discourse_state.archived_summary = archive
        self.discourse_state.discourse_summary = self._format_discourse_summary()

    def _predict_binding_entity(self, text: str) -> str:
        lower = text.strip().lower()
        if not lower:
            return ""
        words = re.findall(r"[A-Za-z']+", lower)
        pronouns = {tok for tok in words if self._is_pronoun_token(tok)}
        if pronouns and self._component_enabled("coreference"):
            if self.discourse_state.focus_stack:
                return self.discourse_state.focus_stack[0]
            if self.discourse_state.topic != "unknown":
                return self.discourse_state.topic
        if lower.startswith("who") and self._component_enabled("registry") and self.relational_state.subject != "unknown":
            return self.relational_state.subject
        if self._component_enabled("coreference") and any(phrase in lower for phrase in ("what about it", "what about them", "what about this", "what about that")):
            if self.discourse_state.focus_stack:
                return self.discourse_state.focus_stack[0]
        if self._component_enabled("coreference") and self.discourse_state.active_entities:
            return self.discourse_state.active_entities[0]
        return self.discourse_state.topic if self.discourse_state.topic != "unknown" else ""

    def _observed_binding_entity(self, text: str) -> str:
        lower = text.strip().lower()
        if not lower:
            return ""
        if self._component_enabled("registry") and self.relational_state.subject != "unknown":
            return self.relational_state.subject
        if self._component_enabled("coreference") and self.discourse_state.active_entities:
            return self.discourse_state.active_entities[0]
        return self.discourse_state.topic if self.discourse_state.topic != "unknown" else ""

    def _predict_next_entity(self) -> str:
        """Predict next entity by matching current L3 soft-state to registered latent states."""
        # Try latent prediction first
        try:
            current_l3 = int(self.agent.l3_soft_state())
            best, best_score = "unknown", -1.0
            current_turn = self.discourse_state.turn_index

            for entity, record in self.discourse_state.entity_registry.items():
                dominant = record.get("dominant_latent_state")
                if dominant is None:
                    continue
                # Latent match: same state = strong signal
                latent_match = 1.0 if int(dominant) == current_l3 else 0.0
                recency = 1.0 / max(1, current_turn - int(record.get("last_turn", 0)) + 1)
                conf = float(record.get("binding_confidence", 0.0))
                stability = float(record.get("stability_score", 0.0))
                score = 0.40 * latent_match + 0.25 * conf + 0.20 * recency + 0.15 * stability
                if score > best_score:
                    best, best_score = entity, score
            if best != "unknown":
                return best
        except Exception:
            pass

        # Fall back to surface heuristic
        best, best_score = "unknown", -1.0
        current_turn = self.discourse_state.turn_index
        for entity, record in self.discourse_state.entity_registry.items():
            recency = 1.0 / max(1, current_turn - int(record.get("last_turn", 0)) + 1)
            score = float(record.get("binding_confidence", 0.0)) * 0.6 + recency * 0.4
            if score > best_score:
                best, best_score = entity, score
        return best

    def _tick_binding_evaluator(self, predicted: str, observed_tokens: List[str]) -> None:
        """
        HPM evaluator step: reinforce bindings that predicted observed tokens,
        decay those that didn't. Updates binding_confidence and stability_score.
        """
        observed_set = {t.lower() for t in observed_tokens}
        for entity, record in self.discourse_state.entity_registry.items():
            appeared = entity.lower() in observed_set
            conf = float(record.get("binding_confidence", 0.0))
            hits = int(record.get("prediction_hits", 0))
            misses = int(record.get("prediction_misses", 0))

            if appeared:
                record["binding_confidence"] = min(1.0, conf + 0.08)
                record["prediction_hits"] = hits + 1
            else:
                record["binding_confidence"] = max(0.0, conf * 0.92)
                record["prediction_misses"] = misses + 1

            # Stability = running prediction accuracy (HPM pattern density proxy)
            total = hits + misses + (1 if appeared else 1)
            record["stability_score"] = (hits + (1 if appeared else 0)) / max(1, total)

        # Extra reinforcement for the entity that was explicitly predicted
        if predicted != "unknown" and predicted in self.discourse_state.entity_registry:
            entry = self.discourse_state.entity_registry[predicted]
            if predicted.lower() in observed_set:
                entry["binding_confidence"] = min(1.0, float(entry["binding_confidence"]) + 0.05)

    def _feed_binding_prediction_to_l3(self, predicted_entity: str, matched: bool) -> None:
        """Feed binding prediction success back into L3 patterns."""
        if predicted_entity == "unknown":
            return
        record = self.discourse_state.entity_registry.get(predicted_entity, {})
        stability = float(record.get("stability_score", 0.0))
        gate_strength = stability * (0.8 if matched else -0.2)
        gate_strength = float(np.clip(gate_strength, 0.0, 1.0))

        feedback = {
            "topdown_gate": gate_strength,
            "topdown_pattern_suppression": 0.3 * gate_strength if matched else 0.0,
            "reward": 0.1 if matched else 0.0,
        }
        if hasattr(self.agent, '_pending_feedback'):
            self.agent._pending_feedback.update(feedback)

    def _update_entity_registry(
        self,
        tokens: List[str],
        *,
        role: str,
        dialogue_act: str,
        sentence_type: str,
        sentence_features: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not tokens:
            return
        sentence_features = sentence_features or {}
        current_turn = self.discourse_state.turn_index
        subject = self.relational_state.subject
        predicate = self.relational_state.predicate
        obj = self.relational_state.object
        for idx, tok in enumerate(tokens):
            entry = self.discourse_state.entity_registry.setdefault(
                tok,
                {
                    "entity": tok,
                    "first_seen_turn": current_turn,
                    "last_turn": current_turn,
                    "last_role": role,
                    "last_dialogue_act": dialogue_act,
                    "last_sentence_type": sentence_type,
                    "last_subject": subject,
                    "last_predicate": predicate,
                    "last_object": obj,
                    "mention_count": 0,
                    "salience": 0.0,
                    "binding_confidence": 0.0,
                    "stability_score": 0.0,
                    "prediction_hits": 0,
                    "prediction_misses": 0,
                    "last_position": idx,
                },
            )
            entry["last_turn"] = current_turn
            entry["last_role"] = "subject" if tok == subject else "object" if tok == obj else role
            entry["last_dialogue_act"] = dialogue_act
            entry["last_sentence_type"] = sentence_type
            entry["last_subject"] = subject
            entry["last_predicate"] = predicate
            entry["last_object"] = obj
            entry["mention_count"] = int(entry.get("mention_count", 0)) + 1
            entry["last_position"] = idx

            # Latent Seam: Record current L3 state for this entity
            l3_state = self.agent.l3_soft_state() if hasattr(self.agent, "l3_soft_state") else None
            if l3_state is not None:
                prev = entry.get("latent_states", [])
                prev.append(int(l3_state))
                entry["latent_states"] = prev[-8:]
                from collections import Counter
                entry["dominant_latent_state"] = Counter(entry["latent_states"]).most_common(1)[0][0]

            current_salience = float(self.discourse_state.entity_salience.get(tok, 0.0))
            entry["salience"] = max(float(entry.get("salience", 0.0)) * 0.88, current_salience)
            if sentence_features:
                entry["sentence_confidence"] = float(sentence_features.get("sentence_confidence", 0.0))

    def _ensure_entity_record(self, entity: str) -> Dict[str, Any]:
        entity = entity.strip().lower()
        if not entity:
            return {}
        entry = self.discourse_state.entity_registry.setdefault(
            entity,
            {
                "entity": entity,
                "first_seen_turn": self.discourse_state.turn_index,
                "last_turn": self.discourse_state.turn_index,
                "last_role": "unknown",
                "last_dialogue_act": "default",
                "last_sentence_type": "fragment",
                "last_subject": "unknown",
                "last_predicate": "unknown",
                "last_object": "unknown",
                "mention_count": 0,
                "salience": 0.0,
                "binding_confidence": 0.0,
                "stability_score": 0.0,
                "prediction_hits": 0,
                "prediction_misses": 0,
                "last_position": 0,
                "possessions": [],
                "states": {},
            },
        )
        entry.setdefault("possessions", [])
        entry.setdefault("states", {})
        return entry

    def _update_possession(self, possessor: str, item: str, *, add: bool = True) -> None:
        possessor = possessor.strip().lower()
        item = item.strip().lower()
        if not possessor or not item:
            return
        entry = self._ensure_entity_record(possessor)
        possessions = list(entry.get("possessions", []))
        if add and item not in possessions:
            possessions.append(item)
        elif not add:
            possessions = [obj for obj in possessions if obj != item]
        entry["possessions"] = possessions
        entry["last_turn"] = self.discourse_state.turn_index
        entry["stability_score"] = float(entry.get("stability_score", 0.0)) + (0.05 if add else -0.03)

    def _apply_action_schema_effects(self, frame: Dict[str, Any], text: str = "", raw_tokens: Optional[List[str]] = None) -> None:
        predicate = self._grammar_lemma(str(frame.get("predicate", "")).strip().lower())
        raw_tokens = list(raw_tokens or self._raw_tokens(text))
        content_tokens = self._semantic_tokens(raw_tokens)
        schema_key = predicate
        for tok in raw_tokens:
            lemma = self._grammar_lemma(tok)
            if lemma in ACTION_SCHEMAS:
                schema_key = lemma
                break
        schema = ACTION_SCHEMAS.get(schema_key)
        if not schema:
            return

        def _known(value: Any) -> str:
            text_value = str(value).strip().lower()
            return text_value if text_value and text_value != "unknown" else ""

        subject = str(frame.get("subject", "")).strip().lower()
        agent = _known(frame.get("agent", "")) or subject
        obj = _known(frame.get("object", ""))
        patient = _known(frame.get("patient", ""))

        if schema_key == "give":
            recipient = patient
            if not recipient:
                if frame.get("voice") == "active" and len(content_tokens) >= 4:
                    recipient = str(content_tokens[2]).strip().lower()
                elif len(content_tokens) >= 3:
                    recipient = str(content_tokens[1]).strip().lower()
            if not recipient and len(raw_tokens) >= 3:
                try:
                    verb_index = next(i for i, tok in enumerate(raw_tokens) if self._grammar_lemma(tok) == "give")
                    tail_tokens = [tok for tok in self._semantic_tokens(raw_tokens[verb_index + 1:]) if tok]
                    if tail_tokens:
                        recipient = tail_tokens[0]
                except StopIteration:
                    pass
            if recipient and recipient == agent and len(content_tokens) >= 3:
                recipient = str(content_tokens[1]).strip().lower()
            if recipient and recipient != "unknown" and obj:
                self._update_possession(recipient, obj, add=True)
                if agent and agent != "unknown":
                    self._update_possession(agent, obj, add=False)
        elif schema_key in {"buy", "receive", "get", "have", "own", "possess"} or schema.get("category") == "transfer":
            if agent and agent != "unknown" and obj:
                self._update_possession(agent, obj, add=True)
        elif schema_key == "break" or schema.get("category") == "state":
            target = obj or subject or patient
            if target:
                entry = self._ensure_entity_record(target)
                states = dict(entry.get("states", {}))
                states["condition"] = "broken"
                entry["states"] = states
                entry["last_turn"] = self.discourse_state.turn_index
        elif schema.get("category") == "possession":
            if agent and agent != "unknown" and obj:
                self._update_possession(agent, obj, add=True)

        if subject and subject != "unknown":
            self._ensure_entity_record(subject)
        if agent and agent != "unknown":
            self._ensure_entity_record(agent)
        if patient and patient != "unknown":
            self._ensure_entity_record(patient)
        if obj:
            self._ensure_entity_record(obj)

    def _entity_possessions(self, entity: str) -> List[str]:
        record = self.discourse_state.entity_registry.get(entity.strip().lower(), {})
        possessions = record.get("possessions", [])
        return [str(item).strip().lower() for item in possessions if item]

    def _resolve_possession_query(self, question_text: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
        lower = question_text.strip().lower()
        tokens = parsed.get("tokens", [])
        content_tokens = parsed.get("content_tokens", [])
        subject_hint = parsed.get("subject_hint", "")
        object_hint = parsed.get("object_hint", "")
        if lower.startswith("what") and any(self._is_possession_verb(tok) for tok in tokens):
            possessor = subject_hint or (content_tokens[0] if content_tokens else "")
            possessions = self._entity_possessions(possessor)
            if possessions:
                answer = possessions[0] if len(possessions) == 1 else ", ".join(possessions)
                return {
                    "matched": True,
                    "answer": answer,
                    "source": "world_model_possession_list",
                    "possessor": possessor,
                }

        if lower.startswith(("does", "do", "did")) and any(self._is_possession_verb(tok) for tok in tokens):
            possessor = subject_hint or (content_tokens[0] if content_tokens else "")
            held = self._entity_possessions(possessor)
            target = object_hint or (content_tokens[-1] if content_tokens else "")
            if possessor and target:
                matched = target in held
                return {
                    "matched": True,
                    "answer": "yes" if matched else "no",
                    "source": "world_model_possession_check",
                    "possessor": possessor,
                    "target": target,
                }

        if lower.startswith("who") and any(self._is_possession_verb(tok) for tok in tokens):
            target = object_hint or (content_tokens[-1] if content_tokens else "")
            if target:
                for entity, record in self.discourse_state.entity_registry.items():
                    possessions = [str(item).strip().lower() for item in record.get("possessions", [])]
                    if target in possessions:
                        return {
                            "matched": True,
                            "answer": entity,
                            "source": "world_model_possession_owner",
                            "target": target,
                        }
        return {"matched": False}

    def _entity_registry_scores(self, question_text: str = "") -> Dict[str, float]:
        scores: Dict[str, float] = {}
        lower = question_text.lower()
        pronouns = {tok.lower() for tok in self.agent._tokenize_words(question_text) if self._is_pronoun_token(tok)}
        for entity, record in self.discourse_state.entity_registry.items():
            if not entity:
                continue
            score = float(record.get("salience", 0.0))
            score += 0.10 * min(5, int(record.get("mention_count", 0)))
            if record.get("last_role") == "subject":
                score += 0.14
            if record.get("last_role") == "object":
                score += 0.04
            if entity == self.discourse_state.topic:
                score += 0.18 * max(0.3, float(self.discourse_state.topic_confidence))
            if entity in self.discourse_state.focus_stack[:2]:
                score += 0.12
            score += 0.15 * float(record.get("stability_score", 0.0))
            if entity in self.discourse_state.active_entities[:2]:
                score += 0.10
            if int(record.get("last_turn", -1)) >= max(0, self.discourse_state.turn_index - 2):
                score += 0.08
            if pronouns:
                score += 0.08 if record.get("last_role") in {"subject", "topic"} else 0.0
            if entity in lower:
                score += 0.10
            if record.get("last_predicate") != "unknown" and record.get("last_predicate") in lower:
                score += 0.05
            scores[entity] = score
        return scores

    def _best_entity_from_registry(self, question_text: str = "") -> str:
        scores = self._entity_registry_scores(question_text)
        if not scores:
            return ""
        ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        return ranked[0][0] if ranked else ""

    def _best_subject_for_query(self, predicate_hint: str = "", object_hint: str = "") -> str:
        predicate_hint = predicate_hint.strip().lower()
        object_hint = object_hint.strip().lower()
        use_registry = self._component_enabled("registry")
        candidates = list(self.discourse_state.entity_registry.items())
        if predicate_hint:
            matching = [
                (entity, record)
                for entity, record in candidates
                if record.get("last_predicate") == predicate_hint
                or record.get("role_bindings", {}).get("predicate") == predicate_hint
            ]
            if matching:
                candidates = matching
        best_entity = ""
        best_score = -1e9
        for entity, record in candidates:
            score = float(record.get("salience", 0.0))
            if use_registry and predicate_hint:
                if record.get("last_predicate") == predicate_hint:
                    score += 1.25
                else:
                    score -= 0.85
            if use_registry and object_hint and record.get("last_object") == object_hint:
                score += 0.12
            if use_registry and object_hint and entity == object_hint:
                score -= 0.95
            if use_registry and record.get("last_role") == "subject":
                score += 0.34
            elif use_registry and record.get("last_role") == "object":
                score -= 0.18
            if use_registry and entity == self.discourse_state.topic:
                score += 0.10 * max(0.2, float(self.discourse_state.topic_confidence))
            if use_registry and int(record.get("last_turn", -1)) >= max(0, self.discourse_state.turn_index - 3):
                score += 0.08
            if use_registry and entity in self.discourse_state.active_entities[:2]:
                score += 0.10
            if score > best_score:
                best_score = score
                best_entity = entity
        return best_entity

    def _best_object_for_query(self, subject_hint: str = "", predicate_hint: str = "") -> str:
        subject_hint = subject_hint.strip().lower()
        predicate_hint = predicate_hint.strip().lower()
        use_registry = self._component_enabled("registry")
        candidates = list(self.discourse_state.entity_registry.items())
        if subject_hint:
            matching = [
                (entity, record)
                for entity, record in candidates
                if record.get("last_subject") == subject_hint
                or record.get("role_bindings", {}).get("subject") == subject_hint
            ]
            if matching:
                candidates = matching
        best_entity = ""
        best_score = -1e9
        for entity, record in candidates:
            score = float(record.get("salience", 0.0))
            candidate_value = entity
            if use_registry and predicate_hint:
                if record.get("last_predicate") == predicate_hint:
                    score += 1.22
                    if record.get("last_object") not in {"", "unknown"}:
                        candidate_value = str(record.get("last_object")).strip().lower() or candidate_value
                else:
                    score -= 0.65
            if use_registry and subject_hint and record.get("last_subject") == subject_hint:
                score += 0.90
                if record.get("last_object") not in {"", "unknown"}:
                    candidate_value = str(record.get("last_object")).strip().lower() or candidate_value
            if use_registry and record.get("last_role") == "object":
                score += 0.28
            elif use_registry and record.get("last_role") == "subject":
                score -= 0.08
            if use_registry and entity == self.discourse_state.topic:
                score += 0.08 * max(0.2, float(self.discourse_state.topic_confidence))
            if use_registry and int(record.get("last_turn", -1)) >= max(0, self.discourse_state.turn_index - 3):
                score += 0.08
            if use_registry and entity in self.discourse_state.active_entities[:2]:
                score += 0.08
            if score > best_score:
                best_score = score
                best_entity = candidate_value
        return best_entity

    def _best_entity_by_property(self, property_hints: Iterable[str]) -> str:
        hints = {hint.strip().lower() for hint in property_hints if hint}
        if not hints:
            return ""
        use_property = self._component_enabled("property")
        use_registry = self._component_enabled("registry")
        copulas = {"is", "are", "was", "were", "be", "been"}
        direct_matches = sorted(
            self.discourse_state.entity_registry.items(),
            key=lambda item: (
                int(item[1].get("last_turn", -1)),
                float(item[1].get("binding_confidence", 0.0)),
                float(item[1].get("salience", 0.0)),
            ),
            reverse=True,
        )
        for candidate, record in direct_matches:
            predicate = str(record.get("last_predicate", "unknown")).lower()
            role_predicate = str(record.get("role_bindings", {}).get("predicate", "unknown")).lower()
            obj = str(record.get("last_object", "unknown")).lower()
            if use_registry and candidate in hints:
                continue
            if use_registry and (predicate in hints or role_predicate in hints or (predicate in copulas and obj in hints)) and (
                record.get("last_role") == "subject" or str(record.get("last_subject", "")).lower() == candidate
            ):
                return candidate
        best_entity = ""
        best_score = -1e9
        for entity, record in self.discourse_state.entity_registry.items():
            score = float(record.get("salience", 0.0))
            score += 0.08 * min(5, int(record.get("mention_count", 0)))
            subject = str(record.get("last_subject", "")).lower()
            predicate = str(record.get("last_predicate", "unknown")).lower()
            obj = str(record.get("last_object", "unknown")).lower()
            role_predicate = str(record.get("role_bindings", {}).get("predicate", "unknown")).lower()
            if use_registry and (predicate in hints or role_predicate in hints):
                score += 1.20
            if use_property and obj in hints:
                score += 1.28
                if predicate in copulas:
                    score += 0.22
                if use_property and entity in hints:
                    score -= 1.10
            if use_property and entity in hints:
                score += 0.08
            if use_registry and subject == entity:
                score += 0.12
            if use_registry and record.get("last_role") == "subject":
                score += 0.10
            elif use_registry and record.get("last_role") == "object":
                score -= 0.05
            if use_registry and int(record.get("last_turn", -1)) >= max(0, self.discourse_state.turn_index - 3):
                score += 0.05
            if score > best_score:
                best_score = score
                best_entity = entity
        return best_entity

    def _parse_query_frame(self, question_text: str) -> Dict[str, Any]:
        lower = question_text.strip().lower()
        tokens = re.findall(r"[A-Za-z']+", lower)
        content_tokens = self._semantic_tokens(tokens)
        negated = any(tok in _DISCOURSE_NEGATIONS or "n't" in tok for tok in tokens)
        predicate_hint = ""
        subject_hint = ""
        object_hint = ""
        if lower.startswith("who"):
            if any(self._is_auxiliary_token(tok) for tok in tokens):
                if content_tokens:
                    predicate_hint = content_tokens[-1]
                if len(content_tokens) >= 2:
                    subject_hint = content_tokens[0]
            elif len(content_tokens) >= 2:
                predicate_hint = content_tokens[0]
                object_hint = content_tokens[1]
            elif content_tokens:
                predicate_hint = content_tokens[0]
        elif lower.startswith("what"):
            if any(self._is_auxiliary_token(tok) for tok in tokens) and content_tokens:
                predicate_hint = content_tokens[-1]
                if len(content_tokens) >= 2:
                    object_hint = content_tokens[0]
            elif len(content_tokens) >= 2:
                predicate_hint = content_tokens[0]
                object_hint = content_tokens[1]
            elif content_tokens:
                predicate_hint = content_tokens[0]
        else:
            if len(content_tokens) >= 2:
                predicate_hint = content_tokens[0]
                object_hint = content_tokens[1]
            elif content_tokens:
                predicate_hint = content_tokens[0]
        return {
            "tokens": tokens,
            "content_tokens": content_tokens,
            "subject_hint": subject_hint,
            "predicate_hint": predicate_hint,
            "object_hint": object_hint,
            "negated": negated,
        }

    def _parse_chain_query_frame(self, question_text: str) -> Dict[str, Any]:
        lower = question_text.strip().lower()
        tokens = re.findall(r"[A-Za-z']+", lower)
        if not tokens or tokens[0] not in {"what", "which"}:
            return {"matched": False}

        property_hint = ""
        for tok in tokens[1:]:
            if tok in _DISCOURSE_STOPWORDS or tok in _DISCOURSE_INTERROGATIVES or self._is_auxiliary_token(tok):
                continue
            property_hint = tok
            break
        copula_match = re.search(r"\b(is|are|was|were|be|been)\b", lower)
        if not copula_match:
            return {"matched": False}
        pivot_phrase = lower[copula_match.end():].strip(" ?.")
        if not pivot_phrase:
            return {"matched": False}
        pivot_tokens = [tok for tok in re.findall(r"[A-Za-z']+", pivot_phrase) if tok]
        if len(pivot_tokens) < 2 and not any(marker in pivot_phrase for marker in (" by ", " that ", " which ", " who ", " whom ", " whose ")):
            return {"matched": False}

        return {
            "matched": True,
            "property_hint": property_hint,
            "pivot_phrase": pivot_phrase,
            "token_count": len(tokens),
        }

    def _strip_article_prefix(self, phrase: str) -> str:
        text = phrase.strip().lower().strip(" ?.")
        text = re.sub(r"^(?:the|a|an)\s+", "", text)
        return text.strip()

    def _split_last_clause_marker(self, phrase: str) -> tuple[str, str]:
        lower = phrase.strip().lower()
        best_idx = -1
        best_marker = ""
        for marker in sorted(_DISCOURSE_CLAUSE_MARKERS, key=len, reverse=True):
            idx = lower.rfind(f" {marker} ")
            if idx > best_idx:
                best_idx = idx
                best_marker = marker
        if best_idx < 0:
            return "", ""
        left = lower[:best_idx].strip()
        right = lower[best_idx + len(best_marker) + 2 :].strip()
        return left, right

    def _split_by_phrase(self, phrase: str) -> tuple[str, str]:
        lower = phrase.strip().lower()
        idx = lower.rfind(" by ")
        if idx < 0:
            return "", ""
        return lower[:idx].strip(), lower[idx + 4 :].strip()

    def _relation_hints_from_phrase(self, phrase: str) -> Dict[str, str]:
        text = self._strip_article_prefix(phrase)
        raw_tokens = [tok for tok in re.findall(r"[A-Za-z']+", text) if tok]
        if not raw_tokens:
            return {"subject_hint": "", "predicate_hint": "", "object_hint": "", "relation_kind": ""}

        if raw_tokens[0] in {
            "thing",
            "animal",
            "person",
            "object",
            "item",
            "entity",
            "one",
            "place",
            "way",
            "shape",
            "colour",
            "color",
            "size",
            "kind",
            "type",
        } and len(raw_tokens) > 1:
            raw_tokens = raw_tokens[1:]

        if raw_tokens and self._is_preposition_token(raw_tokens[-1]):
            raw_tokens = raw_tokens[:-1]
        raw_tokens = [
            tok
            for tok in raw_tokens
            if not self._is_stopword_token(tok)
            and not self._is_auxiliary_token(tok)
            and not self._is_preposition_token(tok)
        ]
        if not raw_tokens:
            return {"subject_hint": "", "predicate_hint": "", "object_hint": "", "relation_kind": ""}

        subject_hint = ""
        predicate_hint = ""
        object_hint = ""

        if len(raw_tokens) == 1:
            predicate_hint = raw_tokens[0]
        else:
            subject_hint = raw_tokens[0]
            predicate_hint = raw_tokens[1]
            if len(raw_tokens) > 2:
                object_hint = raw_tokens[2]

        relation_kind = "object"
        if predicate_hint in {"is", "are", "was", "were", "be", "been"}:
            relation_kind = "property"

        return {
            "subject_hint": subject_hint,
            "predicate_hint": predicate_hint,
            "object_hint": object_hint,
            "relation_kind": relation_kind,
        }

    def _resolve_entity_phrase(self, phrase: str, *, max_hops: int = 2, _depth: int = 0) -> Dict[str, Any]:
        phrase = self._strip_article_prefix(phrase)
        if not phrase:
            return {"entity": "", "hops": 0, "source": "empty"}

        if phrase in self.discourse_state.entity_registry:
            return {"entity": phrase, "hops": 0, "source": "registry"}

        if _depth >= max_hops:
            return {"entity": "", "hops": 0, "source": "depth_cap"}

        relation_phrase, agent_phrase = self._split_by_phrase(phrase)
        if agent_phrase:
            agent_result = self._resolve_entity_phrase(agent_phrase, max_hops=max_hops, _depth=_depth + 1)
            agent_entity = str(agent_result.get("entity", "")).strip().lower()
            if agent_entity:
                relation_hints = self._relation_hints_from_phrase(relation_phrase)
                predicate_hint = relation_hints.get("predicate_hint", "")
                subject_hint = relation_hints.get("subject_hint", "")
                relation_kind = relation_hints.get("relation_kind", "object")
                if relation_kind == "property" and predicate_hint:
                    subject_candidate = subject_hint or agent_entity
                    entity = self._best_subject_for_query(predicate_hint=predicate_hint, object_hint=subject_candidate)
                else:
                    entity = self._best_object_for_query(subject_hint=agent_entity, predicate_hint=predicate_hint)
                    if not entity and subject_hint:
                        entity = self._best_object_for_query(subject_hint=subject_hint, predicate_hint=predicate_hint)
                if entity:
                    return {
                        "entity": entity,
                        "hops": int(agent_result.get("hops", 0)) + 1,
                        "source": "by_chain",
                        "pivot_phrase": relation_phrase,
                        "agent_entity": agent_entity,
                    }

        head, clause = self._split_last_clause_marker(phrase)
        if clause:
            clause_result = self._resolve_entity_phrase(clause, max_hops=max_hops, _depth=_depth + 1)
            if clause_result.get("entity"):
                return {
                    "entity": clause_result["entity"],
                    "hops": int(clause_result.get("hops", 0)),
                    "source": "clause",
                    "pivot_phrase": clause,
                }

        relation_hints = self._relation_hints_from_phrase(phrase)
        predicate_hint = relation_hints.get("predicate_hint", "")
        subject_hint = relation_hints.get("subject_hint", "")
        relation_kind = relation_hints.get("relation_kind", "object")
        entity = ""
        if relation_kind == "property" and predicate_hint:
            entity = self._best_subject_for_query(predicate_hint=predicate_hint, object_hint=subject_hint)
        elif subject_hint or predicate_hint:
            entity = self._best_object_for_query(subject_hint=subject_hint, predicate_hint=predicate_hint)
        if entity:
            return {
                "entity": entity,
                "hops": 1,
                "source": "relation",
                "pivot_phrase": phrase,
            }
        return {"entity": "", "hops": 0, "source": "unresolved"}

    def _calibrate_binding_confidence(self, predicted_entity: str, observed_entity: str, observed_text: str = "") -> Dict[str, Any]:
        predicted = predicted_entity.strip().lower()
        observed = observed_entity.strip().lower()
        matched = bool(predicted and observed and predicted == observed)
        if not matched and observed_text:
            lower = observed_text.lower()
            matched = bool(observed and observed in lower and (not predicted or predicted in lower))

        if matched:
            self.relational_state.confidence = min(1.0, self.relational_state.confidence + 0.12)
        else:
            self.relational_state.confidence = max(0.0, self.relational_state.confidence * 0.82)

        for entity in {predicted, observed}:
            if not entity or entity not in self.discourse_state.entity_registry:
                continue
            entry = self.discourse_state.entity_registry[entity]
            if matched:
                entry["binding_confidence"] = min(1.0, float(entry.get("binding_confidence", 0.0)) + 0.12)
            else:
                entry["binding_confidence"] = max(0.0, float(entry.get("binding_confidence", 0.0)) * 0.88)
        return {
            "predicted_entity": predicted,
            "observed_entity": observed,
            "matched": matched,
            "binding_confidence": float(self.relational_state.confidence),
        }

    def record_binding_feedback(self, predicted_entity: str, observed_entity: str, observed_text: str = "") -> Dict[str, Any]:
        return self._calibrate_binding_confidence(predicted_entity, observed_entity, observed_text=observed_text)

    def _build_prompt(self) -> str:
        lines: List[str] = []
        if self.system_prompt:
            lines.append(self.system_prompt)
        if self.discourse_state.discourse_summary:
            lines.append(f"Focus: {self.discourse_state.discourse_summary}")
        if self.relational_state.proposition:
            lines.append(f"Relation: {self.relational_state.proposition}")
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
        target_sentence_features: Optional[Dict[str, Any]] = None,
        discourse_features: Optional[Dict[str, Any]] = None,
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
        context_features.update(target_sentence_features or {})
        context_features.update(discourse_features or {})

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

        response = self._choose_chat_candidate(
            candidates,
            user_text=user_text,
            dialogue_act=dialogue_act,
            target_sentence_features=target_sentence_features,
        )
        return self._sanitize_response(response)

    def _response_seed_text(self, user_text: str) -> str:
        recent_texts = [turn.text for turn in self._recent_turns() if turn.text.strip()]
        seed_parts: List[str] = []
        if self.system_prompt:
            seed_parts.append(self.system_prompt)
        if self.discourse_state.discourse_summary:
            seed_parts.append(f"Focus: {self.discourse_state.discourse_summary}")
        if self.relational_state.proposition:
            seed_parts.append(f"Relation: {self.relational_state.proposition}")
        if recent_texts:
            seed_parts.extend(recent_texts[-2:])
        else:
            seed_parts.append(user_text.strip())
        seed = " ".join(part for part in seed_parts if part)
        return seed.strip() or user_text.strip()

    def _choose_chat_candidate(
        self,
        candidates: List[str],
        user_text: str,
        dialogue_act: str = "default",
        target_sentence_features: Optional[Dict[str, Any]] = None,
        discourse_state: Optional[DiscourseState] = None,
    ) -> str:
        best_text = ""
        best_score = -1e9
        for text in candidates:
            score = self._chat_response_score(
                text,
                user_text,
                dialogue_act=dialogue_act,
                target_sentence_features=target_sentence_features,
                discourse_state=discourse_state,
            )
            if score > best_score:
                best_score = score
                best_text = text
        return best_text

    def _chat_response_score(
        self,
        text: str,
        user_text: str,
        dialogue_act: str = "default",
        target_sentence_features: Optional[Dict[str, Any]] = None,
        discourse_state: Optional[DiscourseState] = None,
    ) -> float:
        if not text:
            return -1e9
        tokens = self.agent._tokenize_words(text)
        if not tokens:
            return -1e9
        discourse_state = discourse_state or self.discourse_state

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
        sentence_features = self._sentence_features(text) if self.use_sentence_features else {}
        duplicate_penalty = signal_pack.repeat_score
        if recent_context:
            duplicate_penalty = min(1.0, duplicate_penalty + max(
                self.text_signals.analyze(ctx, context_texts=[user_text], dictionary=self.agent.dictionary, grammar=self.agent.grammar).repeat_score
                for ctx in recent_context
            ) * 0.35)

        act_bonus = self._dialogue_act_bonus(dialogue_act, text)
        sentence_bonus = self._sentence_score(dialogue_act, sentence_features) if self.use_sentence_features else 0.0
        target_sentence_bonus = 0.0
        if self.use_sentence_features and target_sentence_features:
            target_sentence_bonus = self._target_sentence_score(sentence_features, target_sentence_features)
        discourse_bonus = self._discourse_bonus(text, user_text, discourse_state)

        return (
            0.48 * plausibility
            + 0.24 * signal_pack.combined_score()
            + 0.18 * uniq_ratio
            + punctuation_bonus
            + act_bonus
            + sentence_bonus
            + target_sentence_bonus
            + discourse_bonus
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

    def _content_tokens(self, text: str) -> List[str]:
        tokens = []
        for tok in self.agent._tokenize_words(text):
            low = tok.lower()
            if not low or self._is_stopword_token(low):
                continue
            if any(ch.isalpha() for ch in low):
                tokens.append(low)
        return tokens

    def _raw_tokens(self, text: str) -> List[str]:
        return [tok.lower() for tok in re.findall(r"[A-Za-z']+", text)]

    def _first_meaningful_token(self, tokens: List[str]) -> str:
        for tok in tokens:
            if tok and not self._is_stopword_token(tok) and not self._is_interrogative_token(tok):
                return tok
        return ""

    def _semantic_tokens(self, tokens: List[str], *, keep_prepositions: bool = False) -> List[str]:
        semantic: List[str] = []
        for tok in tokens:
            if not tok or not any(ch.isalpha() for ch in tok):
                continue
            if self._is_stopword_token(tok) or self._is_interrogative_token(tok) or self._is_auxiliary_token(tok):
                continue
            if not keep_prepositions and self._is_preposition_token(tok):
                continue
            semantic.append(tok)
        return semantic

    def _split_clause_segments(self, raw_tokens: List[str]) -> List[List[str]]:
        segments: List[List[str]] = []
        current: List[str] = []
        for tok in raw_tokens:
            if tok in _DISCOURSE_CLAUSE_MARKERS and current:
                segments.append(current)
                current = []
                continue
            current.append(tok)
        if current:
            segments.append(current)
        return [segment for segment in segments if segment]

    def _infer_clause_frame(
        self,
        raw_tokens: List[str],
        *,
        role: str,
        dialogue_act: str,
        sentence_features: Dict[str, Any],
        fallback_subject: str = "unknown",
        normalize_passive: bool = True,
    ) -> Dict[str, Any]:
        content_tokens = self._semantic_tokens(raw_tokens)
        subject = self._first_meaningful_token(content_tokens)
        predicate = "unknown"
        obj = "unknown"
        voice = "active"
        grammatical_subject = subject if subject != "unknown" else fallback_subject
        agent = subject
        patient = obj

        by_index = raw_tokens.index("by") if "by" in raw_tokens else -1
        passive_aux = normalize_passive and any(tok in {"was", "were", "been"} for tok in raw_tokens) and by_index >= 0
        if passive_aux:
            voice = "passive"
            before_by = raw_tokens[:by_index]
            after_by = raw_tokens[by_index + 1 :]
            patient = self._first_meaningful_token(before_by) or subject
            agent = self._first_meaningful_token(after_by) or subject
            grammatical_subject = patient
            subject = agent
            predicate_candidates = [
                tok
                for tok in before_by
                if not self._is_stopword_token(tok)
                and not self._is_pronoun_token(tok)
                and not self._is_auxiliary_token(tok)
                and not self._is_preposition_token(tok)
            ]
            predicate = predicate_candidates[-1] if predicate_candidates else predicate
            obj = patient

        preposition_index = next(
            (
                idx
                for idx, tok in enumerate(raw_tokens)
                if self._is_preposition_token(tok) and not (passive_aux and tok == "by")
            ),
            -1,
        )
        if not passive_aux and content_tokens:
            if preposition_index > 0:
                predicate_candidates = self._semantic_tokens(raw_tokens[:preposition_index])
                if predicate_candidates:
                    predicate = predicate_candidates[-1]
                preposition_object = self._first_meaningful_token(raw_tokens[preposition_index + 1 :])
                if preposition_object != "unknown":
                    obj = preposition_object
            elif len(content_tokens) == 1:
                predicate = content_tokens[0]
                obj = "unknown"
            elif len(content_tokens) == 2:
                predicate = content_tokens[1]
                obj = "unknown"
            else:
                predicate = content_tokens[1]
                obj = content_tokens[-1]
        if preposition_index > 0 and passive_aux:
            preposition_object = self._first_meaningful_token(raw_tokens[preposition_index + 1 :])
            if preposition_object != "unknown":
                obj = preposition_object

        proposition_parts = [part for part in (predicate, subject, obj) if part and part != "unknown"]
        proposition = ""
        if predicate != "unknown" and subject != "unknown":
            proposition = f"{predicate}({subject})"
        if predicate != "unknown" and obj != "unknown":
            proposition = f"{predicate}({subject}, {obj})" if subject != "unknown" else f"{predicate}({obj})"
        if not proposition and proposition_parts:
            proposition = " ".join(proposition_parts[:3])

        return {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "voice": voice,
            "agent": agent,
            "patient": patient,
            "grammatical_subject": grammatical_subject,
            "proposition": proposition,
            "confidence": 0.0,
            "role_bindings": {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "voice": voice,
                "agent": agent,
                "patient": patient,
                "grammatical_subject": grammatical_subject,
                "dialogue_act": dialogue_act,
                "role": role,
                "sentence_type": str(sentence_features.get("dominant_sentence_type", "fragment")),
            },
        }

    def _rank_tokens(self, tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        scores: Dict[str, float] = {}
        for idx, tok in enumerate(tokens):
            scores[tok] = scores.get(tok, 0.0) + 1.0 + 0.08 * idx
        ranked = sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
        return [tok for tok, _ in ranked]

    def _update_discourse_state(
        self,
        text: str,
        *,
        role: str,
        dialogue_act: str,
        sentence_features: Optional[Dict[str, Any]] = None,
    ) -> None:
        grammar_parse = self._grammar_sentence_parse(text)
        tokens = self._content_tokens(text)
        parsed_tokens = list(getattr(grammar_parse, "tokens", []) or [])
        pronouns = [tok.lower() for tok in parsed_tokens if self._is_pronoun_token(tok)]
        if not pronouns:
            pronouns = [tok.lower() for tok in self.agent._tokenize_words(text) if self._is_pronoun_token(tok)]
        sentence_features = sentence_features or self._sentence_features(text)
        last_sentence_type = str(sentence_features.get("dominant_sentence_type", "fragment"))
        self._update_relational_state(text, tokens=tokens, pronouns=pronouns, role=role, dialogue_act=dialogue_act, sentence_features=sentence_features)
        self._update_entity_registry(
            tokens,
            role=role,
            dialogue_act=dialogue_act,
            sentence_type=last_sentence_type,
            sentence_features=sentence_features,
        )

        if tokens:
            for tok in list(self.discourse_state.entity_salience):
                self.discourse_state.entity_salience[tok] *= 0.86
                if self.discourse_state.entity_salience[tok] < 0.01:
                    self.discourse_state.entity_salience.pop(tok, None)
            for tok in tokens:
                self.discourse_state.entity_salience[tok] = self.discourse_state.entity_salience.get(tok, 0.0) * 0.72 + 1.0

        ranked = sorted(self.discourse_state.entity_salience.items(), key=lambda item: item[1], reverse=True)
        self.discourse_state.active_entities = [tok for tok, _ in ranked[:4]]
        if self.discourse_state.active_entities:
            if self.discourse_state.topic == "unknown" or self.discourse_state.topic not in self.discourse_state.active_entities:
                if pronouns and self.discourse_state.topic != "unknown":
                    self.discourse_state.active_entities = list(dict.fromkeys([self.discourse_state.topic, *self.discourse_state.active_entities]))
                else:
                    self.discourse_state.topic = self.discourse_state.active_entities[0]
            elif self.discourse_state.topic_confidence < 0.25:
                self.discourse_state.topic = self.discourse_state.active_entities[0]

        if pronouns and self.discourse_state.active_entities:
            self.discourse_state.topic_confidence = min(1.0, self.discourse_state.topic_confidence + 0.10 * len(pronouns))
            self.discourse_state.pronoun_candidates = list(dict.fromkeys(pronouns + self.discourse_state.active_entities[:3]))
        elif tokens:
            self.discourse_state.topic_confidence = min(1.0, 0.70 * self.discourse_state.topic_confidence + 0.15 * min(3, len(tokens)))
            self.discourse_state.pronoun_candidates = self.discourse_state.active_entities[:3]
        else:
            self.discourse_state.topic_confidence = max(0.0, self.discourse_state.topic_confidence * 0.92)

        self.discourse_state.last_dialogue_act = dialogue_act or role
        self.discourse_state.last_sentence_type = last_sentence_type
        self.discourse_state.focus_stack = self.discourse_state.active_entities[:3]
        self.discourse_state.turn_index += 1
        self.discourse_state.discourse_summary = self._format_discourse_summary()
        self.situation_state.last_update_role = role
        self.situation_state.last_update_turn = self.discourse_state.turn_index

    def _update_relational_state(
        self,
        text: str,
        *,
        tokens: Optional[List[str]] = None,
        pronouns: Optional[List[str]] = None,
        role: str,
        dialogue_act: str,
        sentence_features: Optional[Dict[str, Any]] = None,
    ) -> None:
        raw_tokens = self._raw_tokens(text)
        grammar_parse = self._grammar_sentence_parse(text)
        tokens = list(tokens or self._content_tokens(text))
        pronouns = list(pronouns or [])
        sentence_features = sentence_features or self._sentence_features(text)
        sentence_type = str(sentence_features.get("dominant_sentence_type", "fragment"))
        segments = self._split_clause_segments(raw_tokens) if self._component_enabled("clause") else [raw_tokens]
        segments = segments or [raw_tokens]
        frames: List[Dict[str, Any]] = []
        inherited_subject = self.relational_state.subject if self.relational_state.subject != "unknown" else self.discourse_state.topic
        use_passive = self._component_enabled("passive")
        use_clause = self._component_enabled("clause")
        for segment in segments:
            frame = self._infer_clause_frame(
                segment,
                role=role,
                dialogue_act=dialogue_act,
                sentence_features=sentence_features,
                fallback_subject=inherited_subject,
                normalize_passive=use_passive,
            )
            if self._component_enabled("coreference") and frame["subject"] == "unknown" and self.discourse_state.active_entities:
                frame["subject"] = self.discourse_state.active_entities[0]
                frame["agent"] = frame["subject"]
                frame["role_bindings"]["subject"] = frame["subject"]
                frame["role_bindings"]["agent"] = frame["subject"]
            if self._component_enabled("registry") and frame["subject"] == "unknown" and role == "assistant" and self.relational_state.subject != "unknown":
                frame["subject"] = self.relational_state.subject
                frame["agent"] = frame["subject"]
                frame["role_bindings"]["subject"] = frame["subject"]
                frame["role_bindings"]["agent"] = frame["subject"]
            frames.append(frame)
            inherited_subject = frame["subject"] if frame["subject"] != "unknown" else inherited_subject

        if self._component_enabled("coreference") and pronouns and self.discourse_state.active_entities and frames:
            frames[-1]["subject"] = self.discourse_state.active_entities[0]
            frames[-1]["agent"] = frames[-1]["subject"]
            frames[-1]["role_bindings"]["subject"] = frames[-1]["subject"]
            frames[-1]["role_bindings"]["agent"] = frames[-1]["subject"]

        if use_clause and frames:
            self.relational_state.binding_stack.extend(frames)
            self.relational_state.binding_stack = self.relational_state.binding_stack[-4:]

        current = frames[-1] if frames else self._infer_clause_frame(
            raw_tokens,
            role=role,
            dialogue_act=dialogue_act,
            sentence_features=sentence_features,
            fallback_subject=inherited_subject,
        )
        grammar_svo: Dict[str, Any] = {}
        grammar_extract = getattr(self.grammar, "extract_svo", None)
        if callable(grammar_extract):
            try:
                grammar_svo = dict(grammar_extract(text) or {})
            except Exception:
                grammar_svo = {}
        elif grammar_parse is not None:
            parsed_tokens = list(getattr(grammar_parse, "tokens", []) or [])
            parsed_pos = list(getattr(grammar_parse, "pos_tags", []) or [])
            parsed_lemmas = list(getattr(grammar_parse, "lemmas", []) or [])
            if parsed_tokens and parsed_pos and parsed_lemmas:
                try:
                    subject = next((tok for tok, pos in zip(parsed_tokens, parsed_pos) if pos in {"PR", "NN", "NNP"}), "")
                    predicate_idx = next((i for i, pos in enumerate(parsed_pos) if pos == "VB"), -1)
                    obj = ""
                    for tok, pos in zip(reversed(parsed_tokens), reversed(parsed_pos)):
                        if pos in {"NN", "NNP", "PR"}:
                            obj = tok
                            break
                    grammar_svo = {
                        "subject": subject,
                        "predicate": parsed_lemmas[predicate_idx] if predicate_idx >= 0 else "",
                        "object": obj,
                        "voice": "active",
                    }
                except Exception:
                    grammar_svo = {}
        if grammar_svo:
            if grammar_svo.get("subject") and current["subject"] == "unknown":
                current["subject"] = str(grammar_svo.get("subject", "")).strip().lower()
                current["agent"] = current["subject"]
                current["role_bindings"]["subject"] = current["subject"]
                current["role_bindings"]["agent"] = current["subject"]
            if grammar_svo.get("predicate") and current["predicate"] == "unknown":
                current["predicate"] = str(grammar_svo.get("predicate", "")).strip().lower()
                current["role_bindings"]["predicate"] = current["predicate"]
            if grammar_svo.get("object") and current["object"] == "unknown":
                current["object"] = str(grammar_svo.get("object", "")).strip().lower()
                current["role_bindings"]["object"] = current["object"]
            if grammar_svo.get("voice") and current["voice"] == "active":
                current["voice"] = str(grammar_svo.get("voice", "active")).strip().lower()
                current["role_bindings"]["voice"] = current["voice"]
        if use_clause and frames:
            if frames[0]["voice"] == "passive":
                current = frames[0]
            elif len(frames) > 1 and frames[0]["subject"] != "unknown" and frames[-1]["predicate"] != "unknown":
                current = dict(frames[-1])
                current["subject"] = frames[0]["subject"]
                current["agent"] = frames[0]["subject"]
                current["grammatical_subject"] = frames[0]["subject"]
                current["role_bindings"] = dict(current.get("role_bindings", {}))
                current["role_bindings"]["subject"] = frames[0]["subject"]
                current["role_bindings"]["agent"] = frames[0]["subject"]
                current["role_bindings"]["grammatical_subject"] = frames[0]["subject"]

        self.relational_state.subject = current["subject"] if current["subject"] != "unknown" else "unknown"
        self.relational_state.predicate = current["predicate"] if current["predicate"] != "unknown" else "unknown"
        self.relational_state.object = current["object"] if current["object"] != "unknown" else "unknown"
        self.relational_state.role_bindings["subject"] = self.relational_state.subject
        self.relational_state.role_bindings["predicate"] = self.relational_state.predicate
        self.relational_state.role_bindings["object"] = self.relational_state.object
        self.relational_state.role_bindings["grammatical_subject"] = current["grammatical_subject"]
        self.relational_state.voice = current["voice"]
        self.relational_state.agent = current["agent"]
        self.relational_state.role_bindings["voice"] = current["voice"]
        self.relational_state.role_bindings["agent"] = current["agent"]
        if current["voice"] == "passive" and current["patient"] != "unknown":
            self.relational_state.role_bindings["patient"] = current["patient"]
        if current["proposition"]:
            self.relational_state.proposition = current["proposition"]
            self.relational_state.proposition_history.append(current["proposition"])
        else:
            self.relational_state.proposition = ""
        if use_clause and len(frames) > 1:
            self.relational_state.role_bindings["main_subject"] = frames[0]["subject"]
            self.relational_state.role_bindings["clause_count"] = str(len(frames))
        if self._component_enabled("coreference") and dialogue_act in {"question", "clarification"} and pronouns:
            self.relational_state.confidence = min(1.0, self.relational_state.confidence + 0.08)
        elif tokens:
            self.relational_state.confidence = min(1.0, 0.68 * self.relational_state.confidence + 0.12)
        else:
            self.relational_state.confidence = max(0.0, self.relational_state.confidence * 0.90)

        self._apply_action_schema_effects(current, text=text, raw_tokens=raw_tokens)

    def _format_discourse_summary(self) -> str:
        parts: List[str] = []
        if self.discourse_state.topic != "unknown":
            parts.append(f"topic={self.discourse_state.topic}")
        if self.discourse_state.active_entities:
            parts.append(f"entities={','.join(self.discourse_state.active_entities[:3])}")
        if self.relational_state.proposition:
            parts.append(f"prop={self.relational_state.proposition}")
        if self.discourse_state.last_dialogue_act:
            parts.append(f"act={self.discourse_state.last_dialogue_act}")
        if self.discourse_state.last_sentence_type:
            parts.append(f"sentence={self.discourse_state.last_sentence_type}")
        if self.discourse_state.pronoun_candidates:
            parts.append(f"refs={','.join(self.discourse_state.pronoun_candidates[:3])}")
        if self.discourse_state.archived_entities:
            parts.append(f"archive={','.join(self.discourse_state.archived_entities[:3])}")
        parts.append(f"confidence={self.discourse_state.topic_confidence:.2f}")
        return "; ".join(parts)

    def _discourse_context_features(self) -> Dict[str, Any]:
        return {
            "discourse_topic": self.discourse_state.topic,
            "topic_confidence": float(self.discourse_state.topic_confidence),
            "active_entity_count": len(self.discourse_state.active_entities),
            "active_entities": list(self.discourse_state.active_entities[:4]),
            "entity_registry_size": len(self.discourse_state.entity_registry),
            "entity_registry_top": [entity for entity, _ in sorted(
                self._entity_registry_scores(self.discourse_state.discourse_summary).items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )[:4]],
            "pronoun_candidates": list(self.discourse_state.pronoun_candidates[:4]),
            "focus_stack": list(self.discourse_state.focus_stack[:4]),
            "discourse_summary": self.discourse_state.discourse_summary,
            "last_dialogue_act": self.discourse_state.last_dialogue_act,
            "last_sentence_type": self.discourse_state.last_sentence_type,
            "turn_index": int(self.discourse_state.turn_index),
            "relational_subject": self.relational_state.subject,
            "relational_predicate": self.relational_state.predicate,
            "relational_object": self.relational_state.object,
            "relational_proposition": self.relational_state.proposition,
            "relational_confidence": float(self.relational_state.confidence),
            "relational_voice": self.relational_state.voice,
            "binding_stack_depth": len(self.relational_state.binding_stack),
        }

    def _lookup_entity_by_relation(self, subject_hint: str = "", predicate_hint: str = "") -> str:
        subject_hint = subject_hint.strip().lower()
        predicate_hint = predicate_hint.strip().lower()
        heuristic_scale = self._relation_heuristic_scale()
        best_entity = ""
        best_score = -1e9
        for entity, record in self.discourse_state.entity_registry.items():
            score = float(record.get("binding_confidence", 0.0)) + float(record.get("salience", 0.0))
            if heuristic_scale > 0.0 and subject_hint and (
                record.get("last_subject") == subject_hint
                or record.get("role_bindings", {}).get("subject") == subject_hint
            ):
                score += 1.0
            if heuristic_scale > 0.0 and predicate_hint and record.get("last_predicate") == predicate_hint:
                score += 1.0
            if heuristic_scale > 0.0 and predicate_hint and record.get("role_bindings", {}).get("predicate") == predicate_hint:
                score += 0.7
            if heuristic_scale > 0.0 and int(record.get("last_turn", -1)) >= max(0, self.discourse_state.turn_index - 4):
                score += 0.08
            if score > best_score:
                best_score = score
                best_entity = entity
        return best_entity

    def _lookup_property_for_entity(self, entity: str, property_hints: Iterable[str]) -> str:
        entity = entity.strip().lower()
        hints = {hint.strip().lower() for hint in property_hints if hint}
        if not entity or not hints:
            return ""
        heuristic_scale = self._relation_heuristic_scale()
        copulas = {"is", "are", "was", "were", "be", "been"}
        direct_record = self.discourse_state.entity_registry.get(entity)
        if direct_record is not None:
            direct_predicate = str(direct_record.get("last_predicate", "unknown")).lower()
            direct_subject = str(direct_record.get("last_subject", "unknown")).lower()
            direct_object = str(direct_record.get("last_object", "unknown")).lower()
            if direct_subject == entity:
                if direct_predicate in copulas and direct_object and direct_object != "unknown":
                    return direct_object
                if direct_predicate not in copulas and direct_predicate not in {"unknown", entity}:
                    return direct_predicate
        if hints.intersection({"color", "colour"}):
            direct_matches = sorted(
                self.discourse_state.entity_registry.items(),
                key=lambda item: (
                    int(item[1].get("last_turn", -1)),
                    float(item[1].get("binding_confidence", 0.0)),
                ),
                reverse=True,
            )
            for candidate, record in direct_matches:
                predicate = str(record.get("last_predicate", "unknown")).lower()
                subject_match = record.get("last_subject") == entity or record.get("role_bindings", {}).get("subject") == entity
                if subject_match:
                    if predicate in copulas:
                        value = str(record.get("last_object", "")).strip().lower()
                        if value and value != "unknown":
                            return value
                    if predicate not in copulas and predicate not in {"unknown", entity}:
                        return predicate
        best_value = ""
        best_score = -1e9
        for candidate, record in self.discourse_state.entity_registry.items():
            score = float(record.get("binding_confidence", 0.0)) + float(record.get("salience", 0.0))
            candidate_value = candidate
            if heuristic_scale > 0.0 and (record.get("last_subject") == entity or record.get("role_bindings", {}).get("subject") == entity):
                score += 0.8
            if heuristic_scale > 0.0 and (record.get("last_object") == entity or record.get("role_bindings", {}).get("object") == entity):
                score += 0.45
            predicate = str(record.get("last_predicate", "unknown")).lower()
            if heuristic_scale > 0.0 and (predicate in hints or record.get("role_bindings", {}).get("predicate") in hints):
                score += 1.0
            if heuristic_scale > 0.0 and hints.intersection({"color", "colour"}) and predicate in copulas and candidate != entity:
                score += 0.55
                if record.get("last_subject") == entity or record.get("role_bindings", {}).get("subject") == entity:
                    candidate_value = str(record.get("last_object", candidate)).strip().lower() or candidate_value
                    score += 0.35
            if heuristic_scale > 0.0 and candidate in hints:
                score += 0.25
            if score > best_score:
                best_score = score
                best_value = candidate_value
        return best_value

    def query_chain(self, question_text: str) -> Dict[str, Any]:
        question_text = question_text.strip()
        lower = question_text.lower()
        if not question_text:
            raise ValueError("question_text must not be empty")

        chain_frame = self._parse_chain_query_frame(question_text)
        if not chain_frame.get("matched"):
            return {
                "question": question_text,
                "answer": "unknown",
                "source": "chain_unavailable",
                "hops": 0,
            }

        subject_hint = str(chain_frame.get("subject_hint", "")).strip().lower()
        predicate_hint = str(chain_frame.get("predicate_hint", "")).strip().lower()
        property_hint = str(chain_frame.get("property_hint", "")).strip().lower()
        pivot_phrase = str(chain_frame.get("pivot_phrase", "")).strip().lower()
        if not pivot_phrase:
            return {
                "question": question_text,
                "answer": "unknown",
                "source": "chain_failed",
                "hops": 1,
                "intermediate_entity": "unknown",
            }
        resolution = self._resolve_entity_phrase(pivot_phrase, max_hops=2)
        intermediate_entity = str(resolution.get("entity", "")).strip().lower()
        if not intermediate_entity and subject_hint and predicate_hint:
            relation_key = self._lookup_entity_by_relation(subject_hint=subject_hint, predicate_hint=predicate_hint)
            if relation_key:
                relation_record = self.discourse_state.entity_registry.get(relation_key, {})
                intermediate_entity = str(relation_record.get("last_object", relation_key)).strip().lower()
                resolution = {
                    "entity": intermediate_entity,
                    "hops": 1,
                    "source": "relation_fallback",
                    "relation_key": relation_key,
                }
        if intermediate_entity == "unknown" or not intermediate_entity:
            intermediate_entity = ""

        answer = self._lookup_property_for_entity(intermediate_entity, {property_hint} if property_hint else set())
        if not answer:
            fallback_property = property_hint or "property"
            answer = self.query(f"What {fallback_property} is {intermediate_entity}?").get("answer", "unknown")
        return {
            "question": question_text,
            "answer": answer or "unknown",
            "source": "chain_resolved",
            "hops": int(resolution.get("hops", 0)) + 1,
            "intermediate_entity": intermediate_entity,
            "property_hint": property_hint,
            "relation_key": resolution.get("relation_key", ""),
            "pivot_source": resolution.get("source", ""),
            "pivot_phrase": pivot_phrase,
        }

    def query(self, question_text: str) -> Dict[str, Any]:
        question_text = question_text.strip()
        if not question_text:
            raise ValueError("question_text must not be empty")

        lower = question_text.lower()
        parsed = self._parse_query_frame(question_text)
        question_tokens = parsed["tokens"]
        content_tokens = parsed["content_tokens"]
        pronoun_tokens = [tok for tok in question_tokens if self._is_pronoun_token(tok)]
        resolution_source = "registry"
        answer = ""
        chain_details: Dict[str, Any] = {"hops": 0}

        if parsed["negated"]:
            answer = "unknown"
            resolution_source = "negated_query"
        else:
            possession_result = self._resolve_possession_query(question_text, parsed)
            if possession_result.get("matched"):
                answer = str(possession_result.get("answer", "")).strip().lower()
                resolution_source = str(possession_result.get("source", "world_model"))
            elif lower.startswith("who"):
                predicate_hint = parsed["predicate_hint"]
                subject_hint = parsed["subject_hint"]
                object_hint = parsed["object_hint"]
                if subject_hint:
                    answer = self._best_object_for_query(subject_hint=subject_hint, predicate_hint=predicate_hint)
                    resolution_source = "registry_object" if answer else "relational_object"
                    if self._component_enabled("registry") and not answer and self.relational_state.object != "unknown":
                        answer = self.relational_state.object
                else:
                    answer = self._best_subject_for_query(predicate_hint=predicate_hint, object_hint=object_hint)
                    resolution_source = "registry_subject" if answer else "relational_subject"
                    if self._component_enabled("registry") and not answer and self.relational_state.subject != "unknown":
                        answer = self.relational_state.subject
            elif pronoun_tokens:
                answer = self._best_entity_from_registry(question_text)
                if self._component_enabled("coreference") and not answer:
                    answer = self.discourse_state.topic
                resolution_source = "coreference"
            elif lower.startswith("what"):
                copula_query = any(self._is_auxiliary_token(tok) for tok in parsed["tokens"])
                if copula_query:
                    answer = self._best_entity_by_property(parsed["content_tokens"])
                    resolution_source = "property" if answer else "relational_property"
                    if self._component_enabled("registry") and not answer and self.relational_state.subject != "unknown":
                        answer = self.relational_state.subject
                else:
                    answer = self._best_subject_for_query(predicate_hint=parsed["predicate_hint"], object_hint=parsed["object_hint"])
                    resolution_source = "registry_subject" if answer else "relational_subject"
                    if self._component_enabled("registry") and not answer and self.relational_state.subject != "unknown":
                        answer = self.relational_state.subject
            elif any(phrase in lower for phrase in ("what is the topic", "what are we talking about", "what is this about")):
                answer = self.discourse_state.topic
                resolution_source = "topic"
            elif self._component_enabled("registry") and self.relational_state.proposition:
                answer = self.relational_state.proposition
                resolution_source = "proposition"
            else:
                answer = self._best_entity_from_registry(question_text) or self.discourse_state.topic

        chain_frame = self._parse_chain_query_frame(question_text)
        if chain_frame.get("matched"):
            chain_result = self.query_chain(question_text)
            if chain_result.get("answer") and chain_result.get("answer") != "unknown":
                answer = chain_result["answer"]
                resolution_source = chain_result.get("source", resolution_source)
            chain_details = dict(chain_result)

        if not answer:
            answer = "unknown"

        predicted_entity = self._predict_binding_entity(question_text)
        prediction_matched = bool(predicted_entity and answer != "unknown" and predicted_entity == answer)
        self.record_binding_feedback(
            predicted_entity=predicted_entity,
            observed_entity=answer,
            observed_text=question_text,
        )
        confidence = float(self.relational_state.confidence)
        registry_entry = self.discourse_state.entity_registry.get(answer)
        if registry_entry is not None:
            confidence = max(confidence, float(registry_entry.get("binding_confidence", 0.0)))
        elif answer == self.discourse_state.topic and self.discourse_state.topic != "unknown":
            confidence = max(confidence, float(self.discourse_state.topic_confidence))
        return {
            "question": question_text,
            "answer": answer,
            "confidence": confidence,
            "source": resolution_source,
            "topic": self.discourse_state.topic,
            "subject": self.relational_state.subject,
            "predicate": self.relational_state.predicate,
            "object": self.relational_state.object,
            "registry_size": len(self.discourse_state.entity_registry),
            "negated": bool(parsed["negated"]),
            "predicted_entity": predicted_entity,
            "prediction_matched": prediction_matched,
            **chain_details,
        }

    def _discourse_feedback_signal(self, text: str, *, role: str, dialogue_act: str) -> Dict[str, Any]:
        return {
            "kind": "discourse",
            "role": role,
            "dialogue_act": dialogue_act,
            **self._discourse_context_features(),
            **self.relational_state.to_dict(),
            **self._sentence_features(text),
        }

    def _discourse_bonus(self, text: str, user_text: str, discourse_state: DiscourseState) -> float:
        if discourse_state.topic == "unknown" or discourse_state.topic_confidence <= 0.0:
            return 0.0
        lower = text.lower()
        topic_hits = 0
        for tok in discourse_state.active_entities[:4]:
            if tok and tok in lower:
                topic_hits += 1
        if topic_hits == 0 and any(self._is_pronoun_token(pron) for pron in user_text.lower().split()):
            # If the user is referring back with pronouns, prefer responses that keep the topic visible.
            if discourse_state.topic in lower or any(tok in lower for tok in discourse_state.focus_stack[:2]):
                topic_hits += 1
        if topic_hits == 0:
            return -0.05 * min(1.0, discourse_state.topic_confidence)
        relation_bonus = 0.0
        if self.relational_state.subject != "unknown" and self.relational_state.subject in lower:
            relation_bonus += 0.06
        if self.relational_state.predicate != "unknown" and self.relational_state.predicate in lower:
            relation_bonus += 0.04
        if self.relational_state.object != "unknown" and self.relational_state.object in lower:
            relation_bonus += 0.04
        return min(0.22, 0.05 * topic_hits * max(0.3, discourse_state.topic_confidence) + relation_bonus)

    def _sentence_features(self, text: str) -> Dict[str, Any]:
        spans = self.sentence_adapter.segment(text)
        sentence_count = len(spans)
        dominant_type = spans[0].sentence_type if spans else "fragment"
        average_confidence = float(sum(span.confidence for span in spans) / max(1, sentence_count)) if spans else 0.0
        return {
            "sentence_count": sentence_count,
            "dominant_sentence_type": dominant_type,
            "sentence_confidence": average_confidence,
        }

    def _sentence_score(self, dialogue_act: str, sentence_features: Dict[str, Any]) -> float:
        sentence_count = int(sentence_features.get("sentence_count", 0))
        dominant_type = str(sentence_features.get("dominant_sentence_type", "fragment"))
        confidence = float(sentence_features.get("sentence_confidence", 0.0))
        score = 0.0
        if sentence_count == 1:
            score += 0.08
        elif sentence_count > 1:
            score -= min(0.15, 0.04 * (sentence_count - 1))
        if confidence > 0.6:
            score += 0.05
        if dialogue_act == "question" and dominant_type == "question":
            score += 0.14
        elif dialogue_act == "greeting" and dominant_type in {"declarative", "exclamation"}:
            score += 0.10
        elif dialogue_act == "request" and dominant_type in {"declarative", "question"}:
            score += 0.08
        elif dialogue_act == "clarification" and dominant_type in {"question", "clarification"}:
            score += 0.12
        elif dialogue_act == "closing" and dominant_type == "closing":
            score += 0.12
        return score

    def _target_sentence_score(
        self,
        sentence_features: Dict[str, Any],
        target_sentence_features: Dict[str, Any],
    ) -> float:
        score = 0.0
        target_count = int(target_sentence_features.get("sentence_count", 0))
        target_type = str(target_sentence_features.get("dominant_sentence_type", "fragment"))
        target_confidence = float(target_sentence_features.get("sentence_confidence", 0.0))
        generated_count = int(sentence_features.get("sentence_count", 0))
        generated_type = str(sentence_features.get("dominant_sentence_type", "fragment"))
        generated_confidence = float(sentence_features.get("sentence_confidence", 0.0))

        if target_count > 0:
            if generated_count == target_count:
                score += 0.18
            else:
                count_gap = abs(generated_count - target_count)
                score -= min(0.14, 0.04 * count_gap)
        if target_type != "fragment" and generated_type == target_type:
            score += 0.14
        if target_confidence > 0.5 and generated_confidence > 0.5:
            score += 0.05
        return score

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
        use_sentence_features: bool = True,
        dictionary: Optional[DictionaryValidator] = None,
        grammar: Optional[GrammarValidator] = None,
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
            use_sentence_features=use_sentence_features,
            dictionary=dictionary,
            grammar=grammar,
        )
        self._question_history: List[str] = []

    def ask(self, seed_text: str | None = None) -> str:
        question = self._generate_question(seed_text=seed_text)
        self._append("assistant", question)
        # Process assistant question (includes discourse update and evaluator bracketing)
        self.observe_text(
            question,
            role="assistant",
            dialogue_act="question",
            feedback_mode="none"
        )
        self._question_history.append(question)
        return question

    def answer_turn(self, answer_text: str) -> ChatResult:
        answer_text = answer_text.strip()
        if not answer_text:
            raise ValueError("answer_text must not be empty")

        prior_questions = [turn.text for turn in self._recent_turns() if turn.role == "assistant" and turn.text.strip()]
        dialogue_act = self._dialogue_act(answer_text)
        self._append("user", answer_text)

        # Process user answer
        learn_stats = self.observe_text(
            answer_text,
            role="user",
            dialogue_act=dialogue_act,
            feedback_mode="target" if self.learn_from_user else "none"
        )

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
            context_texts=[answer_text, self.discourse_state.discourse_summary, *prior_questions],
        )
        self._append("assistant", question_text)
        # Process assistant question
        response_stats = self.observe_text(
            question_text,
            role="assistant",
            dialogue_act="question",
            feedback_mode="target" if self.learn_from_reply else "none",
            feedback_signal=question_signal.to_dict() if question_signal else None
        )

        if question_signal:
            response_stats.update(question_signal.to_dict())
            response_stats["text_signal_score"] = question_signal.combined_score()

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
        context_features.update(self._discourse_context_features())

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
    surface_mode: str = "word",
    library_path: Optional[str] = None,
    checkpoint_dir: str = ".",
    seed_corpus_path: Optional[str] = CHAT_SEED_CORPUS,
) -> List[Dict[str, Any]]:
    """Run a small scripted chat benchmark."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar, surface_mode=surface_mode)

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
        dictionary=dictionary,
        grammar=grammar,
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
    surface_mode: str = "word",
    library_path: Optional[str] = None,
    checkpoint_dir: str = ".",
    seed_corpus_path: Optional[str] = CHAT_SEED_CORPUS,
) -> List[Dict[str, Any]]:
    """Run a small scripted reverse-chat benchmark where the model asks and the user answers."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(num_workers=num_workers, dictionary=dictionary, grammar=grammar, surface_mode=surface_mode)

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
        dictionary=dictionary,
        grammar=grammar,
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


DEFAULT_BINDING_BENCHMARK_CASES: List[Dict[str, Any]] = [
    {
        "statements": ["The cat chased the dog."],
        "question": "Who chased the dog?",
        "expected": "cat",
    },
    {
        "statements": ["The dog was chased by the cat."],
        "question": "Who chased the dog?",
        "expected": "cat",
    },
    {
        "statements": ["The cat sat on the mat.", "The mat is red."],
        "question": "What colour is the thing the cat sat on?",
        "expected": "red",
    },
]


HARD_BINDING_BENCHMARK_CASES: List[Dict[str, Any]] = [
    {
        "statements": [
            "The cat that the dog chased sat on the mat.",
            "The mat was red.",
        ],
        "question": "Who sat on the mat?",
        "expected": "cat",
    },
    {
        "statements": [
            "The scientist who the student admired wrote a book.",
            "The book was blue.",
        ],
        "question": "Who wrote a book?",
        "expected": "scientist",
    },
    {
        "statements": [
            "The robot saw the cat.",
            "It then moved toward the mat.",
        ],
        "question": "What moved toward the mat?",
        "expected": "robot",
    },
    {
        "statements": [
            "The dog was chased by the cat that the child liked.",
            "The child was young.",
        ],
        "question": "Who chased the dog?",
        "expected": "cat",
    },
    {
        "statements": [
            "The trophy that the dog noticed was red.",
            "The dog slept.",
        ],
        "question": "What was red?",
        "expected": "trophy",
    },
]


RELATIONAL_EMERGENCE_BENCHMARK_CASES: List[Dict[str, Any]] = [
    {
        "statements": ["The dog chased the cat."],
        "question": "Who chased the cat?",
        "expected": "dog",
        "relation_class": "active",
    },
    {
        "statements": ["The cat was chased by the dog."],
        "question": "Who chased the cat?",
        "expected": "dog",
        "relation_class": "passive",
    },
    {
        "statements": ["The robot moved toward the mat."],
        "question": "What moved toward the mat?",
        "expected": "robot",
        "relation_class": "active",
    },
    {
        "statements": ["The mat was approached by the robot."],
        "question": "What was approached by the robot?",
        "expected": "mat",
        "relation_class": "passive",
    },
    {
        "statements": ["The cat that the dog chased sat on the mat."],
        "question": "Who sat on the mat?",
        "expected": "cat",
        "relation_class": "nested_active",
    },
    {
        "statements": ["The mat was sat on by the cat that the dog chased."],
        "question": "What was sat on by the cat that the dog chased?",
        "expected": "mat",
        "relation_class": "nested_passive",
    },
]

RELATIONAL_EMERGENCE_CORPUS_SPEC: Dict[str, Any] = {
    "name": "relational_emergence",
    "surface_mode": "word",
    "heuristic_ablation": ["heuristics_on", "heuristics_off"],
    "train_scale_hint": "50k+ relational sentences",
    "families": [
        {"name": "active_voice", "role_pattern": "subject-verb-object"},
        {"name": "passive_voice", "role_pattern": "patient-be-by-agent"},
        {"name": "nested_clause", "role_pattern": "main-clause-with-relative-clause"},
    ],
    "probe_families": [
        "active",
        "passive",
        "nested_active",
        "nested_passive",
    ],
    "latent_probe": {
        "layer": "l3",
        "latent_width": 8,
        "soft_state_bins": 8,
        "soft_state_codes": 16,
    },
}

RELATIONAL_COMPONENT_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "full",
        "use_relational_heuristics": True,
        "use_passive_normalization": True,
        "use_clause_stack": True,
        "use_registry_scoring": True,
        "use_coreference_scoring": True,
        "use_property_scoring": True,
    },
    {
        "name": "no_passive",
        "use_relational_heuristics": True,
        "use_passive_normalization": False,
        "use_clause_stack": True,
        "use_registry_scoring": True,
        "use_coreference_scoring": True,
        "use_property_scoring": True,
    },
    {
        "name": "no_clause",
        "use_relational_heuristics": True,
        "use_passive_normalization": True,
        "use_clause_stack": False,
        "use_registry_scoring": True,
        "use_coreference_scoring": True,
        "use_property_scoring": True,
    },
    {
        "name": "no_registry",
        "use_relational_heuristics": True,
        "use_passive_normalization": True,
        "use_clause_stack": True,
        "use_registry_scoring": False,
        "use_coreference_scoring": True,
        "use_property_scoring": True,
    },
    {
        "name": "all_off",
        "use_relational_heuristics": False,
        "use_passive_normalization": False,
        "use_clause_stack": False,
        "use_registry_scoring": False,
        "use_coreference_scoring": False,
        "use_property_scoring": False,
    },
]


def _relational_probe_kind(question: str) -> str:
    lower = question.strip().lower()
    if not lower:
        return "unknown"
    if "what colour" in lower or "what color" in lower or "what shape" in lower:
        return "property"
    if "who" in lower:
        return "role"
    if "what" in lower:
        return "role_or_property"
    return "other"


def run_binding_evaluator_benchmark(
    corpus_path: str,
    cases: Optional[List[Dict[str, Any]]] = None,
    warmup_chars: int = 256,
    history_window: int = 2,
    response_steps: int = 16,
    num_workers: int = 1,
    use_dict: bool = False,
    surface_mode: str = "word",
    layer_latent_dims: Optional[Dict[str, int]] = None,
    library_path: Optional[str] = None,
    checkpoint_dir: str = ".",
    seed_corpus_path: Optional[str] = CHAT_SEED_CORPUS,
    report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a tiny binding QA benchmark over scripted relation pairs.

    The benchmark measures the predictor/evaluator loop directly:
    - binding prediction counts, hits, and misses while observing statements
    - query accuracy on the corresponding question
    - whether the final answer matches the expected entity/property
    """
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    layered = LayeredAgent(
        num_workers=num_workers,
        dictionary=dictionary,
        grammar=grammar,
        surface_mode=surface_mode,
        layer_latent_dims=layer_latent_dims,
    )

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

    benchmark_cases = cases or DEFAULT_BINDING_BENCHMARK_CASES
    case_reports: List[Dict[str, Any]] = []

    for idx, case in enumerate(benchmark_cases, start=1):
        session = BasicChatSession(
            layered,
            history_window=history_window,
            response_steps=response_steps,
            use_constraints=use_dict,
        )
        statement_reports: List[Dict[str, Any]] = []
        for statement in case.get("statements", []):
            stats = session.observe_text(statement, role="user", feedback_mode="target")
            statement_reports.append({
                "statement": statement,
                "binding_predictions": int(stats.get("binding_predictions", 0)),
                "binding_prediction_hits": int(stats.get("binding_prediction_hits", 0)),
                "binding_prediction_misses": int(stats.get("binding_prediction_misses", 0)),
            })

        query_result = session.query(str(case.get("question", "")))
        expected = str(case.get("expected", "")).strip().lower()
        answer = str(query_result.get("answer", "")).strip().lower()
        answer_correct = bool(expected and answer == expected)

        total_predictions = sum(item["binding_predictions"] for item in statement_reports)
        total_hits = sum(item["binding_prediction_hits"] for item in statement_reports)
        total_misses = sum(item["binding_prediction_misses"] for item in statement_reports)
        prediction_accuracy = float(total_hits / max(1, total_predictions))

        case_reports.append({
            "case": idx,
            "question": case.get("question", ""),
            "expected": case.get("expected", ""),
            "answer": query_result.get("answer", "unknown"),
            "answer_correct": answer_correct,
            "source": query_result.get("source", ""),
            "confidence": float(query_result.get("confidence", 0.0)),
            "predicted_entity": query_result.get("predicted_entity", ""),
            "prediction_matched": bool(query_result.get("prediction_matched", False)),
            "binding_predictions": total_predictions,
            "binding_prediction_hits": total_hits,
            "binding_prediction_misses": total_misses,
            "binding_prediction_accuracy": prediction_accuracy,
            "statement_reports": statement_reports,
        })

    aggregate = {
        "case_count": len(case_reports),
        "avg_answer_accuracy": float(sum(1.0 if row["answer_correct"] else 0.0 for row in case_reports) / max(1, len(case_reports))),
        "avg_binding_prediction_accuracy": float(sum(row["binding_prediction_accuracy"] for row in case_reports) / max(1, len(case_reports))),
        "avg_confidence": float(sum(float(row["confidence"]) for row in case_reports) / max(1, len(case_reports))),
        "avg_prediction_matched_rate": float(sum(1.0 if row["prediction_matched"] else 0.0 for row in case_reports) / max(1, len(case_reports))),
        "binding_prediction_hits": int(sum(int(row["binding_prediction_hits"]) for row in case_reports)),
        "binding_prediction_misses": int(sum(int(row["binding_prediction_misses"]) for row in case_reports)),
        "binding_predictions": int(sum(int(row["binding_predictions"]) for row in case_reports)),
    }

    report = {
        "aggregate": aggregate,
        "cases": case_reports,
        "surface_mode": surface_mode,
        "library_path": resolved_library_path,
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    out_path = report_path or os.path.join(checkpoint_dir, "binding_evaluator_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"Binding evaluator benchmark written: {out_path}")

    return report


def run_binding_width_sweep_benchmark(
    corpus_path: str,
    widths: Optional[List[int]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the binding benchmark across multiple L3 latent widths.

    The first width is treated as the baseline and later widths are compared
    against it. This stays small and reuses the existing benchmark path.
    """
    sweep_widths = list(widths or [8, 12, 16])
    if not sweep_widths:
        raise ValueError("widths must contain at least one latent width")

    cases = kwargs.pop("cases", None)
    benchmark_kwargs = dict(kwargs)

    arms: List[Dict[str, Any]] = []
    for width in sweep_widths:
        report = run_binding_evaluator_benchmark(
            corpus_path=corpus_path,
            cases=cases,
            layer_latent_dims={"l3": int(width)},
            **benchmark_kwargs,
        )
        arms.append({
            "l3_width": int(width),
            "report": report,
        })

    baseline = arms[0]["report"]["aggregate"] if arms else {}
    comparison: List[Dict[str, Any]] = []
    for arm in arms:
        agg = arm["report"]["aggregate"]
        comparison.append({
            "l3_width": arm["l3_width"],
            "avg_answer_accuracy": float(agg.get("avg_answer_accuracy", 0.0)),
            "avg_binding_prediction_accuracy": float(agg.get("avg_binding_prediction_accuracy", 0.0)),
            "avg_confidence": float(agg.get("avg_confidence", 0.0)),
            "delta_answer_accuracy": float(agg.get("avg_answer_accuracy", 0.0) - float(baseline.get("avg_answer_accuracy", 0.0))),
            "delta_binding_prediction_accuracy": float(agg.get("avg_binding_prediction_accuracy", 0.0) - float(baseline.get("avg_binding_prediction_accuracy", 0.0))),
        })

    report = {
        "baseline_width": arms[0]["l3_width"],
        "arms": arms,
        "comparison": comparison,
    }

    checkpoint_dir = str(benchmark_kwargs.get("checkpoint_dir", "."))
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_path = benchmark_kwargs.get("report_path") or os.path.join(checkpoint_dir, "binding_width_sweep_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"Binding width sweep benchmark written: {out_path}")
    return report


def _safe_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    cosine = float(np.dot(a, b) / denom)
    return float(max(0.0, 1.0 - cosine))


def _class_distribution_separation(vectors_by_label: Dict[str, List[np.ndarray]]) -> Dict[str, Any]:
    class_means: Dict[str, np.ndarray] = {}
    within: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for label, vectors in vectors_by_label.items():
        if not vectors:
            continue
        stacked = np.vstack([np.asarray(vec, dtype=np.float32).ravel() for vec in vectors])
        mean_vec = stacked.mean(axis=0)
        mean_vec = mean_vec / (mean_vec.sum() + 1e-12)
        class_means[label] = mean_vec
        counts[label] = len(vectors)
        within[label] = float(np.mean([_safe_cosine_distance(vec, mean_vec) for vec in stacked])) if len(vectors) > 1 else 0.0

    labels = sorted(class_means)
    between: Dict[str, float] = {}
    for idx, left in enumerate(labels):
        for right in labels[idx + 1:]:
            between[f"{left}_vs_{right}"] = _safe_cosine_distance(class_means[left], class_means[right])

    avg_within = float(np.mean(list(within.values()))) if within else 0.0
    avg_between = float(np.mean(list(between.values()))) if between else 0.0
    separation_ratio = float(avg_between / max(1e-12, avg_within)) if between else 0.0
    return {
        "class_means": {label: vec.tolist() for label, vec in class_means.items()},
        "counts": counts,
        "within": within,
        "between": between,
        "avg_within": avg_within,
        "avg_between": avg_between,
        "separation_ratio": separation_ratio,
    }


RELATIONAL_SEGMENTATION_CUES = {
    "was",
    "were",
    "been",
    "by",
    "that",
    "which",
    "who",
    "whom",
    "whose",
    "chased",
    "chase",
    "sat",
    "gave",
    "give",
    "said",
    "say",
    "moved",
    "move",
    "approached",
    "approach",
    "looked",
    "look",
    "saw",
    "see",
    "told",
    "tell",
}


def _token_char_spans(text: str) -> List[Tuple[str, int, int]]:
    spans: List[Tuple[str, int, int]] = []
    for match in re.finditer(r"[A-Za-z']+", text):
        spans.append((match.group(0).lower(), match.start(), match.end()))
    return spans


def _role_segmentation_alignment(path: Sequence[int], text: str) -> Dict[str, Any]:
    spans = _token_char_spans(text)
    if not spans or not path:
        return {
            "boundary_rate": 0.0,
            "background_rate": 0.0,
            "alignment_ratio": 0.0,
            "boundary_markers": 0,
            "background_tokens": 0,
        }

    path_arr = np.asarray(list(path), dtype=np.int32).ravel()
    path_len = min(len(path_arr), len(text))
    if path_len <= 1:
        return {
            "boundary_rate": 0.0,
            "background_rate": 0.0,
            "alignment_ratio": 0.0,
            "boundary_markers": 0,
            "background_tokens": 0,
        }

    marker_positions: List[int] = []
    background_positions: List[int] = []
    for token, start, _ in spans:
        if token in RELATIONAL_SEGMENTATION_CUES:
            marker_positions.append(start)
        else:
            background_positions.append(start)

    def _change_at(pos: int) -> float:
        if pos <= 0 or pos >= path_len:
            return 0.0
        return 1.0 if path_arr[pos] != path_arr[pos - 1] else 0.0

    boundary_rate = float(np.mean([_change_at(pos) for pos in marker_positions])) if marker_positions else 0.0
    background_positions = [pos for pos in background_positions if 0 < pos < path_len]
    background_rate = float(np.mean([_change_at(pos) for pos in background_positions])) if background_positions else 0.0
    return {
        "boundary_rate": boundary_rate,
        "background_rate": background_rate,
        "alignment_ratio": float(boundary_rate / max(1e-12, background_rate)) if marker_positions else 0.0,
        "boundary_markers": len(marker_positions),
        "background_tokens": len(background_positions),
    }


def run_relational_emergence_benchmark(
    corpus_path: str,
    cases: Optional[List[Dict[str, Any]]] = None,
    warmup_chars: int = 256,
    history_window: int = 2,
    response_steps: int = 16,
    num_workers: int = 1,
    use_dict: bool = False,
    surface_mode: str = "word",
    layer_latent_dims: Optional[Dict[str, int]] = None,
    library_path: Optional[str] = None,
    checkpoint_dir: str = ".",
    seed_corpus_path: Optional[str] = CHAT_SEED_CORPUS,
    report_path: Optional[str] = None,
    compare_ablation: bool = True,
    session_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Measure whether relation structure survives without heuristic binding shortcuts."""
    dictionary = NLTKWordList(download=False) if use_dict else None
    grammar = HeuristicGrammarLibrary() if use_dict else None
    benchmark_cases = cases or RELATIONAL_EMERGENCE_BENCHMARK_CASES

    arm_specs = [("heuristics_on", True)]
    if compare_ablation:
        arm_specs.append(("heuristics_off", False))
    session_kwargs = dict(session_kwargs or {})
    if "use_relational_heuristics" in session_kwargs:
        compare_ablation = False

    arms: List[Dict[str, Any]] = []
    matrix: List[Dict[str, Any]] = []
    for arm_name, use_relational_heuristics in arm_specs:
        session_kwargs_for_arm = dict(session_kwargs)
        if "use_relational_heuristics" in session_kwargs_for_arm:
            use_relational_heuristics = bool(session_kwargs_for_arm.pop("use_relational_heuristics"))
        layered = LayeredAgent(
            num_workers=num_workers,
            dictionary=dictionary,
            grammar=grammar,
            surface_mode=surface_mode,
            layer_latent_dims=layer_latent_dims,
        )

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

        case_reports: List[Dict[str, Any]] = []
        voice_vectors: Dict[str, List[np.ndarray]] = {"active": [], "passive": []}
        class_vectors: Dict[str, List[np.ndarray]] = {}

        for idx, case in enumerate(benchmark_cases, start=1):
            session = BasicChatSession(
                layered,
                history_window=history_window,
                response_steps=response_steps,
                use_constraints=use_dict,
                use_relational_heuristics=use_relational_heuristics,
                **session_kwargs_for_arm,
            )
            statement_reports: List[Dict[str, Any]] = []
            for statement in case.get("statements", []):
                l2_start = len(getattr(layered, "_l2_state_history", []))
                stats = session.observe_text(statement, role="user", feedback_mode="target")
                l2_slice = list(getattr(layered, "_l2_state_history", [])[l2_start:])
                l3_path = layered.l3_viterbi_path(l2_slice)
                role_alignment = _role_segmentation_alignment(l3_path, statement)
                statement_reports.append({
                    "statement": statement,
                    "binding_predictions": int(stats.get("binding_predictions", 0)),
                    "binding_prediction_hits": int(stats.get("binding_prediction_hits", 0)),
                    "binding_prediction_misses": int(stats.get("binding_prediction_misses", 0)),
                    "role_segmentation_boundary_rate": float(role_alignment.get("boundary_rate", 0.0)),
                    "role_segmentation_background_rate": float(role_alignment.get("background_rate", 0.0)),
                    "role_segmentation_alignment_ratio": float(role_alignment.get("alignment_ratio", 0.0)),
                    "role_segmentation_boundary_markers": int(role_alignment.get("boundary_markers", 0)),
                })

            l3_distribution = layered.l3_state_distribution()
            l3_soft_state = int(layered.l3_soft_state())
            relation_class = str(case.get("relation_class", "unknown"))
            class_vectors.setdefault(relation_class, []).append(np.asarray(l3_distribution, dtype=np.float32))
            voice_key = "passive" if "passive" in relation_class else "active"
            voice_vectors.setdefault(voice_key, []).append(np.asarray(l3_distribution, dtype=np.float32))

            query_result = session.query(str(case.get("question", "")))
            expected = str(case.get("expected", "")).strip().lower()
            answer = str(query_result.get("answer", "")).strip().lower()
            answer_correct = bool(expected and answer == expected)

            total_predictions = sum(item["binding_predictions"] for item in statement_reports)
            total_hits = sum(item["binding_prediction_hits"] for item in statement_reports)
            total_misses = sum(item["binding_prediction_misses"] for item in statement_reports)
            prediction_accuracy = float(total_hits / max(1, total_predictions))

            case_reports.append({
                "case": idx,
                "relation_class": relation_class,
                "question": case.get("question", ""),
                "expected": case.get("expected", ""),
                "answer": query_result.get("answer", "unknown"),
                "answer_correct": answer_correct,
                "source": query_result.get("source", ""),
                "confidence": float(query_result.get("confidence", 0.0)),
                "predicted_entity": query_result.get("predicted_entity", ""),
                "prediction_matched": bool(query_result.get("prediction_matched", False)),
                "binding_predictions": total_predictions,
                "binding_prediction_hits": total_hits,
                "binding_prediction_misses": total_misses,
                "binding_prediction_accuracy": prediction_accuracy,
                "l3_soft_state": l3_soft_state,
                "l3_distribution": [float(v) for v in np.asarray(l3_distribution, dtype=np.float32).ravel().tolist()],
                "role_segmentation_boundary_rate": float(np.mean([float(item["role_segmentation_boundary_rate"]) for item in statement_reports]) if statement_reports else 0.0),
                "role_segmentation_background_rate": float(np.mean([float(item["role_segmentation_background_rate"]) for item in statement_reports]) if statement_reports else 0.0),
                "role_segmentation_alignment_ratio": float(np.mean([float(item["role_segmentation_alignment_ratio"]) for item in statement_reports]) if statement_reports else 0.0),
                "statement_reports": statement_reports,
            })
            matrix.append({
                "arm": arm_name,
                "use_relational_heuristics": use_relational_heuristics,
                "case": idx,
                "relation_class": relation_class,
                "probe_kind": _relational_probe_kind(str(case.get("question", ""))),
                "expected": case.get("expected", ""),
                "answer": query_result.get("answer", "unknown"),
                "answer_correct": answer_correct,
                "l3_soft_state": l3_soft_state,
                "binding_prediction_accuracy": prediction_accuracy,
                "role_segmentation_alignment_ratio": float(np.mean([float(item["role_segmentation_alignment_ratio"]) for item in statement_reports]) if statement_reports else 0.0),
            })

        voice_separation = _class_distribution_separation(voice_vectors)
        class_separation = _class_distribution_separation(class_vectors)

        aggregate = {
            "case_count": len(case_reports),
            "avg_answer_accuracy": float(sum(1.0 if row["answer_correct"] else 0.0 for row in case_reports) / max(1, len(case_reports))),
            "avg_binding_prediction_accuracy": float(sum(row["binding_prediction_accuracy"] for row in case_reports) / max(1, len(case_reports))),
            "avg_confidence": float(sum(float(row["confidence"]) for row in case_reports) / max(1, len(case_reports))),
            "avg_prediction_matched_rate": float(sum(1.0 if row["prediction_matched"] else 0.0 for row in case_reports) / max(1, len(case_reports))),
            "binding_prediction_hits": int(sum(int(row["binding_prediction_hits"]) for row in case_reports)),
            "binding_prediction_misses": int(sum(int(row["binding_prediction_misses"]) for row in case_reports)),
            "binding_predictions": int(sum(int(row["binding_predictions"]) for row in case_reports)),
            "avg_role_segmentation_boundary_rate": float(sum(float(row["role_segmentation_boundary_rate"]) for row in case_reports) / max(1, len(case_reports))),
            "avg_role_segmentation_background_rate": float(sum(float(row["role_segmentation_background_rate"]) for row in case_reports) / max(1, len(case_reports))),
            "avg_role_segmentation_alignment_ratio": float(sum(float(row["role_segmentation_alignment_ratio"]) for row in case_reports) / max(1, len(case_reports))),
            "active_passive_avg_between": float(voice_separation.get("avg_between", 0.0)),
            "active_passive_avg_within": float(voice_separation.get("avg_within", 0.0)),
            "active_passive_separation_ratio": float(voice_separation.get("separation_ratio", 0.0)),
        }

        arms.append({
            "arm": arm_name,
            "use_relational_heuristics": use_relational_heuristics,
            "aggregate": aggregate,
            "cases": case_reports,
            "active_passive_separation": voice_separation,
            "class_separation": class_separation,
        })

    baseline = arms[0]["aggregate"] if arms else {}
    comparison: List[Dict[str, Any]] = []
    for arm in arms:
        agg = arm["aggregate"]
        comparison.append({
            "arm": arm["arm"],
            "use_relational_heuristics": bool(arm["use_relational_heuristics"]),
            "avg_answer_accuracy": float(agg.get("avg_answer_accuracy", 0.0)),
            "avg_binding_prediction_accuracy": float(agg.get("avg_binding_prediction_accuracy", 0.0)),
            "active_passive_separation_ratio": float(agg.get("active_passive_separation_ratio", 0.0)),
            "delta_answer_accuracy": float(agg.get("avg_answer_accuracy", 0.0) - float(baseline.get("avg_answer_accuracy", 0.0))),
            "delta_binding_prediction_accuracy": float(agg.get("avg_binding_prediction_accuracy", 0.0) - float(baseline.get("avg_binding_prediction_accuracy", 0.0))),
            "delta_active_passive_separation_ratio": float(agg.get("active_passive_separation_ratio", 0.0) - float(baseline.get("active_passive_separation_ratio", 0.0))),
        })

    report = {
        "corpus_spec": RELATIONAL_EMERGENCE_CORPUS_SPEC,
        "baseline_arm": arms[0]["arm"] if arms else "",
        "arms": arms,
        "comparison": comparison,
        "matrix": matrix,
        "surface_mode": surface_mode,
        "library_path": _resolve_chat_library_path(library_path),
        "layer_latent_dims": dict(layer_latent_dims or {}),
    }

    os.makedirs(checkpoint_dir, exist_ok=True)
    out_path = report_path or os.path.join(checkpoint_dir, "relational_emergence_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"Relational emergence benchmark written: {out_path}")
    return report


def run_relational_emergence_sweep_benchmark(
    corpus_path: str,
    sentence_counts: Optional[List[int]] = None,
    corpus_modes: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the relational-emergence benchmark at increasing corpus sizes."""
    from hpm_ai_v4.simulations.build_relational_corpus import build_relational_corpus
    from hpm_ai_v4.simulations.build_relational_mixed_corpus import build_relational_mixed_corpus

    sweep_counts = list(sentence_counts or [5000, 10000, 50000])
    if not sweep_counts:
        raise ValueError("sentence_counts must contain at least one size")
    sweep_modes = list(corpus_modes or ["focused", "mixed"])
    if not sweep_modes:
        raise ValueError("corpus_modes must contain at least one mode")

    benchmark_kwargs = dict(kwargs)
    cases = benchmark_kwargs.pop("cases", None)
    checkpoint_dir = str(benchmark_kwargs.get("checkpoint_dir", "."))
    os.makedirs(checkpoint_dir, exist_ok=True)

    arms: List[Dict[str, Any]] = []
    for mode in sweep_modes:
        for count in sweep_counts:
            scale_corpus_path = os.path.join(checkpoint_dir, f"relational_{mode}_corpus_{int(count)}.txt")
            if mode == "mixed":
                corpus_result = build_relational_mixed_corpus(scale_corpus_path, sentence_count=int(count))
            else:
                corpus_result = build_relational_corpus(scale_corpus_path, sentence_count=int(count))
            report = run_relational_emergence_benchmark(
                corpus_path=corpus_result.corpus_path,
                cases=cases,
                **benchmark_kwargs,
            )
            arms.append({
                "corpus_mode": mode,
                "sentence_count": int(count),
                "corpus_path": corpus_result.corpus_path,
                "family_counts": corpus_result.family_counts,
                "report": report,
            })

    baseline = arms[0]["report"]["arms"][0]["aggregate"] if arms else {}
    comparison: List[Dict[str, Any]] = []
    for arm in arms:
        heuristic_on = arm["report"]["arms"][0]["aggregate"]
        heuristic_off = arm["report"]["arms"][1]["aggregate"] if len(arm["report"]["arms"]) > 1 else {}
        comparison.append({
            "corpus_mode": arm["corpus_mode"],
            "sentence_count": arm["sentence_count"],
            "heuristics_on_answer_accuracy": float(heuristic_on.get("avg_answer_accuracy", 0.0)),
            "heuristics_off_answer_accuracy": float(heuristic_off.get("avg_answer_accuracy", 0.0)),
            "heuristics_on_binding_accuracy": float(heuristic_on.get("avg_binding_prediction_accuracy", 0.0)),
            "heuristics_off_binding_accuracy": float(heuristic_off.get("avg_binding_prediction_accuracy", 0.0)),
            "heuristics_on_separation_ratio": float(heuristic_on.get("active_passive_separation_ratio", 0.0)),
            "heuristics_off_separation_ratio": float(heuristic_off.get("active_passive_separation_ratio", 0.0)),
            "delta_answer_accuracy": float(heuristic_on.get("avg_answer_accuracy", 0.0) - float(baseline.get("avg_answer_accuracy", 0.0))),
            "delta_binding_accuracy": float(heuristic_on.get("avg_binding_prediction_accuracy", 0.0) - float(baseline.get("avg_binding_prediction_accuracy", 0.0))),
            "delta_separation_ratio": float(heuristic_on.get("active_passive_separation_ratio", 0.0) - float(baseline.get("active_passive_separation_ratio", 0.0))),
        })

    report = {
        "baseline_sentence_count": arms[0]["sentence_count"] if arms else 0,
        "baseline_corpus_mode": arms[0]["corpus_mode"] if arms else "",
        "arms": arms,
        "comparison": comparison,
        "corpus_path": corpus_path,
        "corpus_modes": sweep_modes,
    }

    out_path = benchmark_kwargs.get("report_path") or os.path.join(checkpoint_dir, "relational_emergence_sweep_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"Relational emergence sweep benchmark written: {out_path}")
    return report


def run_relational_component_sweep_benchmark(
    corpus_path: str,
    sentence_counts: Optional[List[int]] = None,
    corpus_modes: Optional[List[str]] = None,
    configs: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the relational benchmark across corpus modes and heuristic-component configs."""
    from hpm_ai_v4.simulations.build_relational_corpus import build_relational_corpus
    from hpm_ai_v4.simulations.build_relational_mixed_corpus import build_relational_mixed_corpus

    sweep_counts = list(sentence_counts or [5000, 10000, 50000])
    sweep_modes = list(corpus_modes or ["focused", "mixed"])
    sweep_configs = list(configs or RELATIONAL_COMPONENT_CONFIGS)
    if not sweep_counts:
        raise ValueError("sentence_counts must contain at least one size")
    if not sweep_modes:
        raise ValueError("corpus_modes must contain at least one mode")
    if not sweep_configs:
        raise ValueError("configs must contain at least one configuration")

    benchmark_kwargs = dict(kwargs)
    cases = benchmark_kwargs.pop("cases", None)
    checkpoint_dir = str(benchmark_kwargs.get("checkpoint_dir", "."))
    os.makedirs(checkpoint_dir, exist_ok=True)

    arms: List[Dict[str, Any]] = []
    matrix: List[Dict[str, Any]] = []

    for mode in sweep_modes:
        for count in sweep_counts:
            if mode == "mixed":
                corpus_result = build_relational_mixed_corpus(os.path.join(checkpoint_dir, f"relational_mixed_{int(count)}.txt"), sentence_count=int(count))
            else:
                corpus_result = build_relational_corpus(os.path.join(checkpoint_dir, f"relational_focused_{int(count)}.txt"), sentence_count=int(count))

            for config in sweep_configs:
                session_kwargs = {
                    key: value
                    for key, value in config.items()
                    if key not in {"name"}
                }
                report = run_relational_emergence_benchmark(
                    corpus_path=corpus_result.corpus_path,
                    cases=cases,
                    compare_ablation=False,
                    session_kwargs=session_kwargs,
                    **benchmark_kwargs,
                )
                arm = {
                    "corpus_mode": mode,
                    "sentence_count": int(count),
                    "config": str(config.get("name", "config")),
                    "config_spec": dict(config),
                    "corpus_path": corpus_result.corpus_path,
                    "family_counts": corpus_result.family_counts,
                    "report": report,
                }
                arms.append(arm)
                aggregate = report["arms"][0]["aggregate"] if report.get("arms") else {}
                matrix.append({
                    "corpus_mode": mode,
                    "sentence_count": int(count),
                    "config": str(config.get("name", "config")),
                    "answer_accuracy": float(aggregate.get("avg_answer_accuracy", 0.0)),
                    "binding_accuracy": float(aggregate.get("avg_binding_prediction_accuracy", 0.0)),
                    "separation_ratio": float(aggregate.get("active_passive_separation_ratio", 0.0)),
                    "case_count": int(aggregate.get("case_count", 0)),
                })

    report = {
        "baseline_sentence_count": arms[0]["sentence_count"] if arms else 0,
        "baseline_corpus_mode": arms[0]["corpus_mode"] if arms else "",
        "baseline_config": arms[0]["config"] if arms else "",
        "arms": arms,
        "matrix": matrix,
        "corpus_path": corpus_path,
        "corpus_modes": sweep_modes,
        "configs": [dict(cfg) for cfg in sweep_configs],
        "sentence_counts": sweep_counts,
    }

    out_path = benchmark_kwargs.get("report_path") or os.path.join(checkpoint_dir, "relational_component_sweep_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(f"Relational component sweep benchmark written: {out_path}")
    return report


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
