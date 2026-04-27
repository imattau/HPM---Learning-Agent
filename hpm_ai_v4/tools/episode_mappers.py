"""Canonical mappers from dialogue-style rows into structured episodes."""
from __future__ import annotations

from typing import Any, Dict, Optional

from hpm_ai_v4.agents.reasoning import EpisodeRecord
from hpm_ai_v4.io.adapters import TextAdapter


def _normalize_label(value: Any, fallback: str = "unknown") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return fallback
    return text.replace(" ", "_").replace("-", "_")


def _coerce_reward(row: Dict[str, Any], fallback: float = 0.5) -> float:
    for key in ("reward", "score", "quality", "label"):
        if key in row:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return float(fallback)


def _coerce_context(row: Dict[str, Any]) -> list[int]:
    context = row.get("context", [])
    if isinstance(context, list):
        return [int(v) for v in context if isinstance(v, (int, float)) or str(v).isdigit()]
    if isinstance(context, str):
        return [ord(ch) - 32 for ch in context if 32 <= ord(ch) <= 126]
    return []


def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _infer_dialogue_act(text: Any) -> str:
    lowered = _clean_text(text).lower()
    if not lowered:
        return "default"
    if lowered.endswith("?"):
        return "question"
    if any(lowered.startswith(prefix) for prefix in ("hi", "hello", "hey", "good morning", "good evening")):
        return "greeting"
    if any(phrase in lowered for phrase in ("what do you mean", "clarify", "can you explain", "explain that", "what does that mean")):
        return "clarification"
    if any(lowered.startswith(prefix) for prefix in ("bye", "goodbye", "thanks", "thank you")):
        return "closing"
    if any(lowered.startswith(prefix) for prefix in ("tell me", "give me", "show me", "explain", "help me")):
        return "request"
    return "default"


def map_dialogue_row(
    row: Dict[str, Any],
    *,
    domain: str,
    task_family: str,
    intent: Optional[str] = None,
    action_label: Optional[str] = None,
    outcome_label: Optional[str] = None,
    stage: Optional[str] = None,
    policy: Optional[str] = None,
    tag: str = "observe",
    reward: Optional[float] = None,
) -> EpisodeRecord:
    context = _coerce_context(row)
    canonical_intent = _normalize_label(intent or row.get("intent") or row.get("dialogue_act") or row.get("act"))
    canonical_action = _normalize_label(action_label or row.get("action") or row.get("speaker") or row.get("role"))
    canonical_outcome = _normalize_label(outcome_label or row.get("outcome") or row.get("response_type") or row.get("reply_type"))
    canonical_stage = _normalize_label(stage or row.get("stage") or row.get("phase") or row.get("step"))
    canonical_policy = _normalize_label(policy or row.get("policy") or row.get("strategy"))
    canonical_domain = _normalize_label(domain, domain)
    canonical_task_family = _normalize_label(task_family, task_family)

    metadata = dict(row.get("metadata", {}) or {})
    metadata.update(
        {
            "domain": canonical_domain,
            "task_family": canonical_task_family,
            "intent": canonical_intent,
            "action_label": canonical_action,
            "outcome_label": canonical_outcome,
            "stage": canonical_stage,
            "policy": canonical_policy,
            "source": row.get("source", metadata.get("source", "unknown")),
        }
    )
    episode_action = int(row.get("action", row.get("action_id", 0)) or 0)
    episode_reward = float(reward if reward is not None else _coerce_reward(row, fallback=0.5))
    return EpisodeRecord(
        context=context,
        action=episode_action,
        reward=episode_reward,
        tag=tag,
        metadata=metadata,
        domain=canonical_domain,
        task_family=canonical_task_family,
        intent=canonical_intent,
        action_label=canonical_action,
        outcome_label=canonical_outcome,
        stage=canonical_stage,
        policy=canonical_policy,
    )


def map_ubuntu_chat_episode(row: Dict[str, Any]) -> EpisodeRecord:
    """Map an Ubuntu-style chat row into a structured episode."""
    task_family = row.get("task_family", "troubleshooting")
    intent = row.get("intent") or row.get("dialogue_act") or row.get("label")
    action_label = row.get("speaker") or row.get("role") or row.get("action") or "user_message"
    outcome_label = row.get("outcome") or row.get("response_type") or "agent_response"
    stage = row.get("stage") or row.get("phase") or "troubleshooting_initial"
    return map_dialogue_row(
        row,
        domain="chat",
        task_family=str(task_family),
        intent=str(intent) if intent is not None else None,
        action_label=str(action_label),
        outcome_label=str(outcome_label),
        stage=str(stage),
        policy=row.get("policy"),
        tag="chat",
        reward=row.get("reward"),
    )


def map_dialogstudio_episode(row: Dict[str, Any]) -> EpisodeRecord:
    """Map a DialogStudio-style instruction row into a structured episode."""
    task_family = row.get("task_family") or row.get("task_type") or row.get("instruction_type") or "instruction"
    intent = row.get("intent") or row.get("dialogue_act") or row.get("prompt_type") or "instruction"
    action_label = row.get("action") or row.get("speaker") or "prompt"
    outcome_label = row.get("outcome") or row.get("response_type") or "fulfilled"
    stage = row.get("stage") or row.get("phase") or "instruction_understanding"
    return map_dialogue_row(
        row,
        domain=row.get("domain", "chat"),
        task_family=str(task_family),
        intent=str(intent),
        action_label=str(action_label),
        outcome_label=str(outcome_label),
        stage=str(stage),
        policy=row.get("policy"),
        tag="instruction",
        reward=row.get("reward"),
    )


def map_dailydialog_dialog(row: Dict[str, Any], *, max_turns: int = 8) -> list[EpisodeRecord]:
    """Map a DailyDialog row into a structured episode sequence."""
    raw_turns = row.get("dialog") or row.get("utterances") or []
    turns = [_clean_text(turn) for turn in raw_turns[:max_turns]]
    turns = [turn for turn in turns if turn]
    if not turns:
        return []

    adapter = TextAdapter()
    records: list[EpisodeRecord] = []
    task_family = _normalize_label(row.get("task_family") or "daily_dialog")

    for idx, turn_text in enumerate(turns):
        previous_turns = turns[max(0, idx - 2):idx]
        context_text = " ".join(previous_turns)
        context = adapter.to_observations(context_text, max_length=256) if context_text else []
        role = "user" if idx % 2 == 0 else "assistant"
        next_text = turns[idx + 1] if idx + 1 < len(turns) else ""
        intent = _infer_dialogue_act(turn_text)
        outcome_label = _infer_dialogue_act(next_text) if next_text else "dialogue_end"
        stage = "dialogue_start" if idx == 0 else "dialogue_end" if idx == len(turns) - 1 else "dialogue_middle"
        reward = 0.45
        reward += 0.10 if role == "assistant" else 0.04
        reward += 0.08 if turn_text.endswith("?") else 0.0
        reward += 0.05 if len(turn_text) > 20 else 0.0
        reward = min(1.0, reward)
        record = map_dialogue_row(
            {
                "context": context,
                "action": idx % 2,
                "reward": reward,
                "metadata": {
                    "source": "daily_dialog",
                    "turn_index": idx,
                    "turn_text": turn_text,
                    "next_text": next_text,
                    "speaker": role,
                    "next_speaker": "assistant" if role == "user" else "user",
                },
            },
            domain="chat",
            task_family=task_family,
            intent=intent,
            action_label=role,
            outcome_label=outcome_label,
            stage=stage,
            policy="dialogue",
            tag="daily_dialog",
            reward=reward,
        )
        records.append(record)
    return records
