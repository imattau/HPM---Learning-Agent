from hpm_ai_v4.tools.episode_mappers import map_dialogstudio_episode, map_ubuntu_chat_episode


def test_map_ubuntu_chat_episode_populates_structured_fields():
    episode = map_ubuntu_chat_episode(
        {
            "context": [1, 2, 3, 4],
            "action": 1,
            "intent": "greeting",
            "speaker": "user",
            "outcome": "agent_greeting_response",
            "stage": "troubleshooting_initial",
            "reward": 0.9,
        }
    )

    assert episode.domain == "chat"
    assert episode.task_family == "troubleshooting"
    assert episode.intent == "greeting"
    assert episode.action_label == "user"
    assert episode.outcome_label == "agent_greeting_response"
    assert episode.stage == "troubleshooting_initial"
    assert episode.metadata["domain"] == "chat"
    assert episode.metadata["task_family"] == "troubleshooting"


def test_map_dialogstudio_episode_populates_instruction_fields():
    episode = map_dialogstudio_episode(
        {
            "context": [5, 6, 7],
            "action_id": 2,
            "prompt_type": "instruction",
            "task_type": "summarization",
            "response_type": "fulfilled",
            "phase": "instruction_understanding",
            "reward": 0.8,
        }
    )

    assert episode.domain == "chat"
    assert episode.task_family == "summarization"
    assert episode.intent == "instruction"
    assert episode.action == 2
    assert episode.outcome_label == "fulfilled"
    assert episode.stage == "instruction_understanding"
    assert episode.metadata["intent"] == "instruction"
