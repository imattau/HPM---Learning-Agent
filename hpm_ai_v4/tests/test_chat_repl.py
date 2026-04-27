from hpm_ai_v4.simulations.chat_repl import BasicChatRepl, ReverseChatRepl
from hpm_ai_v4.simulations.chat_simulation import BasicChatSession, ReverseChatSession
from hpm_ai_v4.simulations.layered_agent import LayeredAgent


def _warm_agent(agent: LayeredAgent, text: str) -> None:
    for ch in text:
        raw = 94 if ch == "\n" else ord(ch) - 32
        if 0 <= raw <= 94:
            agent.perceive(raw)


def test_chat_repl_handles_commands_and_chat(tmp_path):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)
    session = BasicChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    repl = BasicChatRepl(session=session, save_path=str(tmp_path / "chat_bundle"))

    help_result = repl.handle_line("/help")
    assert help_result.kind == "help"
    assert "Commands" in help_result.message

    chat_result = repl.handle_line("Hello there.")
    assert chat_result.kind == "chat"
    assert isinstance(chat_result.message, str)
    assert len(chat_result.message) > 0

    history_result = repl.handle_line("/history")
    assert history_result.kind == "history"
    assert "User:" in history_result.message

    reset_result = repl.handle_line("/reset")
    assert reset_result.kind == "reset"
    assert session.history == []

    save_result = repl.handle_line("/save")
    assert save_result.kind == "save"
    assert (tmp_path / "chat_bundle.l1.pkl").exists()

    load_result = repl.handle_line("/load")
    assert load_result.kind == "load"

    quit_result = repl.handle_line("/quit")
    assert quit_result.kind == "quit"
    assert quit_result.continue_running is False


def test_reverse_chat_repl_handles_question_flow(tmp_path):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)
    session = ReverseChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    repl = ReverseChatRepl(session=session, save_path=str(tmp_path / "chat_bundle"))

    opening = session.ask()
    assert opening.endswith("?")

    chat_result = repl.handle_line("I need a summary.")
    assert chat_result.kind == "chat"
    assert isinstance(chat_result.message, str)
    assert chat_result.message.endswith("?")


def test_reverse_chat_repl_reset_can_restart_question_flow(tmp_path):
    agent = LayeredAgent(num_workers=1)
    _warm_agent(agent, "the quick brown fox jumps over the lazy dog. " * 4)
    session = ReverseChatSession(agent, history_window=3, response_steps=16, use_constraints=False)
    repl = ReverseChatRepl(session=session, save_path=str(tmp_path / "chat_bundle"))

    first = session.ask()
    assert first.endswith("?")
    reset_result = repl.handle_line("/reset")
    assert reset_result.kind == "reset"
    assert session.history == []
    next_question = session.ask()
    assert next_question.endswith("?")
