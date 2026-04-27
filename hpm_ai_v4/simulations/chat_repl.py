"""Interactive terminal REPL for basic HPM chat."""
import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hpm_ai_v4.simulations.chat_simulation import (
    BasicChatSession,
    _load_chat_library,
    _resolve_chat_library_path,
    ReverseChatSession,
)
from hpm_ai_v4.simulations.full_simulation import WikipediaStream
from hpm_ai_v4.simulations.layered_agent import LayeredAgent
from hpm_ai_v4.tools.dictionary import NLTKWordList
from hpm_ai_v4.tools.grammar import HeuristicGrammarLibrary
from hpm_ai_v4.simulations.chat_simulation import CHAT_SEED_CORPUS


@dataclass
class ReplResult:
    kind: str
    message: str
    continue_running: bool = True


class BasicChatRepl:
    """Small interactive shell over BasicChatSession."""

    def __init__(
        self,
        session: BasicChatSession,
        save_path: Optional[str] = None,
    ):
        self.session = session
        self.save_path = save_path

    def handle_line(self, line: str) -> ReplResult:
        line = line.strip()
        if not line:
            return ReplResult(kind="noop", message="")

        if line in {"/quit", "/exit"}:
            return ReplResult(kind="quit", message="bye", continue_running=False)
        if line in {"/help", "?"}:
            return ReplResult(
                kind="help",
                message=(
                    "Commands: /help, /history, /reset, /save [path], /load [path], /quit\n"
                    "Anything else is sent as user text."
                ),
            )
        if line == "/history":
            return ReplResult(kind="history", message=self.session.transcript())
        if line == "/reset":
            self.session.reset()
            return ReplResult(kind="reset", message="session reset")
        if line.startswith("/save"):
            path = self._path_arg(line) or self.save_path
            if not path:
                return ReplResult(kind="error", message="No save path provided.")
            self.session.agent.save_bundle(path)
            return ReplResult(kind="save", message=f"saved bundle to {path}")
        if line.startswith("/load"):
            path = self._path_arg(line) or self.save_path
            if not path:
                return ReplResult(kind="error", message="No load path provided.")
            loaded = self.session.agent.load_bundle(path)
            return ReplResult(kind="load", message=f"loaded {loaded} bundle parts from {path}")

        result = self.session.chat_turn(line)
        return ReplResult(kind="chat", message=result.response_text)

    def run(self) -> None:
        print("Basic HPM chat REPL. Type /help for commands.")
        while True:
            try:
                line = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            result = self.handle_line(line)
            if result.message:
                if result.kind == "chat":
                    print(f"assistant> {result.message}")
                else:
                    print(result.message)
            if not result.continue_running:
                break

    def _path_arg(self, line: str) -> Optional[str]:
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            return None
        return parts[1].strip()


def _build_session(
    corpus_path: str,
    warmup_chars: int = 800,
    response_steps: int = 48,
    history_window: int = 6,
    num_workers: int = 1,
    use_dict: bool = True,
    library_path: Optional[str] = None,
    seed_corpus_path: Optional[str] = CHAT_SEED_CORPUS,
) -> BasicChatSession:
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

    return BasicChatSession(
        layered,
        history_window=history_window,
        response_steps=response_steps,
        use_constraints=use_dict,
    )


def run_basic_chat_repl(
    corpus_path: str,
    warmup_chars: int = 800,
    response_steps: int = 48,
    history_window: int = 6,
    num_workers: int = 1,
    use_dict: bool = True,
    library_path: Optional[str] = None,
    seed_corpus_path: Optional[str] = CHAT_SEED_CORPUS,
) -> None:
    session = _build_session(
        corpus_path=corpus_path,
        warmup_chars=warmup_chars,
        response_steps=response_steps,
        history_window=history_window,
        num_workers=num_workers,
        use_dict=use_dict,
        library_path=library_path,
        seed_corpus_path=seed_corpus_path,
    )
    repl = BasicChatRepl(session=session, save_path=library_path)
    repl.run()


class ReverseChatRepl(BasicChatRepl):
    """Interactive shell for question-first chat."""

    def run(self) -> None:
        print("Reverse HPM chat REPL. The model asks, you answer. Type /help for commands.")
        if hasattr(self.session, "ask"):
            opening = self.session.ask()
            if opening:
                print(f"assistant> {opening}")
        while True:
            try:
                line = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            result = self.handle_line(line)
            if result.message:
                if result.kind == "chat":
                    print(f"assistant> {result.message}")
                else:
                    print(result.message)
            if result.kind == "reset" and hasattr(self.session, "ask"):
                opening = self.session.ask()
                if opening:
                    print(f"assistant> {opening}")
            if not result.continue_running:
                break


def _build_reverse_session(
    corpus_path: str,
    warmup_chars: int = 800,
    response_steps: int = 48,
    history_window: int = 6,
    num_workers: int = 1,
    use_dict: bool = True,
    library_path: Optional[str] = None,
    seed_corpus_path: Optional[str] = CHAT_SEED_CORPUS,
) -> ReverseChatSession:
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

    return ReverseChatSession(
        layered,
        history_window=history_window,
        response_steps=response_steps,
        use_constraints=use_dict,
    )


def run_reverse_chat_repl(
    corpus_path: str,
    warmup_chars: int = 800,
    response_steps: int = 48,
    history_window: int = 6,
    num_workers: int = 1,
    use_dict: bool = True,
    library_path: Optional[str] = None,
    seed_corpus_path: Optional[str] = CHAT_SEED_CORPUS,
) -> None:
    session = _build_reverse_session(
        corpus_path=corpus_path,
        warmup_chars=warmup_chars,
        response_steps=response_steps,
        history_window=history_window,
        num_workers=num_workers,
        use_dict=use_dict,
        library_path=library_path,
        seed_corpus_path=seed_corpus_path,
    )
    repl = ReverseChatRepl(session=session, save_path=library_path)
    repl.run()


def _parse_args():
    p = argparse.ArgumentParser(description="Interactive basic HPM chat REPL")
    p.add_argument("--corpus", required=True, help="Path to plain-text corpus file")
    p.add_argument("--warmup-chars", type=int, default=800)
    p.add_argument("--response-steps", type=int, default=48)
    p.add_argument("--history-window", type=int, default=6)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--dict", action="store_true", help="Enable dictionary and grammar validators")
    p.add_argument("--library", default=None, help="Base path to pre-built pattern library (no .pkl suffix)")
    p.add_argument("--seed-corpus", default=CHAT_SEED_CORPUS, help="Optional dialogue seed corpus for warmup")
    p.add_argument("--reverse", action="store_true", help="Run the question-first reverse chat mode")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    runner = run_reverse_chat_repl if args.reverse else run_basic_chat_repl
    runner(
        corpus_path=args.corpus,
        warmup_chars=args.warmup_chars,
        response_steps=args.response_steps,
        history_window=args.history_window,
        num_workers=args.workers,
        use_dict=args.dict,
        library_path=args.library,
        seed_corpus_path=args.seed_corpus,
    )
