"""Local HTTP server for the HPM v2 chat frontend (stdlib only).

Binds to localhost, checks Host and Origin, and serves a JSON API over
``ChatService`` plus the static single-page app in ``web/``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from hpm_ai_v2.chat.service import ChatService, NotFound

WEB_DIR = Path(__file__).parent / "web"
_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
_MAX_BODY = 64 * 1024
_ID = r"([0-9a-f]{32})"
_ROUTES = [
    ("GET", re.compile(r"^/api/config$"), "config"),
    ("GET", re.compile(r"^/api/sources$"), "sources"),
    ("GET", re.compile(r"^/api/sources/([^/]+)$"), "source"),
    ("GET", re.compile(r"^/api/conversations$"), "list_conversations"),
    ("POST", re.compile(r"^/api/conversations$"), "create_conversation"),
    ("GET", re.compile(rf"^/api/conversations/{_ID}$"), "get_conversation"),
    ("PATCH", re.compile(rf"^/api/conversations/{_ID}$"), "rename"),
    ("DELETE", re.compile(rf"^/api/conversations/{_ID}$"), "delete"),
    ("POST", re.compile(rf"^/api/conversations/{_ID}/messages$"), "send"),
    ("POST", re.compile(rf"^/api/conversations/{_ID}/messages/{_ID}/edit$"), "edit"),
]


class _BadRequest(ValueError):
    pass


def make_handler(service: ChatService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "HPMChat/1"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            return

        # -- plumbing ------------------------------------------------------
        def _reply(self, status: int, payload=None, *, body: bytes | None = None, ctype="application/json"):
            if body is None:
                body = json.dumps(payload if payload is not None else {}, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _error(self, status: int, error: str, message: str):
            self._reply(status, {"error": error, "message": message})

        def _stream_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

        def _event(self, name: str, payload) -> None:
            """Write one SSE event; a vanished client must not abort the turn."""
            if getattr(self, "_client_gone", False):
                return
            data = json.dumps(payload, ensure_ascii=False)
            try:
                self.wfile.write(f"event: {name}\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
            except OSError:
                self._client_gone = True

        def _host_ok(self) -> bool:
            raw = self.headers.get("Host") or ""
            host = raw[: raw.index("]") + 1] if raw.startswith("[") else raw.rsplit(":", 1)[0]
            return host in _LOCAL_HOSTS

        def _origin_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if origin is None:
                return True
            return urlsplit(origin).netloc == self.headers.get("Host")

        def _json_body(self) -> dict:
            if "application/json" not in (self.headers.get("Content-Type") or ""):
                raise _BadRequest("Content-Type must be application/json")
            length = int(self.headers.get("Content-Length") or 0)
            if length > _MAX_BODY:
                raise _BadRequest("request body too large")
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                raise _BadRequest("body is not valid JSON") from None
            if not isinstance(data, dict):
                raise _BadRequest("body must be a JSON object")
            return data

        # -- dispatch ------------------------------------------------------
        def _handle(self):
            if not self._host_ok():
                return self._error(403, "forbidden_host", "Only localhost is allowed.")
            path = urlsplit(self.path).path
            if self.command in {"GET", "HEAD"} and path in _STATIC:
                name, ctype = _STATIC[path]
                return self._reply(200, body=(WEB_DIR / name).read_bytes(), ctype=ctype)
            if self.command != "GET" and not self._origin_ok():
                return self._error(403, "forbidden_origin", "Cross-origin writes are not allowed.")
            for method, pattern, action in _ROUTES:
                match = pattern.match(path)
                if match and method == self.command:
                    return self._run(action, match.groups())
            known = any(p.match(path) for _m, p, _a in _ROUTES)
            if known:
                return self._error(405, "method_not_allowed", f"{self.command} is not allowed here.")
            return self._error(404, "not_found", "No such resource.")

        def _run(self, action: str, args: tuple[str, ...]):
            try:
                status, payload = 200, None
                if action == "config":
                    payload = service.config()
                elif action == "sources":
                    payload = service.sources()
                elif action == "source":
                    payload = service.source(args[0])
                elif action == "list_conversations":
                    query = parse_qs(urlsplit(self.path).query).get("q", [""])[0]
                    payload = service.list_conversations(query[:200])
                elif action == "create_conversation":
                    self._json_body()
                    status, payload = 201, service.create_conversation()
                elif action == "get_conversation":
                    payload = service.get_conversation(args[0])
                elif action == "rename":
                    payload = service.rename(args[0], str(self._json_body().get("title", "")))
                elif action == "delete":
                    service.delete(args[0])
                    status, payload = 200, {"deleted": args[0]}
                elif action in {"send", "edit"}:
                    body = self._json_body()
                    text = body.get("text")
                    if not isinstance(text, str) or not text.strip():
                        raise _BadRequest("text must be a non-empty string")
                    if len(text) > 4000:
                        raise _BadRequest("text is too long (4000 characters max)")
                    scope = body.get("scope_source_id") if action == "send" else None
                    if scope is not None and not isinstance(scope, str):
                        raise _BadRequest("scope_source_id must be a string")
                    streaming = "text/event-stream" in (self.headers.get("Accept") or "")
                    if streaming:
                        service.get_conversation(args[0])  # 404 before the stream starts
                        self._stream_start()
                        on_stage = lambda name, label: self._event("stage", {"stage": name, "label": label})
                        try:
                            if action == "send":
                                message = service.send(args[0], text, scope_source_id=scope, on_stage=on_stage)
                            else:
                                message = service.edit(args[0], args[1], text, on_stage=on_stage)
                            self._event("reply", message)
                        except (NotFound, ValueError) as exc:
                            self._event("error", {"error": "bad_request", "message": str(exc)})
                        except Exception as exc:
                            print(f"internal error: {exc!r}", file=sys.stderr)
                            self._event("error", {"error": "internal_error", "message": "The server hit an unexpected error."})
                        return
                    if action == "send":
                        payload = service.send(args[0], text, scope_source_id=scope)
                    else:
                        payload = service.edit(args[0], args[1], text)
                self._reply(status, payload)
            except NotFound as exc:
                self._error(404, "not_found", f"Not found: {exc.args[0] if exc.args else ''}")
            except ValueError as exc:
                self._error(400, "bad_request", str(exc))
            except Exception as exc:  # never leak a traceback to the browser
                print(f"internal error: {exc!r}", file=sys.stderr)
                self._error(500, "internal_error", "The server hit an unexpected error.")

        do_GET = do_HEAD = do_POST = do_PATCH = do_DELETE = _handle

    return Handler


def create_server(service: ChatService, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the chat server only binds to localhost")
    server = ThreadingHTTPServer((host, port), make_handler(service))
    server.daemon_threads = True
    return server


def load_service(checkpoint: Path, data_dir: Path, user_id: str = "default") -> ChatService:
    from hpm_ai_v2.agents.reader_agent import ReaderAgent
    from hpm_ai_v2.training import SupervisedStudyTrainer

    agent = ReaderAgent.load_agent(str(checkpoint))
    trainer = SupervisedStudyTrainer(agent)
    trainer.load_checkpoint_state(checkpoint)
    encoder = getattr(agent.config, "embedding_model", None)
    info = {
        "checkpoint": str(checkpoint),
        "representation": f"semantic encoder: {encoder}" if encoder else "lexical TF-IDF vocabulary",
    }
    return ChatService(trainer, data_dir, user_id=user_id, info=info)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="trained study checkpoint directory")
    parser.add_argument("--data-dir", type=Path, required=True, help="conversations and event log")
    parser.add_argument("--user", default="default")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    service = load_service(args.checkpoint, args.data_dir, args.user)
    server = create_server(service, port=args.port)
    print(f"HPM chat on http://127.0.0.1:{server.server_address[1]}  (data: {args.data_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
