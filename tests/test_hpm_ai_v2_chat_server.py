import json
import threading

import httpx
import pytest

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.chat.server import create_server, load_service
from hpm_ai_v2.chat.service import ChatService
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import SourceDocument, SupervisedStudyTrainer

SOLAR = "Solar panels convert sunlight into electricity. Solar energy can reduce household emissions."


def _trainer(tmp_path):
    source = SourceDocument("solar", SOLAR, "Solar energy")
    config = TextDomainConfig.from_passages([SOLAR], max_vocab=100)
    trainer = SupervisedStudyTrainer(ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / "forest")))
    trainer.ingest([source])
    return trainer


@pytest.fixture
def server(tmp_path):
    service = ChatService(_trainer(tmp_path), tmp_path / "data", info={"representation": "lexical"})
    httpd = create_server(service, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_static_pages_and_read_only_endpoints(server):
    page = httpx.get(server + "/")
    assert page.status_code == 200 and "HPM Study Assistant" in page.text
    assert "Content-Security-Policy" in page.headers
    assert httpx.get(server + "/app.js").headers["content-type"].startswith("text/javascript")
    assert httpx.get(server + "/style.css").status_code == 200
    config = httpx.get(server + "/api/config").json()
    assert config["source_count"] == 1 and config["representation"] == "lexical"
    assert httpx.get(server + "/api/sources").json()[0]["source_id"] == "solar"
    assert httpx.get(server + "/api/sources/solar").json()["text"] == SOLAR
    assert httpx.get(server + "/api/sources/nope").status_code == 404


def test_conversation_flow_over_http(server):
    conv = httpx.post(server + "/api/conversations", json={}).json()
    url = f"{server}/api/conversations/{conv['id']}"
    reply = httpx.post(url + "/messages", json={"text": "What do solar panels convert into electricity?"}).json()
    assert reply["act"] == "answer" and reply["verified"]
    cite = reply["citations"][0]
    assert SOLAR[cite["start"]:cite["end"]] == cite["text"]
    assert len(httpx.get(url).json()["messages"]) == 2
    assert httpx.patch(url, json={"title": "Solar"}).json()["title"] == "Solar"
    assert [c["title"] for c in httpx.get(server + "/api/conversations").json()] == ["Solar"]
    edited = httpx.post(f"{url}/messages/{reply['turn_id']}/edit", json={"text": "Summarize solar energy"}).json()
    assert edited["task_family"] == "summarize"
    assert httpx.delete(url).status_code == 200
    assert httpx.get(url).status_code == 404


def test_error_shapes(server):
    conv = httpx.post(server + "/api/conversations", json={}).json()
    url = f"{server}/api/conversations/{conv['id']}/messages"
    bad = httpx.post(url, json={"text": "   "})
    assert bad.status_code == 400 and bad.json()["error"] == "bad_request"
    assert httpx.post(url, json={"text": "x" * 5000}).status_code == 400
    assert httpx.post(url, json={"text": "hi", "scope_source_id": "missing"}).status_code == 400
    assert httpx.post(url, content="not json", headers={"Content-Type": "application/json"}).status_code == 400
    assert httpx.post(url, content="text=hi", headers={"Content-Type": "text/plain"}).status_code == 400
    assert httpx.get(server + "/api/nothing").status_code == 404
    assert httpx.put(server + "/api/conversations").status_code in {404, 501}
    assert httpx.delete(server + "/api/config").status_code == 405
    assert httpx.post(server + "/api/conversations/" + "0" * 32 + "/messages", json={"text": "hi"}).status_code == 404


def test_rejects_foreign_host_and_cross_origin_writes(server):
    assert httpx.get(server + "/api/config", headers={"Host": "evil.example"}).status_code == 403
    cross = httpx.post(server + "/api/conversations", json={}, headers={"Origin": "http://evil.example"})
    assert cross.status_code == 403 and cross.json()["error"] == "forbidden_origin"
    same = httpx.post(server + "/api/conversations", json={}, headers={"Origin": server})
    assert same.status_code == 201


def test_server_refuses_non_local_bind(tmp_path):
    service = ChatService(_trainer(tmp_path), tmp_path / "data")
    with pytest.raises(ValueError):
        create_server(service, host="0.0.0.0", port=0)


def test_load_service_from_checkpoint_answers_like_the_original(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.save_checkpoint(tmp_path / "ckpt")
    service = load_service(tmp_path / "ckpt", tmp_path / "data2")
    assert service.config()["representation"].startswith("lexical")
    conv = service.create_conversation()
    reply = service.send(conv["id"], "What do solar panels convert into electricity?")
    assert reply["act"] == "answer" and reply["citations"][0]["source_id"] == "solar"


def _sse(response):
    events, name = [], None
    for line in response.iter_lines():
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            events.append((name, json.loads(line[6:])))
    return events


def test_send_streams_stages_then_reply_and_persists(server):
    conv = httpx.post(server + "/api/conversations", json={}).json()
    url = f"{server}/api/conversations/{conv['id']}/messages"
    with httpx.stream("POST", url, json={"text": "What do solar panels convert into electricity?"},
                      headers={"Accept": "text/event-stream"}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse(response)
    assert [n for n, _ in events] == ["stage"] * 4 + ["reply"]
    assert [d["stage"] for n, d in events[:4]] == ["interpret", "plan", "verify", "realize"]
    assert events[-1][1]["act"] == "answer"
    assert len(httpx.get(f"{server}/api/conversations/{conv['id']}").json()["messages"]) == 2


def test_stream_errors_arrive_as_events_or_status(server):
    conv = httpx.post(server + "/api/conversations", json={}).json()
    url = f"{server}/api/conversations/{conv['id']}/messages"
    accept = {"Accept": "text/event-stream"}
    assert httpx.post(url, json={"text": " "}, headers=accept).status_code == 400
    missing = httpx.post(f"{server}/api/conversations/{'0' * 32}/messages", json={"text": "hi"}, headers=accept)
    assert missing.status_code == 404
    edit_url = f"{url}/{'1' * 32}/edit"
    with httpx.stream("POST", edit_url, json={"text": "hi"}, headers=accept) as response:
        events = _sse(response)
    assert events and events[-1][0] == "error"


def test_conversation_search_over_http(server):
    a = httpx.post(server + "/api/conversations", json={}).json()["id"]
    httpx.post(f"{server}/api/conversations/{a}/messages", json={"text": "What do solar panels convert?"})
    assert [c["id"] for c in httpx.get(server + "/api/conversations", params={"q": "solar"}).json()] == [a]
    assert httpx.get(server + "/api/conversations", params={"q": "nothing-here"}).json() == []


SEED = __import__("pathlib").Path(__file__).parents[1] / "hpm_ai_v2/training/curricula/correction_transfer_seed.jsonl"
ASKED = "How do bees show others where food can be found?"
PARAPHRASE = "How do bees communicate food location to hive mates?"


def _seed_checkpoint(tmp_path):
    from hpm_ai_v2.training import load_curriculum_jsonl

    sources, _ = load_curriculum_jsonl(SEED)
    config = TextDomainConfig.from_passages([s.text for s in sources], max_vocab=200)
    agent = ReaderAgent(config, dynamic_vocab=False, inject_blank_priors=False, cold_dir=str(tmp_path / "forest"))
    trainer = SupervisedStudyTrainer(agent)
    trainer.ingest(sources)
    trainer.save_checkpoint(tmp_path / "ckpt")
    return tmp_path / "ckpt"


def _serve(service):
    httpd = create_server(service, port=0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_feedback_over_http_and_learning_survives_a_server_restart(tmp_path):
    checkpoint = _seed_checkpoint(tmp_path)
    httpd, base = _serve(load_service(checkpoint, tmp_path / "data"))
    try:
        conv = httpx.post(base + "/api/conversations", json={}).json()["id"]
        url = f"{base}/api/conversations/{conv}/messages"
        reply = httpx.post(url, json={"text": ASKED}).json()
        assert {c["source_id"] for c in reply["citations"]} == {"visual"} and reply["feedback"] == []

        rows = httpx.get(f"{url}/{reply['turn_id']}/candidates").json()
        assert any(r["current"] for r in rows)
        waggle = next(r for r in rows if r["source_id"] == "waggle" and r["start"] == 0)
        done = httpx.post(f"{url}/{reply['turn_id']}/feedback",
                          json={"signal": "correct", "correction": {k: waggle[k] for k in ("source_id", "start", "end")}})
        assert done.status_code == 200 and done.json()["learned"] and done.json()["feedback"] == ["correct"]
        assert httpx.get(f"{base}/api/conversations/{conv}").json()["messages"][1]["feedback"] == ["correct"]
        assert httpx.post(f"{url}/{reply['turn_id']}/feedback", json={"signal": "up"}).status_code == 200
        assert httpx.post(f"{url}/{reply['turn_id']}/feedback", json={"signal": "down"}).status_code == 400
        assert httpx.post(f"{url}/{reply['turn_id']}/feedback", json={"signal": 3}).status_code == 400
        assert httpx.post(f"{url}/{'9' * 32}/feedback", json={"signal": "up"}).status_code == 400
        after = httpx.post(url, json={"text": PARAPHRASE}).json()
        assert {c["source_id"] for c in after["citations"]} == {"waggle"}
    finally:
        httpd.shutdown(); httpd.server_close()

    httpd, base = _serve(load_service(checkpoint, tmp_path / "data"))    # restart: replay the ledger
    try:
        conv = httpx.post(base + "/api/conversations", json={}).json()["id"]
        again = httpx.post(f"{base}/api/conversations/{conv}/messages", json={"text": PARAPHRASE}).json()
        assert {c["source_id"] for c in again["citations"]} == {"waggle"}
    finally:
        httpd.shutdown(); httpd.server_close()

    from hpm_ai_v2.chat.feedback import LedgerMismatch
    (tmp_path / "other").mkdir()
    other = _seed_checkpoint(tmp_path / "other")
    (other / "study_training_state.json").write_text((other / "study_training_state.json").read_text() + " ")
    with pytest.raises(LedgerMismatch):
        load_service(other, tmp_path / "data")


def test_preferences_over_http(server):
    prefs = httpx.get(server + "/api/preferences").json()
    assert prefs["summary_format"] == {"value": None, "options": ["bullets", "numbered"]}
    assert httpx.post(server + "/api/preferences", json={"key": "summary_format", "value": "bullets"}).json()["summary_format"]["value"] == "bullets"
    assert httpx.post(server + "/api/preferences", json={"key": "summary_format", "value": "poem"}).status_code == 400
    assert httpx.post(server + "/api/preferences", json={"key": "tone", "value": "x"}).status_code == 400
    assert httpx.post(server + "/api/preferences", json={"key": 1, "value": "x"}).status_code == 400
    conv = httpx.post(server + "/api/conversations", json={}).json()["id"]
    reply = httpx.post(f"{server}/api/conversations/{conv}/messages", json={"text": "Summarize solar energy"}).json()
    assert reply["blocks"][0]["type"] == "bullet_list"
