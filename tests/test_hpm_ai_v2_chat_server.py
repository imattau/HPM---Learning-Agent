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
