import hashlib
import json

import pytest

from hpm_ai_v2.agents.reader_agent import ReaderAgent
from hpm_ai_v2.chat import ChatSession
from hpm_ai_v2.chat.calibrate import (
    Item, cited_sources, evaluate, fit_calibration, load_trainer, main as calibrate_main, read_questions, resolve,
)
from hpm_ai_v2.chat.calibration import Calibration, CalibrationMismatch
from hpm_ai_v2.chat.evidence import ExpandingPlanner
from hpm_ai_v2.chat.feedback import file_digest
from hpm_ai_v2.chat.server import load_service, main as server_main
from hpm_ai_v2.domains.text_domain import TextDomainConfig
from hpm_ai_v2.training import SourceDocument, SupervisedStudyTrainer

SOURCES = {
    "solar": "Solar panels convert sunlight into electricity. The solar system of a house includes an inverter and a battery.",
    "wind": "Wind turbines turn moving air into electricity. Offshore wind farms feed the power grid.",
    "bees": "Honeybees communicate food locations through a waggle dance. The hive system keeps the colony warm in winter.",
    "bats": "Bats navigate in darkness using echolocation. Bats sleep during the day in caves.",
    "tides": "Tides rise and fall because of the moon's gravity. Spring tides occur at new and full moon.",
    "yeast": "Yeast ferments sugar into alcohol and carbon dioxide. Bakers use yeast to make bread rise.",
    "volcano": "Volcanoes erupt when magma rises through cracks in the crust. Ash clouds can travel for thousands of kilometres.",
    "glacier": "Glaciers are slow rivers of ice that carve valleys. Melting glaciers raise the level of the sea.",
    "coral": "Coral reefs are built by tiny animals called polyps. Warm water stress causes coral bleaching.",
    "lens": "A convex lens bends light rays to a focus. Cameras and eyes both use a lens to form an image.",
}
VALIDATION = (
    [(f"val-in-{i}", q, s) for i, (q, s) in enumerate([
        ("How do honeybees tell each other where food is?", "waggle dance"), ("Why do tides rise and fall?", "moon's gravity"),
        ("What do bakers use yeast for?", "make bread rise"), ("How do bats find their way in the dark?", "echolocation"),
        ("What causes coral bleaching?", "Warm water stress"), ("How does a convex lens work?", "bends light rays")])]
    + [(f"val-out-{i}", q, None) for i, q in enumerate([
        "How does the immune system fight infection?", "What is the best system for filing taxes?",
        "How does the water system of a city work?", "What is the power of a prime number?", "How do I make a good cup of tea?"])]
)
TEST = (
    [(f"test-in-{i}", q, s) for i, (q, s) in enumerate([
        ("How do volcanoes erupt?", "magma rises through cracks"), ("What do glaciers do to valleys?", "carve valleys")])]
    + [(f"test-out-{i}", q, None) for i, q in enumerate([
        "How does a nuclear power system work?", "How does the weather forecast system predict rain?",
        "What is the level of noise in a library?", "How do I make a paper boat?", "Which system do banks use for night deposits?"])]
)


def _rows(spec):
    return [{"id": i, "question": q, "answer_contains": s} for i, q, s in spec]


def _trainer(tmp_path, name="forest"):
    sources = [SourceDocument(k, v, k) for k, v in SOURCES.items()]
    config = TextDomainConfig.from_passages([s.text for s in sources], max_vocab=300)
    trainer = SupervisedStudyTrainer(ReaderAgent(config, dynamic_vocab=False, cold_dir=str(tmp_path / name)))
    trainer.ingest(sources)
    return trainer


@pytest.fixture
def checkpoint(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.save_checkpoint(tmp_path / "ckpt")
    return tmp_path / "ckpt"


def _write(path, spec):
    path.write_text("\n".join(json.dumps(r) for r in _rows(spec)) + "\n", encoding="utf-8")
    return path


def _tree_hash(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_questions_resolve_to_the_sources_that_contain_their_quoted_answer(tmp_path):
    trainer = _trainer(tmp_path)
    items = resolve(trainer, _rows(VALIDATION))
    assert items[0].acceptable_sources == frozenset({"bees"}) and items[0].answerable
    assert not items[-1].answerable and items[-1].acceptable_sources == frozenset()
    two = resolve(trainer, [{"id": "x", "question": "q", "answer_contains": "electricity"}])
    assert two[0].acceptable_sources == frozenset({"solar", "wind"})       # every source holding the quote is right


def test_a_quote_that_matches_nothing_or_a_repeated_id_is_an_error(tmp_path):
    trainer = _trainer(tmp_path)
    with pytest.raises(ValueError, match="no source contains"):
        resolve(trainer, [{"id": "x", "question": "q", "answer_contains": "not in any source"}])
    path = tmp_path / "dup.jsonl"
    path.write_text(json.dumps({"id": "a", "question": "q", "answer_contains": None}) + "\n" * 1
                    + json.dumps({"id": "a", "question": "r", "answer_contains": None}) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_questions(path)


def test_evaluate_counts_right_wrong_and_abstained_separately(tmp_path):
    trainer = _trainer(tmp_path)
    items = [Item("a", "How do honeybees tell each other where food is?", frozenset({"bees"})),
             Item("b", "How do honeybees tell each other where food is?", frozenset({"tides"})),     # deliberately wrong label
             Item("c", "Who invented the telephone?", frozenset()),
             Item("d", "How does the immune system fight infection?", frozenset())]
    result = evaluate(items, ExpandingPlanner(trainer))
    assert result["counts"] == {"answerable: right source": 1, "answerable: wrong source": 1,
                                "unanswerable: abstained": 1, "unanswerable: answered or clarified": 1}
    assert result["balanced_accuracy"] == pytest.approx((0.5 + 0.5) / 2)


def test_calibration_sets_the_trainers_threshold_and_catches_held_out_junk_that_shares_a_word(tmp_path):
    trainer = _trainer(tmp_path)
    assert trainer.abstain_threshold == 0.0
    held_out = resolve(trainer, _rows(TEST))
    before = evaluate(held_out, ExpandingPlanner(trainer))
    calibration = fit_calibration(trainer, resolve(trainer, _rows(VALIDATION)))
    after = evaluate(held_out, ExpandingPlanner(trainer))
    assert trainer.abstain_threshold == calibration.abstain_threshold > 0.0
    assert before["counts"]["unanswerable: answered or clarified"] == 5       # each shares "system", "power", ...
    # The threshold sits just above the highest validation junk score, so held-out junk scoring above that
    # still gets through: here the library-noise question (0.164 against a 0.158 validation maximum).
    assert after["counts"]["unanswerable: abstained"] == 4
    assert ExpandingPlanner(trainer).plan("t", "What is the level of noise in a library?").act == "answer"
    assert after["counts"]["answerable: right source"] == before["counts"]["answerable: right source"] == 2
    assert calibration.fitted_on["balanced_accuracy_after"] > calibration.fitted_on["balanced_accuracy_before"]


def test_calibration_needs_both_answerable_and_unanswerable_questions(tmp_path):
    trainer = _trainer(tmp_path)
    only_in = [i for i in resolve(trainer, _rows(VALIDATION)) if i.answerable]
    with pytest.raises(ValueError, match="both"):
        fit_calibration(trainer, only_in)


def test_a_calibration_file_round_trips_and_is_bound_to_its_checkpoint(tmp_path):
    trainer = _trainer(tmp_path)
    cal = Calibration(abstain_threshold=0.2, checkpoint_digest="abc", fitted_on={"answerable": 3})
    path = cal.save(tmp_path / "out" / "cal.json")
    assert Calibration.load(path, checkpoint_digest="abc") == cal
    assert Calibration.load(path) == cal                                       # no digest to compare: accepted
    with pytest.raises(CalibrationMismatch):
        Calibration.load(path, checkpoint_digest="different")
    Calibration.load(path).apply(trainer)
    assert trainer.abstain_threshold == 0.2


def test_the_cli_fits_reports_before_and_after_and_leaves_the_checkpoint_untouched(checkpoint, tmp_path, capsys):
    validation, test = _write(tmp_path / "val.jsonl", VALIDATION), _write(tmp_path / "test.jsonl", TEST)
    untouched = _tree_hash(checkpoint)
    out = tmp_path / "cal.json"
    assert calibrate_main([str(checkpoint), "--validation", str(validation), "--test", str(test), "--output", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["test_before"]["counts"]["unanswerable: answered or clarified"] == 5
    assert report["test_after"]["counts"]["unanswerable: abstained"] == 4          # one held-out junk question scores above the fit
    saved = json.loads(out.read_text())
    assert saved["abstain_threshold"] > 0 and saved["checkpoint_digest"] == file_digest(checkpoint / "study_training_state.json")
    assert _tree_hash(checkpoint) == untouched                                 # the pretrained base was not modified


def test_the_server_applies_a_calibration_and_says_so(checkpoint, tmp_path):
    out = tmp_path / "cal.json"
    calibrate_main([str(checkpoint), "--validation", str(_write(tmp_path / "val.jsonl", VALIDATION)), "--output", str(out)])
    plain = load_service(checkpoint, tmp_path / "d1")
    assert plain.trainer.abstain_threshold == 0.0 and plain.config()["abstention"] == "abstention uncalibrated"
    calibrated = load_service(checkpoint, tmp_path / "d2", calibration_path=out)
    threshold = json.loads(out.read_text())["abstain_threshold"]
    assert calibrated.trainer.abstain_threshold == threshold
    assert calibrated.config()["abstention"] == f"abstention calibrated (threshold {threshold:.3f})"
    conv = calibrated.create_conversation()
    assert calibrated.send(conv["id"], "How does the weather forecast system predict rain?")["act"] == "abstain"
    assert calibrated.send(conv["id"], "How do volcanoes erupt?")["act"] == "answer"


def test_a_calibration_for_another_checkpoint_is_refused(checkpoint, tmp_path):
    stale = Calibration(0.2, checkpoint_digest="from-another-checkpoint").save(tmp_path / "stale.json")
    with pytest.raises(CalibrationMismatch):
        load_service(checkpoint, tmp_path / "d", calibration_path=stale)
    assert server_main([str(checkpoint), "--data-dir", str(tmp_path / "d2"), "--calibration", str(stale)]) == 2


def test_the_word_support_check_still_applies_to_a_calibrated_lexical_checkpoint(tmp_path):
    trainer = _trainer(tmp_path)
    fit_calibration(trainer, resolve(trainer, _rows(VALIDATION)))
    # a match that rests only on the question word "what" passes any confidence threshold below its score
    trainer.abstain_threshold = 1e-6
    reply = ChatSession(trainer).send("What is the capital of France?")
    assert reply.abstained and reply.plan.reason in {"no_content_support", "no_retrievable_supported_answer"}
