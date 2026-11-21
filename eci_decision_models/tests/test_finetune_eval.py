"""Tests for local fine-tuned Laya evaluation, using fakes only."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from eci_decision_models import finetune_eval
from eci_decision_models.artifacts import PredictionRecord
from eci_decision_models.finetune_eval import (
    evaluate_pairs,
    load_local_laya,
    main,
    strict_direction_metrics,
)
from eci_decision_models.metrics import compute_binary_metrics, compute_classification_metrics

PROBABILITIES = {"cause": 0.7, "caused_by": 0.2, "none": 0.1}


def _record(
    *,
    record_id: str,
    item_id: int,
    gold_choice: str,
    prediction: str | None,
    status: str = "ok",
    prediction_valid: bool = True,
    probability_valid: bool = True,
    probabilities: tuple[float, float, float] | None = (0.7, 0.2, 0.1),
    error: str | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        run_id="finetune-eval",
        item_id=item_id,
        record_id=record_id,
        model="local-laya",
        attempt=1,
        raw_request_id=f"finetune-eval:{record_id}:1",
        request_hash="request-hash",
        response_hash="response-hash",
        adapter_version="finetune-eval-v1",
        parser_version="eci-parser-v1",
        gold_choice=gold_choice,
        prediction=prediction,
        status=status,
        prediction_valid=prediction_valid,
        probability_valid=probability_valid,
        probabilities=probabilities,
        error=error,
    )


@pytest.mark.parametrize(
    ("gold", "pred", "expected"),
    [
        ("cause", "cause", {"tp": 1, "fp": 0, "fn": 0, "tn": 0}),
        ("cause", "caused_by", {"tp": 0, "fp": 0, "fn": 1, "tn": 0}),
        ("cause", "none", {"tp": 0, "fp": 0, "fn": 1, "tn": 0}),
        ("none", "cause", {"tp": 0, "fp": 1, "fn": 0, "tn": 0}),
        ("none", "none", {"tp": 0, "fp": 0, "fn": 0, "tn": 1}),
        ("caused_by", "caused_by", {"tp": 1, "fp": 0, "fn": 0, "tn": 0}),
    ],
)
def test_strict_direction_single_cases(
    gold: str,
    pred: str,
    expected: dict[str, int],
) -> None:
    metrics = strict_direction_metrics([gold], [pred])

    assert metrics["tp"] == expected["tp"]
    assert metrics["fp"] == expected["fp"]
    assert metrics["fn"] == expected["fn"]
    assert metrics["tn"] == expected["tn"]
    assert metrics["n"] == 1


def test_strict_direction_hand_computed_prf() -> None:
    gold = ["cause", "cause", "cause", "none", "none"]
    pred = ["cause", "caused_by", "none", "cause", "none"]

    metrics = strict_direction_metrics(gold, pred)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 2
    assert metrics["tn"] == 1
    assert metrics["n"] == 5
    assert metrics["precision"] == pytest.approx(1 / 2)
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["f1"] == pytest.approx(2 / 5)


def test_strict_direction_rejects_unknown_choices() -> None:
    with pytest.raises(ValueError):
        strict_direction_metrics(["other"], ["cause"])
    with pytest.raises(ValueError):
        strict_direction_metrics(["cause"], ["other"])
    with pytest.raises(ValueError):
        strict_direction_metrics(["cause"], [])


def test_evaluate_pairs_matches_compute_classification_metrics() -> None:
    pairs = [
        {"record_id": "r1", "gold_choice": "cause"},
        {"record_id": "r2", "gold_choice": "caused_by"},
        {"record_id": "r3", "gold_choice": "none"},
        {"record_id": "r4", "gold_choice": "none"},
        {"record_id": "r5", "gold_choice": "cause"},
    ]
    predictions = [
        {
            "record_id": "r1",
            "choice": "cause",
            "probabilities": PROBABILITIES,
            "status": "ok",
        },
        {
            "record_id": "r2",
            "choice": "cause",
            "probabilities": PROBABILITIES,
            "status": "ok",
        },
        {
            "record_id": "r3",
            "choice": "none",
            "probabilities": {"cause": 0.1, "caused_by": 0.2, "none": 0.7},
            "status": "ok",
        },
        {
            "record_id": "r4",
            "choice": "cause",
            "probabilities": PROBABILITIES,
            "status": "ok",
        },
        {
            "record_id": "r5",
            "choice": None,
            "probabilities": None,
            "status": "invalid_response",
            "error": "response did not contain a valid choice",
        },
    ]
    expected_records = [
        _record(
            record_id="r1",
            item_id=0,
            gold_choice="cause",
            prediction="cause",
        ),
        _record(
            record_id="r2",
            item_id=1,
            gold_choice="caused_by",
            prediction="cause",
        ),
        _record(
            record_id="r3",
            item_id=2,
            gold_choice="none",
            prediction="none",
            probabilities=(0.1, 0.2, 0.7),
        ),
        _record(
            record_id="r4",
            item_id=3,
            gold_choice="none",
            prediction="cause",
        ),
        _record(
            record_id="r5",
            item_id=4,
            gold_choice="cause",
            prediction=None,
            status="invalid_response",
            prediction_valid=False,
            probability_valid=False,
            probabilities=None,
            error="response did not contain a valid choice",
        ),
    ]

    metrics = evaluate_pairs(pairs, cast("list[dict[str, object]]", predictions))
    strict = cast("dict[str, object]", metrics["strict_direction"])

    assert metrics["classification"] == compute_classification_metrics(expected_records)
    assert metrics["binary"] == compute_binary_metrics(expected_records)
    assert metrics["N"] == 5
    assert metrics["valid"] == 4
    assert metrics["fail"] == 1
    assert metrics["coverage"] == pytest.approx(4 / 5)
    assert strict["tp"] == 1
    assert strict["fp"] == 1
    assert strict["fn"] == 2
    assert strict["tn"] == 1


def test_evaluate_pairs_reports_missing_and_extra_predictions() -> None:
    pairs = [{"record_id": "r1", "gold_choice": "cause"}]
    predictions = [
        {"record_id": "other", "choice": "none", "probabilities": PROBABILITIES},
    ]

    metrics = evaluate_pairs(pairs, cast("list[dict[str, object]]", predictions))

    assert metrics["N"] == 1
    assert metrics["valid"] == 0
    assert metrics["matched"] == 0
    assert metrics["missing_predictions"] == 1
    assert metrics["extra_predictions"] == 1
    assert metrics["status_counts"] == {"api_error": 1}


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class _FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], dict[str, object]]] = []

    def system_one(
        self,
        state: dict[str, object],
        questions: dict[str, object],
    ) -> object:
        self.calls.append((state, questions))
        if state["sentence"] == "first sentence":
            return {
                "answers": {
                    "eci_relation": {
                        "choice": "cause",
                        "probabilities": {"cause": 0.8, "caused_by": 0.1, "none": 0.1},
                    },
                },
            }
        return {
            "answers": {
                "eci_relation": {
                    "choice": "none",
                    "probabilities": {"cause": 0.1, "caused_by": 0.2, "none": 0.7},
                },
            },
        }


def _test_row(record_id: str, item_id: int, sentence: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "item_id": item_id,
        "state": {"sentence": sentence, "event1": "e1", "event2": "e2"},
        "questions": {"eci_relation": {"type": "choice"}},
    }


def test_fake_client_end_to_end_writes_expected_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_path = tmp_path / "test.jsonl"
    pairs_path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        test_path,
        [
            _test_row("r1", 1, "first sentence"),
            _test_row("r2", 2, "second sentence"),
        ],
    )
    _write_jsonl(
        pairs_path,
        [
            {"record_id": "r1", "item_id": 1, "gold_choice": "cause"},
            {"record_id": "r2", "item_id": 2, "gold_choice": "none"},
        ],
    )

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model": "fake"}\n', encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"fake-weights")
    output_dir = tmp_path / "eval"

    fake_agent = _FakeAgent()
    loaded: list[tuple[object, str]] = []

    def fake_load(checkpoint_dir: object, device: str = "mps") -> _FakeAgent:
        loaded.append((checkpoint_dir, device))
        return fake_agent

    monkeypatch.setattr(finetune_eval, "load_local_laya", fake_load)

    exit_code = main(
        [
            "--checkpoint",
            str(checkpoint),
            "--test-jsonl",
            str(test_path),
            "--pairs-jsonl",
            str(pairs_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--run-id",
            "test-run",
        ],
    )

    assert exit_code == 0
    assert loaded == [(checkpoint, "cpu")]
    assert len(fake_agent.calls) == 2

    prediction_lines = (
        (output_dir / "predictions.jsonl")
        .read_text(
            encoding="utf-8",
        )
        .splitlines()
    )
    predictions = [json.loads(line) for line in prediction_lines]
    assert len(predictions) == 2
    assert predictions[0]["record_id"] == "r1"
    assert predictions[0]["item_id"] == 1
    assert predictions[0]["choice"] == "cause"
    assert predictions[0]["probabilities"] == [0.8, 0.1, 0.1]
    assert predictions[0]["status"] == "ok"
    assert predictions[0]["error"] is None
    assert predictions[1]["record_id"] == "r2"
    assert predictions[1]["choice"] == "none"

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["N"] == 2
    assert metrics["matched"] == 2
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["classification"]["N"] == 2
    assert metrics["binary"]["N"] == 2
    assert metrics["strict_direction"]["tp"] == 1
    assert metrics["strict_direction"]["tn"] == 1
    assert metrics["status_counts"] == {"ok": 2}

    markdown = (output_dir / "metrics.md").read_text(encoding="utf-8")
    assert "Strict direction (iLIF-style)" in markdown
    assert "Three-class" in markdown

    meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "test-run"
    assert meta["checkpoint"] == str(checkpoint)
    assert (
        meta["checkpoint_config_sha256"]
        == hashlib.sha256(
            (checkpoint / "config.json").read_bytes(),
        ).hexdigest()
    )
    assert (
        meta["checkpoint_model_safetensors_sha256"]
        == hashlib.sha256(
            (checkpoint / "model.safetensors").read_bytes(),
        ).hexdigest()
    )
    assert meta["test_data_sha256"] == hashlib.sha256(test_path.read_bytes()).hexdigest()
    assert meta["device"] == "cpu"
    assert "laya_sdk_version" in meta
    assert "started_at" in meta
    assert "finished_at" in meta
    assert meta["counts"]["test_rows"] == 2
    assert meta["counts"]["predictions"] == 2
    assert meta["counts"]["pairs"] == 2
    assert meta["counts"]["valid"] == 2


def test_load_local_laya_calls_lazy_load_without_revision_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, str]] = []
    fake_agent = object()

    def fake_load(checkpoint_dir: object, device: str) -> object:
        calls.append((checkpoint_dir, device))
        return fake_agent

    fake_module = SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, "laya", fake_module)

    checkpoint = "local-checkpoint-without-40-hex-revision"
    agent = load_local_laya(checkpoint, device="cpu")

    assert agent is fake_agent
    assert calls == [(checkpoint, "cpu")]

    load_local_laya(Path("/tmp/local-checkpoint"))
    assert calls[-1] == ("/tmp/local-checkpoint", "mps")


def test_evaluate_pairs_accepts_item_id_and_sequence_probabilities() -> None:
    pairs = [{"item_id": 7, "gold_choice": "cause"}]
    predictions = [
        {"item_id": 7, "choice": "cause", "probabilities": [0.6, 0.3, 0.1]},
    ]

    metrics = evaluate_pairs(pairs, predictions)
    strict = cast("dict[str, object]", metrics["strict_direction"])

    assert metrics["N"] == 1
    assert metrics["valid"] == 1
    assert strict["tp"] == 1
