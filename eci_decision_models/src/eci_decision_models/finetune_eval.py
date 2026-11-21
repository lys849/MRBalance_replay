"""Evaluate a locally fine-tuned Laya checkpoint on fold-0 test pairs.

The module has three jobs:

* turn interface choices into iLIF-style strict direction counts;
* join gold test pairs with local predictions and reuse the shared
  classification/binary metrics;
* optionally run a local Laya checkpoint and persist a reproducible run.

Nothing here imports the real ``laya`` package at module import time.  The
loader imports it lazily only when the caller explicitly asks to load a local
checkpoint, and tests therefore never download weights or touch the network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Protocol, cast

from eci_decision_models.artifacts import PredictionRecord, atomic_write_text, sha256_file
from eci_decision_models.canonical import CHOICE_IDS
from eci_decision_models.laya import build_laya_request, parse_laya_response
from eci_decision_models.metrics import (
    compute_binary_metrics,
    compute_classification_metrics,
)
from eci_decision_models.probability import PredictionStatus, validate_prediction

_ADAPTER_VERSION: Final = "finetune-eval-v1"
_PARSER_VERSION: Final = "eci-parser-v1"
_DEFAULT_RUN_ID: Final = "finetune-eval"
_DEFAULT_MODEL: Final = "local-laya"
_DEFAULT_DEVICE: Final = "mps"
_PREDICTION_VALID_STATUSES: Final = frozenset(
    {
        PredictionStatus.OK.value,
        PredictionStatus.MISSING_PROBABILITY.value,
        PredictionStatus.INVALID_PROBABILITY.value,
    },
)
_CAUSAL_CHOICES: Final = frozenset({"cause", "caused_by"})
_STATUS_VALUES: Final = frozenset(status.value for status in PredictionStatus)
_TRI_LABEL_TO_CHOICE: Final = {0: "none", 1: "cause", 2: "caused_by"}
_GOLD_KEYS: Final = ("gold_choice", "gold", "label")
_ID_KEYS: Final = ("record_id", "id", "uid")
_RESULT_ID_KEYS: Final = ("record_id", "id", "uid", "item_id", "pair_id")


class LocalAgent(Protocol):
    """Minimal local Laya surface used by the runner."""

    def system_one(
        self,
        state: Mapping[str, object],
        questions: Mapping[str, object],
    ) -> object: ...


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def strict_direction_metrics(
    gold_choices: Sequence[str],
    pred_choices: Sequence[str | None],
) -> dict[str, object]:
    """Return iLIF-style strict direction counts and P/R/F1.

    Gold and predicted choices are in ``{"cause", "caused_by", "none"}``.
    ``None`` is accepted as an invalid prediction and counts as a missed causal
    direction when the gold label is causal; it is neither a false positive nor
    a true negative when the gold label is ``none``.
    """
    if len(gold_choices) != len(pred_choices):
        raise ValueError("gold_choices and pred_choices must have the same length")

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for gold, pred in zip(gold_choices, pred_choices, strict=True):
        if gold not in CHOICE_IDS:
            raise ValueError(f"gold choice {gold!r} must be one of {list(CHOICE_IDS)!r}")
        if pred is not None and pred not in CHOICE_IDS:
            raise ValueError(
                f"predicted choice {pred!r} must be one of {list(CHOICE_IDS)!r} or None",
            )

        if gold in _CAUSAL_CHOICES:
            if pred == gold:
                true_positive += 1
            else:
                false_negative += 1
        elif pred == "none":
            true_negative += 1
        elif pred in _CAUSAL_CHOICES:
            false_positive += 1

    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,
        "precision": precision,
        "recall": recall,
        "f1": _safe_f1(precision, recall),
        "n": len(gold_choices),
    }


def _read_jsonl(path: str | Path) -> list[dict[str, object]]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"unable to read {source}: {exc}") from exc

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload: object = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"{source} line {line_number}: expected a JSON object")
        rows.append({str(key): value for key, value in payload.items()})
    return rows


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    return [cast("str", item) for item in value]


def _event_span(value: object) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    positions: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        positions.append(item)
    return positions


def _esc_texts(row: Mapping[str, object]) -> tuple[str, str, str] | None:
    words = _string_list(row.get("words"))
    events = row.get("events")
    if words is None or not isinstance(events, list) or len(events) != 2:
        return None
    span1 = _event_span(events[0])
    span2 = _event_span(events[1])
    if span1 is None or span2 is None:
        return None
    try:
        sentence = " ".join(words)
        event1 = " ".join(words[index] for index in span1)
        event2 = " ".join(words[index] for index in span2)
    except IndexError as exc:
        raise ValueError("events contain an out-of-range token index") from exc
    return sentence, event1, event2


def _row_texts(row: Mapping[str, object]) -> tuple[str, str, str]:
    esc_texts = _esc_texts(row)
    if esc_texts is not None:
        return esc_texts

    sentence = row.get("sentence")
    event1 = row.get("event1")
    event2 = row.get("event2")
    if isinstance(sentence, str) and isinstance(event1, str) and isinstance(event2, str):
        return sentence, event1, event2

    raise ValueError(
        "test row must contain state/questions, sentence/event1/event2, or ESC words/events",
    )


def load_test_rows(path: str | Path) -> list[dict[str, object]]:
    """Load test JSONL rows and ensure each one carries Laya state/questions.

    Rows may already contain ``state`` and ``questions``.  Otherwise they are
    accepted in either canonical ``sentence``/``event1``/``event2`` form or
    ESC ``words``/``events`` form and converted with the shared request builder.
    """
    rows = _read_jsonl(path)
    prepared: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        state = row.get("state")
        questions = row.get("questions")
        if isinstance(state, Mapping) and isinstance(questions, Mapping):
            prepared_row = dict(row)
            prepared_row["state"] = dict(state)
            prepared_row["questions"] = dict(questions)
            prepared.append(prepared_row)
            continue

        try:
            sentence, event1, event2 = _row_texts(row)
        except ValueError as exc:
            raise ValueError(f"test row {index} is not usable: {exc}") from exc
        request = build_laya_request(sentence, event1, event2)
        prepared_row = dict(row)
        prepared_row["state"] = request.state
        prepared_row["questions"] = request.questions
        prepared.append(prepared_row)
    return prepared


def load_pairs(path: str | Path) -> list[dict[str, object]]:
    """Load gold pair rows used to score the local predictions."""
    return _read_jsonl(path)


def _gold_choice(pair: Mapping[str, object]) -> str:
    for key in _GOLD_KEYS:
        value = pair.get(key)
        if isinstance(value, str) and value in CHOICE_IDS:
            return value

    tri_label = pair.get("tri_causal_label")
    if (
        isinstance(tri_label, int)
        and not isinstance(tri_label, bool)
        and tri_label in _TRI_LABEL_TO_CHOICE
    ):
        return _TRI_LABEL_TO_CHOICE[tri_label]

    value = pair.get("choice")
    if isinstance(value, str) and value in CHOICE_IDS:
        return value

    raise ValueError(
        "test pair must contain a gold choice in one of "
        f"{list(_GOLD_KEYS)!r}, tri_causal_label, or choice",
    )


def _identifier(row: Mapping[str, object], index: int) -> str:
    for key in _ID_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    if "item_id" in row:
        return f"item:{row['item_id']}"
    return f"index:{index}"


def _coerce_probabilities(value: object) -> Mapping[str, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return cast("Mapping[str, float]", {str(key): item for key, item in value.items()})
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return cast("Mapping[str, float]", dict(zip(CHOICE_IDS, value, strict=True)))
    return None


def _normalize_prediction(
    prediction: Mapping[str, object],
) -> tuple[str | None, tuple[float, float, float] | None, str, str | None]:
    status_value = prediction.get("status")
    explicit_status = (
        status_value if isinstance(status_value, str) and status_value in _STATUS_VALUES else None
    )
    error_value = prediction.get("error")
    error = error_value if isinstance(error_value, str) and error_value else None

    if explicit_status == PredictionStatus.API_ERROR.value:
        return None, None, PredictionStatus.API_ERROR.value, error or "prediction failed"

    raw_choice = prediction.get("choice")
    if raw_choice is None:
        raw_choice = prediction.get("prediction")
    if raw_choice is None:
        raw_choice = prediction.get("pred")
    choice = raw_choice if isinstance(raw_choice, str) and raw_choice in CHOICE_IDS else None

    raw_probabilities = prediction.get("probabilities")
    probabilities = _coerce_probabilities(raw_probabilities)
    if raw_probabilities is not None and probabilities is None:
        if choice is None:
            return (
                None,
                None,
                PredictionStatus.INVALID_RESPONSE.value,
                error or "response did not contain a valid choice",
            )
        return (
            choice,
            None,
            PredictionStatus.INVALID_PROBABILITY.value,
            error or "probabilities must be a mapping or a three-element sequence",
        )

    validated = validate_prediction(choice, probabilities)
    return (
        validated.choice,
        validated.probabilities,
        validated.status.value,
        error or validated.error,
    )


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    return str(value)


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prediction_index(
    predictions: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    indexed: dict[str, Mapping[str, object]] = {}
    for index, prediction in enumerate(predictions):
        key = _identifier(prediction, index)
        if key in indexed:
            raise ValueError(f"duplicate prediction identifier {key!r}")
        indexed[key] = prediction
    return indexed


def _item_id(
    pair: Mapping[str, object],
    prediction: Mapping[str, object] | None,
    index: int,
) -> int:
    for source in (pair, prediction):
        if source is None:
            continue
        value = source.get("item_id")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lstrip("-").isdigit():
            return int(value)
    return index


def _make_record(
    pair: Mapping[str, object],
    prediction: Mapping[str, object] | None,
    *,
    index: int,
    run_id: str,
    model: str,
) -> tuple[PredictionRecord, str, str | None]:
    gold_choice = _gold_choice(pair)
    record_id = _identifier(pair, index)
    item_id = _item_id(pair, prediction, index)

    if prediction is None:
        return (
            PredictionRecord(
                run_id=run_id,
                item_id=item_id,
                record_id=record_id,
                model=model,
                attempt=1,
                raw_request_id=f"{run_id}:{record_id}:1",
                request_hash=_sha256_json({"record_id": record_id}),
                response_hash=_sha256_json(None),
                adapter_version=_ADAPTER_VERSION,
                parser_version=_PARSER_VERSION,
                gold_choice=gold_choice,
                prediction=None,
                status=PredictionStatus.API_ERROR.value,
                prediction_valid=False,
                probability_valid=False,
                probabilities=None,
                error=f"no prediction for {record_id!r}",
                control_status="missing_prediction",
            ),
            gold_choice,
            None,
        )

    choice, probabilities, status, error = _normalize_prediction(prediction)
    prediction_valid = status in _PREDICTION_VALID_STATUSES and choice is not None
    probability_valid = status == PredictionStatus.OK.value and probabilities is not None
    request_source = {
        "record_id": record_id,
        "state": pair.get("state"),
        "questions": pair.get("questions"),
    }
    response_source = prediction.get("raw")
    if response_source is None:
        response_source = {
            "choice": choice,
            "probabilities": list(probabilities) if probabilities is not None else None,
            "status": status,
        }

    return (
        PredictionRecord(
            run_id=run_id,
            item_id=item_id,
            record_id=record_id,
            model=model,
            attempt=1,
            raw_request_id=f"{run_id}:{record_id}:1",
            request_hash=_sha256_json(request_source),
            response_hash=_sha256_json(response_source),
            adapter_version=_ADAPTER_VERSION,
            parser_version=_PARSER_VERSION,
            gold_choice=gold_choice,
            prediction=choice,
            status=status,
            prediction_valid=prediction_valid,
            probability_valid=probability_valid,
            probabilities=probabilities,
            error=error,
        ),
        gold_choice,
        choice,
    )


def evaluate_pairs(
    test_pairs: Sequence[Mapping[str, object]],
    predictions: Sequence[Mapping[str, object]],
    *,
    run_id: str = _DEFAULT_RUN_ID,
    model: str = _DEFAULT_MODEL,
) -> dict[str, object]:
    """Join gold pairs with predictions and compute the full metric bundle.

    The join key prefers ``record_id`` then ``id``/``uid``; when neither side
    has an identifier the row position is used.  Missing predictions are kept
    as ``api_error`` records so that coverage stays visible.  Predictions that
    do not correspond to a test pair are reported as ``extra_predictions`` but
    do not affect the metrics.
    """
    indexed = _prediction_index(predictions)
    seen_pairs: set[str] = set()
    records: list[PredictionRecord] = []
    gold_choices: list[str] = []
    pred_choices: list[str | None] = []
    matched = 0

    for index, pair in enumerate(test_pairs):
        key = _identifier(pair, index)
        if key in seen_pairs:
            raise ValueError(f"duplicate test pair identifier {key!r}")
        seen_pairs.add(key)
        prediction = indexed.get(key)
        if prediction is not None:
            matched += 1
        record, gold_choice, pred_choice = _make_record(
            pair,
            prediction,
            index=index,
            run_id=run_id,
            model=model,
        )
        records.append(record)
        gold_choices.append(gold_choice)
        pred_choices.append(pred_choice)

    extra_ids = sorted(set(indexed) - seen_pairs)
    classification = compute_classification_metrics(records)
    binary = compute_binary_metrics(records)
    total = len(records)
    valid = sum(1 for record in records if record.prediction_valid)
    status_counts = Counter(record.status for record in records)
    ordered_status_counts = {status: status_counts[status] for status in sorted(status_counts)}

    return {
        "N": total,
        "valid": valid,
        "fail": total - valid,
        "coverage": (valid / total) if total else 0.0,
        "matched": matched,
        "missing_predictions": total - matched,
        "extra_predictions": len(extra_ids),
        "extra_prediction_ids": extra_ids,
        "status_counts": ordered_status_counts,
        "classification": classification,
        "binary": binary,
        "strict_direction": strict_direction_metrics(gold_choices, pred_choices),
    }


def load_local_laya(
    checkpoint_dir: str | Path,
    device: str = _DEFAULT_DEVICE,
) -> LocalAgent:
    """Load a local Laya checkpoint without any remote revision requirement."""
    laya = importlib.import_module("laya")
    return cast("LocalAgent", laya.load(str(checkpoint_dir), device=device))


def run_local_laya(
    agent: LocalAgent,
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Run one ``system_one`` call per row and parse every response.

    The returned dictionaries carry the identifier fields from the source row so
    that :func:`evaluate_pairs` can join them back to the gold pairs.
    """
    results: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        prediction: dict[str, object] = {}
        for key in _RESULT_ID_KEYS:
            if key in row:
                prediction[key] = row[key]
        if not any(key in prediction for key in _RESULT_ID_KEYS):
            prediction["record_id"] = f"index:{index}"

        state = row.get("state")
        questions = row.get("questions")
        if not isinstance(state, Mapping) or not isinstance(questions, Mapping):
            prediction.update(
                {
                    "raw": None,
                    "choice": None,
                    "probabilities": None,
                    "status": PredictionStatus.INVALID_RESPONSE.value,
                    "error": "row is missing state or questions",
                },
            )
            results.append(prediction)
            continue

        try:
            raw = agent.system_one(state, questions)
        except Exception as exc:  # every model failure stays visible
            prediction.update(
                {
                    "raw": None,
                    "choice": None,
                    "probabilities": None,
                    "status": PredictionStatus.API_ERROR.value,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            results.append(prediction)
            continue

        validated = parse_laya_response(raw)
        prediction.update(
            {
                "raw": _json_safe(raw),
                "choice": validated.choice,
                "probabilities": (
                    list(validated.probabilities) if validated.probabilities is not None else None
                ),
                "status": validated.status.value,
                "error": validated.error,
            },
        )
        results.append(prediction)
    return results


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sdk_version() -> str | None:
    try:
        return importlib.metadata.version("laya")
    except importlib.metadata.PackageNotFoundError:
        return None


def _checkpoint_hash(checkpoint: str | Path, filename: str) -> str | None:
    candidate = Path(checkpoint) / filename
    if not candidate.is_file():
        return None
    return sha256_file(candidate)


def _format_number(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "N/A"
    return str(value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a mapping")
    return cast("Mapping[str, object]", value)


def render_metrics_markdown(metrics: Mapping[str, object], run_id: str) -> str:
    """Render a compact, deterministic Markdown report."""
    classification = _mapping(metrics["classification"])
    binary = _mapping(metrics["binary"])
    strict = _mapping(metrics["strict_direction"])
    status_counts = _mapping(metrics.get("status_counts", {}))

    lines = [
        f"# Fine-tuned Laya evaluation: {run_id}",
        "",
        "## Coverage",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in (
        "N",
        "valid",
        "fail",
        "coverage",
        "matched",
        "missing_predictions",
        "extra_predictions",
    ):
        lines.append(f"| {key} | {_format_number(metrics.get(key))} |")

    lines.extend(
        [
            "",
            "## Three-class",
            "",
            "| metric | value |",
            "| --- | --- |",
            f"| accuracy | {_format_number(classification.get('accuracy'))} |",
            f"| macro_f1 | {_format_number(classification.get('macro_f1'))} |",
            "",
            "| class | precision | recall | f1 | support |",
            "| --- | --- | --- | --- | --- |",
        ],
    )
    per_class = _mapping(classification["per_class"])
    for choice in CHOICE_IDS:
        row = _mapping(per_class[choice])
        lines.append(
            "| "
            + " | ".join(
                [
                    choice,
                    _format_number(row.get("precision")),
                    _format_number(row.get("recall")),
                    _format_number(row.get("f1")),
                    _format_number(row.get("support")),
                ],
            )
            + " |",
        )

    lines.extend(
        [
            "",
            "## Binary existence",
            "",
            "| metric | value |",
            "| --- | --- |",
        ],
    )
    for key in ("precision", "recall", "f1", "accuracy"):
        lines.append(f"| {key} | {_format_number(binary.get(key))} |")

    lines.extend(
        [
            "",
            "## Strict direction (iLIF-style)",
            "",
            "| metric | value |",
            "| --- | --- |",
        ],
    )
    for key in ("tp", "fp", "fn", "tn", "precision", "recall", "f1", "n"):
        lines.append(f"| {key} | {_format_number(strict.get(key))} |")

    lines.extend(["", "## Status", "", "| status | count |", "| --- | --- |"])
    for status in sorted(status_counts):
        lines.append(f"| {status} | {_format_number(status_counts[status])} |")
    lines.append("")
    return "\n".join(lines)


def _write_predictions(path: Path, predictions: Sequence[Mapping[str, object]]) -> None:
    content = "".join(
        json.dumps(
            _json_safe(prediction),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
        for prediction in predictions
    )
    atomic_write_text(path, content)


def _build_meta(
    *,
    run_id: str,
    checkpoint: Path,
    test_jsonl: Path,
    pairs_jsonl: Path,
    device: str,
    command_argv: Sequence[str],
    started_at: str,
    finished_at: str,
    rows: int,
    predictions: int,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "checkpoint": str(checkpoint),
        "checkpoint_config_sha256": _checkpoint_hash(checkpoint, "config.json"),
        "checkpoint_model_safetensors_sha256": _checkpoint_hash(
            checkpoint,
            "model.safetensors",
        ),
        "test_jsonl": str(test_jsonl),
        "test_data_sha256": sha256_file(test_jsonl),
        "pairs_jsonl": str(pairs_jsonl),
        "pairs_data_sha256": sha256_file(pairs_jsonl),
        "command": " ".join(
            [sys.executable, "-m", "eci_decision_models.finetune_eval", *command_argv],
        ),
        "command_argv": list(command_argv),
        "device": device,
        "laya_sdk_version": _sdk_version(),
        "started_at": started_at,
        "finished_at": finished_at,
        "counts": {
            "test_rows": rows,
            "predictions": predictions,
            "pairs": metrics.get("N"),
            "matched": metrics.get("matched"),
            "valid": metrics.get("valid"),
            "fail": metrics.get("fail"),
            "coverage": metrics.get("coverage"),
            "extra_predictions": metrics.get("extra_predictions"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eci_decision_models.finetune_eval",
        description="Evaluate a local fine-tuned Laya checkpoint on fold-0 pairs.",
    )
    parser.add_argument("--checkpoint", required=True, help="Local Laya checkpoint directory")
    parser.add_argument("--test-jsonl", required=True, help="Test rows used for inference")
    parser.add_argument("--pairs-jsonl", required=True, help="Gold pairs used for scoring")
    parser.add_argument("--output-dir", required=True, help="Run output directory")
    parser.add_argument("--device", default=_DEFAULT_DEVICE, help="Local inference device")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run local inference, score it, and persist the run artifacts."""
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    started_at = _utc_now()

    run_id = args.run_id
    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-laya-finetune"

    checkpoint = Path(args.checkpoint)
    test_jsonl = Path(args.test_jsonl)
    pairs_jsonl = Path(args.pairs_jsonl)
    output_dir = Path(args.output_dir)

    rows = load_test_rows(test_jsonl)
    pairs = load_pairs(pairs_jsonl)
    agent = load_local_laya(checkpoint, device=args.device)
    predictions = run_local_laya(agent, rows)
    metrics = evaluate_pairs(pairs, predictions, run_id=run_id)
    finished_at = _utc_now()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(output_dir / "predictions.jsonl", predictions)
    atomic_write_text(
        output_dir / "metrics.json",
        json.dumps(metrics, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(
        output_dir / "metrics.md",
        render_metrics_markdown(metrics, run_id),
    )
    meta = _build_meta(
        run_id=run_id,
        checkpoint=checkpoint,
        test_jsonl=test_jsonl,
        pairs_jsonl=pairs_jsonl,
        device=args.device,
        command_argv=raw_argv,
        started_at=started_at,
        finished_at=finished_at,
        rows=len(rows),
        predictions=len(predictions),
        metrics=metrics,
    )
    atomic_write_text(
        output_dir / "meta.json",
        json.dumps(meta, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
