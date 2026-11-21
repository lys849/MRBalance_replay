"""Offline tests for the single-lane replay runner."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from mrbalance_replay import __version__
from mrbalance_replay.agents import DebateResult
from mrbalance_replay.data import build_question, load_jsonl
from mrbalance_replay.llm import GenerationResult
from mrbalance_replay.manifest import deterministic_gzip
from mrbalance_replay.prompts import (
    DEFAULT_AGENT_ROLES,
    DEFAULT_PROMPT_STYLES,
    ROLES,
)
from mrbalance_replay.runner import (
    RunConfig,
    _select_response,
    build_parser,
    main,
    run,
)

REPLY = "Reasoning (A)"
EXPLANATORY_REPLY = "A careful reading suggests the answer is (C)."
NEW_FIELDS = {
    "gpt_response",
    "consensus_answer",
    "label_consensus",
    "label_prose_naive",
    "label_prose_fixed",
    "gpt_pred_bi_causal_label",
    "n_calls",
    "prompt_tokens",
    "completion_tokens",
}


class FakeClient:
    """Thread-safe offline client with deterministic agent and ranker replies."""

    def __init__(self, reply: str = REPLY, fail_marker: str | None = None) -> None:
        self.reply = reply
        self.fail_marker = fail_marker
        self._lock = threading.Lock()
        self.calls = 0

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        model: str | None = None,
    ) -> GenerationResult:
        del model
        with self._lock:
            self.calls += 1
        text = "\n".join(message["content"] for message in messages)
        if self.fail_marker is not None and self.fail_marker in text:
            raise RuntimeError(f"boom: {self.fail_marker}")
        if len(messages) == 1:
            return GenerationResult(content="[1, 2]", prompt_tokens=1, completion_tokens=1)
        return GenerationResult(content=self.reply, prompt_tokens=1, completion_tokens=1)


class PerItemSequenceClient:
    """Return a deterministic reply sequence per sample, independent of threads."""

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = replies
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        model: str | None = None,
    ) -> GenerationResult:
        del model
        text = "\n".join(message["content"] for message in messages)
        match = re.search(r"item\d+", text)
        key = match.group(0) if match is not None else ""
        with self._lock:
            index = self._counts.get(key, 0)
            self._counts[key] = index + 1
        if len(messages) == 1:
            return GenerationResult(content="[1, 2]", prompt_tokens=1, completion_tokens=1)
        reply = self.replies[index % len(self.replies)]
        return GenerationResult(content=reply, prompt_tokens=1, completion_tokens=1)


def make_item(index: int) -> dict[str, Any]:
    return {
        "item_id": index,
        "words": [f"item{index}", "rain", "flood"],
        "events": [[1], [2]],
        "bi_causal_label": 1,
        "type": "intra",
    }


def write_input(path: Path, count: int) -> Path:
    path.write_text(
        "".join(json.dumps(make_item(index)) + "\n" for index in range(count)),
        encoding="utf-8",
    )
    return path


def read_records(output_dir: Path) -> list[dict[str, Any]]:
    return load_jsonl(output_dir / "response.jsonl")


def test_run_writes_complete_records(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 3)
    output_dir = tmp_path / "output"
    client = FakeClient()

    stats = run(
        RunConfig(input_path=input_path, output_dir=output_dir, concurrency=2),
        client=client,
    )

    records = sorted(read_records(output_dir), key=lambda record: record["item_id"])
    assert len(records) == 3
    assert stats.processed == 3
    assert stats.skipped == 0
    assert stats.failed == 0
    assert stats.n_calls == 9
    assert stats.prompt_tokens == 9
    assert stats.completion_tokens == 9
    assert client.calls == 9

    for index, record in enumerate(records):
        assert set(record) >= set(make_item(index)) | NEW_FIELDS
        assert record["item_id"] == index
        assert record["words"] == make_item(index)["words"]
        assert record["events"] == make_item(index)["events"]
        assert record["gpt_response"] == REPLY
        assert record["consensus_answer"] == "A"
        assert record["label_consensus"] == 1
        assert record["label_prose_naive"] == 1
        assert record["label_prose_fixed"] == 1
        assert record["gpt_pred_bi_causal_label"] == 1
        assert record["n_calls"] == 3
        assert record["prompt_tokens"] == 3
        assert record["completion_tokens"] == 3


def test_resume_skips_existing_items(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 3)
    output_dir = tmp_path / "output"

    first_client = FakeClient()
    run(RunConfig(input_path=input_path, output_dir=output_dir), client=first_client)
    first_records = read_records(output_dir)

    second_client = FakeClient()
    stats = run(RunConfig(input_path=input_path, output_dir=output_dir), client=second_client)

    assert stats.processed == 0
    assert stats.skipped == 3
    assert stats.failed == 0
    assert stats.n_calls == 0
    assert second_client.calls == 0
    assert read_records(output_dir) == first_records


def make_shared_id_item(item_id: int, marker: str) -> dict[str, Any]:
    """CTB-style record: distinct content, colliding ``item_id``."""
    return {
        "item_id": item_id,
        "words": [marker, "rain", "flood"],
        "events": [[1], [2]],
        "bi_causal_label": 1,
        "type": "intra",
    }


def test_duplicate_item_ids_are_all_processed(tmp_path: Path) -> None:
    """CTB reuses item_id across distinct records; each must run on its own."""
    input_path = tmp_path / "input.jsonl"
    items = [make_shared_id_item(7, marker) for marker in ("alpha", "beta", "gamma")]
    input_path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    output_dir = tmp_path / "output"

    stats = run(
        RunConfig(input_path=input_path, output_dir=output_dir, concurrency=3),
        client=FakeClient(),
    )

    records = read_records(output_dir)
    assert stats.processed == 3
    assert stats.skipped == 0
    assert len(records) == 3
    assert {record["words"][0] for record in records} == {"alpha", "beta", "gamma"}
    assert len({record["_record_key"] for record in records}) == 3


def test_resume_does_not_skip_distinct_records_sharing_item_id(tmp_path: Path) -> None:
    """Resume keys on content, not on the non-unique ``item_id``."""
    output_dir = tmp_path / "output"
    first_input = tmp_path / "first.jsonl"
    first_items = [make_shared_id_item(7, "alpha"), make_shared_id_item(7, "beta")]
    first_input.write_text(
        "".join(json.dumps(item) + "\n" for item in first_items), encoding="utf-8"
    )
    run(RunConfig(input_path=first_input, output_dir=output_dir), client=FakeClient())

    second_input = tmp_path / "second.jsonl"
    second_items = [*first_items, make_shared_id_item(7, "gamma")]
    second_input.write_text(
        "".join(json.dumps(item) + "\n" for item in second_items), encoding="utf-8"
    )
    stats = run(RunConfig(input_path=second_input, output_dir=output_dir), client=FakeClient())

    records = read_records(output_dir)
    assert stats.skipped == 2
    assert stats.processed == 1
    assert len(records) == 3
    assert {record["words"][0] for record in records} == {"alpha", "beta", "gamma"}


def test_limit_truncates_input(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 5)
    output_dir = tmp_path / "output"

    stats = run(
        RunConfig(input_path=input_path, output_dir=output_dir, limit=2),
        client=FakeClient(),
    )

    records = read_records(output_dir)
    assert stats.processed == 2
    assert stats.skipped == 0
    # Concurrent workers append in completion order, so compare as a set.
    assert sorted(record["item_id"] for record in records) == [0, 1]


def test_concurrent_run_does_not_lose_or_duplicate_rows(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 8)
    output_dir = tmp_path / "output"
    client = FakeClient()

    stats = run(
        RunConfig(input_path=input_path, output_dir=output_dir, concurrency=4),
        client=client,
    )

    records = read_records(output_dir)
    item_ids = [record["item_id"] for record in records]
    assert stats.processed == 8
    assert len(records) == 8
    assert len(set(item_ids)) == 8
    assert sorted(item_ids) == list(range(8))
    assert client.calls == 24


def test_failed_sample_does_not_abort_others(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 3)
    output_dir = tmp_path / "output"
    client = FakeClient(fail_marker="item1")

    stats = run(
        RunConfig(input_path=input_path, output_dir=output_dir, concurrency=2),
        client=client,
    )

    records = sorted(read_records(output_dir), key=lambda record: record["item_id"])
    assert len(records) == 3
    assert stats.processed == 2
    assert stats.failed == 1
    assert stats.skipped == 0
    assert stats.n_calls == 6

    failed = records[1]
    assert failed["item_id"] == 1
    assert failed["gpt_response"] is None
    assert failed["consensus_answer"] is None
    assert failed["label_consensus"] is None
    assert failed["label_prose_naive"] is None
    assert failed["label_prose_fixed"] is None
    assert failed["gpt_pred_bi_causal_label"] is None
    assert failed["n_calls"] == 0
    assert "error" in failed
    assert "boom" in failed["error"]

    for record in (records[0], records[2]):
        assert record["gpt_pred_bi_causal_label"] == 1
        assert "error" not in record


def test_failed_records_are_retried_by_default(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 2)
    output_dir = tmp_path / "output"

    first = run(
        RunConfig(input_path=input_path, output_dir=output_dir),
        client=FakeClient(fail_marker="item1"),
    )
    assert first.processed == 1
    assert first.failed == 1

    second = run(RunConfig(input_path=input_path, output_dir=output_dir), client=FakeClient())

    assert second.processed == 1
    assert second.skipped == 1
    assert second.failed == 0
    records = read_records(output_dir)
    assert len(records) == 3
    recovered = [record for record in records if record["item_id"] == 1 and "error" not in record]
    assert len(recovered) == 1
    assert recovered[0]["gpt_pred_bi_causal_label"] == 1


def test_retry_failed_can_be_disabled(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 2)
    output_dir = tmp_path / "output"
    run(
        RunConfig(input_path=input_path, output_dir=output_dir),
        client=FakeClient(fail_marker="item1"),
    )

    second = run(
        RunConfig(input_path=input_path, output_dir=output_dir, retry_failed=False),
        client=FakeClient(),
    )

    assert second.processed == 0
    assert second.skipped == 2
    assert second.failed == 0


def test_cli_reads_retry_failed_flag() -> None:
    assert build_parser().parse_args([]).retry_failed is True
    assert build_parser().parse_args(["--no-retry-failed"]).retry_failed is False


def test_select_response_returns_none_without_consensus_answer() -> None:
    for answer in (None, ""):
        result = DebateResult(
            answer=answer,
            activations=0,
            ranker_calls=0,
            prompt_tokens=0,
            completion_tokens=0,
            completions=[["prose (A)"]],
        )
        assert _select_response(result) is None


def test_label_modes_select_consensus_naive_or_fixed(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 1)

    records = {}
    for mode in ("consensus", "naive", "fixed"):
        output_dir = tmp_path / mode
        run(
            RunConfig(input_path=input_path, output_dir=output_dir, label_mode=mode),
            client=FakeClient(reply=EXPLANATORY_REPLY),
        )
        records[mode] = read_records(output_dir)[0]

    for record in records.values():
        assert record["gpt_response"] == EXPLANATORY_REPLY
        assert record["consensus_answer"] == "C"
        assert record["label_consensus"] == 0
        assert record["label_prose_naive"] == 1
        assert record["label_prose_fixed"] == 0

    # Consensus uses the debate letter (C -> 0); naive re-parses the prose and
    # mistakes the article "A" for option A; fixed reads the final (C).
    assert records["consensus"]["gpt_pred_bi_causal_label"] == 0
    assert records["naive"]["gpt_pred_bi_causal_label"] == 1
    assert records["fixed"]["gpt_pred_bi_causal_label"] == 0


def test_same_seed_reproduces_label_sequence(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 6)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    replies = ("A", "B", "C", "B", "A", "C")

    run(
        RunConfig(input_path=input_path, output_dir=first_dir, concurrency=3, seed=11),
        client=PerItemSequenceClient(replies),
    )
    run(
        RunConfig(input_path=input_path, output_dir=second_dir, concurrency=3, seed=11),
        client=PerItemSequenceClient(replies),
    )

    def labels(output_dir: Path) -> list[object]:
        records = sorted(read_records(output_dir), key=lambda record: record["item_id"])
        return [record["gpt_pred_bi_causal_label"] for record in records]

    assert labels(first_dir) == labels(second_dir)


def test_evaluate_prints_metrics_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 1)
    output_dir = tmp_path / "output"

    run(
        RunConfig(input_path=input_path, output_dir=output_dir, evaluate=True),
        client=FakeClient(),
    )

    printed = capsys.readouterr().out
    assert "Run summary:" in printed
    assert "All test" in printed
    assert "Skipped" not in printed


def test_cli_help_exits_without_settings(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "MRBalance replay pipeline" in capsys.readouterr().out


def test_run_config_experiment_switches_default_to_official_behaviour() -> None:
    config = RunConfig()

    assert config.shuffle_roles_each_round is False
    assert config.selection == "listwise"
    assert config.fix_top2_activation is False


def test_cli_parses_experiment_switches() -> None:
    default = build_parser().parse_args([])
    assert default.shuffle_roles_each_round is False
    assert default.selection == "listwise"
    assert default.fix_top2_activation is False

    enabled = build_parser().parse_args(
        ["--shuffle-roles-each-round", "--selection", "importance", "--fix-top2-activation"]
    )
    assert enabled.shuffle_roles_each_round is True
    assert enabled.selection == "importance"
    assert enabled.fix_top2_activation is True


def test_cli_forwards_experiment_switches_to_run_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, RunConfig] = {}

    def fake_run(config: RunConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr("mrbalance_replay.runner.run", fake_run)

    assert (
        main(["--shuffle-roles-each-round", "--selection", "importance", "--fix-top2-activation"])
        == 0
    )
    assert captured["config"].shuffle_roles_each_round is True
    assert captured["config"].selection == "importance"
    assert captured["config"].fix_top2_activation is True


def test_fix_top2_activation_is_forwarded_to_debate_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    class RecordingGraph:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def forward(self, question: str, client: Any) -> DebateResult:
            del question, client
            return DebateResult(
                answer="A",
                activations=0,
                ranker_calls=0,
                prompt_tokens=0,
                completion_tokens=0,
                completions=[["reasoning (A)"]],
            )

    monkeypatch.setattr("mrbalance_replay.runner.DebateGraph", RecordingGraph)
    input_path = write_input(tmp_path / "input.jsonl", 1)

    run(
        RunConfig(input_path=input_path, output_dir=tmp_path / "out", fix_top2_activation=True),
        client=FakeClient(),
    )

    assert captured["fix_top2_activation"] is True


def test_selection_control_changes_call_count(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 1)
    replies = ("(A)", "(B)", "(C)", "(A)", "(B)", "(C)", "(A)", "(B)", "(C)", "(A)")

    listwise_dir = tmp_path / "listwise"
    listwise_stats = run(
        RunConfig(input_path=input_path, output_dir=listwise_dir, selection="listwise"),
        client=PerItemSequenceClient(replies),
    )
    assert listwise_stats.n_calls == 11
    assert read_records(listwise_dir)[0]["n_calls"] == 11

    importance_dir = tmp_path / "importance"
    importance_stats = run(
        RunConfig(input_path=input_path, output_dir=importance_dir, selection="importance"),
        client=PerItemSequenceClient(replies),
    )
    assert importance_stats.n_calls == 10
    assert read_records(importance_dir)[0]["n_calls"] == 10


class RecordingClient:
    """Record every prompt and return a fixed single-agent reply."""

    def __init__(self, reply: str = "(A)") -> None:
        self.reply = reply
        self._lock = threading.Lock()
        self.calls: list[list[dict[str, str]]] = []

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        model: str | None = None,
    ) -> GenerationResult:
        del model
        with self._lock:
            self.calls.append(list(messages))
        return GenerationResult(content=self.reply, prompt_tokens=2, completion_tokens=3)


def test_single_zeroshot_uses_one_call_and_writes_shared_fields(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 1)
    output_dir = tmp_path / "output"
    client = RecordingClient(reply="(A)")

    # Debate-only knobs are deliberately invalid: single mode must ignore them.
    stats = run(
        RunConfig(
            input_path=input_path,
            output_dir=output_dir,
            mode="single_zeroshot",
            agents=0,
            rounds=0,
            selection="not-a-real-selection",
            shuffle_roles_each_round=True,
        ),
        client=client,
    )

    assert stats.processed == 1
    assert stats.n_calls == 1
    assert len(client.calls) == 1
    record = read_records(output_dir)[0]
    assert set(record) >= set(make_item(0)) | NEW_FIELDS
    assert record["gpt_response"] == "(A)"
    assert record["consensus_answer"] == "A"
    assert record["label_consensus"] == 1
    assert record["label_prose_naive"] == 1
    assert record["label_prose_fixed"] == 1
    assert record["gpt_pred_bi_causal_label"] == 1
    assert record["n_calls"] == 1
    assert record["prompt_tokens"] == 2
    assert record["completion_tokens"] == 3


@pytest.mark.parametrize(
    ("mode", "has_cot"),
    [("single_zeroshot", False), ("single_cot", True)],
)
def test_single_agent_prompts(mode: str, has_cot: bool, tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 1)
    output_dir = tmp_path / "output"
    client = RecordingClient()

    run(RunConfig(input_path=input_path, output_dir=output_dir, mode=mode), client=client)

    messages = client.calls[0]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[0]["content"] == ROLES["causal analysis expert"]
    assert "causal analysis expert" in messages[0]["content"]
    assert messages[1]["content"] == build_question(make_item(0)) + (
        "\nLet's think step by step." if has_cot else ""
    )
    assert ("Let's think step by step." in messages[1]["content"]) is has_cot


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("(A)", 1),
        ("(B)", 1),
        ("(C)", 0),
        ("I cannot decide.", None),
    ],
)
def test_single_agent_consensus_label_mapping(
    reply: str, expected: int | None, tmp_path: Path
) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 1)
    output_dir = tmp_path / reply.replace(" ", "_").replace("(", "").replace(")", "")

    run(
        RunConfig(input_path=input_path, output_dir=output_dir, mode="single_zeroshot"),
        client=RecordingClient(reply=reply),
    )

    record = read_records(output_dir)[0]
    expected_answer = {"(A)": "A", "(B)": "B", "(C)": "C"}.get(reply)
    assert record["consensus_answer"] == expected_answer
    assert record["label_consensus"] == expected
    assert record["gpt_pred_bi_causal_label"] == expected
    assert record["n_calls"] == 1


def test_single_agent_limit_and_resume(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 3)
    output_dir = tmp_path / "output"

    first_client = RecordingClient()
    first = run(
        RunConfig(
            input_path=input_path,
            output_dir=output_dir,
            mode="single_zeroshot",
            limit=2,
        ),
        client=first_client,
    )
    assert first.processed == 2
    assert first.n_calls == 2
    assert len(first_client.calls) == 2
    assert len(read_records(output_dir)) == 2

    second_client = RecordingClient()
    second = run(
        RunConfig(input_path=input_path, output_dir=output_dir, mode="single_zeroshot"),
        client=second_client,
    )
    assert second.processed == 1
    assert second.skipped == 2
    assert second.n_calls == 1
    assert len(second_client.calls) == 1
    assert len(read_records(output_dir)) == 3


def test_invalid_mode_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported mode"):
        run(
            RunConfig(
                input_path=tmp_path / "input.jsonl",
                output_dir=tmp_path / "output",
                mode="bogus",
            ),
            client=FakeClient(),
        )


def test_cli_mode_defaults_and_single_mode_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    assert build_parser().parse_args([]).mode == "debate"

    captured: dict[str, RunConfig] = {}

    def fake_run(config: RunConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr("mrbalance_replay.runner.run", fake_run)

    with caplog.at_level(logging.WARNING):
        assert main(["--mode", "single_cot", "--agents", "9", "--rounds", "9"]) == 0

    assert captured["config"].mode == "single_cot"
    assert captured["config"].agents == 9
    assert "ignores" in caplog.text


def test_cli_roles_and_prompt_styles_default_to_module_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.roles == DEFAULT_AGENT_ROLES
    assert args.prompt_styles == DEFAULT_PROMPT_STYLES


def test_cli_roles_and_prompt_styles_allow_surrounding_whitespace() -> None:
    args = build_parser().parse_args(
        [
            "--roles",
            "none, NLPexpert, knowledge graphs expert, causal analysis expert",
            "--prompt-styles",
            " positive , normal,none ,negative",
        ]
    )

    assert args.roles == (
        "none",
        "NLPexpert",
        "knowledge graphs expert",
        "causal analysis expert",
    )
    assert args.prompt_styles == ("positive", "normal", "none", "negative")


def test_cli_rejects_unknown_role_and_lists_valid_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--roles", "bogus"])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "unknown role(s): bogus" in stderr
    assert "knowledge graphs expert" in stderr
    assert "causal analysis expert" in stderr
    assert "NLPexpert" in stderr


def test_cli_rejects_unknown_prompt_style_and_lists_valid_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["--prompt-styles", "bogus"])

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "unknown prompt style(s): bogus" in stderr
    for style in ("none", "positive", "normal", "normal1", "negative"):
        assert style in stderr


def test_cli_rejects_role_count_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--agents", "2", "--roles", "none"])

    assert exc_info.value.code == 2
    assert "entries but --agents is 2" in capsys.readouterr().err


def test_cli_rejects_prompt_style_count_mismatch(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--agents", "2", "--roles", "none,none", "--prompt-styles", "none"])

    assert exc_info.value.code == 2
    assert "entries but --agents is 2" in capsys.readouterr().err


def test_cli_forwards_roles_and_prompt_styles_to_run_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, RunConfig] = {}

    def fake_run(config: RunConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr("mrbalance_replay.runner.run", fake_run)

    assert (
        main(
            [
                "--agents",
                "2",
                "--roles",
                "none, NLPexpert",
                "--prompt-styles",
                "normal, negative",
            ]
        )
        == 0
    )
    assert captured["config"].roles == ("none", "NLPexpert")
    assert captured["config"].prompt_styles == ("normal", "negative")


def test_cli_single_mode_ignores_role_and_style_counts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    captured: dict[str, RunConfig] = {}

    def fake_run(config: RunConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr("mrbalance_replay.runner.run", fake_run)

    with caplog.at_level(logging.WARNING):
        assert (
            main(
                [
                    "--mode",
                    "single_cot",
                    "--agents",
                    "9",
                    "--roles",
                    "none",
                    "--prompt-styles",
                    "negative",
                ]
            )
            == 0
        )

    assert captured["config"].roles == ("none",)
    assert captured["config"].prompt_styles == ("negative",)
    assert "roles/prompt_styles" in caplog.text


def test_manifest_records_metadata_metrics_and_no_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "super-secret-value")
    input_path = write_input(tmp_path / "input.jsonl", 2)
    output_dir = tmp_path / "output"

    stats = run(
        RunConfig(input_path=input_path, output_dir=output_dir, run_id="demo-run"),
        client=FakeClient(),
    )

    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == 1
    assert meta["run_id"] == "demo-run"
    assert meta["created_at"]
    assert meta["started_at"]
    assert meta["finished_at"]
    assert meta["package_version"] == __version__
    assert meta["command"]
    assert set(meta["git"]) == {"commit", "branch", "dirty"}
    assert meta["model"]
    assert meta["reasoning_effort"]

    assert meta["config"]["input_path"] == str(input_path)
    assert meta["config"]["output_dir"] == str(output_dir)
    assert meta["config"]["run_id"] == "demo-run"
    assert meta["config"]["roles"] == list(DEFAULT_AGENT_ROLES)
    assert meta["config"]["prompt_styles"] == list(DEFAULT_PROMPT_STYLES)

    assert meta["data"] == {
        "path": str(input_path),
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "n_records": 2,
    }
    response_path = output_dir / "response.jsonl"
    assert meta["output"]["path"] == str(response_path)
    assert meta["output"]["sha256"] == hashlib.sha256(response_path.read_bytes()).hexdigest()
    assert meta["output"]["n_records"] == 2
    assert meta["output"]["n_failed"] == 0

    assert meta["stats"]["processed"] == 2
    assert meta["stats"]["skipped"] == 0
    assert meta["stats"]["failed"] == 0
    assert meta["stats"]["calls"] == stats.n_calls
    assert meta["stats"]["prompt_tokens"] == stats.prompt_tokens
    assert meta["stats"]["completion_tokens"] == stats.completion_tokens

    assert set(meta["metrics"]) == {"all", "intra", "inter"}
    expected_metric_keys = {
        "precision",
        "recall",
        "f1",
        "pos_recall",
        "neg_recall",
        "accuracy",
        "n_samples",
    }
    for split in meta["metrics"].values():
        assert set(split) == expected_metric_keys
    assert meta["metrics"]["all"]["n_samples"] == 2
    assert meta["metrics"]["all"]["accuracy"] == 1.0

    payload = (output_dir / "run_meta.json").read_text(encoding="utf-8")
    assert "super-secret-value" not in payload
    assert "api_key" not in payload
    assert "OPENAI_API_KEY" not in payload


def test_manifest_run_id_defaults_to_output_dir_name(tmp_path: Path) -> None:
    output_dir = tmp_path / "default-run"
    run(
        RunConfig(input_path=write_input(tmp_path / "input.jsonl", 1), output_dir=output_dir),
        client=FakeClient(),
    )

    meta = json.loads((output_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "default-run"


def test_manifest_is_not_written_when_the_run_raises(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    input_path = write_input(tmp_path / "input.jsonl", 1)

    with pytest.raises(ValueError, match="unsupported mode"):
        run(
            RunConfig(input_path=input_path, output_dir=output_dir, mode="bogus"),
            client=FakeClient(),
        )

    assert not (output_dir / "run_meta.json").exists()


def test_experiment_dir_writes_four_artifacts(tmp_path: Path) -> None:
    input_path = write_input(tmp_path / "input.jsonl", 2)
    output_dir = tmp_path / "output"
    experiment_dir = tmp_path / "experiment"

    run(
        RunConfig(
            input_path=input_path,
            output_dir=output_dir,
            run_id="experiment-run",
            experiment_dir=experiment_dir,
        ),
        client=FakeClient(),
    )

    assert sorted(path.name for path in experiment_dir.iterdir()) == [
        "meta.json",
        "metrics.json",
        "metrics.md",
        "response.jsonl.gz",
    ]
    assert json.loads((experiment_dir / "meta.json").read_text(encoding="utf-8")) == json.loads(
        (output_dir / "run_meta.json").read_text(encoding="utf-8")
    )

    metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"all", "intra", "inter"}
    assert metrics["all"]["n_samples"] == 2
    assert "All test" in (experiment_dir / "metrics.md").read_text(encoding="utf-8")

    response_bytes = (output_dir / "response.jsonl").read_bytes()
    with gzip.open(experiment_dir / "response.jsonl.gz", "rb") as handle:
        assert handle.read() == response_bytes


def test_deterministic_gzip_produces_identical_bytes(tmp_path: Path) -> None:
    source = tmp_path / "response.jsonl"
    source.write_bytes(b'{"a": 1}\n{"b": 2}\n')
    first = tmp_path / "first.gz"
    second = tmp_path / "second.gz"

    deterministic_gzip(source=source, destination=first)
    deterministic_gzip(source=source, destination=second)

    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rb") as handle:
        assert handle.read() == source.read_bytes()


def test_cli_manifest_options_default_to_none() -> None:
    args = build_parser().parse_args([])

    assert args.run_id is None
    assert args.experiment_dir is None


def test_cli_forwards_manifest_options_to_run_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, RunConfig] = {}

    def fake_run(config: RunConfig) -> None:
        captured["config"] = config

    monkeypatch.setattr("mrbalance_replay.runner.run", fake_run)

    experiment_dir = tmp_path / "experiment"
    assert main(["--run-id", "cli-run", "--experiment-dir", str(experiment_dir)]) == 0
    assert captured["config"].run_id == "cli-run"
    assert captured["config"].experiment_dir == experiment_dir
