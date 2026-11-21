"""Single-lane runner for the MRBalance replay pipeline.

The runner loads ECI JSONL samples, executes one independent debate graph per
sample, and appends the predictions to ``<output_dir>/response.jsonl``.  It is
designed for long, restartable experiments: resume is keyed on a content
fingerprint of each input record (``item_id`` is NOT unique in CTB — distinct
records share it), samples run concurrently, and one failing sample never
aborts the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import threading
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TextIO, cast

from .agents import DebateGraph, DebateResult
from .config import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT, load_settings
from .data import build_question, load_jsonl
from .evaluation.metrics import compute_metrics, format_report
from .llm import LLMClient, OpenAIClient
from .manifest import utc_now, write_manifest
from .parsing import extract_final_label, parse_single_choice
from .prompts import DEFAULT_AGENT_ROLES, DEFAULT_PROMPT_STYLES, PROMPT_STYLES, ROLES

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("reference/MRBalance/data/test/ESC_intra_data.json")
DEFAULT_OUTPUT_DIR = Path("outputs/esc")
DEFAULT_AGENTS = 4
DEFAULT_ROUNDS = 3
DEFAULT_CONCURRENCY = 8
DEFAULT_LABEL_MODE = "consensus"
_LABEL_MODES = ("consensus", "naive", "fixed")
DEFAULT_SELECTION = "listwise"
_SELECTIONS = ("listwise", "importance")
DEFAULT_MODE = "debate"
_MODES = ("debate", "single_zeroshot", "single_cot")
_SINGLE_MODES = ("single_zeroshot", "single_cot")
SINGLE_AGENT_ROLE = "causal analysis expert"


@dataclass
class RunConfig:
    """Configuration for one replay run."""

    input_path: Path = DEFAULT_INPUT
    output_dir: Path = DEFAULT_OUTPUT_DIR
    run_id: str | None = None
    experiment_dir: Path | None = None
    mode: str = DEFAULT_MODE
    model: str | None = None
    reasoning_effort: str | None = None
    agents: int = DEFAULT_AGENTS
    rounds: int = DEFAULT_ROUNDS
    roles: Sequence[str] = DEFAULT_AGENT_ROLES
    prompt_styles: Sequence[str] = DEFAULT_PROMPT_STYLES
    concurrency: int = DEFAULT_CONCURRENCY
    limit: int | None = None
    label_mode: str = DEFAULT_LABEL_MODE
    seed: int = 0
    evaluate: bool = False
    retry_failed: bool = True
    shuffle_roles_each_round: bool = False
    selection: str = DEFAULT_SELECTION
    fix_top2_activation: bool = False


@dataclass
class RunStats:
    """Aggregate counters for a completed run."""

    processed: int = 0
    skipped: int = 0
    failed: int = 0
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_seconds: float = 0.0

    @property
    def attempted(self) -> int:
        """Samples that actually entered the debate (successful or failed)."""
        return self.processed + self.failed

    @property
    def average_seconds(self) -> float:
        """Average wall-clock seconds per attempted sample."""
        if self.attempted == 0:
            return 0.0
        return self.elapsed_seconds / self.attempted

    @property
    def total_calls(self) -> int:
        """Alias for ``n_calls`` used in human-readable reports."""
        return self.n_calls


@dataclass(frozen=True)
class _Outcome:
    failed: bool
    n_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _resolve_client(config: RunConfig, client: LLMClient | None) -> tuple[LLMClient, str, str]:
    """Return the client, model name and reasoning effort for the run."""

    if client is not None:
        return (
            client,
            config.model or DEFAULT_MODEL,
            config.reasoning_effort or DEFAULT_REASONING_EFFORT,
        )

    settings = load_settings()
    model = config.model or settings.model
    reasoning_effort = (
        config.reasoning_effort
        if config.reasoning_effort is not None
        else settings.reasoning_effort
    )
    effective_settings = replace(settings, model=model, reasoning_effort=reasoning_effort)
    return OpenAIClient(effective_settings), model, reasoning_effort


# Keys the runner adds to each output row; stripped when fingerprinting so that
# rows written by earlier versions (without ``_record_key``) still match.
_ADDED_KEYS = frozenset(
    {
        "gpt_response",
        "consensus_answer",
        "label_consensus",
        "label_prose_naive",
        "label_prose_fixed",
        "gpt_pred_bi_causal_label",
        "n_calls",
        "prompt_tokens",
        "completion_tokens",
        "error",
        "_record_key",
    }
)


def _record_key(item: dict[str, Any]) -> str:
    """Stable fingerprint of an input record's content.

    ``item_id`` cannot be used: CTB assigns the same id to distinct records.
    """
    payload = {key: value for key, value in item.items() if key not in _ADDED_KEYS}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


def _existing_record_keys(path: Path, *, retry_failed: bool = True) -> set[str]:
    """Return fingerprints of rows that should block a repeat run.

    Rows carrying an ``error`` field are excluded by default so a failed sample
    is retried, and retried again on every subsequent run until it succeeds.
    """

    if not path.exists():
        return set()
    keys: set[str] = set()
    for item in load_jsonl(path):
        if retry_failed and item.get("error"):
            continue
        keys.add(_record_key(item))
    return keys


def _write_record(handle: TextIO, lock: threading.Lock, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with lock:
        handle.write(line)
        handle.flush()


def _consensus_label(answer: str | None) -> int | None:
    """Map the debate consensus letter to a binary label.

    This is the official-equivalent label: ``Muty_agent_debate.py`` stores the
    consensus letter in ``gpt_response`` and then tokenizes it.  A/B mean a
    causal relation, C means no relation, and missing/other answers stay
    ``None``.
    """

    if answer in ("A", "B"):
        return 1
    if answer == "C":
        return 0
    return None


def _select_response(result: DebateResult) -> str | None:
    """Pick the raw model reply that supports the final consensus answer.

    ``DebateResult.answer`` is the parsed option (A/B/C).  The raw reply is kept
    for offline prose analysis (``naive``/``fixed``); it is no longer used to
    define the official-equivalent label.

    When the graph cannot produce a consensus answer, this returns ``None``.
    The official reference turns an empty answer into "C" and therefore label
    0; we keep ``None`` so the missing consensus is visible in the output.
    """

    answer = result.answer
    if answer is None or answer == "":
        return None

    match: str | None = None
    fallback: str | None = None
    for replies in result.completions:
        for reply in replies:
            if reply is None:
                continue
            fallback = reply
            if parse_single_choice(reply) == answer:
                match = reply
    if match is not None:
        return match
    return fallback


def _label_fields(
    response: str | None,
    consensus_answer: str | None,
    config: RunConfig,
) -> dict[str, Any]:
    """Build the shared label columns used by debate and single-agent runs."""

    label_consensus = _consensus_label(consensus_answer)
    label_prose_naive = extract_final_label(response, "naive")
    label_prose_fixed = extract_final_label(response, "fixed")
    selected_label = {
        "consensus": label_consensus,
        "naive": label_prose_naive,
        "fixed": label_prose_fixed,
    }[config.label_mode]
    return {
        "gpt_response": response,
        "consensus_answer": consensus_answer,
        "label_consensus": label_consensus,
        "label_prose_naive": label_prose_naive,
        "label_prose_fixed": label_prose_fixed,
        "gpt_pred_bi_causal_label": selected_label,
    }


def _single_agent_messages(question: str, mode: str) -> list[dict[str, str]]:
    """Build the single-agent baseline prompt.

    ``single_cot`` appends the standard zero-shot-COT line.  The official
    ``ECI.py`` zero-shot-COT branch is an empty stub, so this is our own
    implementation rather than a reference port.
    """

    user_content = question
    if mode == "single_cot":
        user_content = f"{question}\nLet's think step by step."
    return [
        {"role": "system", "content": ROLES[SINGLE_AGENT_ROLE]},
        {"role": "user", "content": user_content},
    ]


def _run_single_agent(
    item: dict[str, Any],
    config: RunConfig,
    model: str,
    client: LLMClient,
) -> tuple[str, str | None, int, int, int]:
    """Run exactly one LLM call and return reply, answer, calls and tokens."""

    question = build_question(item)
    generation = client.generate(_single_agent_messages(question, config.mode), model=model)
    response = generation.content
    consensus_answer = parse_single_choice(response)
    return (
        response,
        consensus_answer,
        1,
        generation.prompt_tokens,
        generation.completion_tokens,
    )


def _process_item(
    item: dict[str, Any],
    config: RunConfig,
    model: str,
    client: LLMClient,
    handle: TextIO,
    lock: threading.Lock,
) -> _Outcome:
    item_id = item.get("item_id")
    try:
        response: str | None
        consensus_answer: str | None
        if config.mode in _SINGLE_MODES:
            response, consensus_answer, n_calls, prompt_tokens, completion_tokens = (
                _run_single_agent(item, config, model, client)
            )
        else:
            question = build_question(item)
            # The graph mutates state during forward/backward, so every sample
            # gets its own instance.  Reusing one graph across threads would
            # corrupt it.  Seed each sample independently from its content
            # fingerprint so runs are reproducible even when completion order
            # varies.
            sample_seed = config.seed ^ int(_record_key(item)[:8], 16)
            graph = DebateGraph(
                agents=config.agents,
                rounds=config.rounds,
                agent_roles=config.roles,
                prompt_styles=config.prompt_styles,
                model=model,
                shuffle_roles_each_round=config.shuffle_roles_each_round,
                selection=config.selection,
                fix_top2_activation=config.fix_top2_activation,
                rng=random.Random(sample_seed),
            )
            # ``LLMClientLike`` declares mutable result attributes while
            # ``GenerationResult`` is frozen; the runtime interface is identical.
            result = graph.forward(question, cast(Any, client))
            response = _select_response(result)
            consensus_answer = result.answer
            n_calls = result.calls
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens

        record = dict(item)
        record.update(_label_fields(response, consensus_answer, config))
        record["n_calls"] = n_calls
        record["prompt_tokens"] = prompt_tokens
        record["completion_tokens"] = completion_tokens
        record["_record_key"] = _record_key(item)
        _write_record(handle, lock, record)
        return _Outcome(
            failed=False,
            n_calls=n_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception:
        logger.exception("sample %s failed", item_id)
        record = dict(item)
        record["gpt_response"] = None
        record["consensus_answer"] = None
        record["label_consensus"] = None
        record["label_prose_naive"] = None
        record["label_prose_fixed"] = None
        record["gpt_pred_bi_causal_label"] = None
        record["n_calls"] = 0
        record["prompt_tokens"] = 0
        record["completion_tokens"] = 0
        record["_record_key"] = _record_key(item)
        record["error"] = traceback.format_exc()
        try:
            _write_record(handle, lock, record)
        except Exception:
            logger.exception("failed to write error record for sample %s", item_id)
        return _Outcome(failed=True)


def _print_summary(stats: RunStats) -> None:
    print(
        "Run summary: "
        f"processed={stats.processed} skipped={stats.skipped} failed={stats.failed} "
        f"calls={stats.n_calls} "
        f"prompt_tokens={stats.prompt_tokens} "
        f"completion_tokens={stats.completion_tokens} "
        f"elapsed={stats.elapsed_seconds:.2f}s "
        f"avg_per_sample={stats.average_seconds:.2f}s"
    )


def run(config: RunConfig, client: LLMClient | None = None) -> RunStats:
    """Execute one replay run and return aggregate statistics."""

    if config.mode not in _MODES:
        raise ValueError(f"unsupported mode: {config.mode!r}")
    if config.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if config.label_mode not in _LABEL_MODES:
        raise ValueError(f"unsupported label_mode: {config.label_mode!r}")
    if config.mode == "debate":
        if config.agents <= 0:
            raise ValueError("agents must be positive")
        if config.rounds <= 0:
            raise ValueError("rounds must be positive")
        if config.selection not in _SELECTIONS:
            raise ValueError(f"unsupported selection: {config.selection!r}")
        _validate_roles(config.roles, config.agents)
        _validate_prompt_styles(config.prompt_styles, config.agents)

    started_at = utc_now()
    started = time.perf_counter()
    random.seed(config.seed)
    resolved_client, model, reasoning_effort = _resolve_client(config, client)

    response_path = Path(config.output_dir) / "response.jsonl"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    items = load_jsonl(config.input_path)
    if config.limit is not None:
        items = items[: config.limit]

    existing_keys = _existing_record_keys(response_path, retry_failed=config.retry_failed)
    stats = RunStats()
    pending: list[dict[str, Any]] = []
    for item in items:
        if _record_key(item) in existing_keys:
            stats.skipped += 1
        else:
            pending.append(item)

    lock = threading.Lock()
    with (
        response_path.open("a", encoding="utf-8") as handle,
        ThreadPoolExecutor(max_workers=config.concurrency) as executor,
    ):
        futures = {
            executor.submit(
                _process_item,
                item,
                config,
                model,
                resolved_client,
                handle,
                lock,
            ): item
            for item in pending
        }
        for future in as_completed(futures):
            outcome = future.result()
            if outcome.failed:
                stats.failed += 1
            else:
                stats.processed += 1
            stats.n_calls += outcome.n_calls
            stats.prompt_tokens += outcome.prompt_tokens
            stats.completion_tokens += outcome.completion_tokens

    stats.elapsed_seconds = time.perf_counter() - started
    finished_at = utc_now()
    _print_summary(stats)

    # Metrics always feed the manifest (and the experiment artifacts); the
    # ``--evaluate`` flag only controls whether they are printed.
    report = compute_metrics(load_jsonl(response_path))
    if config.evaluate:
        print(format_report(report))

    write_manifest(
        config=config,
        stats=stats,
        report=report,
        model=model,
        reasoning_effort=reasoning_effort,
        started_at=started_at,
        finished_at=finished_at,
    )

    return stats


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated option, stripping whitespace around items."""
    return [part.strip() for part in value.split(",")]


def _role_list(value: str) -> tuple[str, ...]:
    roles = _split_csv(value)
    invalid = [role for role in roles if role not in ROLES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown role(s): {', '.join(invalid)}; valid roles: {', '.join(ROLES)}"
        )
    return tuple(roles)


def _prompt_style_list(value: str) -> tuple[str, ...]:
    styles = _split_csv(value)
    invalid = [style for style in styles if style not in PROMPT_STYLES]
    if invalid:
        raise argparse.ArgumentTypeError(
            "unknown prompt style(s): "
            f"{', '.join(invalid)}; valid styles: {', '.join(PROMPT_STYLES)}"
        )
    return tuple(styles)


def _validate_roles(roles: Sequence[str], agents: int) -> None:
    invalid = [role for role in roles if role not in ROLES]
    if invalid:
        raise ValueError(f"unknown role(s): {', '.join(invalid)}; valid roles: {', '.join(ROLES)}")
    if len(roles) != agents:
        raise ValueError(f"roles has {len(roles)} entries but agents is {agents}")


def _validate_prompt_styles(prompt_styles: Sequence[str], agents: int) -> None:
    invalid = [style for style in prompt_styles if style not in PROMPT_STYLES]
    if invalid:
        raise ValueError(
            "unknown prompt style(s): "
            f"{', '.join(invalid)}; valid styles: {', '.join(PROMPT_STYLES)}"
        )
    if len(prompt_styles) != agents:
        raise ValueError(f"prompt_styles has {len(prompt_styles)} entries but agents is {agents}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MRBalance replay pipeline over an ECI JSONL file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="input JSONL dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for response.jsonl (default: %(default)s)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run identifier recorded in the manifest (default: output_dir name)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help=(
            "also write meta.json, metrics.json, metrics.md and "
            "response.jsonl.gz into this directory"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=_MODES,
        default=DEFAULT_MODE,
        help="pipeline mode: multi-agent debate or single-agent baseline (default: %(default)s)",
    )
    parser.add_argument("--model", default=None, help="override the model from Settings")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="override MRBALANCE_REASONING_EFFORT from Settings",
    )
    parser.add_argument(
        "--agents",
        type=_positive_int,
        default=DEFAULT_AGENTS,
        help="number of debate agents; ignored by single-agent modes (default: %(default)s)",
    )
    parser.add_argument(
        "--rounds",
        type=_positive_int,
        default=DEFAULT_ROUNDS,
        help="number of debate rounds; ignored by single-agent modes (default: %(default)s)",
    )
    parser.add_argument(
        "--roles",
        type=_role_list,
        default=DEFAULT_AGENT_ROLES,
        help=(
            "comma-separated debate agent roles; entry count must match --agents "
            f"(default: {', '.join(DEFAULT_AGENT_ROLES)})"
        ),
    )
    parser.add_argument(
        "--prompt-styles",
        type=_prompt_style_list,
        default=DEFAULT_PROMPT_STYLES,
        help=(
            "comma-separated prompt styles; entry count must match --agents "
            f"(default: {', '.join(DEFAULT_PROMPT_STYLES)})"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
        help="number of samples processed in parallel (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=_non_negative_int,
        default=None,
        help="only process the first N samples",
    )
    parser.add_argument(
        "--label-mode",
        choices=_LABEL_MODES,
        default=DEFAULT_LABEL_MODE,
        help="prediction label source (default: %(default)s)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="compute and print metrics after the run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for debate shuffling (default: %(default)s)",
    )
    parser.add_argument(
        "--retry-failed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="re-run samples whose previous row has an error (default: %(default)s)",
    )
    parser.add_argument(
        "--shuffle-roles-each-round",
        action="store_true",
        help=(
            "resample agent roles from the role pool at the start of every round "
            "(paper setting; official code keeps roles fixed)"
        ),
    )
    parser.add_argument(
        "--selection",
        choices=_SELECTIONS,
        default=DEFAULT_SELECTION,
        help=(
            "round-3 top-2 selection: official listwise LLM ranker or the "
            "approximate paper-style importance scores (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--fix-top2-activation",
        action="store_true",
        help=(
            "compare selected agent ids instead of shuffled positions when "
            "activating the round-3 top-2 agents (official code has this defect; "
            "default keeps the official behaviour)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "debate":
        if len(args.roles) != args.agents:
            parser.error(f"--roles has {len(args.roles)} entries but --agents is {args.agents}")
        if len(args.prompt_styles) != args.agents:
            parser.error(
                f"--prompt-styles has {len(args.prompt_styles)} entries "
                f"but --agents is {args.agents}"
            )
    else:
        logger.warning(
            "mode=%s ignores agents/rounds/roles/prompt_styles/selection/shuffle_roles_each_round",
            args.mode,
        )
    config = RunConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        run_id=args.run_id,
        experiment_dir=args.experiment_dir,
        mode=args.mode,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        agents=args.agents,
        rounds=args.rounds,
        roles=args.roles,
        prompt_styles=args.prompt_styles,
        concurrency=args.concurrency,
        limit=args.limit,
        label_mode=args.label_mode,
        seed=args.seed,
        evaluate=args.evaluate,
        retry_failed=args.retry_failed,
        shuffle_roles_each_round=args.shuffle_roles_each_round,
        selection=args.selection,
        fix_top2_activation=args.fix_top2_activation,
    )
    run(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
