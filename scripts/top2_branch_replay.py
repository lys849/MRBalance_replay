"""Paired branch replay for the round-3 top-2 activation defect.

Take the ESC samples that actually reach round 3 in the archived ``official-esc``
run, replay their first two rounds once, then branch the third round into the
official (position-vs-agent-id) and fixed (agent-id) activation rules from the
same graph state.  Only the round-3 participants differ between the branches.

Usage::

    uv run python scripts/top2_branch_replay.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from mrbalance_replay.agents import DebateGraph
from mrbalance_replay.config import load_settings
from mrbalance_replay.data import build_question, load_jsonl
from mrbalance_replay.llm import GenerationResult, OpenAIClient
from mrbalance_replay.prompts import DEFAULT_AGENT_ROLES, DEFAULT_PROMPT_STYLES
from mrbalance_replay.runner import _record_key

AGENTS = 4
ROUNDS = 3
SEED = 0
# Rounds 0 and 1 activate every agent, then the listwise ranker runs once.
PRE_CALLS = 2 * AGENTS + 1
ARCHIVE_CELLS = 10  # n_calls for a sample that reached round 3

MessageKey = tuple[tuple[str, str], ...]


def message_key(messages: Sequence[dict[str, str]]) -> MessageKey:
    return tuple((message["role"], message["content"]) for message in messages)


def label_of(answer: str | None) -> int | None:
    if answer in ("A", "B"):
        return 1
    if answer == "C":
        return 0
    return None


def active_agents(graph: DebateGraph) -> tuple[int, ...]:
    start = AGENTS * (ROUNDS - 1)
    return tuple(idx % AGENTS for idx in range(start, start + AGENTS) if graph.nodes[idx].active)


def make_graph(
    fix: bool,
    sample_seed: int,
    model: str,
    roles: Sequence[str],
    prompt_styles: Sequence[str],
    *,
    force_rounds: bool = False,
) -> DebateGraph:
    graph = DebateGraph(
        agents=AGENTS,
        rounds=ROUNDS,
        agent_roles=roles,
        prompt_styles=prompt_styles,
        model=model,
        shuffle_roles_each_round=False,
        selection="listwise",
        fix_top2_activation=fix,
        rng=random.Random(sample_seed),
    )
    if force_rounds:
        # Disable early consensus so every sample reaches the round-3 branch.
        graph.check_consensus = lambda idxs, idx_mask: (False, None)  # type: ignore[method-assign]
    return graph


class Recorder:
    """Pass calls through and record them in order."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[tuple[MessageKey, GenerationResult]] = []

    def generate(
        self, messages: Sequence[dict[str, str]], model: str | None = None
    ) -> GenerationResult:
        result = cast(GenerationResult, self.inner.generate(messages, model=model))
        self.calls.append((message_key(messages), result))
        return result


class ReplayThenLive:
    """Replay the first ``len(pre)`` results, then reuse the memo or call live."""

    def __init__(
        self,
        pre: Sequence[GenerationResult],
        memo: dict[MessageKey, GenerationResult],
        inner: Any,
    ) -> None:
        self.pre = list(pre)
        self.memo = memo
        self.inner = inner
        self.n = 0

    def generate(
        self, messages: Sequence[dict[str, str]], model: str | None = None
    ) -> GenerationResult:
        self.n += 1
        if self.n <= len(self.pre):
            return self.pre[self.n - 1]
        key = message_key(messages)
        if key in self.memo:
            return self.memo[key]
        result = cast(GenerationResult, self.inner.generate(messages, model=model))
        self.memo[key] = result
        return result


@dataclass
class SampleOutcome:
    key: str
    item_id: Any
    gold: int
    reached_round3: bool
    freeze_answer: str | None
    freeze_calls: int
    buggy_answer: str | None
    fixed_answer: str | None
    buggy_agents: tuple[int, ...]
    fixed_agents: tuple[int, ...]
    selection_differs: bool
    replay_consistent: bool


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reference/MRBalance/data/test/ESC_intra_data.json"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("experiments/official-esc/response.jsonl.gz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/top2-pilot/branch19.json"),
    )
    parser.add_argument(
        "--roles",
        default=",".join(DEFAULT_AGENT_ROLES),
        help="comma-separated agent roles (default: official 4x knowledge graphs expert)",
    )
    parser.add_argument(
        "--prompt-styles",
        default=",".join(DEFAULT_PROMPT_STYLES),
        help="comma-separated first-round prompt styles",
    )
    parser.add_argument(
        "--force-rounds",
        action="store_true",
        help="disable early consensus so the round-3 branch is always reached",
    )
    parser.add_argument(
        "--freeze-only",
        action="store_true",
        help="only replay rounds 0-2 once, to measure round-3 membership stability",
    )
    args = parser.parse_args(argv)

    keys: set[str] = set()
    with gzip.open(args.archive, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record["n_calls"] >= ARCHIVE_CELLS:
                    keys.add(record["_record_key"])

    items = [item for item in load_jsonl(args.input) if _record_key(item) in keys]
    print(f"archived round-3 samples: {len(keys)}, found in input: {len(items)}")

    settings = load_settings()
    model = settings.model
    roles = tuple(part.strip() for part in args.roles.split(","))
    prompt_styles = tuple(part.strip() for part in args.prompt_styles.split(","))
    live = OpenAIClient(settings)
    outcomes: list[SampleOutcome] = []

    for position, item in enumerate(items, 1):
        key = _record_key(item)
        sample_seed = SEED ^ int(key[:8], 16)
        question = build_question(item)
        gold = int(item["bi_causal_label"])

        recorder = Recorder(live)
        frozen = make_graph(
            False, sample_seed, model, roles, prompt_styles, force_rounds=args.force_rounds
        )
        frozen_result = frozen.forward(question, cast(Any, recorder))
        reached = frozen_result.calls >= ARCHIVE_CELLS and len(recorder.calls) >= PRE_CALLS
        if not reached:
            print(
                f"[{position}/{len(items)}] {key[:8]} skipped: "
                f"stopped at {frozen_result.calls} calls"
            )
            outcomes.append(
                SampleOutcome(
                    key=key,
                    item_id=item.get("item_id"),
                    gold=gold,
                    reached_round3=False,
                    freeze_answer=frozen_result.answer,
                    freeze_calls=frozen_result.calls,
                    buggy_answer=None,
                    fixed_answer=None,
                    buggy_agents=(),
                    fixed_agents=(),
                    selection_differs=False,
                    replay_consistent=False,
                )
            )
            continue

        if args.freeze_only:
            outcomes.append(
                SampleOutcome(
                    key=key,
                    item_id=item.get("item_id"),
                    gold=gold,
                    reached_round3=True,
                    freeze_answer=frozen_result.answer,
                    freeze_calls=frozen_result.calls,
                    buggy_answer=None,
                    fixed_answer=None,
                    buggy_agents=(),
                    fixed_agents=(),
                    selection_differs=False,
                    replay_consistent=False,
                )
            )
            print(
                f"[{position}/{len(items)}] {key[:8]} reached round 3 ({frozen_result.calls} calls)"
            )
            continue

        pre = [result for _key, result in recorder.calls[:PRE_CALLS]]
        memo: dict[MessageKey, GenerationResult] = dict(recorder.calls[PRE_CALLS:])

        buggy_graph = make_graph(
            False, sample_seed, model, roles, prompt_styles, force_rounds=args.force_rounds
        )
        buggy_result = buggy_graph.forward(question, cast(Any, ReplayThenLive(pre, memo, live)))

        fixed_graph = make_graph(
            True, sample_seed, model, roles, prompt_styles, force_rounds=args.force_rounds
        )
        fixed_result = fixed_graph.forward(question, cast(Any, ReplayThenLive(pre, memo, live)))

        buggy_agents = active_agents(buggy_graph)
        fixed_agents = active_agents(fixed_graph)
        consistent = buggy_result.answer == frozen_result.answer
        outcome = SampleOutcome(
            key=key,
            item_id=item.get("item_id"),
            gold=gold,
            reached_round3=True,
            freeze_answer=frozen_result.answer,
            freeze_calls=frozen_result.calls,
            buggy_answer=buggy_result.answer,
            fixed_answer=fixed_result.answer,
            buggy_agents=buggy_agents,
            fixed_agents=fixed_agents,
            selection_differs=buggy_agents != fixed_agents,
            replay_consistent=consistent,
        )
        outcomes.append(outcome)
        print(
            f"[{position}/{len(items)}] {key[:8]} gold={gold} "
            f"agents {buggy_agents}->{fixed_agents} "
            f"answer {buggy_result.answer}->{fixed_result.answer} "
            f"consistent={consistent}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(outcome) for outcome in outcomes], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    tested = [outcome for outcome in outcomes if outcome.reached_round3]
    flips = [
        outcome
        for outcome in tested
        if label_of(outcome.buggy_answer) != label_of(outcome.fixed_answer)
    ]
    print("\n=== summary ===")
    print(f"reached round 3 on replay: {len(tested)}/{len(outcomes)}")
    print(f"selection differs: {sum(o.selection_differs for o in tested)}/{len(tested)}")
    consistent_count = sum(o.replay_consistent for o in tested)
    print(f"replay consistent with frozen run: {consistent_count}/{len(tested)}")
    print(f"label flips: {len(flips)}/{len(tested)}")
    for outcome in flips:
        print(
            f"  {outcome.key[:8]} gold={outcome.gold} "
            f"{outcome.buggy_agents}->{outcome.fixed_agents} "
            f"{outcome.buggy_answer}->{outcome.fixed_answer}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
