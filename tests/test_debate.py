"""Offline behaviour tests for the MRBalance debate engine."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from mrbalance_replay.agents import DebateGraph, Neuron, parse_ranks
from mrbalance_replay.prompts import PROMPT_STYLES, ROLES

QUESTION = (
    "Input: The heavy rain caused flooding in the city.\n"
    'Question: is there a causal relationship between "heavy rain" and '
    '"flooding in the city" ?\n'
    "(A) cause\n"
    "(B) caused by\n"
    "(C) none"
)


@dataclass
class FakeGeneration:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class FakeClient:
    """Deterministic offline client that distinguishes ranker from agent calls."""

    def __init__(
        self,
        answers: Sequence[str] | Callable[[int], str] = "ABC",
        ranker_response: str = "[1, 2]",
    ) -> None:
        self._answers = answers
        self.ranker_response = ranker_response
        self.calls: list[Sequence[dict[str, str]]] = []
        self.activation_calls: list[Sequence[dict[str, str]]] = []
        self.ranker_calls: list[Sequence[dict[str, str]]] = []
        self.activation_count = 0

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        model: str | None = None,
    ) -> FakeGeneration:
        del model
        self.calls.append(messages)
        if len(messages) == 1:
            self.ranker_calls.append(messages)
            return FakeGeneration(self.ranker_response, 1, 2)

        self.activation_calls.append(messages)
        if callable(self._answers):
            reply = self._answers(self.activation_count)
        else:
            reply = self._answers[self.activation_count % len(self._answers)]
        self.activation_count += 1
        return FakeGeneration(f"reasoning ({reply})", 3, 5)


def make_graph(**kwargs: Any) -> DebateGraph:
    return DebateGraph(agents=4, rounds=3, **kwargs)


def test_graph_shape_matches_reference() -> None:
    graph = make_graph()

    assert len(graph.nodes) == 12
    assert len(graph.edges) == 2 * 4 * 4
    assert [edge.id for edge in graph.edges] == list(range(32))
    assert len(graph.nodes[0].to_edges) == 4
    assert len(graph.nodes[0].from_edges) == 0
    assert len(graph.nodes[-1].from_edges) == 4
    assert len(graph.nodes[-1].to_edges) == 0


def test_early_consensus_stops_after_three_calls() -> None:
    graph = make_graph()
    client = FakeClient(answers="A")

    result = graph.forward(QUESTION, client)

    assert result.answer == "A"
    assert result.activations == 3
    assert result.ranker_calls == 0
    assert result.calls == 3
    assert len(client.calls) == 3
    assert result.prompt_tokens == 9
    assert result.completion_tokens == 15


def test_no_consensus_runs_full_four_four_two_plus_ranker() -> None:
    graph = make_graph()
    client = FakeClient(answers="ABC")

    result = graph.forward(QUESTION, client)

    assert result.activations == 10
    assert result.ranker_calls == 1
    assert result.calls == 11
    assert len(client.activation_calls) == 10
    assert len(client.ranker_calls) == 1
    assert len(client.calls) == 11
    assert result.prompt_tokens == 10 * 3 + 1
    assert result.completion_tokens == 10 * 5 + 2


def test_consensus_threshold_is_strictly_greater_than_two_thirds() -> None:
    graph = make_graph()
    mask = list(range(4))

    for idx, answer in enumerate(["A", "B", "A", "C"]):
        graph.nodes[idx].answer = answer
    reached, consensus_answer = graph.check_consensus([0, 1, 2, 3], mask)
    assert reached is False
    assert consensus_answer is None

    graph.nodes[3].answer = "A"
    reached, consensus_answer = graph.check_consensus([0, 1, 2, 3], mask)
    assert reached is True
    assert consensus_answer == "A"


@pytest.mark.parametrize(
    ("completion", "max_num", "expected"),
    [
        ("I prefer [1, 2] here.", 4, [0, 1]),
        ("[3, 1]", 4, [2, 0]),
        ("[1, 9] then [7, 5]", 4, [3, 3]),
        ("out of order [2, 4] and finally [3, 1]", 4, [2, 0]),
    ],
)
def test_parse_ranks_normal_and_out_of_order(
    completion: str, max_num: int, expected: list[int]
) -> None:
    assert parse_ranks(completion, max_num=max_num) == expected


def test_parse_ranks_falls_back_to_random_pair_on_failure() -> None:
    random.seed(0)

    tops = parse_ranks("no numbers here", max_num=4)

    assert len(tops) == 2
    assert len(set(tops)) == 2
    assert all(0 <= top < 4 for top in tops)


def test_parse_ranks_fallback_uses_injected_rng() -> None:
    first = parse_ranks("no numbers here", max_num=4, rng=random.Random(7))
    second = parse_ranks("no numbers here", max_num=4, rng=random.Random(7))

    assert first == second


def test_graph_rng_makes_shuffles_reproducible() -> None:
    random.seed(1)
    first_graph = DebateGraph(agents=4, rounds=3, rng=random.Random(5))
    first_result = first_graph.forward(QUESTION, FakeClient(answers="ABC"))

    random.seed(999)
    second_graph = DebateGraph(agents=4, rounds=3, rng=random.Random(5))
    second_result = second_graph.forward(QUESTION, FakeClient(answers="ABC"))

    assert first_result.completions == second_result.completions
    assert first_result.answer == second_result.answer


class _FixedClient:
    def __init__(self, content: str | None) -> None:
        self.content = content

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        model: str | None = None,
    ) -> Any:
        del messages, model
        return SimpleNamespace(content=self.content, prompt_tokens=0, completion_tokens=0)


@pytest.mark.parametrize("content", ["", None])
def test_neuron_replaces_empty_reply_with_random_letter(content: str | None) -> None:
    neuron = Neuron("none", rng=random.Random(0))

    neuron.activate(QUESTION, _FixedClient(content))

    # The reference replaces the empty reply with a bare letter; its answer
    # parser still requires parentheses, so ``answer`` legitimately stays None.
    assert neuron.reply in {"A", "B", "C"}


def test_prompt_style_only_first_round_by_default() -> None:
    graph = make_graph()
    client = FakeClient(answers="ABC")

    graph.forward(QUESTION, client)

    assert len(client.activation_calls) >= 7
    for messages in client.activation_calls[4:7]:
        content = messages[-1]["content"]
        assert content.startswith("\nHere is the question:\n")
        assert PROMPT_STYLES["positive"] not in content
        assert PROMPT_STYLES["normal"] not in content
        assert PROMPT_STYLES["normal1"] not in content
        assert PROMPT_STYLES["negative"] not in content


def test_prompt_style_all_rounds_includes_style_after_round_zero() -> None:
    graph = make_graph(prompt_style_all_rounds=True)
    client = FakeClient(answers="ABC")

    graph.forward(QUESTION, client)

    non_empty_styles = [
        PROMPT_STYLES[style] for style in ("positive", "normal", "normal1", "negative")
    ]
    assert len(client.activation_calls) >= 7
    for messages in client.activation_calls[4:7]:
        content = messages[-1]["content"]
        assert any(content.startswith(style) for style in non_empty_styles)


def test_backward_normalizes_matching_nodes_and_guards_zero() -> None:
    graph = make_graph()
    client = FakeClient(answers="A")
    result = graph.forward(QUESTION, client)

    importances = graph.backward(result.answer)

    active = [idx for idx in range(4) if graph.nodes[idx].active]
    assert len(active) == 3
    for idx in active:
        assert graph.nodes[idx].importance == pytest.approx(1 / 3)
    for idx in range(4, 12):
        assert importances[idx] == 0.0
    assert sum(importances) == pytest.approx(1.0)

    with pytest.warns(RuntimeWarning):
        guarded = graph.backward("B")
    assert all(importance == 0.0 for importance in guarded)


def test_backward_propagates_importance_to_earlier_rounds() -> None:
    graph = make_graph()
    client = FakeClient(answers="ABC")
    result = graph.forward(QUESTION, client)

    importances = graph.backward(result.answer)

    # Each fully active layer normalizes to 1; importance flows backwards.
    assert sum(importances[:4]) == pytest.approx(1.0)
    assert sum(importances[4:8]) == pytest.approx(1.0)
    assert sum(importances[8:12]) == pytest.approx(1.0)
    assert any(importance > 0 for importance in importances[:4])
    assert any(importance > 0 for importance in importances[4:8])


def test_fix_top2_activation_selects_different_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reverse_shuffle(items: list[Any]) -> None:
        items.reverse()

    monkeypatch.setattr(random, "shuffle", reverse_shuffle)

    faithful = make_graph(fix_top2_activation=False)
    faithful_client = FakeClient(answers="ABC")
    faithful.forward(QUESTION, faithful_client)

    fixed = make_graph(fix_top2_activation=True)
    fixed_client = FakeClient(answers="ABC")
    fixed.forward(QUESTION, fixed_client)

    faithful_active = {idx for idx in range(8, 12) if faithful.nodes[idx].active}
    fixed_active = {idx for idx in range(8, 12) if fixed.nodes[idx].active}

    # The reference compares the shuffled position (2, 3) with selected agent
    # ids, activating the last two nodes in the shuffled loop.  The fix
    # compares actual agent ids, activating agents 2 and 3 (nodes 10 and 11).
    assert faithful_active == {8, 9}
    assert fixed_active == {10, 11}
    assert faithful_active != fixed_active
    assert faithful_client.activation_count == fixed_client.activation_count == 10


def test_roles_stay_fixed_without_shuffle() -> None:
    roles = ("knowledge graphs expert", "causal analysis expert", "NLPexpert")
    graph = make_graph(
        agent_roles=roles,
        shuffle_roles_each_round=False,
        rng=random.Random(0),
    )
    client = FakeClient(answers="ABC")

    result = graph.forward(QUESTION, client)

    assert result.calls == 11
    for agent in range(4):
        assert len({graph.nodes[agent + 4 * rid].role for rid in range(3)}) == 1

    # The system message sent to the client is the prompt of the agent's role.
    for rid in range(3):
        active = [idx for idx in range(4 * rid, 4 * (rid + 1)) if graph.nodes[idx].active]
        start = rid * 4
        sent = client.activation_calls[start : start + len(active)]
        assert sorted(call[0]["content"] for call in sent) == sorted(
            ROLES[graph.nodes[idx].role] for idx in active
        )


def test_shuffle_roles_each_round_is_seeded_and_changes_roles() -> None:
    roles = ("knowledge graphs expert", "causal analysis expert", "NLPexpert")

    def role_grid(seed: int) -> list[list[str]]:
        graph = make_graph(
            agent_roles=roles,
            shuffle_roles_each_round=True,
            rng=random.Random(seed),
        )
        graph.forward(QUESTION, FakeClient(answers="ABC"))
        return [[graph.nodes[rid * 4 + agent].role for agent in range(4)] for rid in range(3)]

    first = role_grid(0)
    second = role_grid(0)

    per_agent = [{first[rid][agent] for rid in range(3)} for agent in range(4)]
    assert any(len(role_set) > 1 for role_set in per_agent)
    assert len({role for row in first for role in row}) > 1
    assert second == first


def test_shuffle_roles_each_round_reaches_the_client() -> None:
    roles = ("knowledge graphs expert", "causal analysis expert", "NLPexpert")
    graph = make_graph(
        agent_roles=roles,
        shuffle_roles_each_round=True,
        rng=random.Random(0),
    )
    client = FakeClient(answers="ABC")

    graph.forward(QUESTION, client)

    system_prompts = {call[0]["content"] for call in client.activation_calls}
    assert len(system_prompts) > 1
    assert system_prompts <= {ROLES[role] for role in roles}


def test_selection_listwise_calls_ranker_before_round_three() -> None:
    graph = make_graph(selection="listwise")
    client = FakeClient(answers="ABC")

    result = graph.forward(QUESTION, client)

    assert result.ranker_calls == 1
    assert len(client.ranker_calls) == 1
    assert result.calls == 11


def test_selection_importance_skips_ranker_and_picks_previous_majority() -> None:
    graph = make_graph(selection="importance")
    client = FakeClient(answers="ABC")

    result = graph.forward(QUESTION, client)

    assert result.ranker_calls == 0
    assert result.activations == 10
    assert result.calls == 10
    assert client.ranker_calls == []

    previous = [idx for idx in range(4, 8) if graph.nodes[idx].active]
    assert len(previous) == 4
    answers = [graph.nodes[idx].get_answer() for idx in previous]
    majority = Counter(answers).most_common(1)[0][0]
    expected = {idx % 4 for idx in previous if graph.nodes[idx].get_answer() == majority}
    selected = {idx % 4 for idx in range(8, 12) if graph.nodes[idx].active}

    assert len(expected) == 2
    assert selected == expected


def test_invalid_selection_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown selection"):
        DebateGraph(agents=4, rounds=3, selection="bogus")
