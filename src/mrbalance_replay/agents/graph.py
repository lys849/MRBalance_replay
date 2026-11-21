"""Multi-agent debate graph, equivalent to the reference ``LLMLP``.

The reference file is ``reference/MRBalance/CausalDebate/LLMLP.py``.  The
three-round forward pass and the importance backward pass are reproduced
faithfully.  Two switches deliberately expose known reference defects so that
callers can choose between paper-faithful replay and corrected behaviour.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..prompts import DEFAULT_AGENT_ROLES, DEFAULT_PROMPT_STYLES
from .neuron import Edge, LLMClientLike, Neuron, listwise_ranker_2
from .parsing import most_frequent, parse_single_choice

ACTIVATION_MAP = {"listwise": 0, "trueskill": 1, "window": 2, "none": -1}
SELECTIONS = ("listwise", "importance")


@dataclass
class DebateResult:
    """Outcome of one forward pass plus usage counters."""

    answer: str | None
    activations: int
    ranker_calls: int
    prompt_tokens: int
    completion_tokens: int
    completions: list[list[str | None]] = field(default_factory=list)

    @property
    def calls(self) -> int:
        """Total LLM calls (agent activations plus listwise ranker calls)."""
        return self.activations + self.ranker_calls

    @property
    def resp_cnt(self) -> int:
        """Alias matching the reference return counter."""
        return self.calls


def _cycle(values: Sequence[str], n: int) -> list[str]:
    if not values:
        raise ValueError("values must not be empty")
    return [values[i % len(values)] for i in range(n)]


class DebateGraph:
    """A layered debate graph with ``agents * rounds`` LLM neurons.

    Two experiment switches control the paper-vs-official-code differences:

    ``shuffle_roles_each_round``
        Default ``False`` keeps the official code behaviour: every agent keeps
        the same role in all rounds.  When ``True``, the roles of all agents are
        resampled from the deduplicated ``agent_roles`` pool at the start of
        each round (including ``"none"`` if present), using ``self.rng``.  This
        is the paper's Section 5.3 role assignment and is NOT what the official
        repository implements by default.

    ``selection``
        Default ``"listwise"`` calls the official listwise LLM ranker before
        round 3.  ``"importance"`` is an *approximate* implementation of the
        paper's team-member optimization: it does not call the ranker and picks
        the two agent ids with the highest importance scores from the previous
        round.  The scores use the same edge-weight backward formula as
        :meth:`backward`, with the previous round's majority answer as the base
        result, over the subgraph up to that round.  The paper does not publish
        its exact importance/scoring details, so this is a documented
        approximation rather than an official-code behaviour.  Ties are broken
        by ascending agent id.
    """

    def __init__(
        self,
        agents: int = 4,
        rounds: int = 3,
        activation: str = "listwise",
        qtype: str = "single_choice",
        *,
        agent_roles: Sequence[str] | None = None,
        prompt_styles: Sequence[str] | None = None,
        model: str = "gpt-3.5-turbo",
        models: Sequence[str] | None = None,
        prompt_style_all_rounds: bool = False,
        fix_top2_activation: bool = False,
        shuffle_roles_each_round: bool = False,
        selection: str = "listwise",
        rng: random.Random | None = None,
    ) -> None:
        if agents <= 0:
            raise ValueError("agents must be positive")
        if activation not in ACTIVATION_MAP:
            raise ValueError(f"unknown activation: {activation}")
        if ACTIVATION_MAP[activation] != 0:
            raise NotImplementedError("only listwise activation is implemented")
        if qtype != "single_choice":
            raise NotImplementedError("only single_choice qtype is implemented")
        if selection not in SELECTIONS:
            raise ValueError(f"unknown selection: {selection!r}")

        self.agents = agents
        self.rounds = rounds
        self.activation_mode = activation
        self.activation_cost = 1
        self.qtype = qtype
        self.model = model
        role_pool_source = agent_roles if agent_roles is not None else DEFAULT_AGENT_ROLES
        # Preserve order while removing duplicates; ``"none"`` is a valid role.
        self.role_pool = list(dict.fromkeys(role_pool_source))
        self.agent_roles = _cycle(role_pool_source, agents)
        self.prompt_styles = _cycle(
            prompt_styles if prompt_styles is not None else DEFAULT_PROMPT_STYLES, agents
        )
        self.models = _cycle(models if models is not None else [model], agents)
        self.prompt_style_all_rounds = prompt_style_all_rounds
        self.fix_top2_activation = fix_top2_activation
        self.shuffle_roles_each_round = shuffle_roles_each_round
        self.selection = selection
        # ``random`` (the module) is the reference default; tests and concurrent
        # callers can inject a private ``random.Random`` instance instead.
        self.rng: Any = rng or random
        self.cmp_res: Callable[[str | None, str | None], bool] = lambda x, y: x == y
        self.ans_parser: Callable[[str], str | None] = parse_single_choice

        self.nodes: list[Neuron] = []
        self.edges: list[Edge] = []
        self._init_nn()

    def _init_nn(self) -> None:
        for idx in range(self.agents):
            self.nodes.append(self._new_neuron(idx))

        agents_last_round = self.nodes[: self.agents]
        for _rid in range(1, self.rounds):
            for idx in range(self.agents):
                node = self._new_neuron(idx)
                self.nodes.append(node)
                for a1 in agents_last_round:
                    self.edges.append(Edge(a1, node, edge_id=len(self.edges)))
            agents_last_round = self.nodes[-self.agents :]

    def _new_neuron(self, idx: int) -> Neuron:
        node = Neuron(
            self.agent_roles[idx],
            self.models[idx],
            prompt_style=self.prompt_styles[idx],
            qtype=self.qtype,
            ans_parser=self.ans_parser,
            rng=self.rng,
        )
        node.prompt_style_all_rounds = self.prompt_style_all_rounds
        return node

    def _assign_round_roles(self, rid: int) -> None:
        """Resample every agent's role before activating round ``rid``."""
        if not self.shuffle_roles_each_round:
            return
        start = self.agents * rid
        for offset in range(self.agents):
            self.nodes[start + offset].role = self.rng.choice(self.role_pool)

    def zero_grad(self) -> None:
        for edge in self.edges:
            edge.zero_weight()

    def check_consensus(
        self,
        idxs: Sequence[int],
        idx_mask: Sequence[int],
    ) -> tuple[bool, str | None]:
        """Strict reference threshold: count must exceed ``floor(2/3 * mask)``."""
        candidates = [self.nodes[idx].get_answer() for idx in idxs]
        consensus_answer, ca_cnt = most_frequent(candidates, self.cmp_res)
        if ca_cnt > math.floor(2 / 3 * len(idx_mask)):
            print(f"Consensus answer: {consensus_answer}")
            return True, consensus_answer
        return False, None

    def set_allnodes_deactivated(self) -> None:
        for node in self.nodes:
            node.deactivate()

    def _get_completions(self) -> list[list[str | None]]:
        completions: list[list[str | None]] = [[] for _ in range(self.agents)]
        for rid in range(self.rounds):
            for idx in range(self.agents * rid, self.agents * (rid + 1)):
                if self.nodes[idx].active:
                    completions[idx % self.agents].append(self.nodes[idx].get_reply())
                else:
                    completions[idx % self.agents].append(None)
        return completions

    def _result(
        self,
        answer: str | None,
        activations: int,
        ranker_calls: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> DebateResult:
        return DebateResult(
            answer=answer,
            activations=activations,
            ranker_calls=ranker_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            completions=self._get_completions(),
        )

    def _activate(
        self,
        node_idx: int,
        question: str,
        client: LLMClientLike,
        activations: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> tuple[int, int, int]:
        node = self.nodes[node_idx]
        node.activate(question, client)
        activations += 1
        prompt_tokens += node.prompt_tokens
        completion_tokens += node.completion_tokens
        return activations, prompt_tokens, completion_tokens

    def forward(self, question: str, client: LLMClientLike) -> DebateResult:
        assert self.rounds > 2

        activations = 0
        ranker_calls = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        self.set_allnodes_deactivated()

        # Round 0: every agent answers independently.
        self._assign_round_roles(0)
        loop_indices = list(range(self.agents))
        self.rng.shuffle(loop_indices)
        activated_indices: list[int] = []
        for idx, node_idx in enumerate(loop_indices):
            activations, total_prompt_tokens, total_completion_tokens = self._activate(
                node_idx,
                question,
                client,
                activations,
                total_prompt_tokens,
                total_completion_tokens,
            )
            activated_indices.append(node_idx)

            if idx >= math.floor(2 / 3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return self._result(
                        reply,
                        activations,
                        ranker_calls,
                        total_prompt_tokens,
                        total_completion_tokens,
                    )

        # Round 1: every agent revises after seeing round-0 replies.
        self._assign_round_roles(1)
        loop_indices = list(range(self.agents, self.agents * 2))
        self.rng.shuffle(loop_indices)
        activated_indices = []
        for idx, node_idx in enumerate(loop_indices):
            activations, total_prompt_tokens, total_completion_tokens = self._activate(
                node_idx,
                question,
                client,
                activations,
                total_prompt_tokens,
                total_completion_tokens,
            )
            activated_indices.append(node_idx)

            if idx >= math.floor(2 / 3 * self.agents):
                reached, reply = self.check_consensus(activated_indices, list(range(self.agents)))
                if reached:
                    return self._result(
                        reply,
                        activations,
                        ranker_calls,
                        total_prompt_tokens,
                        total_completion_tokens,
                    )

        idx_mask = list(range(self.agents))
        idxs = list(range(self.agents, self.agents * 2))
        for rid in range(2, self.rounds):
            self._assign_round_roles(rid)
            if self.agents >= 2:
                if self.selection == "listwise":
                    replies = [self.nodes[idx].get_reply() for idx in idxs]
                    indices = list(range(len(replies)))
                    self.rng.shuffle(indices)
                    shuffled_replies = [replies[idx] for idx in indices]

                    tops, prompt_tokens, completion_tokens = listwise_ranker_2(
                        [reply if reply is not None else "" for reply in shuffled_replies],
                        question,
                        client,
                        self.qtype,
                        self.models[0],
                        rng=self.rng,
                    )
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    idx_mask = [idxs[indices[x]] % self.agents for x in tops]
                    ranker_calls += self.activation_cost
                else:
                    idx_mask = self._select_importance_agent_ids(rid - 1, idxs)

            loop_indices = list(range(self.agents * rid, self.agents * (rid + 1)))
            self.rng.shuffle(loop_indices)
            idxs = []
            for idx, node_idx in enumerate(loop_indices):
                # Reference defect: compares the shuffled *position* to the
                # selected agent id.  ``fix_top2_activation`` compares the
                # node's actual agent id instead.  Importance-based selection
                # always matches actual agent ids so the paper-style top-2
                # choice is honoured regardless of the defect switch.
                if self.selection == "importance" or self.fix_top2_activation:
                    selected = node_idx % self.agents in idx_mask
                else:
                    selected = idx in idx_mask
                if not selected:
                    continue

                print(rid, idx)
                activations, total_prompt_tokens, total_completion_tokens = self._activate(
                    node_idx,
                    question,
                    client,
                    activations,
                    total_prompt_tokens,
                    total_completion_tokens,
                )
                idxs.append(node_idx)
                if len(idxs) > math.floor(2 / 3 * len(idx_mask)):
                    reached, reply = self.check_consensus(idxs, idx_mask)
                    if reached:
                        return self._result(
                            reply,
                            activations,
                            ranker_calls,
                            total_prompt_tokens,
                            total_completion_tokens,
                        )

        final_answer = most_frequent([self.nodes[idx].get_answer() for idx in idxs], self.cmp_res)[
            0
        ]
        return self._result(
            final_answer,
            activations,
            ranker_calls,
            total_prompt_tokens,
            total_completion_tokens,
        )

    def _compute_importances(self, end_round: int, result: str | None) -> list[float]:
        """Compute reference importance scores without mutating node state.

        The formula matches :meth:`backward`: nodes in the last active round
        that match ``result`` share ``1 / n``, and earlier rounds receive
        ``sum(edge.weight * downstream.importance)`` over outgoing edges.
        """
        importances = [0.0 for _ in self.nodes]
        node_index = {node: idx for idx, node in enumerate(self.nodes)}
        flag_last = False
        for rid in range(end_round, -1, -1):
            round_indices = range(self.agents * rid, self.agents * (rid + 1))
            if not flag_last:
                active_indices = [idx for idx in round_indices if self.nodes[idx].active]
                if not active_indices:
                    continue
                flag_last = True

                matching = [
                    idx
                    for idx in active_indices
                    if self.cmp_res(self.nodes[idx].get_answer(), result)
                ]
                if matching:
                    ave_w = 1 / len(matching)
                else:
                    warnings.warn(
                        "backward found no active node matching the final result; "
                        "setting all importances to 0",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    ave_w = 0.0

                for idx in round_indices:
                    if self.nodes[idx].active and self.cmp_res(
                        self.nodes[idx].get_answer(), result
                    ):
                        importances[idx] = ave_w
                    else:
                        importances[idx] = 0.0
            else:
                for idx in round_indices:
                    importances[idx] = 0.0
                    if self.nodes[idx].active:
                        for edge in self.nodes[idx].to_edges:
                            importances[idx] += edge.weight * importances[node_index[edge.a2]]

        return importances

    def _select_importance_agent_ids(
        self, previous_rid: int, previous_indices: Sequence[int]
    ) -> list[int]:
        """Approximate the paper's top-2 selection without calling the ranker.

        The previous round's majority answer seeds the same backward formula as
        :meth:`backward`.  Because the previous round is the last layer of that
        subgraph, all agents agreeing with the majority share the same base
        score; ties are broken by ascending agent id.
        """
        active = [idx for idx in previous_indices if self.nodes[idx].active]
        if not active:
            return []

        majority, _count = most_frequent(
            [self.nodes[idx].get_answer() for idx in active], self.cmp_res
        )
        scores = self._compute_importances(previous_rid, majority)
        ranked = sorted(active, key=lambda idx: (-scores[idx], idx % self.agents))
        selected: list[int] = []
        for idx in ranked:
            agent_id = idx % self.agents
            if agent_id not in selected:
                selected.append(agent_id)
            if len(selected) == 2:
                break
        return selected

    def backward(self, result: str | None) -> list[float]:
        """Propagate importance from the final answer back through the graph.

        The reference divides by the number of matching nodes without a guard.
        When no active node in the last active round matches ``result`` we set
        all importances to zero and emit a warning instead of crashing.
        """
        importances = self._compute_importances(self.rounds - 1, result)
        for node, importance in zip(self.nodes, importances, strict=True):
            node.importance = importance
        return importances
