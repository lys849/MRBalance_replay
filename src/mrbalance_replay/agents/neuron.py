"""LLM neuron and edge primitives ported from the reference MRBalance code.

The reference file is ``reference/MRBalance/CausalDebate/LLM_Neuron.py``.  The
activation flow, weight parsing, and listwise ranking are kept faithful.  The
only structural change is dependency injection: an :class:`LLMClientLike` is
passed in instead of calling a module-level OpenAI helper.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from ..prompts import ROLES, construct_message, construct_ranking_message
from .parsing import parse_ranks, parse_single_choice


class GenerationResultLike(Protocol):
    """Structural stand-in for ``mrbalance_replay.llm.GenerationResult``."""

    content: str
    prompt_tokens: int
    completion_tokens: int


class LLMClientLike(Protocol):
    """Structural stand-in for ``mrbalance_replay.llm.LLMClient``."""

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        model: str | None = None,
    ) -> GenerationResultLike: ...


def _find_array(text: str) -> list[int]:
    """Find the last ``[[...]]`` weights array, as the reference does."""
    matches = re.findall(r"\[\[(.*?)\]\]", text)
    if not matches:
        return []

    last_match = matches[-1].replace(" ", "")

    def convert(x: str) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    try:
        return list(map(convert, last_match.split(",")))
    except Exception:
        return []


class Neuron:
    """A single debate agent, equivalent to the reference ``LLMNeuron``."""

    def __init__(
        self,
        role: str,
        model: str = "gpt-3.5-turbo",
        prompt_style: str = "positive",
        qtype: str = "single_choice",
        ans_parser: Callable[[str], str | None] = parse_single_choice,
        rng: random.Random | None = None,
    ) -> None:
        self.role = role
        self.tuning = prompt_style
        self.model = model
        self.qtype = qtype
        self.ans_parser = ans_parser
        self.reply: str | None = None
        self.answer: str | None = ""
        self.active = False
        self.importance = 0.0
        self.question: str | None = None
        self.to_edges: list[Edge] = []
        self.from_edges: list[Edge] = []
        self.prompt_style_all_rounds = False
        self.weights_parser = _find_array
        self.rng: Any = rng or random
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def edges(self) -> list[Edge]:
        """Incoming edges, matching the reference ``from_edges`` collection."""
        return self.from_edges

    @property
    def prompt_tuning(self) -> str:
        """Alias kept for parity with the reference attribute name."""
        return self.tuning

    def get_reply(self) -> str | None:
        return self.reply

    def get_answer(self) -> str | None:
        return self.answer

    def deactivate(self) -> None:
        self.active = False
        self.reply = None
        self.answer = ""
        self.question = None
        self.importance = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def get_context(self) -> tuple[list[dict[str, str]], list[tuple[str, int]]]:
        sys_prompt = ROLES[self.role]
        contexts = [{"role": "system", "content": sys_prompt}]

        formers: list[tuple[str, int]] = []
        for eid, edge in enumerate(self.from_edges):
            if edge.a1.reply is not None and edge.a1.active:
                formers.append((edge.a1.reply, eid))
        return contexts, formers

    def activate(self, question: str, client: LLMClientLike) -> None:
        """Activate this neuron once, exactly following the reference flow."""
        self.question = question
        self.active = True

        contexts, formers = self.get_context()
        original_idxs = [mess[1] for mess in formers]
        self.rng.shuffle(formers)
        shuffled_idxs = [mess[1] for mess in formers]
        replies = [mess[0] for mess in formers]

        contexts.append(
            construct_message(
                replies,
                question,
                self.tuning,
                force_style=self.prompt_style_all_rounds,
            )
        )

        result = client.generate(contexts, model=self.model)
        self.reply = result.content
        self.prompt_tokens = result.prompt_tokens
        self.completion_tokens = result.completion_tokens

        if self.reply is None or self.reply == "":
            self.reply = self.rng.choice(["A", "B", "C"])

        self.answer = self.ans_parser(self.reply)

        weights = self.weights_parser(self.reply)
        if len(weights) != len(replies):
            print("miss match!")
            weights = [0 for _ in range(len(replies))]

        shuffled_pairs = list(zip(shuffled_idxs, weights, replies, strict=False))
        sorted_pairs = sorted(shuffled_pairs, key=lambda x: original_idxs.index(x[0]))
        weights = [weight for _, weight, _ in sorted_pairs]
        sorted_eids = [eid for eid, _, _ in sorted_pairs]

        for lp, eid in enumerate(sorted_eids):
            self.from_edges[eid].weight = (
                weights[lp] / 5 if 0 < weights[lp] <= 5 else (1 if weights[lp] > 5 else 0)
            )

        total = sum(self.from_edges[eid].weight for eid in sorted_eids)
        if total > 0:
            for eid in sorted_eids:
                self.from_edges[eid].weight /= total
        else:
            for eid in sorted_eids:
                self.from_edges[eid].weight = 1 / len(sorted_eids)

    def get_conversation(self) -> list[dict[str, str]]:
        if not self.active:
            return []

        contexts, formers = self.get_context()
        contexts.append(
            construct_message(
                [mess[0] for mess in formers],
                self.question or "",
                self.prompt_tuning,
                force_style=self.prompt_style_all_rounds,
            )
        )
        contexts.append({"role": "assistant", "content": self.reply or ""})
        return contexts


class Edge:
    """A directed weighted edge between two neurons."""

    def __init__(self, a1: Neuron, a2: Neuron, edge_id: int = -1) -> None:
        self.weight = 0.0
        self.a1 = a1
        self.a2 = a2
        self.id = edge_id
        self.a1.to_edges.append(self)
        self.a2.from_edges.append(self)

    def zero_weight(self) -> None:
        self.weight = 0.0


def listwise_ranker_2(
    responses: Sequence[str],
    question: str,
    client: LLMClientLike,
    qtype: str = "single_choice",
    model: str | None = None,
    *,
    rng: random.Random | None = None,
) -> tuple[list[int], int, int]:
    """Ask the LLM to pick the best two responses (reference behaviour)."""
    del qtype  # retained for signature parity with the reference

    assert len(responses) > 2
    message = construct_ranking_message(responses, question)
    result = client.generate([message], model=model)
    return (
        parse_ranks(result.content, max_num=len(responses), rng=rng),
        result.prompt_tokens,
        result.completion_tokens,
    )
