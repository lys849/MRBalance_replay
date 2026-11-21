"""Rank parsing for the debate engine.

``parse_single_choice`` and ``most_frequent`` live in the package-level
:mod:`mrbalance_replay.parsing` module; they are re-exported here so debate
internals keep a single import path.
"""

import random
import re
from typing import Any

from ..parsing import most_frequent, parse_single_choice

__all__ = ["most_frequent", "parse_ranks", "parse_single_choice"]


def parse_ranks(
    completion: str,
    max_num: int = 4,
    rng: random.Random | None = None,
) -> list[int]:
    """Parse the last ``[i, j]`` listwise ranking pair (reference behaviour).

    The reference regex only accepts single digits in ``1..7`` and converts the
    result from 1-based to 0-based before clipping.  A parse failure falls back
    to a random sample of two distinct indices, which is intentional.
    """
    content = completion
    pattern = r"\[([1234567]),\s*([1234567])\]"
    matches = re.findall(pattern, content)

    try:
        match = matches[-1]
        tops = [int(match[0]) - 1, int(match[1]) - 1]

        def clip(x: int) -> int:
            if x < 0:
                return 0
            if x > max_num - 1:
                return max_num - 1
            return x

        tops = [clip(x) for x in tops]
    except Exception:
        print("error in parsing ranks")
        rng_obj: Any = rng if rng is not None else random
        tops = rng_obj.sample(list(range(max_num)), 2)

    return tops
