"""Verbatim prompt assets for the MRBalance debate engine."""

from .roles import (
    DEFAULT_AGENT_ROLES,
    ROLES,
    SYSTEM_PROMPT,
    roles,
    system_prompt,
)
from .templates import (
    DEFAULT_PROMPT_STYLES,
    PROMPT_STYLES,
    construct_message,
    construct_ranking_message,
    prompt_roles_intra,
)

__all__ = [
    "DEFAULT_AGENT_ROLES",
    "DEFAULT_PROMPT_STYLES",
    "PROMPT_STYLES",
    "ROLES",
    "SYSTEM_PROMPT",
    "construct_message",
    "construct_ranking_message",
    "prompt_roles_intra",
    "roles",
    "system_prompt",
]
