# ruff: noqa: E501
"""Persona definitions ported verbatim from the reference MRBalance implementation.

The reference file is ``reference/MRBalance/CausalDebate/prompt_roles.py``.  Do not
reword, trim, or otherwise "improve" these strings: they are part of the paper's
reproduction fidelity.
"""

ROLES: dict[str, str] = {
    "knowledge graphs expert": "You are an expert in knowledge graphs and are responsible for building a graph of cause and effect from events. You'll explain how these events relate to each other in the knowledge graph and reason whether they are causal or not, with an emphasis on the correlations and dependencies between events.",
    "causal analysis expert": "You are a causal analysis expert, specializing in identifying and reasoning about causal relationships between events. You need to accurately find causally related event pairs from the provided text and explain their logical connections.",
    "NLPexpert": "You are an expert in natural language processing, focusing on extracting causal cues from language. You will analyze whether there is a causal connection between two events based on the explanation of causal triggers, temporal relationships, and character actions, and make judgments based on the logical structure in the language.",
    "none": "",
}

SYSTEM_PROMPT = """Can you answer the following question as accurately as possible?"""

DEFAULT_AGENT_ROLES: tuple[str, ...] = (
    "knowledge graphs expert",
    "knowledge graphs expert",
    "knowledge graphs expert",
    "knowledge graphs expert",
)

# Lower-case aliases matching the names used by the reference implementation.
roles = ROLES
system_prompt = SYSTEM_PROMPT
