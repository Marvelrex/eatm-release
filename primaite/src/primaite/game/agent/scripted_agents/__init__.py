# © Crown-owned copyright 2025, Defence Science and Technology Laboratory UK
"""Agents that automatically choose their behaviour according to scripted rules."""
from primaite.game.agent import interface
from primaite.game.agent.scripted_agents import (
    abstract_tap,
    data_manipulation_bot,
    llm_agent,
    probabilistic_agent,
    random_agent,
    TAP001,
    TAP003,
)
from primaite.game.agent.scripted_agents.cage4 import cage4_blue, cage4_red

__all__ = (
    "abstract_tap",
    "cage4_blue",
    "cage4_red",
    "data_manipulation_bot",
    "interface",
    "llm_agent",
    "probabilistic_agent",
    "random_agent",
    "TAP001",
    "TAP003",
)
