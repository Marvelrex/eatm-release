from __future__ import annotations

from CybORG.Agents.LLMAgents.config.config_vars import AGENT_MODE, AGENT_MODE_LLM_BASELINE
from CybORG.Agents.LLMAgents.llm_agent import DefenderAgent
from CybORG.Agents.LLMAgents.llm_policy import LLMDefenderPolicy


def select_policy_class(agent_mode: str):
    if agent_mode == AGENT_MODE_LLM_BASELINE:
        return LLMDefenderPolicy
    raise ValueError(f"Unknown agent_mode='{agent_mode}'. Expected {AGENT_MODE_LLM_BASELINE}")


def create_defender_agent(name: str, agent_mode: str = AGENT_MODE) -> DefenderAgent:
    policy_class = select_policy_class(agent_mode)
    return DefenderAgent(name=name, policy=policy_class, obs_space=[])
