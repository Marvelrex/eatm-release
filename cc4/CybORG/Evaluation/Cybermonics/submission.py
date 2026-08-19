from __future__ import annotations
import os

from CybORG import CybORG
from CybORG.Agents import BaseAgent

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import Policy
from ray.tune import register_env
import torch

from CybORG.Shared.MetricsCallback import MetricsCallback

### Import custom agents here ###
from CybORG.Agents.CybermonicAgents.cage4 import load
from CybORG.Agents.Wrappers.CybermonicWrappers.graph_wrapper import GraphWrapper

from CybORG.Agents.LLMAgents.llm_agent import DefenderAgent, RLLib_shim
from CybORG.Agents.LLMAgents.agent_factory import create_defender_agent
from CybORG.Agents.LLMAgents.comm_vector import CommVectorGenerator as cvg
from CybORG.Shared.Enums import TernaryEnum

from CybORG.Agents.Wrappers.CybermonicWrappers.observation_graph import ObservationGraph
from CybORG.Agents.Wrappers import BaseWrapper
from CybORG.Agents.Wrappers.BlueFixedActionWrapper import EMPTY_MESSAGE

from CybORG.Agents.LLMAgents.config.config_vars import (
    AGENT_MODE,
    ALL_LLM_AGENTS,
    BLUE_AGENT_NAME,
    INVALID_RL_AGENTS,
    NO_LLM_AGENTS,
    RANDOM_RL_AGENTS,
    SLEEP_RL_AGENTS,
    SUB_NAME,
    SUB_TEAM,
    SUB_TECHNIQUE,
)
from CybORG.Agents.SimpleAgents.ConstantAgent import InvalidActionAgent, SleepAgent
from CybORG.Agents.SimpleAgents.RandomAgent import cc4BlueRandomAgent
from CybORG.Simulator.Actions.Action import Sleep

# Use the no-graph non-LLM fast path (SleepAgent-style) when either the
# Sleep-only or InvalidAction-only baseline is requested.
CONSTANT_RL_AGENTS = SLEEP_RL_AGENTS or INVALID_RL_AGENTS or RANDOM_RL_AGENTS

import wandb

class Submission:
    # Submission name
    NAME: str = SUB_NAME
    
    # Name of your team
    TEAM: str = SUB_TEAM

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = SUB_TECHNIQUE

    if ALL_LLM_AGENTS:  # Are you nuts?
        AGENTS: dict[str, BaseAgent] = {
            f"blue_agent_{agent}": create_defender_agent(f"blue_agent_{agent}", AGENT_MODE)
            for agent in range(5)
        }
    elif INVALID_RL_AGENTS:
        AGENTS = {
            f"blue_agent_{i}": InvalidActionAgent(name=f"blue_agent_{i}")
            for i in range(5)
        }
    elif RANDOM_RL_AGENTS:
        AGENTS = {
            f"blue_agent_{i}": cc4BlueRandomAgent(name=f"blue_agent_{i}")
            for i in range(5)
        }
    elif SLEEP_RL_AGENTS:
        AGENTS = {
            f"blue_agent_{i}": SleepAgent(name=f"blue_agent_{i}")
            for i in range(5)
        }
    else:
        AGENTS = {
            f"blue_agent_{i}": load(f'{os.path.dirname(__file__)}/weights/gnn_ppo-{i}.pt')
            for i in range(5)
        }

    if not NO_LLM_AGENTS:
        AGENTS[BLUE_AGENT_NAME] = create_defender_agent(BLUE_AGENT_NAME, AGENT_MODE)
    

    @classmethod
    def wrap(cls, env: CybORG, path: str = ""):
        return CombinedWrapper(env)

    
class CombinedWrapper(BaseWrapper):
    """Unified wrapper that calls parallel_step() exactly ONCE per outer step and fans the
    observations out to the LLM-side and RL-side post-processing."""

    def __init__(self, env):
        super().__init__(env)

        self.metrics_callback = MetricsCallback()

        # PhaseWrapper handles the LLM agent(s).
        self.phase_wrapper = PhaseWrapper(env)
        # GraphWrapper provides GNN-based observation processing for RL agents.
        # We only use its action_translator and observation post-processing,
        # NOT its step() method (to avoid double-stepping).
        self.graph_wrapper = GraphWrapper(env)

    def step(self, actions):
        """Single-step: combine actions, step once, fan out observations."""

        # ── 1. Split & translate actions ──
        phase_actions = {}
        graph_actions = {}
        if ALL_LLM_AGENTS:
            phase_actions = actions
        elif CONSTANT_RL_AGENTS:
            # All non-LLM agents are ConstantAgents (Sleep or InvalidAction)
            # — pass their actions directly (no graph wrapper translation
            # needed). When NO_LLM_AGENTS=True, BLUE_AGENT_NAME is also a
            # constant agent, so include it in constant_actions instead of
            # dropping it.
            if NO_LLM_AGENTS:
                constant_actions = {k: v for k, v in actions.items()
                                    if k.startswith("blue_agent_")}
            else:
                phase_actions = {k: v for k, v in actions.items() if k == BLUE_AGENT_NAME}
                constant_actions = {k: v for k, v in actions.items()
                                    if k.startswith("blue_agent_") and k != BLUE_AGENT_NAME}
            combined_actions = {**phase_actions, **constant_actions}
        elif NO_LLM_AGENTS:
            graph_actions = {k: v for k, v in actions.items()
                            if k.startswith("blue_agent_")}
        else:
            phase_actions = {k: v for k, v in actions.items() if k == BLUE_AGENT_NAME}
            graph_actions = {k: v for k, v in actions.items()
                            if k.startswith("blue_agent_") and k != BLUE_AGENT_NAME}

        if not CONSTANT_RL_AGENTS:
            translated_graph_actions = {
                agent_name: self.graph_wrapper.action_translator(agent_name, action_id)
                for agent_name, action_id in graph_actions.items()
            }
            combined_actions = {**phase_actions, **translated_graph_actions}

        # ── 2. Pre-step: messages, metrics ──
        ec = self.env.unwrapped.environment_controller
        blue_agents = [a for a in self.env.agents if "blue" in a]

        combined_observations_before = {
            a: self.phase_wrapper.observation_change(a, self.env.get_observation(a))
            for a in blue_agents
        }
        self.metrics_callback.on_step(combined_observations_before, combined_actions, self.env)

        messages = {
            a: cvg.create_comm_message(
                ec.get_last_observation(a), ec.get_last_action(a),
                self.env.environment_controller.hostname_ip_map,
            )
            for a in Submission.AGENTS
        }

        # ── 3. SINGLE parallel_step with ALL actions ──
        obs, rews, dones, info = self.env.parallel_step(
            combined_actions, messages=messages,
        )

        # ── 4. Post-process: phase observations for LLM agent(s) ──
        for agent_name in dones:
            if "blue" in agent_name:
                dones[agent_name] = False

        if ALL_LLM_AGENTS:
            self.phase_wrapper.agents = [
                a for a, done in dones.items() if "blue" in a and not done
            ]
        elif NO_LLM_AGENTS:
            self.phase_wrapper.agents = []
        else:
            self.phase_wrapper.agents = [BLUE_AGENT_NAME]

        phase_obs = {
            agent: self.phase_wrapper.observation_change(agent, o)
            for agent, o in obs.items()
            if agent in self.phase_wrapper.agents
        }

        # Team rewards (all blue agents, same team value)
        team_rewards = {
            agent: sum(reward.values())
            for agent, reward in rews.items()
            if "blue" in agent
        }

        # Rewards returned to evaluation: phase agents get their value,
        # graph agents get theirs (all currently identical since team reward).
        phase_rewards = {a: team_rewards[a] for a in self.phase_wrapper.agents if a in team_rewards}
        graph_rewards_out = {a: team_rewards[a] for a in graph_actions if a in team_rewards}

        # ── 5. Post-process: graph observations for RL agents ──
        graph_obs = {}
        if graph_actions and not CONSTANT_RL_AGENTS:
            graph_obs = self.graph_wrapper.build_graph_observations(obs)

        # ── 6. Merge outputs ──
        if CONSTANT_RL_AGENTS:
            # Include constant-agent observations so eval loop keeps calling them.
            if NO_LLM_AGENTS:
                constant_obs = {a: obs[a] for a in obs if "blue" in a}
            else:
                constant_obs = {a: obs[a] for a in obs if "blue" in a and a != BLUE_AGENT_NAME}
            constant_rewards = {a: team_rewards[a] for a in constant_obs if a in team_rewards}
            observations = {**constant_obs, **phase_obs}
            rewards = {**constant_rewards, **phase_rewards}
        elif NO_LLM_AGENTS:
            observations = {**graph_obs}
            rewards = {**graph_rewards_out}
        else:
            observations = {**graph_obs, **phase_obs}
            rewards = {**graph_rewards_out, **phase_rewards}
        terminated = {a: dones.get(a, False) for a in observations}
        truncated = {a: dones.get(a, False) for a in observations}

        return observations, rewards, terminated, truncated, {}

    def reset(self, *args, **kwargs):
        """Reset: call env.reset() once, then initialize both wrappers."""
        # Reset the underlying env ONCE
        self.env.reset(*args, **kwargs)

        # Initialize PhaseWrapper state
        if ALL_LLM_AGENTS:
            self.phase_wrapper.agents = [a for a in self.env.agents if "blue" in a]
        else:
            self.phase_wrapper.agents = [BLUE_AGENT_NAME]
        phase_obs = {
            a: self.phase_wrapper.observation_change(a, self.env.get_observation(a))
            for a in self.phase_wrapper.agents
        }

        # Initialize GraphWrapper state (observation graphs, messages)
        graph_obs = {}
        if not CONSTANT_RL_AGENTS:
            graph_obs = self.graph_wrapper.build_reset_observations()

        self.metrics_callback.on_reset(self.env)

        if CONSTANT_RL_AGENTS:
            # Constant agents (Sleep / InvalidAction) don't need graph
            # observations; just return phase obs plus dummy obs so the
            # eval loop can call get_action on them. Under NO_LLM_AGENTS
            # BLUE_AGENT_NAME is also a constant agent — include it.
            constant_obs = {
                f"blue_agent_{i}": self.phase_wrapper.observation_change(
                    f"blue_agent_{i}", self.env.get_observation(f"blue_agent_{i}"),
                )
                for i in range(5)
                if NO_LLM_AGENTS or f"blue_agent_{i}" != BLUE_AGENT_NAME
            }
            observations = {**constant_obs, **phase_obs}
        elif NO_LLM_AGENTS:
            observations = {**graph_obs}
        elif ALL_LLM_AGENTS:
            observations = {**phase_obs}
        else:
            observations = {**graph_obs, **phase_obs}

        return observations, {}
    
    def action_space(self, agent_name: str) -> Space:
        """Returns the discrete space corresponding to the given agent."""
        if ALL_LLM_AGENTS or CONSTANT_RL_AGENTS or agent_name == BLUE_AGENT_NAME:
            return self.phase_wrapper.action_space(agent_name)
        if hasattr(self.graph_wrapper, "action_space"):
            return self.graph_wrapper.action_space(agent_name)
        return self.phase_wrapper.action_space(agent_name)
    
class PhaseWrapper(BaseWrapper):
    def __init__(self, env: CybORG = None):
        super().__init__(env)

    def observation_change(self, agent: str, observation: dict):
        state = self.env.environment_controller.state
        observation["phase"] = state.mission_phase
        return observation

    def reset(self, *args, **kwargs) -> tuple[dict[str, Any], dict[str, dict]]:
        self.env.reset(*args, **kwargs)
        self.agents = [a for a in self.env.agents if BLUE_AGENT_NAME in a]
        if ALL_LLM_AGENTS or CONSTANT_RL_AGENTS:
            self.agents = [a for a in self.env.agents]
        observations = {a: self.observation_change(a, self.env.get_observation(a)) for a in self.agents}
        return observations, {}

    def step(
        self,
        actions: dict[str, Action] = {},
        messages: dict[str, Any] = None,
        **kwargs,
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:

        # Communication Vector Generation for each agent
        ec = self.env.unwrapped.environment_controller
        if messages is None:
            messages = {
                a: cvg.create_comm_message(ec.get_last_observation(a), ec.get_last_action(a), self.env.environment_controller.hostname_ip_map)
                for a, v in Submission.AGENTS.items()
            }

        obs, rews, dones, info = self.env.parallel_step(
            actions, messages=messages, **kwargs
        )

        for agent_name, status in dones.items():
            if "blue" in agent_name:
                dones[agent_name] = False

        if ALL_LLM_AGENTS or CONSTANT_RL_AGENTS:
            self.agents = [
            agent for agent, done in dones.items() if "blue" in agent and not done
        ]
        else:
            self.agents = [BLUE_AGENT_NAME]

        observations = {agent: self.observation_change(agent, o) for agent, o in obs.items() if agent in self.agents}

        rewards = {
            agent: sum(reward.values())
            for agent, reward in rews.items()
            if agent in self.agents
        }
        team_rewards = {
            agent: sum(reward.values())
            for agent, reward in rews.items()
            if "blue" in agent
        }

        terminated = {agent: done for agent, done in dones.items() if agent in self.agents}
        truncated = {agent: done for agent, done in dones.items() if agent in self.agents}

        return observations, rewards, terminated, truncated, {}

    def action_space(self, agent_name: str) -> Space:
        """Returns the discrete space corresponding to the given agent."""
        return self.env.get_action_space(agent_name)

class MessageWrapper(BaseWrapper):
    def step(self,
        actions: dict[str, Action] = {},
        messages: dict[str, Any] = None,
        **kwargs,
    ) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        if messages is None:
            env = self.env.unwrapped.environment_controller
            messages = {
                a: v.get_message(env.get_last_observation(a), env.get_message_space(a))
                for a, v in Submission.AGENTS.items()
            }
        return super().step(actions, messages, **kwargs)
