import numpy as np
from gym import Space
from CybORG.Agents import BaseAgent
from ray.rllib.policy.policy import Policy
from CybORG.Agents.LLMAgents.obs_wrapper import EMPTY_MESSAGE, EnterpriseObsWrapper

def RLLib_shim(env, msg_agents: dict):
    return EnterpriseObsWrapper(env, msg_agents=msg_agents)

class DefenderAgent(BaseAgent):
    def __init__(self, name: str, policy: Policy, obs_space: np.ndarray):
        super().__init__(name)
        self.policy = policy(obs_space, None, {"agent_name": name})
        self.obs_space = obs_space
        self.end_episode()

    def get_action(self, observation: np.ndarray, action_space: Space):
        self.last_action, _, _ = self.policy.compute_single_action(
            obs=observation,
            prev_action=self.last_action,
            action_space=action_space,
        )
        self.step += 1
        return self.last_action

    def end_episode(self):
        self.step = 0
        self.last_action = None
        policy_end_episode = getattr(self.policy, "end_episode", None)
        if callable(policy_end_episode):
            policy_end_episode()

    def get_message(self, message_data: tuple[set], message_space: Space) -> np.ndarray:
        return np.ones(EMPTY_MESSAGE.shape, dtype=bool)
