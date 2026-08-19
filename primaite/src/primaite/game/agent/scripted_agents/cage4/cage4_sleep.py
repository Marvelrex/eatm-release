from __future__ import annotations

from typing import Dict, Tuple

from gymnasium.core import ObsType
from pydantic import Field

from primaite.game.agent.interface import AbstractScriptedAgent


class CAGE4SleepAgent(AbstractScriptedAgent, discriminator="cage4-sleep-agent"):

    config: "CAGE4SleepAgent.ConfigSchema" = Field(default_factory=lambda: CAGE4SleepAgent.ConfigSchema())

    class ConfigSchema(AbstractScriptedAgent.ConfigSchema):
        type: str = "cage4-sleep-agent"

    def get_action(self, obs: ObsType, timestep: int = 0) -> Tuple[str, Dict]:
        return "do-nothing", {}
