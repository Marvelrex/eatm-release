from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from gymnasium.core import ObsType
from pydantic import Field

from primaite import getLogger
from primaite.game.agent.interface import AbstractScriptedAgent
from primaite.game.agent.scripted_agents.cage4 import comms

_LOGGER = getLogger(__name__)

__all__ = ("CAGE4BlueAgent",)


def _dig(d: Optional[Dict], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


class CAGE4BlueAgent(AbstractScriptedAgent, discriminator="cage4-blue-agent"):

    config: "CAGE4BlueAgent.ConfigSchema" = Field(default_factory=lambda: CAGE4BlueAgent.ConfigSchema())

    last_state: Optional[Dict] = None
    blocked_positions: set = Field(default_factory=set)
    bot_killed: bool = False
    decision_log: list = Field(default_factory=list)

    class AgentSettingsSchema(AbstractScriptedAgent.AgentSettingsSchema):
        zone_index: int = 0
        comms_channel: str = "cage4"
        zones: List[Dict[str, Any]] = Field(default_factory=list)
        folder: str = "database"
        file: str = "database.db"
        bot_app: str = "data-manipulation-bot"
        decoy_service: str = "web-server"
        beachhead_ip: str = ""
        block_router: str = "hq_router"
        scan_interval: int = 5
        server: str = ""
        server_ip: str = ""
        service: str = "database-service"
        client: str = ""
        block_position: int = 1

    class ConfigSchema(AbstractScriptedAgent.ConfigSchema):
        type: str = "cage4-blue-agent"
        agent_settings: "CAGE4BlueAgent.AgentSettingsSchema" = Field(
            default_factory=lambda: CAGE4BlueAgent.AgentSettingsSchema())

    def update_observation(self, state: Dict) -> ObsType:
        self.last_state = state
        return super().update_observation(state)

    def _reset(self) -> None:
        self.blocked_positions = set()
        self.bot_killed = False

    def _zones(self) -> List[Dict[str, Any]]:
        s = self.config.agent_settings
        if s.zones:
            return s.zones
        return [{"name": s.server, "server": s.server, "ip": s.server_ip,
                 "kind": "db" if s.client else "web", "service": s.service,
                 "client": s.client, "block_position": s.block_position}]

    def _corrupt(self, z: Dict[str, Any]) -> bool:
        if z.get("kind") != "db":
            return False
        s = self.config.agent_settings
        health = _dig(self.last_state, "network", "nodes", z["server"], "file_system",
                      "folders", s.folder, "files", s.file, "health_status")
        return health in (2, 3)

    def get_action(self, obs: ObsType, timestep: int = 0) -> Tuple[str, Dict]:
        s = self.config.agent_settings
        if timestep == 0:
            self._reset()
        zones = self._zones()
        compromised = [z for z in zones if self._corrupt(z)]
        detected = bool(compromised)
        peer_detect = comms.detection_vector(s.comms_channel)
        others_compromised = any(d for i, d in enumerate(peer_detect) if i != s.zone_index)
        comms.post(s.comms_channel, s.zone_index, detected=detected, level=3 if detected else 0)

        policy, action, options = self._decide(s, timestep, zones, compromised, others_compromised)
        self.decision_log.append({
            "timestep": timestep, "agent": self.config.ref, "agent_index": s.zone_index,
            "policy": policy, "caos_action": {"action": action, "options": options},
            "cc4_to_caos": f"{policy} -> {action}",
            "message_byte": comms.message_byte(s.comms_channel, s.zone_index)})
        return action, options

    def _decide(self, s, timestep: int, zones, compromised, others_compromised
                ) -> Tuple[str, str, Dict]:
        if compromised:
            z = compromised[0]
            if not self.bot_killed and z.get("client"):
                self.bot_killed = True
                return ("Remove", "node-application-close",
                        {"node_name": z["client"], "application_name": s.bot_app})
            self.bot_killed = False
            return ("Restore", "node-folder-restore",
                    {"node_name": z["server"], "folder_name": s.folder})
        if others_compromised and s.beachhead_ip:
            for z in zones:
                pos = z.get("block_position")
                if pos and pos not in self.blocked_positions:
                    self.blocked_positions.add(pos)
                    return ("BlockTrafficZone", "router-acl-add-rule", {
                        "target_router": s.block_router, "position": pos, "permission": "DENY",
                        "protocol_name": "ALL", "src_ip": s.beachhead_ip, "src_wildcard": "NONE",
                        "src_port": "ALL", "dst_ip": z["ip"], "dst_wildcard": "NONE", "dst_port": "ALL"})
        if self.blocked_positions and not others_compromised:
            pos = self.blocked_positions.pop()
            return ("AllowTrafficZone", "router-acl-remove-rule",
                    {"target_router": s.block_router, "position": pos})
        if timestep % s.scan_interval == 0:
            z = zones[(timestep // s.scan_interval) % len(zones)]
            return ("Analyse", "node-os-scan", {"node_name": z["server"]})
        if s.decoy_service and timestep % (s.scan_interval * 3) == 1:
            return ("DeployDecoy", "node-service-start",
                    {"node_name": zones[0]["server"], "service_name": s.decoy_service})
        return ("do-nothing", "do-nothing", {})
