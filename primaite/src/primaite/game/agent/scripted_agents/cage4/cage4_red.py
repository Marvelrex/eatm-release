from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from gymnasium.core import ObsType
from pydantic import Field

from primaite import getLogger
from primaite.game.agent.interface import AbstractScriptedAgent

_LOGGER = getLogger(__name__)

__all__ = ("CAGE4RedAgent", "reset_activation")

_ACTIVATION: Dict[str, set] = {}


def reset_activation(channel: str) -> None:
    _ACTIVATION.pop(channel, None)

DISCOVER, AGGR_SVC, STEALTH_SVC, DISCOVER_DECEPT, EXPLOIT, PRIV_ESC, IMPACT, DEGRADE, WITHDRAW = range(9)

_SUCCESS: Dict[str, List[Optional[str]]] = {
    "K":  ["KD", "S",  "S",  None, None, None, None, None, None],
    "KD": ["KD", "SD", "SD", None, None, None, None, None, None],
    "S":  ["SD", None, None, "S",  "U",  None, None, None, None],
    "SD": ["SD", None, None, "SD", "UD", None, None, None, None],
    "U":  ["UD", None, None, None, None, "R",  None, None, "S"],
    "UD": ["UD", None, None, None, None, "RD", None, None, "SD"],
    "R":  ["RD", None, None, None, None, None, "R",  "R",  "S"],
    "RD": ["RD", None, None, None, None, None, "RD", "RD", "SD"],
    "F":  ["F",  None, None, None, None, None, None, None, None],
}
_FAILURE: Dict[str, List[Optional[str]]] = {
    "K":  ["K",  "K",  "K",  None, None, None, None, None, None],
    "KD": ["KD", "KD", "KD", None, None, None, None, None, None],
    "S":  ["S",  None, None, "S",  "S",  None, None, None, None],
    "SD": ["SD", None, None, "SD", "SD", None, None, None, None],
    "U":  ["U",  None, None, None, None, "U",  None, None, "U"],
    "UD": ["UD", None, None, None, None, "UD", None, None, "UD"],
    "R":  ["R",  None, None, None, None, None, "R",  "R",  "R"],
    "RD": ["RD", None, None, None, None, None, "RD", "RD", "RD"],
    "F":  ["F",  None, None, None, None, None, None, None, None],
}
_PROB: Dict[str, List[Optional[float]]] = {
    "K":  [0.5,  0.25, 0.25, None, None, None, None, None, None],
    "KD": [None, 0.5,  0.5,  None, None, None, None, None, None],
    "S":  [0.25, None, None, 0.25, 0.5,  None, None, None, None],
    "SD": [None, None, None, 0.25, 0.75, None, None, None, None],
    "U":  [0.5,  None, None, None, None, 0.5,  None, None, 0.0],
    "UD": [None, None, None, None, None, 1.0,  None, None, 0.0],
    "R":  [0.5,  None, None, None, None, None, 0.25, 0.25, 0.0],
    "RD": [None, None, None, None, None, None, 0.5,  0.5,  0.0],
    "F":  [None, None, None, None, None, None, None, None, None],
}


class CAGE4RedAgent(AbstractScriptedAgent, discriminator="cage4-red-agent"):

    config: "CAGE4RedAgent.ConfigSchema" = Field(default_factory=lambda: CAGE4RedAgent.ConfigSchema())

    last_state: Optional[Dict] = None
    host_states: Dict[str, str] = Field(default_factory=dict)
    last_act: Optional[Tuple[str, int]] = None
    rng: Any = None
    awake: bool = False
    engaged: bool = False
    decision_log: List[Dict] = Field(default_factory=list)

    class AgentSettingsSchema(AbstractScriptedAgent.AgentSettingsSchema):
        beachhead: str = "red_beachhead"
        targets: List[Dict[str, Any]] = Field(default_factory=list)
        seed: int = 2025
        decide_every: int = 1
        home_zone: str = ""
        activation_channel: str = "cage4_red"
        activates_after: Optional[str] = None

    class ConfigSchema(AbstractScriptedAgent.ConfigSchema):
        type: str = "cage4-red-agent"
        agent_settings: "CAGE4RedAgent.AgentSettingsSchema" = Field(
            default_factory=lambda: CAGE4RedAgent.AgentSettingsSchema())

    def update_observation(self, state: Dict) -> ObsType:
        self.last_state = state
        return super().update_observation(state)

    def _reset(self) -> None:
        self.host_states = {t["server_ip"]: "K" for t in self.config.agent_settings.targets}
        self.last_act = None
        self.pending_eviction = []
        self.awake = False
        self.engaged = False
        self.rng = random.Random(self.config.agent_settings.seed)

    def _last_success(self) -> Optional[bool]:
        if not self.history:
            return None
        status = getattr(self.history[-1].response, "status", None)
        if status == "success":
            return True
        if status in ("failure", "unreachable"):
            return False
        return None

    def _target(self, ip: str) -> Dict[str, Any]:
        return next(t for t in self.config.agent_settings.targets if t["server_ip"] == ip)

    def _db_corrupt(self, t: Dict[str, Any]) -> bool:
        st = self.last_state or {}
        node = (((st.get("network", {}) or {}).get("nodes", {}) or {}).get(t["server"], {}) or {})
        files = (((node.get("file_system", {}) or {}).get("folders", {}) or {}
                  ).get(t.get("folder", ""), {}) or {}).get("files", {}) or {}
        return (files.get(t.get("file", ""), {}) or {}).get("health_status") in (2, 3)

    _FOOTHOLD_STATES = ("S", "SD", "U", "UD", "R", "RD")
    pending_eviction: List[str] = Field(default_factory=list)

    def notify_blue_recovery(self, server_ip: str, user_only: bool = False) -> bool:
        st = self.host_states.get(server_ip)
        targets = ("U", "UD") if user_only else self._FOOTHOLD_STATES
        if st in targets:
            self.host_states[server_ip] = "KD"
            self.pending_eviction.append(server_ip)
            return True
        return False

    def _is_active(self) -> bool:
        s = self.config.agent_settings
        if s.activates_after is None:
            return True
        has_foothold = any(st in self._FOOTHOLD_STATES for st in self.host_states.values())
        if self.engaged and not has_foothold:
            self.awake = False
            self.engaged = False
            _ACTIVATION.get(s.activation_channel, set()).discard(s.home_zone)
        if self.awake:
            return True
        if s.activates_after in _ACTIVATION.get(s.activation_channel, set()):
            self.awake = True
            return True
        return False

    def _post_foothold(self) -> None:
        s = self.config.agent_settings
        if not s.home_zone:
            return
        has_foothold = any(st in self._FOOTHOLD_STATES for st in self.host_states.values())
        if has_foothold:
            self.engaged = True
        if s.activates_after is None or has_foothold:
            _ACTIVATION.setdefault(s.activation_channel, set()).add(s.home_zone)

    def get_action(self, obs: ObsType, timestep: int = 0) -> Tuple[str, Dict]:
        s = self.config.agent_settings
        if timestep == 0 or self.rng is None:
            self._reset()
        rec: Dict[str, Any] = {"timestep": timestep, "agent": self.config.ref, "role": "red",
                               "target": None, "action_name": None, "state": None, "caos_action": None}

        if not self._is_active():
            rec["action_name"] = "Sleep(inactive)"
            rec["caos_action"] = {"action": "do-nothing", "options": {}}
            self.decision_log.append(rec)
            return "do-nothing", {}

        if self.pending_eviction:
            rec["blue_evicted"] = list(self.pending_eviction)
            self.pending_eviction = []

        if self.last_act is not None:
            ip, act = self.last_act
            ok = self._last_success()
            if ok is not None and ip in self.host_states:
                matrix = _SUCCESS if ok else _FAILURE
                nxt = matrix[self.host_states[ip]][act]
                if nxt is not None:
                    self.host_states[ip] = nxt

        self._post_foothold()

        candidates = [ip for ip, st in self.host_states.items() if st != "F" and any(
            p for p in _PROB[st] if p)]
        if not candidates:
            self.last_act = None
            rec["caos_action"] = {"action": "do-nothing", "options": {}}
            self.decision_log.append(rec)
            return "do-nothing", {}
        ip = self.rng.choice(candidates)
        st = self.host_states[ip]
        opts = [(i, p) for i, p in enumerate(_PROB[st]) if p]
        idx = self._weighted_choice(opts)
        if idx is None:
            self.last_act = None
            rec["caos_action"] = {"action": "do-nothing", "options": {}}
            self.decision_log.append(rec)
            return "do-nothing", {}

        action, options = self._realise(self._target(ip), idx)
        self.last_act = (ip, idx)
        rec.update({"target": ip, "action_name": _ACTION_NAMES[idx], "state": st,
                    "caos_action": {"action": action, "options": options}})
        self.decision_log.append(rec)
        return action, options

    def _weighted_choice(self, options: List[Tuple[int, float]]) -> Optional[int]:
        total = sum(p for _, p in options)
        if total <= 0:
            return None
        r, acc = self.rng.random() * total, 0.0
        for i, p in options:
            acc += p
            if r <= acc:
                return i
        return options[-1][0]

    def _realise(self, t: Dict[str, Any], idx: int) -> Tuple[str, Dict]:
        bh = self.config.agent_settings.beachhead
        ip, server, client = t["server_ip"], t["server"], t.get("client")
        svc, kind = t.get("service"), t.get("kind", "db")
        if idx == DISCOVER:
            return "node-nmap-ping-scan", {"source_node": bh, "target_ip_address": ip, "show": False}
        if idx == AGGR_SVC:
            return "node-nmap-port-scan", {"source_node": bh, "target_ip_address": ip, "show": False}
        if idx == STEALTH_SVC:
            return "node-network-service-recon", {"source_node": bh, "target_ip_address": ip, "show": False}
        if idx == DISCOVER_DECEPT:
            return "node-service-scan", {"node_name": server, "service_name": svc}
        if idx == EXPLOIT:
            if client and t.get("client_app"):
                return "node-application-execute", {"node_name": client, "application_name": t["client_app"]}
            return "node-session-remote-login", {"node_name": bh, "remote_ip": ip,
                                                 "username": t.get("admin_user", "admin"),
                                                 "password": t.get("admin_password", "admin")}
        if idx == PRIV_ESC:
            if client and t.get("client_app"):
                return "node-application-execute", {"node_name": client, "application_name": t["client_app"]}
            return "node-session-remote-login", {"node_name": bh, "remote_ip": ip,
                                                 "username": t.get("admin_user", "admin"),
                                                 "password": t.get("admin_password", "admin")}
        if idx == IMPACT:
            if kind == "db" and client and t.get("bot_app"):
                return "node-application-execute", {"node_name": client, "application_name": t["bot_app"]}
            return "node-service-stop", {"node_name": server, "service_name": svc}
        if idx == DEGRADE:
            return "do-nothing", {}
        return "do-nothing", {}


_ACTION_NAMES = ["DiscoverRemoteSystems", "AggressiveServiceDiscovery", "StealthServiceDiscovery",
                 "DiscoverDeception", "ExploitRemoteService", "PrivilegeEscalate", "Impact",
                 "DegradeServices", "Withdraw"]
