from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, List, Optional

from gymnasium.core import ObsType
from pydantic import Field

from primaite.game.agent.interface import AbstractScriptedAgent
from primaite.game.agent.scripted_agents.cage4 import comms
from primaite.game.agent.scripted_agents.cage4 import families as host_selector
from primaite.game.agent.scripted_agents.llm_agent import LLMAgent, _AbstractLLMAgent, _parse_action_dict

__all__ = ("CAGE4BlueLLMAgent",)

_BLUE_AGENT_NETWORKS = {
    0: "restricted_zone_a",
    1: "operational_zone_a",
    2: "restricted_zone_b",
    3: "operational_zone_b",
    4: "public_access_zone, admin_network, office_network",
}

_ACTIONS = [
    ("Remove", "Attempt to remove red from the host by destroying MALICIOUS PROCESSES, files and "
               "services. Stops all processes identified as malicious by Analyse."),
    ("Restore", "Restore the system to a known good state. This has significant consequences for "
                "system availability. Ideal when a host has a high-level (privileged) compromise."),
    ("BlockTrafficZone", "Block traffic to/from a subnet. If green agents are trying to communicate "
                         "to that zone this will cause availability issues (penalised)."),
    ("AllowTrafficZone", "Allow traffic to/from a subnet. Only has an effect if the zone is currently "
                         "blocked."),
    ("DeployDecoy", "Set up a decoy service on the host. Green agents never access decoys, so any "
                    "access is an unambiguous indicator of red activity; red cannot exploit a decoy."),
    ("Analyse", "Analyse the host for suspicious activity / malware to confirm whether red is present. "
                "Run Analyse when you suspect red agents have been active in your zone."),
]

_DESCRIPTION = (
    "# DESCRIPTION\n"
    "You are an expert in cyber security defending an enterprise network. Analyse the network state "
    "and choose the best action from the list of actions based on their descriptions.\n\n"
    "You are BLUE AGENT {n}, the network defender of: {zones}. Each step you receive an OBSERVATION of "
    "your assigned zone and the inter-agent COMMUNICATION VECTORS summarising every defender's zone. "
    "Analyse the state and choose the single best action.\n\n"
    "Respond EXACTLY with one response as a JSON dictionary with these keys:\n"
    "- action: the INDEX (integer) of ONLY ONE action from `## AVAILABLE ACTIONS`. Each action already "
    "targets the correct host in your assigned zone (BLUE AGENT {n}) or, for Block/AllowTrafficZone, the "
    "attacker's reach to your zone — you only choose WHICH action.\n"
    "- reason: ONLY ONE concise reason for your choice.\n\n"
    "Scoring (native CAGE-4 penalty, best 0): each green-user work/service failure and each red impact "
    "in your zones reduces the score."
)

_NETWORK_STRUCTURE = (
    "## NETWORK STRUCTURE:\n"
    "- 4 Networks: 2 Deployed Networks (A & B), HQ Network, Contractor Network\n"
    "- Security Zones:\n"
    "  * Deployed Networks: Restricted Zone + Operational Zone each\n"
    "  * HQ Network: Public Access, Admin, Office Zones\n"
    "  * Contractor Network: UAV Control Zone (undefended)\n\n"
    "## DEFENSE SETUP:\n"
    "- 5 Network Defenders: 2 in each Deployed Network (1 per zone), 1 in HQ (covers all HQ zones); "
    "the Contractor Network is undefended.\n"
    "- The red team starts in, and keeps a constant presence in, the Contractor Network, and pivots "
    "outward zone by zone. At most 1 red agent per zone.\n"
    "- This is a single static mission phase (no phase switching in this instance)."
)

_COMMVECTOR_FORMAT = (
    "# COMMVECTOR FORMAT\n"
    "The COMMUNICATION VECTORS summarise the status of each defender's zone. Each BLUE AGENT broadcasts "
    "an 8-bit message every step:\n"
    "- Bits 0-4: malicious activity detected in BLUE AGENT 0..4's network (1) or not (0)\n"
    "- Bits 5-6: compromise level of YOUR network: 00 none / 01 netscan or remote exploit / 10 "
    "user-level / 11 admin-level\n"
    "- Bit 7: waiting for an action to complete (1) or not (0)\n"
    "Use a neighbour's ALERT bit to decide whether to BlockTrafficZone before red reaches you."
)


_FAMILY_KEYWORDS = [
    ("blocktrafficzone", "BlockTrafficZone"), ("allowtrafficzone", "AllowTrafficZone"),
    ("deploydecoy", "DeployDecoy"), ("block", "BlockTrafficZone"), ("allow", "AllowTrafficZone"),
    ("decoy", "DeployDecoy"), ("remove", "Remove"), ("restore", "Restore"),
    ("analyse", "Analyse"), ("analyze", "Analyse"), ("sleep", "Sleep"), ("monitor", "Sleep"),
]


def match_native_action(text: str, catalog: List[Dict[str, Any]]) -> Optional[int]:
    t = (text or "").lower()
    family = next((fam for key, fam in _FAMILY_KEYWORDS if key in t), None)
    if family is None or family == "Sleep":
        return None
    host = t.split("host:", 1)[1].split()[0].strip().strip('"\'.,)') if "host:" in t else ""
    subnet = t.split("subnet:", 1)[1].split()[0].strip().strip('"\'.,)') if "subnet:" in t else ""
    subnet = subnet[:-len("_subnet")] if subnet.endswith("_subnet") else subnet
    for e in catalog:
        if e["family"] != family:
            continue
        if family in ("BlockTrafficZone", "AllowTrafficZone"):
            if subnet and e["zone"] == subnet:
                return e["index"]
        elif host and e["host"] == host:
            return e["index"]
    return None


class CAGE4BlueLLMAgent(LLMAgent, discriminator="cage4-blue-llm-agent"):

    config: "CAGE4BlueLLMAgent.ConfigSchema" = Field(
        default_factory=lambda: CAGE4BlueLLMAgent.ConfigSchema())

    last_state: Optional[Dict] = None

    class AgentSettingsSchema(_AbstractLLMAgent.AgentSettingsSchema):
        zone_index: int = 0
        comms_channel: str = "cage4"
        zones: List[Dict[str, Any]] = Field(default_factory=list)
        server: str = ""
        folder: str = "database"
        file: str = "database.db"
        native_actions: bool = False
        action_zones: Dict[int, str] = Field(default_factory=dict)

    class ConfigSchema(AbstractScriptedAgent.ConfigSchema):
        type: str = "cage4-blue-llm-agent"
        agent_settings: "CAGE4BlueLLMAgent.AgentSettingsSchema" = Field(
            default_factory=lambda: CAGE4BlueLLMAgent.AgentSettingsSchema())

    def update_observation(self, state: Dict) -> ObsType:
        self.last_state = state
        return super().update_observation(state)

    def _db_zones(self) -> List[Dict[str, Any]]:
        s = self.config.agent_settings
        return s.zones or ([{"server": s.server, "folder": s.folder, "file": s.file}] if s.server else [])

    def _compromised_servers(self) -> List[str]:
        s = self.config.agent_settings
        out = []
        for z in self._db_zones():
            cur = self.last_state
            for k in ("network", "nodes", z.get("server"), "file_system", "folders",
                      z.get("folder", s.folder), "files", z.get("file", s.file)):
                if not isinstance(cur, dict):
                    cur = None
                    break
                cur = cur.get(k)
            if isinstance(cur, dict) and cur.get("health_status") in (2, 3):
                out.append(z.get("server"))
        return out

    def _zone_detected(self) -> bool:
        return bool(self._compromised_servers())

    @cached_property
    def _native_catalog(self) -> List[Dict[str, Any]]:
        az = self.config.agent_settings.action_zones
        cat = []
        for i, (act, opts) in sorted(self.action_manager.action_map.items()):
            cat.append({"index": i, "family": host_selector.family_of(act),
                        "host": str(opts.get("node_name") or "").lower(),
                        "zone": str(az.get(i) or "").lower()})
        return cat

    @cached_property
    def _native_system_text(self) -> str:
        n = self.config.agent_settings.zone_index
        zones = _BLUE_AGENT_NETWORKS.get(n, f"zone_{n}")
        param = {"BlockTrafficZone": "subnet:<subnet_id>", "AllowTrafficZone": "subnet:<subnet_id>"}
        fams = {e["family"] for e in self._native_catalog}
        legend = "## AVAILABLE ACTIONS\n" + "\n".join(
            f"- {name} {param.get(name, 'host:<hostname>')}: {desc}"
            for name, desc in _ACTIONS if name in fams)
        hosts = sorted({e["host"] for e in self._native_catalog
                        if e["host"] and e["family"] not in ("BlockTrafficZone", "AllowTrafficZone")})
        subnets = sorted({e["zone"] for e in self._native_catalog
                          if e["zone"] and e["family"] in ("BlockTrafficZone", "AllowTrafficZone")})
        targets = ("## YOUR TARGETS\n"
                   f"- <hostname> must be one of: {', '.join(hosts) or '-'}\n"
                   f"- <subnet_id> must be one of: {', '.join(subnets) or '-'}")
        ex_host = hosts[0] if hosts else "host_0"
        candidates = [
            ("Remove", f'- {{"action": "Remove host:{ex_host}", "reason": "Host has been compromised"}}'),
            ("Restore", f'- {{"action": "Restore host:{ex_host}", "reason": "Privileged compromise detected"}}'),
            ("BlockTrafficZone", f'- {{"action": "BlockTrafficZone subnet:{subnets[0] if subnets else "-"}", '
                                 '"reason": "Neighbour alert - cutting the attack path"}'),
            ("Analyse", f'- {{"action": "Analyse host:{ex_host}", "reason": "Suspicious activity reported"}}'),
        ]
        examples = ("## EXAMPLE RESPONSES\n"
                    + "\n".join(line for fam, line in candidates if fam in fams)[:1200])
        description = (
            "# DESCRIPTION\n"
            "You are an expert in cyber security defending an enterprise network. Analyse the network "
            "state and choose the best action and parameter from the list of actions based on their "
            "description.\n\n"
            f"You are BLUE AGENT {n}, the network defender of: {zones}. Each step you receive an "
            "OBSERVATION of your assigned zone and the inter-agent COMMUNICATION VECTORS summarising "
            "every defender's zone.\n\n"
            "Respond EXACTLY with one response as a JSON dictionary with the following keys:\n"
            "- action: ONLY ONE action from `## AVAILABLE ACTIONS`, always including the required "
            "parameter. For <hostname>, you can ONLY execute an action on hosts from `## YOUR TARGETS`. "
            "For <subnet_id>, choose a subnet from `## YOUR TARGETS`.\n"
            "- reason: ONLY ONE reason for your choice.\n\n"
            "An action that cannot be parsed, or whose target is not yours, does NOTHING this step "
            "(Sleep). Scoring (native CAGE-4 penalty, best 0): each green-user work/service failure "
            "and each red impact in your zones reduces the score.")
        networks = "## BLUE AGENT NETWORKS\n" + "\n".join(
            f"- BLUE AGENT {i}: {z}" for i, z in _BLUE_AGENT_NETWORKS.items()) + f"\nYou are BLUE AGENT {n}."
        parts = [description, legend, targets, examples, _NETWORK_STRUCTURE, networks, _COMMVECTOR_FORMAT]
        if self.config.agent_settings.mission_brief:
            parts.append("## YOUR MISSION\n" + self.config.agent_settings.mission_brief)
        return "\n\n".join(parts)

    def _response_instruction(self) -> str:
        if not self.config.agent_settings.native_actions:
            return super()._response_instruction()
        return ('Respond with ONLY a JSON object in this exact format (no prose, no code fence):\n'
                '{"action": "<Action> host:<hostname>" or "<Action> subnet:<subnet_id>", '
                '"reason": "<one concise sentence>"}')

    def get_action(self, obs: ObsType, timestep: int = 0):
        s = self.config.agent_settings
        if not s.native_actions:
            return super().get_action(obs, timestep)
        if s.decide_every > 1 and (timestep % s.decide_every):
            return "do-nothing", {}

        reasoning, usage, error, prompt, response, action_str = "", None, None, None, None, ""
        index = None
        try:
            user = self._build_user_message(obs, timestep)
            prompt = {"system": self._native_system_text, "user": user}
            content, usage = self._chat(self._native_system_text, user)
            response = content
            args = _parse_action_dict(content)
            action_str = str(args.get("action", ""))
            reasoning = str(args.get("reason", args.get("reasoning", "")))
            index = match_native_action(action_str, self._native_catalog)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"

        if index is None:
            action_name, options = "do-nothing", {}
        else:
            action_name, options = self.action_manager.get_action(index)
        self.decision_log.append({
            "timestep": timestep, "agent": self.config.ref, "model": s.model,
            "action_index": index, "action": action_name, "native_action": action_str,
            "usage": usage, "reasoning": reasoning[:200], "prompt": prompt,
            "response": response, "error": error,
        })
        return action_name, options

    @cached_property
    def _system_text(self) -> str:
        n = self.config.agent_settings.zone_index
        zones = _BLUE_AGENT_NETWORKS.get(n, f"zone_{n}")
        legend = "## ACTION FAMILIES\n" + "\n".join(f"- {name}: {desc}" for name, desc in _ACTIONS)
        lines = []
        for i, (act, opts) in sorted(self.action_manager.action_map.items()):
            tgt = (opts.get("node_name") or opts.get("dst_ip")
                   or (f"pos{opts['position']}" if opts.get("position") is not None else None)
                   or opts.get("target_router") or "-")
            lines.append(f"[{i}] {host_selector.family_of(act)} on {tgt}")
        actions = "## AVAILABLE ACTIONS (choose exactly one index)\n" + "\n".join(lines)
        examples = ('## RESPONSE FORMAT\nReply with ONLY a JSON object:\n'
                    '{"action": <index from AVAILABLE ACTIONS>, "reason": "<one concise reason>"}')
        networks = "## BLUE AGENT NETWORKS\n" + "\n".join(
            f"- BLUE AGENT {i}: {z}" for i, z in _BLUE_AGENT_NETWORKS.items()) + f"\nYou are BLUE AGENT {n}."
        parts = [
            _DESCRIPTION.format(n=n, zones=zones),
            legend, actions, examples, _NETWORK_STRUCTURE, networks, _COMMVECTOR_FORMAT,
        ]
        if self.config.agent_settings.mission_brief:
            parts.append("## YOUR MISSION\n" + self.config.agent_settings.mission_brief)
        return "\n\n".join(parts)

    def _pre_analysis(self, obs: ObsType, timestep: int) -> str:
        s = self.config.agent_settings
        compromised = self._compromised_servers()
        comms.post(s.comms_channel, s.zone_index, detected=bool(compromised), level=3 if compromised else 0)
        vec_lines = [f"Commvector Blue Agent {i} Message: {comms.message_byte(s.comms_channel, i)}"
                     for i in range(comms.NUM_AGENTS)]
        activity = ("\n".join(f"- {srv}: ALERT - red detected (database corrupted), addressable with "
                              "Remove/Restore" for srv in compromised) if compromised else "- None")
        native_obs = ("# COMMUNICATION VECTORS (current round)\n" + "\n".join(vec_lines)
                      + "\n\n# SUSPICIOUS ACTIVITY (your zone)\n" + activity)
        base = super()._pre_analysis(obs, timestep)
        return (base + "\n\n" if base else "") + native_obs
