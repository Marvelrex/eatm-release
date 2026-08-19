from __future__ import annotations

import json
import os
from abc import abstractmethod
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from gymnasium.core import ObsType
from pydantic import Field

from primaite import getLogger
from primaite.game.agent.interface import AbstractScriptedAgent

_LOGGER = getLogger(__name__)

__all__ = ("LLMAgent", "LLMRedAgent")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_COORDINATION_BLACKBOARD: Dict[str, List[str]] = {}
_BLACKBOARD_MAXLEN = 12

_BLUE_SYSTEM = (
    "You are an autonomous BLUE-team cyber-defence agent in a network simulation. Each timestep you "
    "analyse the observation and choose exactly ONE action_index from the action catalogue - you "
    "cannot do anything outside it.\n\n"
    "Objective: maximise long-term reward = keep critical assets healthy (file integrity, services "
    "responding) AND keep the network usable for legitimate users. Both unnecessary disruptive actions "
    "(shutting nodes, disabling NICs, blocking legitimate traffic) and undetected attacker activity "
    "reduce reward.\n\n"
    "Use the deterministic threat assessment and observation as evidence to choose the action you "
    "judge best. When you see indicators (failed integrity, anomalous events, stopped services, "
    "negative reward trend), a targeted response is available: scan to confirm, then remediate "
    "(repair/restore/restart) or contain the specific source. If teammates are listed, coordinate: "
    "cover different assets, do not duplicate their actions."
)

_RED_SYSTEM = (
    "You are an autonomous RED-team agent (attacker) in a network simulation. Each timestep you "
    "analyse the observation and choose exactly ONE action_index from the action catalogue - you "
    "cannot do anything outside it.\n\n"
    "Objective: achieve the goal in your mission brief (e.g. corrupt or exfiltrate the target "
    "database) while remaining effective and persistent. Recon before committing your payload; if an "
    "attempt fails, adapt and retry from a different angle or wait. Use do-nothing only to bide time "
    "for a better opportunity. Defenders are active and may remediate your effects, so press your "
    "advantage when you have it."
)


def _parse_action_dict(text: str) -> Dict[str, Any]:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        nl = s.find("\n")
        if nl != -1 and s[:nl].strip().lower() in ("json", ""):
            s = s[nl + 1:]
        s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, depth = s.find("{"), 0
    if start != -1:
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(s[start:i + 1])
    raise RuntimeError(f"could not parse a JSON action dict from response: {text[:200]!r}")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class _AbstractLLMAgent(AbstractScriptedAgent):

    decision_log: List[Dict] = Field(default_factory=list)

    class AgentSettingsSchema(AbstractScriptedAgent.AgentSettingsSchema):

        model: str = "openai/gpt-4o-mini"
        temperature: float = 0.0
        max_tokens: int = 1024

        history_length: int = 6
        decide_every: int = 1
        fallback_action: int = 0
        mission_brief: str = ""
        system_prompt: Optional[str] = None
        log_reasoning: bool = True
        coordination_channel: Optional[str] = None
        openrouter_referer: Optional[str] = None
        openrouter_title: Optional[str] = "PrimAITE"

    @abstractmethod
    def _default_system(self) -> str:
        pass

    def _pre_analysis(self, obs: ObsType, timestep: int) -> str:
        return ""

    @cached_property
    def _action_catalogue(self) -> str:
        lines = []
        for idx, (action, options) in sorted(self.action_manager.action_map.items()):
            opt = ", ".join(f"{k}={v}" for k, v in options.items()) if options else "-"
            lines.append(f"  [{idx}] {action} ({opt})")
        return ("ACTION CATALOGUE (choose exactly one index):\n" + "\n".join(lines))

    @cached_property
    def _valid_indices(self) -> List[int]:
        return sorted(self.action_manager.action_map.keys())

    @cached_property
    def _system_text(self) -> str:
        base = self.config.agent_settings.system_prompt or self._default_system()
        parts = [base]
        if self.config.agent_settings.mission_brief:
            parts.append("MISSION BRIEF:\n" + self.config.agent_settings.mission_brief)
        parts.append(self._action_catalogue)
        return "\n\n".join(parts)

    def get_action(self, obs: ObsType, timestep: int = 0) -> Tuple[str, Dict]:
        settings = self.config.agent_settings
        fallback = settings.fallback_action if settings.fallback_action in self.action_manager.action_map else 0

        if settings.decide_every > 1 and (timestep % settings.decide_every):
            return self.action_manager.get_action(fallback)

        reasoning, usage, error, prompt, response = "", None, None, None, None
        try:
            result = self._decide(obs, timestep)
            index = int(result["action_index"])
            reasoning = str(result.get("reasoning", ""))
            usage, prompt, response = result.get("usage"), result.get("prompt"), result.get("response")
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            _LOGGER.warning(f"{self.config.ref}: LLM decision failed ({error}); using fallback.")
            index = fallback

        if index not in self.action_manager.action_map:
            _LOGGER.warning(f"{self.config.ref}: LLM returned invalid action index {index}; using fallback.")
            index = fallback

        action_name, options = self.action_manager.get_action(index)
        self.decision_log.append({
            "timestep": timestep, "agent": self.config.ref, "model": settings.model,
            "action_index": index, "action": action_name, "usage": usage,
            "reasoning": reasoning[:200], "prompt": prompt, "response": response, "error": error,
        })
        if settings.log_reasoning and reasoning:
            self.logger.info(f"t={timestep} chose [{index}] {action_name}: {reasoning}")
        if settings.coordination_channel:
            self._post_to_channel(settings.coordination_channel, timestep, action_name, options, reasoning)
        return action_name, options

    def _decide(self, obs: ObsType, timestep: int) -> Dict[str, Any]:
        user_text = self._build_user_message(obs, timestep)
        return self._query_llm(system=self._system_text, user=user_text)

    def _build_user_message(self, obs: ObsType, timestep: int) -> str:
        obs_json = json.dumps(_jsonable(obs), indent=2, default=str)

        history_lines = []
        for item in self.history[-self.config.agent_settings.history_length :]:
            params = ", ".join(f"{k}={v}" for k, v in item.parameters.items()) if item.parameters else ""
            status = getattr(item.response, "status", "?")
            reward = "" if item.reward is None else f" reward={item.reward:+.3f}"
            history_lines.append(f"  t={item.timestep}: {item.action}({params}) -> {status}{reward}")
        history_text = "\n".join(history_lines) if history_lines else "  (no prior actions)"

        sections = [f"TIMESTEP: {timestep}", f"RECENT ACTIONS (oldest first):\n{history_text}"]

        analysis = self._pre_analysis(obs, timestep)
        if analysis:
            sections.append(analysis)

        channel = self.config.agent_settings.coordination_channel
        if channel:
            peers = self._peer_notes(channel)
            sections.append("TEAMMATE ACTIONS (other agents on your channel):\n" + ("\n".join(peers) if peers else "  (none yet)"))

        sections.append(f"CURRENT OBSERVATION (CAOS observation space):\n{obs_json}")
        sections.append(self._response_instruction())
        return "\n\n".join(sections)

    def _response_instruction(self) -> str:
        return ('Respond with ONLY a JSON object in this exact format (no prose, no code fence):\n'
                '{"action": <action_index>, "reason": "<one concise sentence>"}')

    def _post_to_channel(self, channel: str, timestep: int, action: str, options: Dict, reasoning: str) -> None:
        target = options.get("node_name") or options.get("application_name") or "-"
        note = f"[{self.config.ref}] t={timestep}: {action}({target}) - {reasoning[:80]}"
        buf = _COORDINATION_BLACKBOARD.setdefault(channel, [])
        buf.append(note)
        del buf[:-_BLACKBOARD_MAXLEN]

    def _peer_notes(self, channel: str) -> List[str]:
        prefix = f"[{self.config.ref}]"
        return [n for n in _COORDINATION_BLACKBOARD.get(channel, []) if not n.startswith(prefix)]

    def _chat(self, system: str, user: str) -> Tuple[str, Optional[Dict[str, int]]]:
        client = self._client
        settings = self.config.agent_settings

        headers = {}
        if settings.openrouter_referer:
            headers["HTTP-Referer"] = settings.openrouter_referer
        if settings.openrouter_title:
            headers["X-Title"] = settings.openrouter_title

        response = client.chat.completions.create(
            model=settings.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            extra_headers=headers or None,
        )
        content = response.choices[0].message.content or ""
        u = getattr(response, "usage", None)
        usage = {k: int(getattr(u, k, 0) or 0)
                 for k in ("prompt_tokens", "completion_tokens", "total_tokens")} if u else None
        return content, usage

    def _query_llm(self, system: str, user: str) -> Dict[str, Any]:
        content, usage = self._chat(system, user)
        args = _parse_action_dict(content)
        idx = args.get("action", args.get("action_index"))
        if idx is None:
            raise RuntimeError(f"LLM response had no 'action' field: {content[:200]!r}")
        reason = str(args.get("reason", args.get("reasoning", "")))
        return {"action_index": int(idx), "reasoning": reason,
                "usage": usage, "prompt": {"system": system, "user": user}, "response": content}

    @cached_property
    def _client(self):
        import openai

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set in the environment")
        return openai.OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)


class LLMAgent(_AbstractLLMAgent, discriminator="llm-agent"):

    config: "LLMAgent.ConfigSchema" = Field(default_factory=lambda: LLMAgent.ConfigSchema())

    class ConfigSchema(AbstractScriptedAgent.ConfigSchema):

        type: str = "llm-agent"
        agent_settings: "_AbstractLLMAgent.AgentSettingsSchema" = Field(
            default_factory=lambda: _AbstractLLMAgent.AgentSettingsSchema()
        )

    def _default_system(self) -> str:
        return _BLUE_SYSTEM

    def _pre_analysis(self, obs: ObsType, timestep: int) -> str:
        recent = self.history[-self.config.agent_settings.history_length :]
        if not recent:
            return ""
        reward_trend = sum(h.reward for h in recent if h.reward is not None)
        failures = [h for h in recent if getattr(h.response, "status", "") == "failure"]
        notes = [f"reward over last {len(recent)} step(s): {reward_trend:+.2f}"]
        if reward_trend < 0:
            notes.append("NEGATIVE trend - an asset is likely compromised or unavailable; investigate/remediate")
        if failures:
            notes.append(f"{len(failures)} recent action(s) FAILED (target may be down/unreachable)")
        return "DETERMINISTIC THREAT ASSESSMENT (tool output): " + "; ".join(notes)


class LLMRedAgent(_AbstractLLMAgent, discriminator="llm-red-agent"):

    config: "LLMRedAgent.ConfigSchema" = Field(default_factory=lambda: LLMRedAgent.ConfigSchema())

    class ConfigSchema(AbstractScriptedAgent.ConfigSchema):

        type: str = "llm-red-agent"
        agent_settings: "_AbstractLLMAgent.AgentSettingsSchema" = Field(
            default_factory=lambda: _AbstractLLMAgent.AgentSettingsSchema()
        )

    def _default_system(self) -> str:
        return _RED_SYSTEM

    def _pre_analysis(self, obs: ObsType, timestep: int) -> str:
        recent = self.history[-self.config.agent_settings.history_length :]
        if not recent:
            return ""
        successes = sum(1 for h in recent if getattr(h.response, "status", "") == "success" and h.action != "do-nothing")
        fails = sum(1 for h in recent if getattr(h.response, "status", "") == "failure")
        return (
            "DETERMINISTIC RECON (tool output): "
            f"recent offensive actions - {successes} succeeded, {fails} failed over last {len(recent)} step(s)."
        )
