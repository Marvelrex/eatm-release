from __future__ import annotations

from typing import Dict, List

_BOARD: Dict[str, Dict[int, Dict[str, int]]] = {}

NUM_AGENTS = 5


def reset(channel: str) -> None:
    _BOARD.pop(channel, None)


def post(channel: str, agent_index: int, detected: bool, level: int = 0, waiting: bool = False) -> None:
    _BOARD.setdefault(channel, {})[agent_index] = {
        "detected": 1 if detected else 0,
        "level": max(0, min(3, int(level))),
        "waiting": 1 if waiting else 0,
    }


def detection_vector(channel: str, num_agents: int = NUM_AGENTS) -> List[int]:
    board = _BOARD.get(channel, {})
    return [board.get(i, {}).get("detected", 0) for i in range(num_agents)]


def message_byte(channel: str, agent_index: int) -> List[int]:
    board = _BOARD.get(channel, {})
    rec = board.get(agent_index, {"detected": 0, "level": 0, "waiting": 0})
    lvl = rec["level"]
    return detection_vector(channel) + [(lvl >> 1) & 1, lvl & 1, rec["waiting"]]
