import os


SUB_NAME = "LLM and RL Agent"
SUB_TEAM = "Anonymous"
SUB_TECHNIQUE = "LLM+RL"

BLUE_AGENT_NAME = "blue_agent_4"
ALL_LLM_AGENTS = False
NO_LLM_AGENTS = False

CONFIG_MODEL_PATH = "config/model/dummy.yml"
STRATEGY_PROMPT_PATH = "config/prompts/acd2025/base.yml"

ENV_VAR_MODEL = "CAGE4_MODEL_CONFIG"
ENV_VAR_PROMPT = "CAGE4_PROMPTS_CONFIG"
ENV_VAR_MODEL_BACKEND = "CAGE4_MODEL_BACKEND"
ENV_VAR_MODEL_NAME = "CAGE4_MODEL_NAME"

INCLUDE_PROMPT_CAGE4_RULES = False
INCLUDE_PROMPT_COMMVECTOR_RULES = False
COMMVECTOR_RULES_PROMPT_PATH = "config/prompts/env_rules/commvector_rules.yml"
CAGE4_RULES_PROMPT_PATH = "config/prompts/env_rules/cage4_rules_v2.yml"

DEBUG_MODE = True
TOTAL_STEPS_PROGRESS_BAR = 1000


AGENT_MODE_LLM_BASELINE = "llm_baseline"
ENV_VAR_AGENT_MODE = "CAGE4_AGENT_MODE"


_ENV_BOOL_TRUE = {"1", "true", "yes", "y", "on"}


def _get_env_bool(var_name: str, default: bool) -> bool:
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in _ENV_BOOL_TRUE


def _get_env_int(var_name: str, default: int) -> int:
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _get_env_float(var_name: str, default: float) -> float:
    raw_value = os.getenv(var_name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _parse_agent_mode(raw_mode: str | None) -> str:
    candidate = (raw_mode or AGENT_MODE_LLM_BASELINE).strip().lower()
    if candidate in {AGENT_MODE_LLM_BASELINE}:
        return candidate
    return AGENT_MODE_LLM_BASELINE


AGENT_MODE = _parse_agent_mode(os.getenv(ENV_VAR_AGENT_MODE, AGENT_MODE_LLM_BASELINE))

BLUE_AGENT_NAME = os.getenv("CAGE4_BLUE_AGENT_NAME", BLUE_AGENT_NAME)
ALL_LLM_AGENTS = _get_env_bool("CAGE4_ALL_LLM_AGENTS", ALL_LLM_AGENTS)
NO_LLM_AGENTS = _get_env_bool("CAGE4_NO_LLM_AGENTS", NO_LLM_AGENTS)
SLEEP_RL_AGENTS = _get_env_bool("CAGE4_SLEEP_RL_AGENTS", False)
INVALID_RL_AGENTS = _get_env_bool("CAGE4_INVALID_RL_AGENTS", False)
RANDOM_RL_AGENTS = _get_env_bool("CAGE4_RANDOM_RL_AGENTS", False)
