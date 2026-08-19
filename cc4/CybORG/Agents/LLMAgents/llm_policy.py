import os

from ray.rllib.policy.policy import Policy
import tqdm
import wandb

from CybORG.Simulator.Actions import (
    Sleep, Remove, Restore, DeployDecoy, Analyse
)
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
    BlockTrafficZone, AllowTrafficZone
)
from CybORG.Agents.LLMAgents.llm_adapter.config_loader import ConfigLoader
from CybORG.Agents.LLMAgents.llm_adapter.utils.constants import CAGE4_SUBNETS
from CybORG.Agents.LLMAgents.llm_adapter.model_manager import ModelManager

from CybORG.Agents.LLMAgents.llm_adapter.utils.logger import Logger
from CybORG.Agents.LLMAgents.llm_adapter import obs_formatter

from CybORG.Agents.LLMAgents.config.config_vars import (
    CONFIG_MODEL_PATH, CAGE4_RULES_PROMPT_PATH, COMMVECTOR_RULES_PROMPT_PATH, STRATEGY_PROMPT_PATH,
    ENV_VAR_MODEL, ENV_VAR_PROMPT, DEBUG_MODE, BLUE_AGENT_NAME, TOTAL_STEPS_PROGRESS_BAR, 
    INCLUDE_PROMPT_CAGE4_RULES, INCLUDE_PROMPT_COMMVECTOR_RULES
)

base_path = os.path.dirname(__file__)
default_config_path = os.path.join(base_path, CONFIG_MODEL_PATH)
cage4_rules_path = os.path.join(base_path, CAGE4_RULES_PROMPT_PATH)
commvector_rules_path = os.path.join(base_path, COMMVECTOR_RULES_PROMPT_PATH)
default_prompts_path = os.path.join(base_path, STRATEGY_PROMPT_PATH)
model_config = os.environ.get(ENV_VAR_MODEL, default_config_path)
strategy_prompt_path = os.environ.get(ENV_VAR_PROMPT, default_prompts_path)

Logger.set_debug_mode(DEBUG_MODE)

_VALID_SUBNET_NAMES = frozenset(CAGE4_SUBNETS)
_ZONE_TO_SUBNET = {
    "restricted_zone_a": "restricted_zone_a_subnet",
    "operational_zone_a": "operational_zone_a_subnet",
    "restricted_zone_b": "restricted_zone_b_subnet",
    "operational_zone_b": "operational_zone_b_subnet",
    "public_access_zone": "public_access_zone_subnet",
    "admin_network": "admin_network_subnet",
    "office_network": "office_network_subnet",
}


def normalize_subnet_name(name: str | None) -> str | None:
    if name is None:
        return None
    if name in _VALID_SUBNET_NAMES:
        return name
    if name in _ZONE_TO_SUBNET:
        return _ZONE_TO_SUBNET[name]
    return name


_llm_calls_file = None
_llm_calls_dir = None

def _get_llm_calls_file():
    global _llm_calls_file, _llm_calls_dir
    run_log_dir = os.environ.get("CAGE4_RUN_LOG_DIR")
    if run_log_dir and run_log_dir != _llm_calls_dir:
        if _llm_calls_file:
            _llm_calls_file.close()
        path = os.path.join(run_log_dir, "llm_calls.log")
        _llm_calls_file = open(path, "a")
        _llm_calls_dir = run_log_dir
    return _llm_calls_file

class LLMDefenderPolicy(Policy):
    def __init__(self, observation_space, action_space, config={}):
        super().__init__(observation_space, action_space, {})
        self.name = config.get("agent_name", None) 
        self.subnet = (CAGE4_SUBNETS)[int(self.name[-1])]
        self.model_manager = ModelManager(ConfigLoader.load_model_configuration(model_config))
        self.prompts = []
        self.cage4_rules_prompt = []
        self.commvector_rules_prompt = []
        self.current_episode_messages = []
        self.last_action: str = None
        self._blocked_subnets: set[str] = set()

        self._load_all_prompts()
        
        self.step = 0
        self.progress_bar = tqdm.tqdm(total=TOTAL_STEPS_PROGRESS_BAR, desc="Steps")

    def end_episode(self):
        self._blocked_subnets.clear()

    def _safe_wandb_log(self, payload: dict):
        try:
            run = getattr(wandb, "run", None)
            if run is None:
                return
            step = getattr(run, "step", None)
            if step is None:
                wandb.log(payload)
            else:
                wandb.log(payload, step=step)
        except Exception as e:
            Logger.debug(f"Skipping wandb.log due to logging error: {e}")
    
    def _load_all_prompts(self):
        strategy_prompt = ConfigLoader.load_prompts(strategy_prompt_path)
        
        self.prompts = self._init_strategy_prompt(strategy_prompt)
        if INCLUDE_PROMPT_CAGE4_RULES:
            self.cage4_rules_prompt = ConfigLoader.load_prompts(cage4_rules_path)
        if INCLUDE_PROMPT_COMMVECTOR_RULES:
            self.commvector_rules_prompt = ConfigLoader.load_prompts(commvector_rules_path)

    def _init_strategy_prompt(self, prompt: list[dict]):
        msg = f"AGENT_NAME: {self.name}\n"
        if INCLUDE_PROMPT_CAGE4_RULES:
            msg += f"You will find the rules of the environment under `# ENVIRONMENT RULES`. "
        if INCLUDE_PROMPT_COMMVECTOR_RULES:
            msg += f"You will find the structure of the communication vectors under `# COMMVECTOR FORMAT`. "
        msg += "Prioritize this `# DESCRIPTION` section to generate your answer. Remember your `AGENT_NAME` is the BLUE AGENT you are. \n"
        msg += "IMPORTANT: Return ONLY ONE action from `## AVAILABLE ACTIONS`\n"
        msg += "# INSTRUCTIONS\n"
        
        prompt[0]["content"] = msg + prompt[0].get("content")
        return prompt
        
    def extract_action(self, response: str):

        actions = [
            "action_invalid", "error_action_extraction", "remove", "restore",
            "blocktrafficzone", "allowtrafficzone", "deploydecoy", "analyse", "sleep"
        ]

        action_log = {f"{self.name}_{action}": 0 for action in actions}
        
        try:
            Logger.debug(f"Processing response: {response}")
            lower_response = response.lower()

            if "action\":" not in lower_response:
                Logger.debug("No 'action' key in response, defaulting to Sleep.")
                self.last_action = "Sleep"
                return Sleep()

            raw_action = lower_response.split("action\":")[1].split(",")[0].strip().strip('"')
            response_action = raw_action.split("}")[0].strip().strip('"')
            lower_response_action = response_action.lower()

            hostname = None
            target_subnet = None

            if "host:" in lower_response_action:
                hostname = lower_response_action.split("host:")[1].split()[0].strip().strip('"').rstrip("}")
                Logger.debug(f"Extracted hostname: {hostname}")

            if "subnet:" in lower_response_action:
                target_subnet = lower_response_action.split("subnet:")[1].split()[0].strip().strip('"').rstrip("}")
                target_subnet = normalize_subnet_name(target_subnet)
                Logger.debug(f"Extracted subnet: {target_subnet}")

            if "remove" in lower_response_action and hostname:
                action_log[f"{self.name}_remove"] = 1
                self._safe_wandb_log(action_log)
                Logger.debug(f"Creating Remove action with hostname: {hostname}")
                self.last_action = f"Remove host:{hostname}"
                return Remove(session=0, agent=self.name, hostname=hostname)

            elif "restore" in lower_response_action and hostname:
                action_log[f"{self.name}_restore"] = 1
                self._safe_wandb_log(action_log)
                Logger.debug(f"Creating Restore action with hostname: {hostname}")
                self.last_action = f"Restore host:{hostname}"
                return Restore(session=0, agent=self.name, hostname=hostname)

            elif "blocktrafficzone" in lower_response_action and target_subnet:
                if target_subnet in self._blocked_subnets:
                    Logger.debug(f"BlockTrafficZone skipped: {target_subnet} already blocked")
                    self.last_action = "Sleep"
                    return Sleep()
                action_log[f"{self.name}_blocktrafficzone"] = 1
                self._safe_wandb_log(action_log)
                Logger.debug(f"Creating BlockTrafficZone action with subnet: {target_subnet}")
                self.last_action = f"BlockTrafficZone subnet:{target_subnet}"
                self._blocked_subnets.add(target_subnet)
                return BlockTrafficZone(session=0, agent=self.name, from_subnet=self.subnet, to_subnet=target_subnet)

            elif "allowtrafficzone" in lower_response_action and target_subnet:
                if target_subnet not in self._blocked_subnets:
                    Logger.debug(f"AllowTrafficZone skipped: {target_subnet} not blocked")
                    self.last_action = "Sleep"
                    return Sleep()
                action_log[f"{self.name}_allowtrafficzone"] = 1
                self._safe_wandb_log(action_log)
                Logger.debug(f"Creating AllowTrafficZone action with subnet: {target_subnet}")
                self.last_action = f"AllowTrafficZone subnet:{target_subnet}"
                self._blocked_subnets.discard(target_subnet)
                return AllowTrafficZone(session=0, agent=self.name, from_subnet=self.subnet, to_subnet=target_subnet)

            elif "deploydecoy" in lower_response_action and hostname:
                action_log[f"{self.name}_deploydecoy"] = 1
                self._safe_wandb_log(action_log)
                Logger.debug(f"Creating DeployDecoy action with hostname: {hostname}")
                self.last_action = f"DeployDecoy host:{hostname}"
                return DeployDecoy(session=0, agent=self.name, hostname=hostname)

            elif "analyse" in lower_response_action and hostname:
                action_log[f"{self.name}_analyse"] = 1
                self._safe_wandb_log(action_log)
                Logger.debug(f"Creating Analyse action with hostname: {hostname}")
                self.last_action = f"Analyse host:{hostname}"
                return Analyse(session=0, agent=self.name, hostname=hostname)
            elif "sleep" in lower_response_action:
                self.last_action = "Sleep"
                action_log[f"{self.name}_sleep"] = 1
                self._safe_wandb_log(action_log)
                Logger.debug("Creating Sleep action")
                return Sleep()
            action_log[f"{self.name}_action_invalid"] = 1
            self._safe_wandb_log(action_log)
            Logger.debug("No valid action pattern found or missing required parameters, defaulting to Sleep")
            self.last_action = "Sleep"
            return Sleep()

        except Exception as e:
            Logger.error(f"Error extracting action: {e}. Defaulting to Sleep.")
            self.last_action = "Sleep"
            action_log[f"{self.name}_error_action_extraction"] = 1
            self._safe_wandb_log(action_log)

            return Sleep()

    def generate_response(self, messages):
        response = self.model_manager.generate_response(messages)

        return response

    def _safe_generate_response(self, messages):
        try:
            return self.generate_response(messages)
        except Exception as e:
            Logger.error(
                f"LLM backend request failed ({type(e).__name__}): {e}. "
                "Falling back to Sleep response."
            )
            return '{"action": "Sleep", "reason": "LLM backend exception fallback"}'

    def compute_single_action(self, obs=None, prev_action=None, **kwargs):
        Logger.new_episode()
        obs_message = obs_formatter.format_observation(obs, self.last_action, self.name)
        self.current_episode_messages = []
        response = ""
        
        if self.prompts:
            self.current_episode_messages.append(self.prompts[0])
            if INCLUDE_PROMPT_CAGE4_RULES:
                self.current_episode_messages.append(self.cage4_rules_prompt[0])
            if INCLUDE_PROMPT_COMMVECTOR_RULES:
                self.current_episode_messages.append(self.commvector_rules_prompt[0])
            self.current_episode_messages.append({"role": "user", "content": obs_message})
            response = self._safe_generate_response(self.current_episode_messages)

        if len(self.prompts) > 1 and not INCLUDE_PROMPT_CAGE4_RULES and not INCLUDE_PROMPT_COMMVECTOR_RULES:
            Logger.info("Continuing conversation with additional prompts")
            for prompt in self.prompts[1:]:
                assistant_response = {"role": "assistant", "content": response}
                self.current_episode_messages.append(assistant_response)
                self.current_episode_messages.append(prompt)
                response = self._safe_generate_response(self.current_episode_messages)

        assistant_response = {"role": "assistant", "content": response}
        self.current_episode_messages.append(assistant_response)
        
        Logger.success("Conversation:")
        for msg in self.current_episode_messages:
            if INCLUDE_PROMPT_CAGE4_RULES:
                if msg == self.cage4_rules_prompt[0]:
                    Logger.conversation_message(msg["role"], "[CAGE4_RULES PROMPT]")
                    continue
            if INCLUDE_PROMPT_COMMVECTOR_RULES:
                if msg == self.commvector_rules_prompt[0]:
                    Logger.conversation_message(msg["role"], "[COMMUNICATION VECTOR RULES PROMPT]")
                    continue
            Logger.conversation_message(msg["role"], msg["content"])
        Logger.success("Conversation complete")
        Logger.success(f"Final Assistant response:\n[ASSISTANT] {response}")

        try:
            action = self.extract_action(response)
        except Exception as e:
            Logger.error(f"Error extracting action, defaulting to Sleep: {e}")
            action = Sleep()

        Logger.success(f"Final action: {action.__class__.__name__}")

        f = _get_llm_calls_file()
        if f:
            try:
                f.write(f"{'='*80}\n")
                f.write(f"STEP {self.step} | AGENT {self.name}\n")
                f.write(f"{'='*80}\n")
                for msg in self.current_episode_messages:
                    role = msg["role"].upper()
                    content = msg["content"]
                    f.write(f"\n--- {role} ---\n{content}\n")
                f.write(f"\n>>> ACTION: {action.__class__.__name__}\n\n")
                f.flush()
            except Exception:
                pass

        self.step += 1
        self.progress_bar.update(1)

        return action, [], {}

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None,
                       prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        state_out = []
        info_out = {}

        print(f"\nProcessing batch of {len(obs_batch)} observations")

        for i, obs in enumerate(obs_batch):
            action, state, info = self.compute_single_action(
                obs,
                state_batches[i] if state_batches else None,
                prev_action_batch[i] if prev_action_batch is not None else None,
                prev_reward_batch[i] if prev_reward_batch is not None else None,
                info_batch[i] if info_batch else None
            )
            actions.append(action)
            if state:
                state_out.append(state)
            if info:
                for k, v in info.items():
                    if k not in info_out:
                        info_out[k] = []
                    info_out[k].append(v)

        print(f"Batch processing complete. Actions generated: {len(actions)}")
        return actions, state_out, info_out


    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass
