import os
import yaml


ENV_VAR_MODEL_BACKEND = "CAGE4_MODEL_BACKEND"
ENV_VAR_MODEL_NAME = "CAGE4_MODEL_NAME"


class ConfigLoader:
    @staticmethod
    def load_model_configuration(config_path) -> dict:
        with open(config_path, 'r') as file:
            cfg = yaml.safe_load(file) or {}

        backend_override = os.getenv(ENV_VAR_MODEL_BACKEND, "").strip()
        model_name_override = os.getenv(ENV_VAR_MODEL_NAME, "").strip()

        if backend_override:
            cfg["backend"] = backend_override
        if model_name_override:
            cfg["model_name"] = model_name_override

        return cfg
    
    @staticmethod
    def load_prompts(prompts_path) -> dict:
        with open(prompts_path, 'r') as file:
            parsed = yaml.safe_load(file)
        prompts = parsed.get('prompts', [])
        return prompts
