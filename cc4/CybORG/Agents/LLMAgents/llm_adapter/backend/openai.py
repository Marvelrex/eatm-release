from openai import OpenAI
from typing import List, Dict
from CybORG.Agents.LLMAgents.llm_adapter.backend.model_backend import ModelBackend
import weave

from CybORG.Agents.LLMAgents.llm_adapter.utils.logger import Logger
from CybORG.Agents.LLMAgents.llm_adapter.utils.token_tracker import TokenUsageTracker
class OpenAIBackend(ModelBackend):

    def __init__(self, hyperparams: dict, api_key: str):
        self.openai_client = OpenAI(api_key=api_key) 
        self.model_name = hyperparams["model_name"].lower()
        self.temperature = hyperparams["generate"]["temperature"]
        self.max_tokens = hyperparams["generate"]["max_new_tokens"]

    @weave.op
    def generate(self, messages: List[Dict[str, str]]) -> str:
        formatted_prompt = self._format_messages_history(messages)
        raw_response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": formatted_prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        TokenUsageTracker.record(getattr(raw_response, "usage", None))
        response = raw_response.choices[0].message.content
        return self._format_response(response)


class NewOpenAIBackend(ModelBackend):

    def __init__(self, hyperparams: dict, api_key: str):
        self.openai_client = OpenAI(api_key=api_key)
        self.model_name = hyperparams["model_name"].lower()
        self.temperature = hyperparams["generate"]["temperature"]
        self.max_tokens = hyperparams["generate"]["max_new_tokens"]

    @weave.op
    def generate(self, messages: List[Dict[str, str]]) -> str:
        formatted_prompt = self._format_messages_history(messages)
        raw_response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": formatted_prompt}]
        )
        TokenUsageTracker.record(getattr(raw_response, "usage", None))
        response = raw_response.choices[0].message.content
        Logger.info(f"Generated response: {response}")
        return self._format_response(response)
