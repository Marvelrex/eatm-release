from __future__ import annotations

from typing import Dict, List
import os
import time

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
import weave

from CybORG.Agents.LLMAgents.llm_adapter.backend.model_backend import ModelBackend
from CybORG.Agents.LLMAgents.llm_adapter.utils.logger import Logger
from CybORG.Agents.LLMAgents.llm_adapter.utils.token_tracker import TokenUsageTracker


def normalize_openrouter_model_name(model_name: str) -> str:
    normalized = (model_name or "").strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "gpt41-nano": "openai/gpt-4.1-nano",
        "gpt-41-nano": "openai/gpt-4.1-nano",
        "gpt-4.1-nano": "openai/gpt-4.1-nano",
    }
    return aliases.get(normalized, model_name)


class OpenRouterBackend(ModelBackend):

    def __init__(self, hyperparams: dict, api_key: str):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter backend")

        self.base_url = hyperparams.get("base_url", os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))

        raw_model_name = str(hyperparams.get("model_name", "")).strip()
        self.model_name = normalize_openrouter_model_name(raw_model_name)

        generate_cfg = hyperparams.get("generate", {})
        self.temperature = float(generate_cfg.get("temperature", 0.9))
        self.max_tokens = int(generate_cfg.get("max_new_tokens", 128))
        self.request_timeout_s = float(
            generate_cfg.get(
                "request_timeout_s",
                os.getenv("CAGE4_OPENROUTER_TIMEOUT_S", 45),
            )
        )
        self.max_api_retries = max(
            0,
            int(
                generate_cfg.get(
                    "api_retries",
                    os.getenv("CAGE4_OPENROUTER_API_RETRIES", 2),
                )
            ),
        )
        self.retry_backoff_s = max(
            0.0,
            float(
                generate_cfg.get(
                    "retry_backoff_s",
                    os.getenv("CAGE4_OPENROUTER_RETRY_BACKOFF_S", 1.5),
                )
            ),
        )
        self.openai_client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.request_timeout_s,
            max_retries=self.max_api_retries,
        )

        self.headers = {
            "HTTP-Referer": hyperparams.get(
                "http_referer",
                os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/cage-paper/cage-4"),
            ),
            "X-Title": hyperparams.get(
                "x_title",
                os.getenv("OPENROUTER_X_TITLE", "CAGE Project"),
            ),
        }
        self.extra_body = dict(hyperparams.get("extra_body", {}))

    @weave.op
    def generate(self, messages: List[Dict[str, str]]) -> str:
        formatted_prompt = self._format_messages_history(messages)
        attempts = self.max_api_retries + 1

        for attempt in range(attempts):
            try:
                raw_response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": formatted_prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_headers=self.headers,
                    extra_body=self.extra_body,
                )
                TokenUsageTracker.record(getattr(raw_response, "usage", None))
                response = raw_response.choices[0].message.content
                if response is None:
                    finish_reason = getattr(raw_response.choices[0], "finish_reason", "unknown")
                    Logger.error(
                        f"OpenRouter returned empty content (finish_reason={finish_reason}, "
                        f"model={self.model_name}). Likely reasoning-token budget exhausted "
                        f"(try increasing max_new_tokens). Returning Sleep fallback."
                    )
                    return '{"action": "Sleep", "reason": "empty LLM content"}'
                Logger.info(f"Generated response: {response}")
                return self._format_response(response)
            except (APITimeoutError, APIConnectionError, RateLimitError, APIError) as e:
                is_last = attempt >= (attempts - 1)
                if not is_last:
                    delay = self.retry_backoff_s * (2 ** attempt)
                    Logger.error(
                        f"OpenRouter request failed ({type(e).__name__}): {e}. "
                        f"Retrying in {delay:.1f}s ({attempt + 1}/{attempts - 1})"
                    )
                    time.sleep(delay)
                    continue
                Logger.error(
                    f"OpenRouter request failed after {attempts} attempts ({type(e).__name__}): {e}. "
                    "Returning Sleep fallback response."
                )
                return '{"action": "Sleep", "reason": "OpenRouter timeout/connection fallback"}'
            except Exception as e:
                Logger.error(
                    f"Unexpected OpenRouter backend error ({type(e).__name__}): {e}. "
                    "Returning Sleep fallback response."
                )
                return '{"action": "Sleep", "reason": "OpenRouter unexpected error fallback"}'

        return '{"action": "Sleep", "reason": "OpenRouter exhausted retries fallback"}'
