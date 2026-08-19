# Third-party code in this directory

- `CybORG/` is the TTCP CAGE Challenge 4 simulator (Cyber Operations Research Gym), MIT-licensed; see `LICENSE.txt` and `README.md`.
- `CybORG/Agents/LLMAgents/llm_adapter/`, `llm_policy.py`, `llm_agent.py`, `obs_wrapper.py`, `comm_vector/`, most of `config/`,
  `CybORG/Agents/CybermonicAgents/`, `CybORG/Agents/Wrappers/CybermonicWrappers/` (`graph_wrapper.py` extended with
  `build_graph_observations` / `build_reset_observations` so the combined wrapper steps the simulator once per turn), `CybORG/Evaluation/Cybermonics/` and
  `CybORG/Evaluation/llamagym/` are taken from the public *llms-are-acd* baseline (Apache License 2.0), partly modified.
- `CybORG/Evaluation/Cybermonics/weights/*.pt` and the Cybermonics agent code are the public CAGE-4 Cybermonics submission,
  redistributed as shipped by *llms-are-acd*.
- `CybORG/Shared/MetricsCallback.py` is a no-op stand-in for a hook the baseline's submission expects.
- `CybORG/Evaluation/evaluation.py` (as modified), the combined wrapper in `CybORG/Evaluation/Cybermonics/submission.py`,
  `CybORG/Agents/LLMAgents/llm_adapter/backend/openrouter.py`, `agent_factory.py`, the prompt files under `config/prompts/acd2025/`
  and `scripts/` are new in this work.
