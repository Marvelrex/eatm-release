# EATM artifact: environments, harness, recorded runs and learned trees

Anonymous artifact accompanying a double-blind submission. It contains the two evaluation
environments with their harnesses, the baseline defenders, the recorded runs (per-episode
decision logs) and the learned action trees behind the reported PrimAITE numbers. The
defender implementation itself (the action-tree memory and its update rule) is withheld during
review and will be released on acceptance; what is here lets a reader check the environments,
the protocol, the prompts, and the decisions the defender actually made.

```
cc4/         CAGE Challenge 4 (CybORG) fork: evaluation harness, LLM-baseline defender, RL teammates
primaite/    PrimAITE fork with the CC4 port (multi- and single-network topologies) and its harness
artifacts/   recorded runs: reward curves, one full decision log per arm, learned trees (PrimAITE)
```

No API keys are included; every LLM-backed run reads `OPENROUTER_API_KEY` from the
environment or from a local `.env` you create.

## artifacts/ — recorded runs

`artifacts/primaite/<seed>/<arm>/` mirrors the run directories of the PrimAITE campaign
(three seed rounds: `seed2025`, `seed42`, `seed818`). Arm names: `bm_<topo>_de10_<model>` are
the EATM arms (decide every 10 steps), `bm_<topo>_de10_acd_<model>` the unguided-LLM (ACD) arms,
`bm_baseline_<topo>` the tree-only arm, `bm_<topo>_sleep` the do-nothing floor,
`bm_<topo>_de10_info_qwen3.7` the information-ablation arm.

| File | Content |
|---|---|
| `curve_<model>.json` | per-episode summary used for every table and figure: `reward` (native CC4 blue reward), `hq_reward` / `hq_restores` / `hq_covered` for the multi-zone agent's zones, `llm_answered`, `fell_back`, `cost_usd`; all three seeds |
| `<model>_ep<N>.json` | the **last episode** of the arm in `seed2025`, in full: `meta` (settings, usage, per-agent and per-zone rewards), `trajectory` (per step: red action, corrupted DBs, reward), `blue_decision_logs[agent]` (one record per step; records on decision steps carry the ranked `menu`, the rendered `prompt` (`system`, `user`), the model `response`, the parsed `choice_num`, the `chosen_arm`, the concrete `action`, `threat`/`pressure` context, teammate `comms` and token `usage`), `red_decision_logs`, and `eatm_action_graph` (the tree snapshot each defender saw at the end of the episode) |
| `tree_<model>.json` | the learned action tree after the arm's last episode: `action_nodes[agent][family][zone][context]` and `sleep_nodes[agent][context]` each holding `visit_count`, `mean_advantage`, `variance`, `recent_advantage`, `mean_time_to_effect`, `opportunity_cost`; plus `baseline_state` and `metadata` |
| `seed_summary.md` | the cross-seed roll-up (model mean minus same-round tree-only mean) |

Paths inside the logs are relative (`results/...`); the runs were produced by
`primaite/run_cage4.py` through `primaite/tools/llm_benchmark.py`.

## cc4/ — CAGE Challenge 4 (CybORG)

Fork of the public `cage-challenge-4` with the *llms-are-acd* extensions (LLM defender, Cybermonics
RL teammates); see `cc4/THIRD_PARTY_NOTICES.md`. What is ours here: the evaluation harness
(`CybORG/Evaluation/evaluation.py`: per-step and per-episode logging, resume, cost accounting,
multiple runs), the single-step combined wrapper in `CybORG/Evaluation/Cybermonics/submission.py`,
the OpenRouter backend in `CybORG/Agents/LLMAgents/llm_adapter/backend/openrouter.py`, the
instrumented FSM red, and the prompts under `CybORG/Agents/LLMAgents/config/prompts/`.

```bash
cd cc4
python -m venv .venv && . .venv/bin/activate        # Python 3.10 or 3.11
pip install -r Requirements.txt && pip install -e .
export OPENROUTER_API_KEY=...
# Scenario B roster: one LLM defender at blue_agent_4, four fixed Cybermonics RL teammates
CAGE4_AGENT_MODE=llm_baseline \
python -m CybORG.Evaluation.evaluation CybORG/Evaluation/Cybermonics \
  --model-config CybORG/Agents/LLMAgents/config/model/gpt-5.4-openrouter.yml \
  --max-eps 10 --episode-steps 500 --output logs/outputs/run1
# Scenario A roster (all five defenders are LLM): add CAGE4_ALL_LLM_AGENTS=1
# Do-nothing teammates: CAGE4_SLEEP_RL_AGENTS=1 ; no model call at all: --model-config .../dummy.yml
```

`--model-config` picks the backbone (one YAML per model under `config/model/`); `--model-name` /
`--model-backend` override single fields. Every run writes `episodes.jsonl`, `step_log.jsonl`,
`summary.json`, `actions.txt` and `llm_calls.log` under `--output`. `scripts/` holds the plot
helpers that read those files.

## primaite/ — PrimAITE port of the CC4 game

Fork of PrimAITE with the CC4 game rebuilt on top of it. The two topologies
(`multi` = seven zones over five defenders, `single` = five zones under one defender) are
`src/primaite/config/_package_data/cage4_fixed_topology.yaml` and `cage4_single_network.yaml`.

| Path | Role |
|---|---|
| `src/primaite/game/agent/scripted_agents/cage4/` | CC4 rules on PrimAITE: scripted blue, unguided-LLM blue, the CybORG FSM red ported verbatim, comms, action-family table |
| `src/primaite/game/agent/scripted_agents/llm_agent.py` | OpenRouter-backed LLM agent |
| `run_cage4.py` | assembles the game (reward machine, mission phases, restore cost) and runs one episode, writing the JSON log format described above |
| `tools/llm_benchmark.py` | multi-episode driver for the ACD arms (`--blue llm`) and the do-nothing floor (`--blue sleep`) |
| `tools/plot_*.py`, `tools/summarize_seeds.py`, `tools/verify_*.py` | figures, per-seed roll-up, number checks over the artifact directories |

```powershell
cd primaite
python -m venv venv; .\venv\Scripts\Activate.ps1   # Python 3.10 to 3.12
pip install -e ".[llm]"                            # add ",dev" for pytest
$env:OPENROUTER_API_KEY = "..."
python run_cage4.py --blue scripted --topo multi --steps 500          # heuristic defenders, free
python run_cage4.py --blue sleep --topo single --steps 500            # do-nothing floor, free
python tools\llm_benchmark.py --blue llm --models openai/gpt-5.6-terra --episodes 10 --steps 500 --decide-every 10 --topo multi
```

Only the scripted red agents are seeded (`--seed`, default 2025); green traffic and LLM sampling
(temperature 0.7) are not, so individual episodes are not bit-reproducible.
