import inspect
import logging
import time
from statistics import mean, stdev
import ipaddress

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from datetime import datetime

import json

import sys
import os


MODEL_PRESETS: dict[str, tuple[str, str]] = {
    "sonnet":  ("openrouter", "anthropic/claude-sonnet-4.5"),
    "opus":    ("openrouter", "anthropic/claude-opus-4.1"),
    "o3mini":  ("openrouter", "openai/o3-mini"),
    "o3-mini": ("openrouter", "openai/o3-mini"),
    "gpt5":    ("openrouter", "openai/gpt-5"),
    "gpt-5":   ("openrouter", "openai/gpt-5"),
    "gpt5.4":  ("openrouter", "openai/gpt-5"),
    "gpt-5.4": ("openrouter", "openai/gpt-5"),
    "nano":    ("openrouter", "openai/gpt-4.1-nano"),
    "gpt-4.1-nano": ("openrouter", "openai/gpt-4.1-nano"),
}


ZONE_TO_BLUE_AGENT = {
    "restricted_zone_a": "blue_agent_0",
    "operational_zone_a": "blue_agent_1",
    "restricted_zone_b": "blue_agent_2",
    "operational_zone_b": "blue_agent_3",
    "public_access_zone": "blue_agent_4",
    "admin_network": "blue_agent_4",
    "office_network": "blue_agent_4",
    "contractor_network": "shared_unowned",
    "internet": "shared_unowned",
}


def _normalize_subnet_name(subnet_key) -> str | None:
    if subnet_key is None:
        return None
    if hasattr(subnet_key, "value"):
        return str(subnet_key.value)
    return str(subnet_key)


def _subnet_to_zone(subnet_name: str | None) -> str | None:
    if not subnet_name:
        return None
    if subnet_name.endswith("_subnet"):
        return subnet_name[: -len("_subnet")]
    return subnet_name


def _assign_blue_agent_for_zone(zone: str | None) -> str | None:
    if not zone:
        return None
    return ZONE_TO_BLUE_AGENT.get(zone, "unknown_owner")


def _infer_zone_from_target_text(target_text: str | None) -> str | None:
    if not target_text:
        return None
    for zone in ZONE_TO_BLUE_AGENT:
        if target_text.startswith(zone) or f"{zone}_subnet" in target_text:
            return zone
    return None


def _build_target_resolution_cache(ec):
    state = getattr(ec, "state", None)
    host_to_meta: dict[str, dict] = {}
    ip_to_meta: dict[str, dict] = {}
    subnet_entries: list[tuple] = []
    subnet_cidr_map_serializable: dict[str, str] = {}

    if state is None:
        return {
            "host_to_meta": host_to_meta,
            "ip_to_meta": ip_to_meta,
            "subnet_entries": subnet_entries,
            "subnet_cidr_map": subnet_cidr_map_serializable,
        }

    for subnet_key, cidr in getattr(ec, "subnet_cidr_map", {}).items():
        subnet_name = _normalize_subnet_name(subnet_key)
        if not subnet_name:
            continue
        zone = _subnet_to_zone(subnet_name)
        assigned_blue_agent = _assign_blue_agent_for_zone(zone)
        cidr_str = str(cidr)
        subnet_cidr_map_serializable[subnet_name] = cidr_str
        try:
            network = ipaddress.ip_network(cidr_str, strict=False)
        except ValueError:
            continue
        subnet_entries.append((network, subnet_name, zone, assigned_blue_agent))

    hostname_subnet_map = getattr(state, "hostname_subnet_map", {})
    for hostname, ip in getattr(state, "hostname_ip_map", {}).items():
        hostname_str = str(hostname)
        ip_str = str(ip)
        subnet_name = _normalize_subnet_name(hostname_subnet_map.get(hostname))
        zone = _subnet_to_zone(subnet_name)
        assigned_blue_agent = _assign_blue_agent_for_zone(zone)
        meta = {
            "target_hostname": hostname_str,
            "target_ip": ip_str,
            "target_subnet": subnet_name,
            "target_zone": zone,
            "assigned_blue_agent": assigned_blue_agent,
        }
        host_to_meta[hostname_str] = meta
        ip_to_meta[ip_str] = meta

    return {
        "host_to_meta": host_to_meta,
        "ip_to_meta": ip_to_meta,
        "subnet_entries": subnet_entries,
        "subnet_cidr_map": subnet_cidr_map_serializable,
    }


def _resolve_ip_to_zone_metadata(ip_str: str, cache: dict) -> dict:
    if ip_str in cache["ip_to_meta"]:
        return dict(cache["ip_to_meta"][ip_str])

    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return {}

    best_match = None
    for network, subnet_name, zone, assigned_blue_agent in cache["subnet_entries"]:
        if ip_obj in network:
            if best_match is None or network.prefixlen > best_match[0].prefixlen:
                best_match = (network, subnet_name, zone, assigned_blue_agent)

    if best_match is None:
        return {}

    _, subnet_name, zone, assigned_blue_agent = best_match
    return {
        "target_hostname": None,
        "target_ip": ip_str,
        "target_subnet": subnet_name,
        "target_zone": zone,
        "assigned_blue_agent": assigned_blue_agent,
    }


def _resolve_cidr_to_zone_metadata(cidr_str: str, cache: dict) -> dict:
    try:
        target_network = ipaddress.ip_network(cidr_str, strict=False)
    except ValueError:
        return {}

    best_match = None
    for network, subnet_name, zone, assigned_blue_agent in cache["subnet_entries"]:
        if target_network == network:
            best_match = (network, subnet_name, zone, assigned_blue_agent)
            break
        if target_network.subnet_of(network) or network.subnet_of(target_network):
            if best_match is None or network.prefixlen > best_match[0].prefixlen:
                best_match = (network, subnet_name, zone, assigned_blue_agent)

    if best_match is None:
        return {}

    _, subnet_name, zone, assigned_blue_agent = best_match
    return {
        "target_hostname": None,
        "target_ip": None,
        "target_subnet": subnet_name,
        "target_zone": zone,
        "assigned_blue_agent": assigned_blue_agent,
    }


def _extract_target_kind_and_value(action_obj, action_str: str) -> tuple[str, str | None]:
    if action_obj is not None:
        if hasattr(action_obj, "hostname"):
            hostname = getattr(action_obj, "hostname", None)
            if hostname is not None:
                return "hostname", str(hostname)
        if hasattr(action_obj, "ip_address"):
            ip_value = getattr(action_obj, "ip_address", None)
            if ip_value is not None:
                return "ip", str(ip_value)
        if hasattr(action_obj, "subnet"):
            subnet_value = getattr(action_obj, "subnet", None)
            if subnet_value is not None:
                return "subnet", str(subnet_value)

    parts = action_str.split(" ", 1)
    if len(parts) < 2:
        return "none", None

    target = parts[1].strip()
    if not target:
        return "none", None

    try:
        ipaddress.ip_network(target, strict=False)
        if "/" in target:
            return "subnet", target
    except ValueError:
        pass

    try:
        ipaddress.ip_address(target)
        return "ip", target
    except ValueError:
        return "hostname", target


def _resolve_red_action_detail(action_obj, action_str: str, agent_name: str, cache: dict) -> dict:
    target_kind, target_value = _extract_target_kind_and_value(action_obj, action_str)
    action_type = action_obj.__class__.__name__ if action_obj is not None else action_str.split(" ", 1)[0]

    detail = {
        "action": action_str,
        "action_type": action_type,
        "target_kind": target_kind,
        "target_raw": target_value,
        "target_hostname": None,
        "target_ip": None,
        "target_subnet": None,
        "target_zone": None,
        "assigned_blue_agent": None,
        "resolved": False,
        "source_red_agent": agent_name,
    }

    if target_value is None:
        return detail

    resolved = {}
    if target_kind == "hostname":
        if target_value in cache["host_to_meta"]:
            resolved = dict(cache["host_to_meta"][target_value])
        else:
            inferred_zone = _infer_zone_from_target_text(target_value)
            resolved = {
                "target_hostname": target_value,
                "target_ip": None,
                "target_subnet": None,
                "target_zone": inferred_zone,
                "assigned_blue_agent": _assign_blue_agent_for_zone(inferred_zone),
            }
    elif target_kind == "ip":
        resolved = _resolve_ip_to_zone_metadata(target_value, cache)
    elif target_kind == "subnet":
        resolved = _resolve_cidr_to_zone_metadata(target_value, cache)

    if resolved:
        detail.update(resolved)
        detail["resolved"] = detail.get("target_zone") is not None

    if detail["target_zone"] is None:
        inferred_zone = _infer_zone_from_target_text(target_value)
        detail["target_zone"] = inferred_zone
        detail["assigned_blue_agent"] = _assign_blue_agent_for_zone(inferred_zone)
        detail["resolved"] = detail["target_zone"] is not None

    return detail


def rmkdir(path: str):
    """Recursive mkdir"""
    partial_path = ""
    for p in path.split("/"):
        partial_path += p + "/"

        if os.path.exists(partial_path):
            if os.path.isdir(partial_path):
                continue
            if os.path.isfile(partial_path):
                raise RuntimeError(f"Cannot create {partial_path} (exists as file).")

        os.mkdir(partial_path)


def load_submission(source: str):
    """Load submission from a directory or zip file"""
    sys.path.insert(0, source)

    if source.endswith(".zip"):
        try:
            # Load submission from zip.
            from submission.submission import Submission
        except ImportError as e:
            raise ImportError(
                """
                Error loading submission from zip.
                Please ensure the zip contains the path submission/submission.py
                """
            ).with_traceback(e.__traceback__)
    else:
        # Load submission normally
        from submission import Submission

    # Remove submission from path.
    sys.path.remove(source)
    return Submission


def configure_runtime_env(
    model_config: str | None = None,
    prompt_config: str | None = None,
    model_backend: str | None = None,
    model_name: str | None = None,
):
    if model_config:
        os.environ["CAGE4_MODEL_CONFIG"] = os.path.abspath(model_config)
    if prompt_config:
        os.environ["CAGE4_PROMPTS_CONFIG"] = os.path.abspath(prompt_config)
    if model_backend:
        os.environ["CAGE4_MODEL_BACKEND"] = model_backend
    if model_name:
        os.environ["CAGE4_MODEL_NAME"] = model_name


def run_output_dir(base_output: str, run_index: int, run_prefix: str = "run") -> str:
    return os.path.join(base_output, f"{run_prefix}_{run_index:03d}")


PRICING_PER_M = {
    "openai/o3-mini":              {"input": 1.10, "output": 4.40},
    "openai/gpt-4.1-nano":         {"input": 0.10, "output": 0.40},
    "openai/gpt-5.4":              {"input": 2.50, "output": 15.00},
    "openai/gpt-5.4-mini":          {"input": 0.40, "output": 1.60},
    "anthropic/claude-sonnet-4.6":  {"input": 3.00, "output": 15.00},
    "anthropic/claude-opus-4.6":    {"input": 5.00, "output": 25.00},
    "anthropic/claude-opus-4.7":    {"input": 5.00, "output": 25.00},
    "anthropic/claude-haiku-4.5":   {"input": 0.80, "output": 4.00},
    "deepseek/deepseek-v3.2":      {"input": 0.30, "output": 0.88},
    "google/gemini-3-flash-preview": {"input": 0.15, "output": 0.60},
}


def _compute_cost(prompt_tokens, completion_tokens, model_name):
    pricing = PRICING_PER_M.get(model_name, {})
    input_cost = prompt_tokens * pricing.get("input", 0) / 1e6
    output_cost = completion_tokens * pricing.get("output", 0) / 1e6
    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(input_cost + output_cost, 6),
        "currency": "USD",
        "model": model_name,
        "pricing_per_M": pricing,
    }


def _resolve_model_name():
    name = os.environ.get("CAGE4_MODEL_NAME")
    if name:
        return name
    cfg_path = os.environ.get("CAGE4_MODEL_CONFIG")
    if cfg_path and os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path) as fh:
                cfg = yaml.safe_load(fh) or {}
            return cfg.get("model_name")
        except Exception:
            pass
    return None


def _collect_model_params() -> dict:
    params: dict = {
        "backend": os.environ.get("CAGE4_MODEL_BACKEND"),
        "model_name": os.environ.get("CAGE4_MODEL_NAME"),
        "model_config_path": os.environ.get("CAGE4_MODEL_CONFIG"),
        "agent_mode": os.environ.get("CAGE4_AGENT_MODE"),
        "all_llm_agents": os.environ.get("CAGE4_ALL_LLM_AGENTS"),
    }
    cfg_path = params["model_config_path"]
    if cfg_path and os.path.exists(cfg_path):
        try:
            import yaml
            with open(cfg_path, "r") as fh:
                cfg = yaml.safe_load(fh) or {}
            params["config"] = cfg
            gen = cfg.get("generate", {}) if isinstance(cfg, dict) else {}
            params["temperature"] = gen.get("temperature")
            params["max_new_tokens"] = gen.get("max_new_tokens")
        except Exception as exc:
            params["config_load_error"] = str(exc)
    return params


def run_evaluation(submission, log_path, max_eps=100, episode_length=500, write_to_file=True, seed=None, run_index=0, attacker_log_path=None, auto_stop=None, resume=False):

    cyborg_version = CYBORG_VERSION
    EPISODE_LENGTH = max(2, int(episode_length))
    scenario = "Scenario4"

    version_header = f"CybORG v{cyborg_version}, {scenario}"
    author_header = f"Author: {submission.NAME}, Team: {submission.TEAM}, Technique: {submission.TECHNIQUE}"

    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=EPISODE_LENGTH,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    wrapped_cyborg = submission.wrap(cyborg)
    
    print(version_header)
    print(author_header)
    print(
        f"Using agents {submission.AGENTS}, if this is incorrect please update the code to load in your agent"
    )

    resume_from = 0
    if write_to_file:
        if not log_path.endswith("/"):
            log_path += "/"
        print(f"Results will be saved to {log_path}")
        if resume:
            ep_file = log_path + "episodes.jsonl"
            if os.path.exists(ep_file):
                with open(ep_file) as f:
                    resume_from = sum(1 for line in f if line.strip())
                print(f"[resume] Found {resume_from} existing episodes, continuing from episode {resume_from}")
        else:
            with open(log_path + "episodes.jsonl", "w") as episode_log:
                episode_log.write("")

    start = datetime.now()

    try:
        from CybORG.Agents.LLMAgents.llm_adapter.utils.token_tracker import TokenUsageTracker
        TokenUsageTracker.reset()
    except Exception:
        TokenUsageTracker = None

    total_reward = []
    agent_total_rewards = {}
    zone_total_rewards = {}
    actions_log = []
    obs_log = []
    auto_stop_n = int(auto_stop) if auto_stop else 0
    best_episode_reward = None
    no_improve_streak = 0
    early_stopped = False
    episodes_completed = 0
    step_log_file = None
    attacker_log_file = None
    if write_to_file:
        step_log_file = open(log_path + "step_log.jsonl", "a" if resume else "w")
        if attacker_log_path:
            os.makedirs(os.path.dirname(attacker_log_path), exist_ok=True)
            attacker_log_file = open(attacker_log_path, "w")

    for i in range(resume_from, resume_from + max_eps):
        episode_start_time = datetime.now()
        episode_tokens_before = TokenUsageTracker.snapshot() if TokenUsageTracker else None
        observations, _ = wrapped_cyborg.reset()
        episode_target_cache = _build_target_resolution_cache(cyborg.environment_controller)

        episode_host_counts = {}
        try:
            ec_state = cyborg.environment_controller.state
            subnet_host_counts: dict[str, int] = {}
            for hostname in ec_state.hosts:
                sn = ec_state.hostname_subnet_map.get(hostname)
                sn_str = sn.value if sn else "unknown"
                subnet_host_counts[sn_str] = subnet_host_counts.get(sn_str, 0) + 1
            episode_host_counts = subnet_host_counts
        except Exception:
            pass

        r = []
        agent_ep_rewards = {}
        zone_ep_rewards = {}
        a = []
        o = []
        count = 0
        for j in range(EPISODE_LENGTH):
            if (
                hasattr(cyborg, "environment_controller")
                and getattr(cyborg.environment_controller, "step_count", 0) >= EPISODE_LENGTH
            ):
                break
            actions = {
                agent_name: agent.get_action(
                    observations[agent_name], wrapped_cyborg.action_space(agent_name)
                )
                for agent_name, agent in submission.AGENTS.items()
                if agent_name in wrapped_cyborg.agents
            }
            observations, rew, term, trunc, info = wrapped_cyborg.step(actions)
            done = {
                agent: term.get(agent, False) or trunc.get(agent, False)
                for agent in wrapped_cyborg.agents
            }
            if hasattr(cyborg, "environment_controller") and getattr(cyborg.environment_controller, "done", False):
                break
            if all(done.values()):
                break
            r.append(mean(rew.values()))
            for agent_name, agent_rew in rew.items():
                agent_ep_rewards[agent_name] = agent_ep_rewards.get(agent_name, 0.0) + float(agent_rew)
            zone_rews = info.get("zone_rewards") if isinstance(info, dict) else None
            if not zone_rews:
                zone_rews = {}
            for agent_name, zr in zone_rews.items():
                zone_ep_rewards[agent_name] = zone_ep_rewards.get(agent_name, 0.0) + float(zr)

            if step_log_file:
                ec = cyborg.environment_controller
                blue_actions = {}
                for agent_name in sorted(a_name for a_name in ec.action if "blue" in a_name):
                    raw = ec.action.get(agent_name)
                    act = raw[0] if isinstance(raw, (list, tuple)) else raw
                    blue_actions[agent_name] = str(act) if act else "Sleep"
                blue_success = {}
                for agent_name in sorted(a_name for a_name in observations if "blue" in a_name):
                    obs_data = observations.get(agent_name, {})
                    success_val = obs_data.get("success") if isinstance(obs_data, dict) else None
                    blue_success[agent_name] = str(success_val) if success_val is not None else "UNKNOWN"
                red_actions = {}
                red_action_details = {}
                for agent_name in sorted(a_name for a_name in ec.action if "red" in a_name):
                    raw = ec.action.get(agent_name)
                    act = raw[0] if isinstance(raw, (list, tuple)) else raw
                    act_str = str(act) if act else "Sleep"
                    if act_str != "Sleep":
                        red_actions[agent_name] = act_str
                        red_action_details[agent_name] = _resolve_red_action_detail(
                            action_obj=act,
                            action_str=act_str,
                            agent_name=agent_name,
                            cache=episode_target_cache,
                        )

                host_status = {}
                try:
                    st = ec.state
                    for hostname in sorted(st.hosts):
                        subnet_name = st.hostname_subnet_map.get(hostname)
                        subnet_str = subnet_name.value if subnet_name else ""
                        red_sessions = 0
                        red_agents_on_host = []
                        for agent_name_s, sess_map in st.sessions.items():
                            if "red" not in agent_name_s:
                                continue
                            for _sid, sess in sess_map.items():
                                if sess.hostname == hostname and sess.active:
                                    red_sessions += 1
                                    if agent_name_s not in red_agents_on_host:
                                        red_agents_on_host.append(agent_name_s)
                        host_obj = st.hosts[hostname]
                        ioc_files = []
                        for f in getattr(host_obj, "files", []) or []:
                            fname = getattr(f, "name", None) or (f.get("File Name") if isinstance(f, dict) else None)
                            if fname in ("cmd.sh", "cmd.exe", "escalate.sh", "escalate.exe"):
                                ioc_files.append(fname)
                        status = "compromised" if red_sessions > 0 else "clean"
                        entry = {"subnet": subnet_str, "status": status}
                        if red_sessions > 0:
                            entry["red_sessions"] = red_sessions
                            entry["red_agents"] = red_agents_on_host
                        if ioc_files:
                            entry["ioc_files"] = ioc_files
                        host_status[hostname] = entry
                except Exception:
                    pass

                step_record = {
                    "run": run_index,
                    "episode": i,
                    "step": j,
                    "team_reward": round(r[-1], 4),
                    "zone_rewards": {k: round(v, 4) for k, v in zone_rews.items()} if zone_rews else {},
                    "blue_actions": blue_actions,
                    "blue_success": blue_success,
                    "red_actions": red_actions,
                    "red_action_details": red_action_details,
                    "host_status": host_status,
                }
                if j == 0 and episode_host_counts:
                    step_record["episode_host_counts"] = episode_host_counts
                step_log_file.write(json.dumps(step_record) + "\n")
                step_log_file.flush()
                if attacker_log_file:
                    attacker_record = dict(step_record)
                    attacker_record["subnet_cidr_map"] = episode_target_cache.get("subnet_cidr_map", {})
                    attacker_record["zone_to_blue_agent"] = ZONE_TO_BLUE_AGENT
                    attacker_log_file.write(json.dumps(attacker_record) + "\n")
                    attacker_log_file.flush()

            if write_to_file:
                a.append(
                    {
                        agent_name: cyborg.get_last_action(agent_name)
                        for agent_name in wrapped_cyborg.agents
                    }
                )
                o.append(
                    {
                        agent_name: observations[agent_name]
                        for agent_name in observations.keys()
                    }
                )
        episode_total_reward = sum(r)
        total_reward.append(episode_total_reward)
        for agent_name, agent_rew in agent_ep_rewards.items():
            agent_total_rewards.setdefault(agent_name, []).append(agent_rew)
        for agent_name, zr in zone_ep_rewards.items():
            zone_total_rewards.setdefault(agent_name, []).append(zr)

        episode_end_time = datetime.now()
        episode_elapsed = (episode_end_time - episode_start_time).total_seconds()
        episode_tokens_after = TokenUsageTracker.snapshot() if TokenUsageTracker else None
        episode_tokens = None
        episode_cost = None
        if episode_tokens_before and episode_tokens_after:
            episode_tokens = {
                k: episode_tokens_after[k] - episode_tokens_before[k]
                for k in episode_tokens_after
            }
            resolved_model = _resolve_model_name()
            if resolved_model:
                episode_cost = _compute_cost(
                    episode_tokens.get("prompt_tokens", 0),
                    episode_tokens.get("completion_tokens", 0),
                    resolved_model,
                )

        if write_to_file:
            actions_log.append(a)
            obs_log.append(o)
            episode_record = {
                "episode_index": i,
                "episode_reward": episode_total_reward,
                "agent_rewards": {k: round(v, 4) for k, v in sorted(
                    (zone_ep_rewards if zone_ep_rewards else agent_ep_rewards).items()
                )},
                "zone_rewards": {k: round(v, 4) for k, v in sorted(zone_ep_rewards.items())} if zone_ep_rewards else None,
                "host_counts": episode_host_counts if episode_host_counts else None,
                "steps_observed": len(r),
                "sim_step_count": int(getattr(cyborg.environment_controller, "step_count", 0)),
                "tokens": episode_tokens,
                "cost": episode_cost,
                "time": {
                    "start": episode_start_time.isoformat(),
                    "end": episode_end_time.isoformat(),
                    "elapsed_seconds": round(episode_elapsed, 2),
                },
            }
            with open(log_path + "episodes.jsonl", "a") as episode_log:
                episode_log.write(json.dumps(episode_record) + "\n")

        episodes_completed = i + 1

        if auto_stop_n > 0:
            if best_episode_reward is None or episode_total_reward > best_episode_reward:
                best_episode_reward = episode_total_reward
                no_improve_streak = 0
            else:
                no_improve_streak += 1
                print(
                    f"[auto-stop] episode {i}: reward {episode_total_reward:.2f} did not "
                    f"improve on best {best_episode_reward:.2f} "
                    f"({no_improve_streak}/{auto_stop_n})"
                )
                if no_improve_streak >= auto_stop_n:
                    print(
                        f"[auto-stop] stopping run_index={run_index}: no improvement for "
                        f"{auto_stop_n} consecutive episodes (best={best_episode_reward:.2f})"
                    )
                    early_stopped = True
                    break

    if step_log_file:
        step_log_file.close()
    if attacker_log_file:
        attacker_log_file.close()

    end = datetime.now()
    difference = end - start

    reward_mean = mean(total_reward)
    reward_stdev = stdev(total_reward) if len(total_reward) > 1 else 0.0
    reward_string = (
        f"Average reward is: {reward_mean} with a standard deviation of {reward_stdev}"
    )
    print(reward_string)

    print(f"File took {difference} amount of time to finish evaluation")
    if write_to_file:
        print(f"Saving results to {log_path}")
        with open(log_path + "summary.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            data.write(f"Using agents {submission.AGENTS}")

        with open(log_path + "full.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act, obs, sum_rew in zip(actions_log, obs_log, total_reward):
                data.write(
                    f"actions: {act},\n observations: {obs},\n total reward: {sum_rew}\n"
                )
        
        with open(log_path + "actions.txt", "w") as data:
            data.write(version_header + "\n")
            data.write(author_header + "\n")
            data.write(reward_string + "\n")
            for act in zip(actions_log):
                data.write(
                    f"actions: {act}"
                )

        token_totals = TokenUsageTracker.snapshot() if TokenUsageTracker else None
        model_params = _collect_model_params()

        with open(log_path + "summary.json", "w") as output:
            data = {
                "submission": {
                    "author": submission.NAME,
                    "team": submission.TEAM,
                    "technique": submission.TECHNIQUE,
                },
                "parameters": {
                    "seed": seed,
                    "episode_length": EPISODE_LENGTH,
                    "max_episodes": max_eps,
                    "auto_stop": auto_stop_n if auto_stop_n > 0 else None,
                    "episodes_completed": episodes_completed,
                    "early_stopped": early_stopped,
                },
                "model": model_params,
                "tokens": token_totals,
                "cost": _compute_cost(
                    token_totals.get("prompt_tokens", 0),
                    token_totals.get("completion_tokens", 0),
                    _resolve_model_name() or "unknown",
                ) if token_totals else None,
                "time": {
                    "start": str(start),
                    "end": str(end),
                    "elapsed": str(difference),
                    "elapsed_seconds": difference.total_seconds(),
                },
                "reward": {
                    "mean": reward_mean,
                    "stdev": reward_stdev,
                },
                "agents": {
                    agent: str(submission.AGENTS[agent]) for agent in submission.AGENTS
                },
                "agent_rewards": {
                    agent_name: {
                        "mean": mean(rewards_list),
                        "stdev": stdev(rewards_list) if len(rewards_list) > 1 else 0.0,
                    }
                    for agent_name, rewards_list in sorted(
                        (zone_total_rewards if zone_total_rewards else agent_total_rewards).items()
                    )
                },
                "zone_rewards": {
                    agent_name: {
                        "mean": mean(rewards_list),
                        "stdev": stdev(rewards_list) if len(rewards_list) > 1 else 0.0,
                    }
                    for agent_name, rewards_list in sorted(zone_total_rewards.items())
                } if zone_total_rewards else None,
            }
            json.dump(data, output, indent=2)

        with open(log_path + "scores.txt", "w") as scores:
            scores.write(f"reward_mean: {reward_mean}\n")
            scores.write(f"reward_stdev: {reward_stdev}\n")

    agent_reward_summary = {
        agent_name: {
            "mean": mean(rewards_list),
            "stdev": stdev(rewards_list) if len(rewards_list) > 1 else 0.0,
        }
        for agent_name, rewards_list in sorted(
            (zone_total_rewards if zone_total_rewards else agent_total_rewards).items()
        )
    }

    zone_reward_summary = {
        agent_name: {
            "mean": mean(rewards_list),
            "stdev": stdev(rewards_list) if len(rewards_list) > 1 else 0.0,
        }
        for agent_name, rewards_list in sorted(zone_total_rewards.items())
    } if zone_total_rewards else {}

    return {
        "reward_mean": reward_mean,
        "reward_stdev": reward_stdev,
        "episodes": max_eps,
        "episodes_completed": episodes_completed,
        "early_stopped": early_stopped,
        "episode_length": EPISODE_LENGTH,
        "seed": seed,
        "log_path": log_path,
        "agent_rewards": agent_reward_summary,
        "zone_rewards": zone_reward_summary,
        "tokens": TokenUsageTracker.snapshot() if TokenUsageTracker else None,
        "elapsed_seconds": difference.total_seconds(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CybORG Evaluation Script")
    parser.add_argument("submission_path", type=str)
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default=None,
        help="Output path (positional, legacy).",
    )
    parser.add_argument(
        "--output",
        "--outpupt",
        dest="output_override",
        type=str,
        default=None,
        help="Output path (preferred flag form). '--outpupt' is accepted as a compatibility alias.",
    )
    parser.add_argument(
        "--append-timestamp",
        dest="append_timestamp",
        action="store_true",
        default=True,
        help="Append timestamp to output path (default: enabled).",
    )
    parser.add_argument(
        "--no-append-timestamp",
        dest="append_timestamp",
        action="store_false",
        help="Disable timestamp suffix on output path.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing run: append to episodes.jsonl instead of overwriting.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Set the seed for CybORG"
    )
    parser.add_argument("--max-eps", type=int, default=100, help="Max episodes to run")
    parser.add_argument(
        "--auto-stop",
        type=int,
        default=None,
        help=(
            "Early-stop a run if the overall episode reward fails to strictly improve "
            "over N consecutive episodes. Disabled when omitted or <= 0."
        ),
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=500,
        help="Maximum number of steps per episode",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of separate runs to execute. Each run gets its own folder.",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="run",
        help="Prefix used for per-run output directories",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=sorted(MODEL_PRESETS.keys()),
        metavar="PRESET",
        help=(
            "Preset model alias — sets backend + model_name in one shot. "
            "Presets: " + ", ".join(sorted(set(MODEL_PRESETS.keys()))) + ". "
            "Explicit --model-backend/--model-name take precedence over the preset."
        ),
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Optional model YAML path (sets CAGE4_MODEL_CONFIG)",
    )
    parser.add_argument(
        "--model-backend",
        type=str,
        default=None,
        help="Optional backend override (sets CAGE4_MODEL_BACKEND)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model name override (sets CAGE4_MODEL_NAME), e.g. openai/gpt-4.1-nano",
    )
    parser.add_argument(
        "--prompt-config",
        type=str,
        default=None,
        help="Optional prompt YAML path (sets CAGE4_PROMPTS_CONFIG)",
    )
    args = parser.parse_args()
    chosen_output_path = args.output_override or args.output_path
    if not chosen_output_path:
        parser.error("output path is required via positional output_path or --output/--outpupt")

    args.output_path = os.path.abspath(chosen_output_path)
    args.submission_path = os.path.abspath(args.submission_path)

    if not args.output_path.endswith("/"):
        args.output_path += "/"

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.append_timestamp:
        args.output_path += run_timestamp + "/"

    rmkdir(args.output_path)

    console_log_path = os.path.join(args.output_path, "console.log")
    _console_log_file = open(console_log_path, "a" if args.resume else "w")

    class _Tee:
        def __init__(self, stream, log_file):
            self._stream = stream
            self._log_file = log_file
        def write(self, data):
            self._stream.write(data)
            try:
                self._log_file.write(data)
                self._log_file.flush()
            except Exception:
                pass
        def flush(self):
            self._stream.flush()
            try:
                self._log_file.flush()
            except Exception:
                pass
        def __getattr__(self, name):
            return getattr(self._stream, name)

    sys.stdout = _Tee(sys.stdout, _console_log_file)
    sys.stderr = _Tee(sys.stderr, _console_log_file)

    agent_mode = os.environ.get("CAGE4_AGENT_MODE", "llm_baseline").strip().lower()

    try:
        from CybORG.Agents.LLMAgents.llm_adapter.utils.logger import Logger as RichLogger
        rich_log_path = os.path.join(args.output_path, "console.log")
        RichLogger.set_log_file(rich_log_path)
    except Exception:
        pass

    preset_backend = preset_model_name = None
    if args.model:
        preset_backend, preset_model_name = MODEL_PRESETS[args.model]
        print(f"[model] preset '{args.model}' -> backend={preset_backend} model_name={preset_model_name}")

    resolved_backend = args.model_backend or preset_backend
    resolved_model_name = args.model_name or preset_model_name

    configure_runtime_env(
        model_config=args.model_config,
        prompt_config=args.prompt_config,
        model_backend=resolved_backend,
        model_name=resolved_model_name,
    )

    runs = max(1, int(args.runs))
    run_results = []

    for run_idx in range(runs):
        run_path = run_output_dir(args.output_path, run_idx, run_prefix=args.run_prefix)
        if not run_path.endswith("/"):
            run_path += "/"
        rmkdir(run_path)

        os.environ["CAGE4_RUN_LOG_DIR"] = run_path
        submission = load_submission(args.submission_path)
        run_seed = (args.seed + run_idx) if args.seed is not None else None
        attacker_logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "Attacker_logs")
        attacker_log_path = os.path.join(attacker_logs_dir, f"Run {run_idx}_{run_timestamp}.jsonl")
        result = run_evaluation(
            submission,
            max_eps=args.max_eps,
            episode_length=args.episode_steps,
            log_path=run_path,
            seed=run_seed,
            run_index=run_idx,
            attacker_log_path=attacker_log_path,
            auto_stop=args.auto_stop,
            resume=args.resume,
        )
        result["run_index"] = run_idx
        run_results.append(result)
        print(f"Attacker log saved -> logs/Attacker_logs/Run {run_idx}_{run_timestamp}.jsonl")

        if result.get("early_stopped"):
            print(
                f"[auto-stop] halting all remaining runs after run_index={run_idx} "
                f"(--auto-stop={args.auto_stop} triggered)."
            )
            break

    if run_results:
        llm_agent_name = os.environ.get("CAGE4_BLUE_AGENT_NAME", "blue_agent_4")
        sleep_agent_names = [f"blue_agent_{i}" for i in range(5) if f"blue_agent_{i}" != llm_agent_name]
        agent_delta_per_run = []
        for r in run_results:
            ar = r.get("agent_rewards", {})
            llm_mean = ar.get(llm_agent_name, {}).get("mean")
            sleep_means = [ar.get(a, {}).get("mean") for a in sleep_agent_names]
            if llm_mean is not None and all(s is not None for s in sleep_means):
                agent_delta_per_run.append(llm_mean - mean(sleep_means))

        agent_delta_summary = {}
        if agent_delta_per_run:
            agent_delta_summary = {
                "llm_agent": llm_agent_name,
                "per_run_delta": [round(d, 2) for d in agent_delta_per_run],
                "mean_delta": round(mean(agent_delta_per_run), 2),
                "stdev_delta": round(stdev(agent_delta_per_run), 2) if len(agent_delta_per_run) > 1 else 0.0,
            }

        def _sum_key(key):
            return sum(
                int((r.get("tokens") or {}).get(key, 0))
                for r in run_results
            )

        totals_tokens = {
            "prompt_tokens": _sum_key("prompt_tokens"),
            "completion_tokens": _sum_key("completion_tokens"),
            "total_tokens": _sum_key("total_tokens"),
            "calls": _sum_key("calls"),
        }
        totals_elapsed_seconds = sum(
            float(r.get("elapsed_seconds") or 0.0) for r in run_results
        )

        with open(os.path.join(args.output_path, "multi_run_summary.json"), "w") as output:
            json.dump(
                {
                    "runs": run_results,
                    "overall_reward_mean_of_means": mean(r["reward_mean"] for r in run_results),
                    "overall_reward_stdev_of_means": (
                        stdev(r["reward_mean"] for r in run_results) if len(run_results) > 1 else 0.0
                    ),
                    "totals": {
                        "tokens": totals_tokens,
                        "elapsed_seconds": totals_elapsed_seconds,
                    },
                    "model": _collect_model_params(),
                    "agent_delta": agent_delta_summary,
                    "parameters": {
                        "max_eps": args.max_eps,
                        "auto_stop": args.auto_stop,
                        "episode_steps": args.episode_steps,
                        "runs": runs,
                        "run_prefix": args.run_prefix,
                        "seed": args.seed,
                        "model_config": os.environ.get("CAGE4_MODEL_CONFIG"),
                        "model_backend": os.environ.get("CAGE4_MODEL_BACKEND"),
                        "model_name": os.environ.get("CAGE4_MODEL_NAME"),
                    },
                },
                output,
                indent=2,
            )

        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")

        advantage_script = os.path.join(scripts_dir, "visualize_agent_advantage.py")
        if os.path.exists(advantage_script):
            advantage_png = os.path.join(args.output_path, "agent_advantage.png")
            try:
                import subprocess
                subprocess.run(
                    [sys.executable, advantage_script, args.output_path,
                     "--agent", llm_agent_name, "--output", advantage_png],
                    timeout=60, check=False,
                )
            except Exception as e:
                print(f"Warning: agent advantage visualization failed: {e}")
