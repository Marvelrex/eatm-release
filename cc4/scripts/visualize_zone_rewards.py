#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ZONE_AGENT_MAP = {
    "blue_agent_0": "restricted_zone_a_subnet",
    "blue_agent_1": "operational_zone_a_subnet",
    "blue_agent_2": "restricted_zone_b_subnet",
    "blue_agent_3": "operational_zone_b_subnet",
    "blue_agent_4": ["public_access_zone_subnet", "admin_network_subnet", "office_network_subnet"],
}

ZONE_LABELS = {
    "restricted_zone_a_subnet":   "Restricted Zone A (Agent 0)",
    "operational_zone_a_subnet":  "Operational Zone A (Agent 1)",
    "restricted_zone_b_subnet":   "Restricted Zone B (Agent 2)",
    "operational_zone_b_subnet":  "Operational Zone B (Agent 3)",
    "public_access_zone_subnet":  "Public Access Zone (Agent 4)",
    "admin_network_subnet":       "Admin Network (Agent 4)",
    "office_network_subnet":      "Office Network (Agent 4)",
    "contractor_network_subnet":  "Contractor Network (no agent)",
}

HOST_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94",
]

RUN_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def short_name(hostname: str, subnet: str) -> str:
    prefix = subnet + "_"
    return hostname[len(prefix):] if hostname.startswith(prefix) else hostname


def load_run(run_dir: Path) -> dict:
    zone_rewards = defaultdict(list)
    ep_file = run_dir / "episodes.jsonl"
    if ep_file.exists():
        eps = []
        for line in ep_file.read_text().splitlines():
            if line.strip():
                eps.append(json.loads(line))
        eps.sort(key=lambda e: e["episode_index"])
        for ep in eps:
            for agent, reward in ep["zone_rewards"].items():
                zone_rewards[agent].append(reward)

    comp_counts = defaultdict(int)
    total_steps = defaultdict(int)
    step_file = run_dir / "step_log.jsonl"
    if step_file.exists():
        for line in step_file.open():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ep = row["episode"]
            for host, info in row["host_status"].items():
                subnet = info["subnet"]
                total_steps[(ep, subnet)] += 1
                if info["status"] == "compromised":
                    comp_counts[(ep, subnet, host)] += 1

    all_eps = set()
    subnet_hosts = defaultdict(set)
    for key in comp_counts:
        ep, subnet, host = key
        all_eps.add(ep)
        subnet_hosts[subnet].add(host)
    for key in total_steps:
        ep, subnet = key
        all_eps.add(ep)

    if step_file.exists():
        first_line = step_file.open().readline().strip()
        if first_line:
            first = json.loads(first_line)
            for host, info in first["host_status"].items():
                subnet_hosts[info["subnet"]].add(host)

    n_eps = max(all_eps) + 1 if all_eps else 0
    host_comp = defaultdict(lambda: defaultdict(list))
    for subnet, hosts in subnet_hosts.items():
        for host in sorted(hosts):
            for ep in range(n_eps):
                host_comp[subnet][host].append(comp_counts.get((ep, subnet, host), 0))

    return dict(zone_rewards), dict(host_comp)


def plot_zone_figure(zone_subnet: str, agent_key: str,
                     all_zone_rewards: list[list[float]],
                     all_host_comp: list[dict],
                     run_boundaries: list[int],
                     out_dir: Path):
    label = ZONE_LABELS.get(zone_subnet, zone_subnet)

    flat_rewards = []
    for run_rewards in all_zone_rewards:
        flat_rewards.extend(run_rewards)
    n_global = len(flat_rewards)

    if n_global == 0:
        return

    merged_host_comp = defaultdict(list)
    all_hosts = set()
    for run_data in all_host_comp:
        if zone_subnet in run_data:
            for host in run_data[zone_subnet]:
                all_hosts.add(host)

    all_hosts = sorted(all_hosts)
    for run_data in all_host_comp:
        zone_data = run_data.get(zone_subnet, {})
        n_eps_run = max((len(v) for v in zone_data.values()), default=0)
        for host in all_hosts:
            if host in zone_data:
                merged_host_comp[host].extend(zone_data[host])
            else:
                merged_host_comp[host].extend([0] * n_eps_run)

    n_hosts = len(all_hosts)
    short_names = [short_name(h, zone_subnet) for h in all_hosts]

    fig, (ax_rew, ax_stack, ax_bar) = plt.subplots(
        3, 1, figsize=(16, 13),
        gridspec_kw={"height_ratios": [1, 1, 0.8]},
    )

    global_idx = 0
    for ri, run_rewards in enumerate(all_zone_rewards):
        color = RUN_COLORS[ri % len(RUN_COLORS)]
        xs = list(range(global_idx, global_idx + len(run_rewards)))
        ax_rew.scatter(xs, run_rewards, color=color, s=12, alpha=0.5)
        global_idx += len(run_rewards)

    arr = np.array(flat_rewards)
    window = max(3, len(all_zone_rewards[0]) // 2) if all_zone_rewards else 3
    if len(arr) >= window:
        rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
        rolling_x = np.arange(window - 1, len(arr))
        ax_rew.plot(rolling_x, rolling, color="black", linewidth=1.5,
                    label=f"Rolling mean (w={window})")

    mean_val = np.mean(arr)
    ax_rew.axhline(mean_val, color="red", linestyle="--", alpha=0.5,
                    label=f"Mean: {mean_val:.0f}")

    for b in run_boundaries:
        ax_rew.axvline(b - 0.5, color="gray", linestyle=":", alpha=0.3)

    ax_rew.set_ylabel("Zone Reward")
    ax_rew.set_title(f"{label} — Zone Reward per Episode", fontsize=12, fontweight="bold")
    ax_rew.legend(fontsize=8, loc="lower right")
    ax_rew.grid(True, alpha=0.2)

    x = np.arange(n_global)
    stack_data = np.zeros((n_hosts, n_global))
    for hi, host in enumerate(all_hosts):
        vals = merged_host_comp[host]
        stack_data[hi, :len(vals)] = vals

    colors = [HOST_COLORS[i % len(HOST_COLORS)] for i in range(n_hosts)]
    ax_stack.stackplot(x, stack_data, labels=short_names, colors=colors, alpha=0.7)

    for b in run_boundaries:
        ax_stack.axvline(b - 0.5, color="gray", linestyle=":", alpha=0.3)

    ax_stack.set_ylabel("Compromised Steps")
    ax_stack.set_xlabel("Global Episode Index")
    ax_stack.set_title(f"{label} — Per-Host Compromised Steps per Episode (stacked)",
                       fontsize=11, fontweight="bold")
    ax_stack.legend(fontsize=6, ncol=min(5, n_hosts), loc="upper right")
    ax_stack.grid(True, alpha=0.2)

    mean_comp = []
    for host in all_hosts:
        vals = merged_host_comp[host]
        mean_comp.append(np.mean(vals) if vals else 0)

    bars = ax_bar.barh(range(n_hosts), mean_comp,
                       color=colors, alpha=0.7, height=0.6)
    ax_bar.set_yticks(range(n_hosts))
    ax_bar.set_yticklabels(short_names, fontsize=8)
    ax_bar.set_xlabel("Mean Compromised Steps per Episode")
    ax_bar.set_title(f"{label} — Average Compromised Steps per Host",
                     fontsize=11, fontweight="bold")
    ax_bar.grid(True, alpha=0.2, axis="x")

    for i, val in enumerate(mean_comp):
        ax_bar.text(val + 0.5, i, f"{val:.1f}", va="center", fontsize=8)

    fig.tight_layout()
    safe = zone_subnet.replace("_subnet", "")
    out_path = out_dir / f"zone_host_rewards_{safe}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    run_dirs = sorted(exp_dir.glob("run_*"))
    if not run_dirs:
        print(f"No run_* directories found in {exp_dir}")
        sys.exit(1)

    print(f"Processing {len(run_dirs)} runs...")

    all_zone_rewards = defaultdict(list)
    all_host_comp = []
    eps_per_run = []

    for rd in run_dirs:
        print(f"  Loading {rd.name}...")
        zr, hc = load_run(rd)
        for agent, rewards in zr.items():
            all_zone_rewards[agent].append(rewards)
            eps_per_run_candidate = len(rewards)
        all_host_comp.append(hc)
        if zr:
            first_agent = list(zr.keys())[0]
            eps_per_run.append(len(zr[first_agent]))

    run_boundaries = []
    cumsum = 0
    for n in eps_per_run:
        cumsum += n
        run_boundaries.append(cumsum)

    out_dir = exp_dir / "host_figures"
    out_dir.mkdir(exist_ok=True)

    for agent, subnets in ZONE_AGENT_MAP.items():
        if isinstance(subnets, str):
            subnets = [subnets]
        for subnet in subnets:
            rewards_per_run = all_zone_rewards.get(agent, [])
            if not rewards_per_run:
                continue
            print(f"Plotting {ZONE_LABELS.get(subnet, subnet)}...")
            plot_zone_figure(subnet, agent, rewards_per_run, all_host_comp,
                             run_boundaries[:-1], out_dir)

    subnet = "contractor_network_subnet"
    dummy_rewards = []
    for rd in run_dirs:
        ep_file = rd / "episodes.jsonl"
        if ep_file.exists():
            eps = []
            for line in ep_file.read_text().splitlines():
                if line.strip():
                    eps.append(json.loads(line))
            eps.sort(key=lambda e: e["episode_index"])
            dummy_rewards.append([e["episode_reward"] for e in eps])
    if dummy_rewards:
        print(f"Plotting {ZONE_LABELS[subnet]}...")
        plot_zone_figure(subnet, None, dummy_rewards, all_host_comp,
                         run_boundaries[:-1], out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
