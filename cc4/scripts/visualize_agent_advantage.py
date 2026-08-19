#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


ZONE_TYPES = {
    "Restricted": ["blue_agent_0", "blue_agent_2"],
    "Operational": ["blue_agent_1", "blue_agent_3"],
    "Public / Admin / Office": ["blue_agent_4"],
}

ZONE_TYPE_COLORS = {
    "Restricted": "#e53935",
    "Operational": "#1e88e5",
    "Public / Admin / Office": "#43a047",
}

AGENT_ZONE_LABELS = {
    "blue_agent_0": "restricted_zone_a",
    "blue_agent_1": "operational_zone_a",
    "blue_agent_2": "restricted_zone_b",
    "blue_agent_3": "operational_zone_b",
    "blue_agent_4": "public / admin / office",
}


def find_run_dirs(experiment_dir: str):
    base = Path(experiment_dir)
    candidates = [base]
    for child in sorted(base.iterdir()):
        if child.is_dir() and re.match(r"\d{8}_\d{6}", child.name):
            candidates.insert(0, child)
    for candidate in candidates:
        run_dirs = sorted(
            [d for d in candidate.iterdir() if d.is_dir() and re.match(r"run_\d+", d.name)],
            key=lambda d: int(re.search(r"\d+", d.name).group()),
        )
        if run_dirs:
            return run_dirs, candidate
    return [], base


def load_all_zone_rewards(run_dirs):
    agent_rewards = {f"blue_agent_{i}": [] for i in range(5)}

    for run_dir in run_dirs:
        episodes_file = run_dir / "episodes.jsonl"
        if not episodes_file.exists():
            continue

        run_data = {f"blue_agent_{i}": [] for i in range(5)}
        for line in episodes_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            ep = json.loads(line)
            zr = ep.get("zone_rewards") or {}
            for agent in run_data:
                run_data[agent].append(zr.get(agent, 0.0))

        for agent in agent_rewards:
            agent_rewards[agent].append(run_data[agent])

    return agent_rewards


def detect_ag_agent(run_dirs):
    for run_dir in run_dirs:
        for f in run_dir.iterdir():
            m = re.match(r"alignment_(blue_agent_\d+)\.jsonl", f.name)
            if m:
                return m.group(1)
    return os.environ.get("CAGE4_BLUE_AGENT_NAME", "blue_agent_4")


def plot_advantage(agent_rewards, ag_agent, output_path, run_dirs):
    n_runs = len(run_dirs)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"LLM agent advantage — {ag_agent} ({AGENT_ZONE_LABELS.get(ag_agent, '?')})\n"
        f"{n_runs} runs, zone reward comparison vs same-type baseline",
        fontsize=13, fontweight="bold",
    )

    ag_zone_type = None
    for zt, agents in ZONE_TYPES.items():
        if ag_agent in agents:
            ag_zone_type = zt
            break

    ax = axes[0, 0]
    for zone_type, agents in ZONE_TYPES.items():
        color = ZONE_TYPE_COLORS[zone_type]
        for agent in agents:
            runs = agent_rewards[agent]
            if not runs:
                continue
            max_eps = max(len(r) for r in runs)
            avg_per_ep = []
            for ep_idx in range(max_eps):
                vals = [r[ep_idx] for r in runs if ep_idx < len(r)]
                avg_per_ep.append(mean(vals) if vals else 0.0)

            is_ag = (agent == ag_agent)
            ax.plot(
                range(len(avg_per_ep)), avg_per_ep,
                color=color,
                linewidth=2.5 if is_ag else 1.2,
                linestyle="-" if is_ag else "--",
                alpha=1.0 if is_ag else 0.6,
                marker="o" if is_ag else None,
                markersize=3 if is_ag else 0,
                label=f"{agent} {'(AG)' if is_ag else '(Sleep)'} — {zone_type}",
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Zone Reward (avg across runs)")
    ax.set_title("Per-Episode Zone Rewards by Agent")
    ax.legend(fontsize=7, loc="lower left")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    ax = axes[0, 1]
    if ag_zone_type and len(ZONE_TYPES[ag_zone_type]) > 1:
        baseline_agents = [a for a in ZONE_TYPES[ag_zone_type] if a != ag_agent]
        ag_runs = agent_rewards[ag_agent]
        baseline_runs_all = [agent_rewards[a] for a in baseline_agents]

        max_eps = max(len(r) for r in ag_runs) if ag_runs else 0
        advantage_per_ep = []
        ag_avg = []
        baseline_avg = []
        for ep_idx in range(max_eps):
            ag_vals = [r[ep_idx] for r in ag_runs if ep_idx < len(r)]
            bl_vals = []
            for bl_runs in baseline_runs_all:
                bl_vals.extend([r[ep_idx] for r in bl_runs if ep_idx < len(r)])

            ag_m = mean(ag_vals) if ag_vals else 0.0
            bl_m = mean(bl_vals) if bl_vals else 0.0
            ag_avg.append(ag_m)
            baseline_avg.append(bl_m)
            advantage_per_ep.append(ag_m - bl_m)

        eps = range(len(advantage_per_ep))
        colors_bar = ["#4caf50" if v >= 0 else "#f44336" for v in advantage_per_ep]
        ax.bar(eps, advantage_per_ep, color=colors_bar, alpha=0.7, edgecolor="white")
        if len(advantage_per_ep) >= 3:
            z = np.polyfit(list(eps), advantage_per_ep, 1)
            trend = np.poly1d(z)
            ax.plot(eps, trend(list(eps)), "--", color="#ff9800", linewidth=2,
                    label=f"Trend: {z[0]:+.1f}/ep")
        overall_adv = mean(advantage_per_ep)
        ax.axhline(overall_adv, color="#9c27b0", linewidth=1.5, linestyle=":",
                   label=f"Mean advantage: {overall_adv:+.0f}")
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward Advantage (AG - Baseline)")
        ax.set_title(f"AG Advantage over {', '.join(baseline_agents)} (same zone type)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f"No same-type baseline\nfor {ag_agent}", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        ax.set_title("AG Advantage (no baseline available)")

    ax = axes[1, 0]
    run_indices = list(range(n_runs))

    for zone_type, agents in ZONE_TYPES.items():
        color = ZONE_TYPE_COLORS[zone_type]
        for agent in agents:
            runs = agent_rewards[agent]
            if not runs:
                continue
            run_means = [mean(r) if r else 0.0 for r in runs]
            is_ag = (agent == ag_agent)
            ax.plot(
                run_indices[:len(run_means)], run_means,
                color=color,
                linewidth=2.5 if is_ag else 1.2,
                linestyle="-" if is_ag else "--",
                marker="o" if is_ag else "s",
                markersize=6 if is_ag else 4,
                alpha=1.0 if is_ag else 0.6,
                label=f"{agent} {'(AG)' if is_ag else '(Sleep)'}",
            )

    ax.set_xlabel("Run")
    ax.set_ylabel("Mean Zone Reward per Episode")
    ax.set_title("Per-Run Mean Zone Reward by Agent")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")

    ax = axes[1, 1]
    ax.axis("off")

    table_rows = []
    table_rows.append(["Agent", "Zone Type", "Mode", "Mean Reward", "Std"])

    for zone_type, agents in ZONE_TYPES.items():
        for agent in agents:
            runs = agent_rewards[agent]
            if not runs:
                continue
            all_eps = [v for r in runs for v in r]
            is_ag = (agent == ag_agent)
            m = mean(all_eps) if all_eps else 0.0
            from statistics import stdev as std_fn
            s = std_fn(all_eps) if len(all_eps) > 1 else 0.0
            table_rows.append([
                agent,
                zone_type,
                "LLM" if is_ag else "Sleep",
                f"{m:.0f}",
                f"{s:.0f}",
            ])

    if ag_zone_type and len(ZONE_TYPES[ag_zone_type]) > 1:
        baseline_agents = [a for a in ZONE_TYPES[ag_zone_type] if a != ag_agent]
        ag_all = [v for r in agent_rewards[ag_agent] for v in r]
        bl_all = []
        for ba in baseline_agents:
            bl_all.extend([v for r in agent_rewards[ba] for v in r])
        ag_mean = mean(ag_all) if ag_all else 0.0
        bl_mean = mean(bl_all) if bl_all else 0.0
        adv = ag_mean - bl_mean
        table_rows.append(["", "", "", "", ""])
        table_rows.append(["ADVANTAGE", ag_zone_type, "AG - Sleep", f"{adv:+.0f}",
                           f"({adv/abs(bl_mean)*100:+.1f}%)" if bl_mean != 0 else "N/A"])

    table = ax.table(
        cellText=table_rows[1:],
        colLabels=table_rows[0],
        loc="center",
        cellLoc="center",
        colWidths=[0.22, 0.22, 0.18, 0.18, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for row_idx, row in enumerate(table_rows[1:], start=1):
        if row[2] == "LLM":
            for col in range(5):
                table[row_idx, col].set_facecolor("#e3f2fd")
        elif row[0] == "ADVANTAGE":
            adv_val = float(row[3].replace("+", ""))
            color = "#c8e6c9" if adv_val > 0 else "#ffcdd2"
            for col in range(5):
                table[row_idx, col].set_facecolor(color)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Advantage plot saved to {output_path}")
    plt.close()

    if ag_zone_type and len(ZONE_TYPES[ag_zone_type]) > 1:
        baseline_agents = [a for a in ZONE_TYPES[ag_zone_type] if a != ag_agent]
        ag_all = [v for r in agent_rewards[ag_agent] for v in r]
        bl_all = []
        for ba in baseline_agents:
            bl_all.extend([v for r in agent_rewards[ba] for v in r])
        ag_mean = mean(ag_all)
        bl_mean = mean(bl_all)
        print(f"\n  AG agent ({ag_agent}):       mean zone reward = {ag_mean:.0f}")
        print(f"  Baseline ({', '.join(baseline_agents)}): mean zone reward = {bl_mean:.0f}")
        print(f"  Advantage: {ag_mean - bl_mean:+.0f} ({(ag_mean - bl_mean)/abs(bl_mean)*100:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Visualize LLM agent advantage")
    parser.add_argument("experiment_dir", type=str, help="Experiment output directory")
    parser.add_argument("--agent", type=str, default=None,
                        help="LLM agent name (auto-detected if omitted)")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    run_dirs, base_dir = find_run_dirs(args.experiment_dir)
    if not run_dirs:
        print(f"No run directories found in {args.experiment_dir}")
        return

    ag_agent = args.agent or detect_ag_agent(run_dirs)
    print(f"LLM agent: {ag_agent} ({AGENT_ZONE_LABELS.get(ag_agent, '?')})")
    print(f"Runs: {len(run_dirs)}")

    agent_rewards = load_all_zone_rewards(run_dirs)
    output_path = args.output or str(base_dir / "agent_advantage.png")
    plot_advantage(agent_rewards, ag_agent, output_path, run_dirs)


if __name__ == "__main__":
    main()
