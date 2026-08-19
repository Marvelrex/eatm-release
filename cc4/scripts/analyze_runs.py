#!/usr/bin/env python3

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


AGENT_ZONES = {
    "blue_agent_0": "restricted_zone_a",
    "blue_agent_1": "operational_zone_a",
    "blue_agent_2": "restricted_zone_b",
    "blue_agent_3": "operational_zone_b",
    "blue_agent_4": "public_access_zone / admin / office",
}

ACTION_FAMILIES = [
    "Sleep", "Analyse", "Remove", "Restore",
    "DeployDecoy", "BlockTrafficZone", "AllowTrafficZone",
]

FAMILY_COLORS = {
    "Sleep":             "#9e9e9e",
    "Analyse":           "#2196f3",
    "Remove":            "#f44336",
    "Restore":           "#4caf50",
    "DeployDecoy":       "#ff9800",
    "BlockTrafficZone":  "#9c27b0",
    "AllowTrafficZone":  "#00bcd4",
    "Other":             "#795548",
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


def detect_agent(run_dirs):
    for run_dir in run_dirs:
        for f in run_dir.iterdir():
            m = re.match(r"alignment_(blue_agent_\d+)\.jsonl", f.name)
            if m:
                return m.group(1)
    return "blue_agent_4"


def load_episodes(run_dir: Path):
    episodes_file = run_dir / "episodes.jsonl"
    if not episodes_file.exists():
        return []
    episodes = []
    for line in episodes_file.read_text().strip().split("\n"):
        if line.strip():
            episodes.append(json.loads(line))
    return episodes


def load_alignment(run_dir: Path, agent_name: str):
    alignment_file = run_dir / f"alignment_{agent_name}.jsonl"
    if not alignment_file.exists():
        return []
    records = []
    for line in alignment_file.read_text().strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records


def extract_family(action_str: str) -> str:
    if not action_str:
        return "Sleep"
    first_word = action_str.split()[0].split("@")[0]
    for fam in ACTION_FAMILIES:
        if first_word.lower() == fam.lower():
            return fam
    return "Other"


def compute_run_stats(run_dirs, agent_name: str):
    stats = []
    for run_dir in run_dirs:
        episodes = load_episodes(run_dir)
        alignment = load_alignment(run_dir, agent_name)

        zone_rewards = [
            ep.get("zone_rewards", {}).get(agent_name, 0.0)
            for ep in episodes
            if ep.get("zone_rewards")
        ]

        team_rewards = [ep.get("episode_reward", 0.0) for ep in episodes]

        action_counts = Counter()
        for rec in alignment:
            family = extract_family(rec.get("llm_action", "Sleep"))
            action_counts[family] += 1

        aligned = sum(1 for rec in alignment if rec.get("aligned", False))
        total_actions = len(alignment)

        parse_ok = sum(1 for rec in alignment if rec.get("parse_ok", False))

        stats.append({
            "run_dir": str(run_dir),
            "run_index": int(re.search(r"\d+", run_dir.name).group()),
            "zone_rewards": zone_rewards,
            "zone_reward_mean": mean(zone_rewards) if zone_rewards else 0.0,
            "zone_reward_std": stdev(zone_rewards) if len(zone_rewards) > 1 else 0.0,
            "team_rewards": team_rewards,
            "team_reward_mean": mean(team_rewards) if team_rewards else 0.0,
            "action_counts": dict(action_counts),
            "total_actions": total_actions,
            "alignment_rate": aligned / max(1, total_actions),
            "parse_rate": parse_ok / max(1, total_actions),
            "sleep_fraction": action_counts.get("Sleep", 0) / max(1, total_actions),
            "active_fraction": 1.0 - action_counts.get("Sleep", 0) / max(1, total_actions),
        })

    return stats


def plot_analysis(stats, agent_name: str, output_path: str):
    if not stats:
        print("No run data found.")
        return

    n_runs = len(stats)
    run_indices = [s["run_index"] for s in stats]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        f"LLM-agent learning analysis — {agent_name} ({AGENT_ZONES.get(agent_name, '?')})\n"
        f"{n_runs} runs",
        fontsize=14, fontweight="bold",
    )

    ax = axes[0, 0]
    zone_means = [s["zone_reward_mean"] for s in stats]
    zone_stds = [s["zone_reward_std"] for s in stats]
    ax.errorbar(run_indices, zone_means, yerr=zone_stds, fmt="o-", color="#1976d2",
                capsize=3, markersize=5, label="Agent zone reward")
    if n_runs >= 3:
        z = np.polyfit(run_indices, zone_means, 1)
        trend = np.poly1d(z)
        ax.plot(run_indices, trend(run_indices), "--", color="#f44336", alpha=0.7,
                label=f"Trend: {z[0]:+.1f}/run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Zone Reward (per episode)")
    ax.set_title("Agent's Own-Zone Reward Trend")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax = axes[0, 1]
    team_means = [s["team_reward_mean"] for s in stats]
    ax.plot(run_indices, team_means, "s-", color="#9e9e9e", markersize=5, label="Team reward")
    ax.plot(run_indices, zone_means, "o-", color="#1976d2", markersize=5, label="Agent zone reward")
    ax.set_xlabel("Run")
    ax.set_ylabel("Reward (per episode)")
    ax.set_title("Team vs Agent Zone Reward")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax = axes[0, 2]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, n_runs))
    for i, s in enumerate(stats):
        zr = s["zone_rewards"]
        if zr:
            ax.plot(range(len(zr)), zr, alpha=0.5, color=cmap[i], linewidth=0.8,
                    label=f"Run {s['run_index']}" if i % max(1, n_runs // 5) == 0 else None)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Zone Reward")
    ax.set_title("Per-Episode Zone Rewards (all runs)")
    ax.legend(fontsize=7, loc="lower left")

    ax = axes[1, 0]
    families_present = set()
    for s in stats:
        families_present.update(s["action_counts"].keys())
    families_ordered = [f for f in ACTION_FAMILIES if f in families_present]
    if "Other" in families_present:
        families_ordered.append("Other")

    bottoms = np.zeros(n_runs)
    for fam in families_ordered:
        counts = [s["action_counts"].get(fam, 0) for s in stats]
        fracs = [c / max(1, s["total_actions"]) for c, s in zip(counts, stats)]
        color = FAMILY_COLORS.get(fam, "#795548")
        ax.bar(run_indices, fracs, bottom=bottoms, color=color, label=fam, edgecolor="white",
               linewidth=0.5)
        bottoms += np.array(fracs)

    ax.set_xlabel("Run")
    ax.set_ylabel("Action Fraction")
    ax.set_title("Action Distribution per Run")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax = axes[1, 1]
    sleep_fracs = [s["sleep_fraction"] for s in stats]
    active_fracs = [s["active_fraction"] for s in stats]
    ax.plot(run_indices, sleep_fracs, "o-", color="#9e9e9e", label="Sleep %")
    ax.plot(run_indices, active_fracs, "s-", color="#4caf50", label="Active %")
    if n_runs >= 3:
        z_active = np.polyfit(run_indices, active_fracs, 1)
        trend_active = np.poly1d(z_active)
        ax.plot(run_indices, trend_active(run_indices), "--", color="#f44336", alpha=0.7,
                label=f"Active trend: {z_active[0]:+.3f}/run")
    ax.set_xlabel("Run")
    ax.set_ylabel("Fraction")
    ax.set_title("Sleep vs Active Actions")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax = axes[1, 2]
    ax.axis("off")

    overall_zone_mean = mean(s["zone_reward_mean"] for s in stats)
    first_half = stats[:n_runs // 2] if n_runs >= 4 else stats[:1]
    second_half = stats[n_runs // 2:] if n_runs >= 4 else stats[-1:]
    first_zone = mean(s["zone_reward_mean"] for s in first_half)
    second_zone = mean(s["zone_reward_mean"] for s in second_half)
    improvement = second_zone - first_zone

    avg_alignment = mean(s["alignment_rate"] for s in stats)
    avg_parse = mean(s["parse_rate"] for s in stats)
    avg_active = mean(s["active_fraction"] for s in stats)

    table_data = [
        ["Agent", agent_name],
        ["Zone", AGENT_ZONES.get(agent_name, "?")],
        ["Runs", str(n_runs)],
        ["", ""],
        ["Avg zone reward", f"{overall_zone_mean:.1f}"],
        ["First-half zone avg", f"{first_zone:.1f}"],
        ["Second-half zone avg", f"{second_zone:.1f}"],
        ["Improvement", f"{improvement:+.1f}"],
        ["", ""],
        ["Avg active action %", f"{avg_active:.1%}"],
        ["Avg parse success", f"{avg_parse:.1%}"],
        ["Avg tree alignment", f"{avg_alignment:.1%}"],
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.5, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    imp_color = "#c8e6c9" if improvement > 0 else "#ffcdd2" if improvement < 0 else "#fff9c4"
    for col in range(2):
        table[8, col].set_facecolor(imp_color)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Analysis saved to {output_path}")
    plt.close()


def save_summary_json(stats, agent_name: str, output_path: str):
    n_runs = len(stats)
    first_half = stats[:n_runs // 2] if n_runs >= 4 else stats[:1]
    second_half = stats[n_runs // 2:] if n_runs >= 4 else stats[-1:]

    summary = {
        "agent": agent_name,
        "zone": AGENT_ZONES.get(agent_name, "unknown"),
        "n_runs": n_runs,
        "per_run": [
            {
                "run_index": s["run_index"],
                "zone_reward_mean": round(s["zone_reward_mean"], 2),
                "zone_reward_std": round(s["zone_reward_std"], 2),
                "team_reward_mean": round(s["team_reward_mean"], 2),
                "active_fraction": round(s["active_fraction"], 4),
                "sleep_fraction": round(s["sleep_fraction"], 4),
                "alignment_rate": round(s["alignment_rate"], 4),
                "parse_rate": round(s["parse_rate"], 4),
                "action_counts": s["action_counts"],
            }
            for s in stats
        ],
        "overall": {
            "zone_reward_mean": round(mean(s["zone_reward_mean"] for s in stats), 2),
            "first_half_zone_mean": round(mean(s["zone_reward_mean"] for s in first_half), 2),
            "second_half_zone_mean": round(mean(s["zone_reward_mean"] for s in second_half), 2),
            "improvement": round(
                mean(s["zone_reward_mean"] for s in second_half) -
                mean(s["zone_reward_mean"] for s in first_half), 2
            ),
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LLM-agent learning across runs")
    parser.add_argument("experiment_dir", type=str, help="Experiment output directory")
    parser.add_argument("--agent", type=str, default=None,
                        help="Agent name (auto-detected from alignment logs if omitted)")
    parser.add_argument("--output", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    run_dirs, base_dir = find_run_dirs(args.experiment_dir)
    if not run_dirs:
        print(f"No run directories found in {args.experiment_dir}")
        return

    agent_name = args.agent or detect_agent(run_dirs)
    print(f"Analyzing {len(run_dirs)} runs for {agent_name} ({AGENT_ZONES.get(agent_name, '?')})")

    stats = compute_run_stats(run_dirs, agent_name)

    output_png = args.output or str(base_dir / "learning_analysis.png")
    output_json = str(Path(output_png).with_suffix(".json"))

    plot_analysis(stats, agent_name, output_png)
    save_summary_json(stats, agent_name, output_json)


if __name__ == "__main__":
    main()
