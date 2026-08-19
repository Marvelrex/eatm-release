#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

AGENT_LABELS = {
    "blue_agent_0": "Agent 0\n(Restricted A)",
    "blue_agent_1": "Agent 1\n(Operational A)",
    "blue_agent_2": "Agent 2\n(Restricted B)",
    "blue_agent_3": "Agent 3\n(Operational B)",
    "blue_agent_4": "Agent 4\n(Public/Admin/Office)",
}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
          "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"]


def load_episodes(run_dir: Path) -> list[dict]:
    ep_file = run_dir / "episodes.jsonl"
    if not ep_file.exists():
        return []
    episodes = []
    for line in ep_file.read_text().splitlines():
        line = line.strip()
        if line:
            episodes.append(json.loads(line))
    episodes.sort(key=lambda e: e["episode_index"])
    return episodes


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    run_dirs = sorted(exp_dir.glob("run_*"))
    if not run_dirs:
        print(f"No run_* directories found in {exp_dir}")
        sys.exit(1)

    agents = list(AGENT_LABELS.keys())
    agent_data = {a: [] for a in agents}

    for run_dir in run_dirs:
        episodes = load_episodes(run_dir)
        if not episodes:
            continue
        for agent in agents:
            rewards = [ep["zone_rewards"].get(agent, 0) for ep in episodes]
            agent_data[agent].append(rewards)

    n_runs = len(agent_data[agents[0]])
    n_agents = len(agents)

    fig, axes = plt.subplots(n_agents, 1, figsize=(16, 3.5 * n_agents), sharex=True)
    if n_agents == 1:
        axes = [axes]

    for ax, agent in zip(axes, agents):
        global_idx = 0
        all_rewards = []
        run_boundaries = []

        for ri, run_rewards in enumerate(agent_data[agent]):
            run_start = global_idx
            color = COLORS[ri % len(COLORS)]
            for r in run_rewards:
                ax.scatter(global_idx, r, color=color, s=10, alpha=0.6, zorder=3)
                all_rewards.append(r)
                global_idx += 1
            run_boundaries.append((run_start, global_idx - 1, ri))

        arr = np.array(all_rewards)
        window = max(3, len(agent_data[agent][0]) // 2) if agent_data[agent] else 3
        if len(arr) >= window:
            rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
            rolling_x = np.arange(window - 1, len(arr))
            ax.plot(rolling_x, rolling, color="black", linewidth=1.2, label=f"Rolling mean (w={window})")

        mean_val = np.mean(arr)
        ax.axhline(mean_val, color="red", linestyle="--", alpha=0.5, linewidth=1,
                    label=f"Mean: {mean_val:.0f}")

        for run_start, run_end, ri in run_boundaries:
            if ri > 0:
                ax.axvline(run_start - 0.5, color="gray", linestyle=":", alpha=0.3)

        ax.set_ylabel("Zone Reward")
        ax.set_title(AGENT_LABELS[agent].replace("\n", " — "), fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Global Episode Index")
    fig.suptitle("Per-Agent Zone Rewards Across All Runs", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out1 = exp_dir / "per_agent_episodes.png"
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    fig, axes = plt.subplots(1, n_agents, figsize=(4 * n_agents, 6), sharey=True)
    if n_agents == 1:
        axes = [axes]

    for ax, agent in zip(axes, agents):
        box_data = agent_data[agent]
        run_means = [np.mean(r) for r in box_data]

        bp = ax.boxplot(box_data, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=1.2),
                        whiskerprops=dict(color="gray"),
                        capprops=dict(color="gray"),
                        flierprops=dict(marker="o", markersize=2, alpha=0.4))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(COLORS[i % len(COLORS)])
            patch.set_alpha(0.4)

        for i, m in enumerate(run_means):
            ax.plot(i + 1, m, marker="D", color="black", markersize=4, zorder=5)

        if len(run_means) > 1:
            x = np.arange(1, len(run_means) + 1)
            slope, intercept = np.polyfit(x, run_means, 1)
            ax.plot(x, intercept + slope * x, color="red", linestyle="--",
                    alpha=0.6, linewidth=1, label=f"Trend ({slope:+.1f}/run)")
            ax.legend(fontsize=7)

        ax.set_title(AGENT_LABELS[agent], fontsize=10, fontweight="bold")
        ax.set_xlabel("Run")
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("Zone Reward")
    fig.suptitle("Per-Agent Zone Reward Distribution by Run", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out2 = exp_dir / "per_agent_boxplots.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    mean_matrix = np.zeros((n_runs, n_agents))
    for j, agent in enumerate(agents):
        for i, run_rewards in enumerate(agent_data[agent]):
            mean_matrix[i, j] = np.mean(run_rewards)

    fig, ax = plt.subplots(figsize=(10, max(6, n_runs * 0.4)))
    im = ax.imshow(mean_matrix, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(n_agents))
    ax.set_xticklabels([AGENT_LABELS[a].replace("\n", "\n") for a in agents], fontsize=9)
    ax.set_yticks(range(n_runs))
    ax.set_yticklabels([f"Run {i}" for i in range(n_runs)], fontsize=8)

    for i in range(n_runs):
        for j in range(n_agents):
            ax.text(j, i, f"{mean_matrix[i, j]:.0f}", ha="center", va="center",
                    fontsize=7, color="black")

    plt.colorbar(im, ax=ax, label="Mean Zone Reward")
    ax.set_title("Mean Zone Reward per Agent per Run", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out3 = exp_dir / "per_agent_heatmap.png"
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
