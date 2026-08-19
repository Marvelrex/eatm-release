#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def load_run_team_rewards(run_dir: Path) -> list[float]:
    ep_file = run_dir / "episodes.jsonl"
    if not ep_file.exists():
        return []
    eps = []
    for line in ep_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        eps.append(json.loads(line))
    eps.sort(key=lambda e: e["episode_index"])
    return [e["episode_reward"] for e in eps]


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    run_dirs = sorted(exp_dir.glob("run_*"))
    if not run_dirs:
        print(f"No run_* dirs in {exp_dir}")
        sys.exit(1)

    print(f"Processing {len(run_dirs)} runs...")

    fig, (ax_lines, ax_box) = plt.subplots(
        1, 2, figsize=(16, 6),
        gridspec_kw={"width_ratios": [2.5, 1]},
    )

    all_rewards = []
    run_means = []
    run_labels = []

    for ri, rd in enumerate(run_dirs):
        rewards = load_run_team_rewards(rd)
        if not rewards:
            continue
        color = RUN_COLORS[ri % len(RUN_COLORS)]
        xs = np.arange(len(rewards))
        ax_lines.plot(xs, rewards, color=color, marker="o", markersize=4,
                      linewidth=1.2, alpha=0.8, label=rd.name)
        all_rewards.append(rewards)
        run_means.append(np.mean(rewards))
        run_labels.append(rd.name.replace("run_", ""))

    if all_rewards:
        arr_all = np.array(all_rewards)
        if arr_all.ndim == 2:
            mean_curve = arr_all.mean(axis=0)
            ax_lines.plot(np.arange(len(mean_curve)), mean_curve,
                          color="black", linewidth=2.5, linestyle="--",
                          label=f"Across-run mean")
        overall_mean = np.mean([r for rs in all_rewards for r in rs])
        ax_lines.axhline(overall_mean, color="red", linestyle=":",
                         alpha=0.6, linewidth=1.2,
                         label=f"Overall mean: {overall_mean:.0f}")

    ax_lines.set_xlabel("Episode Index")
    ax_lines.set_ylabel("Team Sum Reward (episode_reward)")
    ax_lines.set_title(f"{exp_dir.name} — Team Sum Reward per Episode by Run",
                       fontsize=12, fontweight="bold")
    ax_lines.legend(fontsize=7, ncol=2, loc="lower right")
    ax_lines.grid(True, alpha=0.25)

    if all_rewards:
        bp = ax_box.boxplot(all_rewards, patch_artist=True, showmeans=True,
                            meanprops={"marker": "D", "markerfacecolor": "black",
                                       "markeredgecolor": "black", "markersize": 5})
        for patch, color in zip(bp["boxes"],
                                [RUN_COLORS[i % len(RUN_COLORS)] for i in range(len(all_rewards))]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_box.set_xticklabels(run_labels, rotation=45, fontsize=7)
        ax_box.set_xlabel("Run")
        ax_box.set_ylabel("Team Sum Reward")
        ax_box.set_title("Per-Run Distribution", fontsize=11, fontweight="bold")
        ax_box.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    out_path = exp_dir / "team_sum_performance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
