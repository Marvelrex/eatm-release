#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_run_team_rewards(run_dir: Path) -> list[float]:
    ep_file = run_dir / "episodes.jsonl"
    if not ep_file.exists():
        return []
    rewards = []
    for line in ep_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rewards.append(json.loads(line)["episode_reward"])
    return rewards


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    run_dirs = sorted(exp_dir.glob("run_*"))
    if not run_dirs:
        print(f"No run_* dirs in {exp_dir}")
        sys.exit(1)

    labels, means, stds, n_eps_list = [], [], [], []
    for rd in run_dirs:
        rewards = load_run_team_rewards(rd)
        if not rewards:
            continue
        labels.append(rd.name.replace("run_", ""))
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))
        n_eps_list.append(len(rewards))

    n = len(labels)
    if n == 0:
        print("No data")
        sys.exit(1)

    overall_mean = np.mean(means)
    overall_std = np.std(means)

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * n + 4), 6))
    xs = np.arange(n)
    ax.errorbar(xs, means, yerr=stds, fmt="o", color="#1f77b4",
                ecolor="gray", elinewidth=1.2, capsize=4,
                markersize=8, label="Run mean ± episode stdev")
    ax.axhline(overall_mean, color="red", linestyle="--", alpha=0.7,
               label=f"Across-run mean: {overall_mean:.1f}")
    ax.fill_between(xs, overall_mean - overall_std, overall_mean + overall_std,
                    color="red", alpha=0.08,
                    label=f"± across-run stdev ({overall_std:.1f})")

    for x, m in zip(xs, means):
        ax.text(x, m, f"  {m:.0f}", fontsize=7, va="center")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, fontsize=8)
    ax.set_xlabel(f"Run ({n} runs, {n_eps_list[0]} eps each)")
    ax.set_ylabel("Mean Team Reward (avg of episode_reward)")
    ax.set_title(f"{exp_dir.name} — Per-Run Average Team Reward",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = exp_dir / "run_avg_team_reward.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"  {n} runs | across-run mean = {overall_mean:.1f} ± {overall_std:.1f}")


if __name__ == "__main__":
    main()
