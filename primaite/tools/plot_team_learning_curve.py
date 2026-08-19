from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def episode_files(d: Path):
    return [p for p in sorted(d.glob("ep*.json")) if not p.name.endswith(".steps.jsonl")]


def main(dpath: str, label: str = None):
    d = Path(dpath)
    eps = episode_files(d)
    if not eps:
        print("no episode json in", d)
        return
    label = label or d.name
    xs, native, corrupt, maxc = [], [], [], None
    for i, ep in enumerate(eps, 1):
        m = json.loads(ep.read_text(encoding="utf-8"))["meta"]
        xs.append(i)
        native.append(m.get("native_blue_reward"))
        corrupt.append(m.get("total_corrupt_steps"))
        maxc = m.get("max_corrupt_steps", maxc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(xs, native, marker="o", ms=6, lw=2.2, color="#2b6cb0")
    ax1.axhline(0, color="#999", lw=1, ls=":")
    ax1.set_xlabel("episode"); ax1.set_ylabel("native CC4 blue reward  (≤0, best 0)")
    ax1.set_title("Team defence score per episode\n(higher / closer to 0 = better)", fontsize=10)
    ax1.set_xticks(xs); ax1.grid(alpha=0.3)

    ax2.plot(xs, corrupt, marker="s", ms=6, lw=2.2, color="#c0392b")
    if maxc:
        ax2.axhline(maxc, color="#c0392b", lw=1, ls=":", alpha=0.5)
        ax2.text(xs[0], maxc, f" max={maxc}", fontsize=8, color="#c0392b", va="bottom")
    ax2.set_xlabel("episode"); ax2.set_ylabel("total corrupt-steps  (lower = better)")
    ax2.set_title("Attacker success per episode\n(lower = better defence)", fontsize=10)
    ax2.set_xticks(xs); ax2.set_ylim(bottom=0); ax2.grid(alpha=0.3)

    fig.suptitle(f"Team learning curve — {label}  ({len(eps)} episodes)", fontsize=12, fontweight="bold")
    out = d / "team_learning_curve.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("episode :", " ".join(f"{x:>4}" for x in xs))
    print("native  :", " ".join(f"{v:>4.0f}" for v in native))
    print("corrupt :", " ".join(f"{v:>4}" for v in corrupt))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/cc4_eatm_vs_acd_20260620_214758/eatm",
         sys.argv[2] if len(sys.argv) > 2 else "EATM (CC4)")
