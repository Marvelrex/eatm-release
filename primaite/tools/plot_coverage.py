import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ["seed42", "seed2025", "seed818"]
ARMS = [("sleep", "Sleep floor"), ("tree", "tree-only"), ("qwen3.7-flash", "Qwen3.7"),
        ("deepseek-v4-flash-0731", "DeepSeek-v4"), ("glm-5.2", "GLM-5.2"),
        ("gpt-5.6-terra", "GPT-5.6 Terra"), ("claude-opus-5", "Claude Opus 5")]
LEGACY = {"qwen3.7-flash": "info_qwen3.7", "gpt-5.6-terra": "terra"}
ZONES = {"multi": 3, "single": 5}
CONTROL, LLM = "#6b6a66", "#2a78d6"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d8d6d2"


def covered(topo, arm):
    out = []
    for s in SEEDS:
        if arm == "tree":
            names = [f"bm_{topo}_baseline", f"bm_baseline_{topo}"]
        elif arm == "sleep":
            names = [f"bm_{topo}_sleep"]
        else:
            names = [f"bm_{topo}_de10_{arm}", f"bm_{topo}_de10_{LEGACY.get(arm, arm)}"]
        d = next((ROOT / "results" / s / n for n in names
                  if (ROOT / "results" / s / n).exists()), None)
        if not d:
            continue
        cs = sorted(d.glob("curve_*.json"))
        if cs:
            out += [e["hq_covered"] for e in json.load(open(cs[0]))["episodes"] if e][:10]
    return out


def main():
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--out", default="Paper/fig_coverage.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8))
    for ax, topo in zip(axes, ("multi", "single")):
        rows = [(nice, covered(topo, key)) for key, nice in ARMS]
        rows = [(n, v) for n, v in rows if v]
        n_zone = ZONES[topo]
        for i, (name, vals) in enumerate(rows):
            colour = CONTROL if name in ("tree-only", "Sleep floor") else LLM
            m = statistics.mean(vals)
            ax.plot([min(vals), max(vals)], [i, i], color=RULE, lw=2.4, zorder=1,
                    solid_capstyle="round",
                    label="range over 30 episodes" if i == 0 else None)
            ax.plot(m, i, "o", ms=8, color=colour, mec="white", mew=1.0, zorder=3)
            ax.annotate(f"{m:.2f}", xy=(m, i), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=colour)
        ax.axvline(n_zone, color=RULE, ls="--", lw=1.2, zorder=0)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r[0] for r in rows] if topo == "multi" else [],
                           fontsize=9, color=INK)
        ax.set_xlim(-0.3, n_zone + 0.45)
        ax.set_ylim(-0.5, len(rows) - 0.1)
        ax.set_xticks(range(n_zone + 1))
        ax.set_xlabel(f"zones given at least one Restore   (of {n_zone})",
                      fontsize=9, color=INK)
        ax.set_title(f"{topo}-network topology", fontsize=10.5, color=INK, pad=14)
        ax.grid(axis="x", alpha=0.22)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}\n")
    for topo in ("multi", "single"):
        for key, nice in ARMS:
            v = covered(topo, key)
            if v:
                print(f"  {topo:<7}{nice:<16}mean {statistics.mean(v):.2f}"
                      f"  range {min(v)}-{max(v)}  of {ZONES[topo]}")


if __name__ == "__main__":
    main()
