import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ["seed42", "seed2025", "seed818"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"]
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#b8b6b2"

CC4 = [("GPT-5.4 mini", 165), ("o3-mini", 66), ("Gemini 3 Flash", -116),
       ("Haiku 4.5", -240), ("DeepSeek-v3.2", -404), ("GPT-5.4", -769)]
PORTS = [("claude-opus-5", "Claude Opus 5"), ("gpt-5.6-terra", "GPT-5.6 Terra"),
         ("glm-5.2", "GLM-5.2"), ("deepseek-v4-flash-0731", "DeepSeek-v4"),
         ("qwen3.7-flash", "Qwen3.7")]
LEGACY = {"qwen3.7-flash": "info_qwen3.7", "gpt-5.6-terra": "terra"}


def mean_reward(seed, topo, arm):
    if arm == "tree":
        names = [f"bm_{topo}_baseline", f"bm_baseline_{topo}"]
    else:
        names = [f"bm_{topo}_de10_{arm}", f"bm_{topo}_de10_{LEGACY.get(arm, arm)}"]
    d = next((ROOT / "results" / seed / n for n in names
              if (ROOT / "results" / seed / n).exists()), None)
    if not d:
        return None
    cs = sorted(d.glob("curve_*.json"))
    if not cs:
        return None
    r = [e["reward"] for e in json.load(open(cs[0]))["episodes"] if e][:10]
    return statistics.mean(r) if len(r) == 10 else None


def port_deltas(topo):
    out = []
    for key, nice in PORTS:
        per = [mean_reward(s, topo, key) for s in SEEDS]
        ctl = [mean_reward(s, topo, "tree") for s in SEEDS]
        if all(v is not None for v in per + ctl):
            out.append((nice, statistics.mean(p - c for p, c in zip(per, ctl))))
    return sorted(out, key=lambda x: -x[1])


def main():
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--out", default="Paper/fig_delta_llm.png")
    args = ap.parse_args()

    groups = [("CAGE 4\n(CybORG)", sorted(CC4, key=lambda x: -x[1])),
              ("PrimAITE\nmulti-network", port_deltas("multi")),
              ("PrimAITE\nsingle-network", port_deltas("single"))]

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for gi, (label, rows) in enumerate(groups):
        colour = SERIES[gi]
        for j, (name, v) in enumerate(rows):
            x = gi + (j - (len(rows) - 1) / 2) * 0.14
            ax.plot([x, x], [0, v], color=colour, lw=2.0, zorder=3, solid_capstyle="butt")
            ax.plot(x, v, "o", ms=7.5, color=colour, mec="white", mew=1.0, zorder=4)
        lo = min(v for _n, v in rows)
        hi = max(v for _n, v in rows)
        ax.annotate(f"{lo:+,.0f} to {hi:+,.0f}", xy=(gi, hi), xytext=(0, 11),
                    textcoords="offset points", ha="center", fontsize=9.5, color=colour,
                    weight="bold")
        ax.annotate(f"{len(rows)} backbones", xy=(gi, hi), xytext=(0, 25),
                    textcoords="offset points", ha="center", fontsize=8, color=MUTED)
    ax.axhline(0, color=INK, lw=1.1, zorder=2)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=10, color=INK)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("$\\Delta_{\\mathrm{LLM}}$   (reward the LLM adds\n"
                  "on top of the tree-only control)", fontsize=9.5, color=INK)
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    ax.margins(y=0.22)
    ax.annotate("below zero, the tree alone scored better", xy=(0.012, 0.012),
                xycoords="axes fraction", fontsize=8.5, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}\n")
    for label, rows in groups:
        flat = label.replace("\n", " ")
        print(f"{flat:<28}" + "  ".join(f"{n}={v:+,.0f}" for n, v in rows))


if __name__ == "__main__":
    main()
