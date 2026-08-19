import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ["seed42", "seed2025", "seed818"]
LEGACY = {"qwen3.7-flash": "info_qwen3.7", "gpt-5.6-terra": "terra"}
MODELS = [("claude-opus-5", "Claude Opus 5", "#0072B2"),
          ("gpt-5.6-terra", "GPT-5.6 Terra", "#D55E00"),
          ("glm-5.2", "GLM-5.2", "#009E73"),
          ("deepseek-v4-flash-0731", "DeepSeek-v4", "#CC79A7"),
          ("qwen3.7-flash", "Qwen3.7", "#E69F00")]
CONTROL, SLEEP = "#6b6a66", "#b4b2ae"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d8d6d2"
N = 10


def curves(topo, arm):
    if arm == "tree":
        names = [f"bm_{topo}_baseline", f"bm_baseline_{topo}"]
    elif arm == "sleep":
        names = [f"bm_{topo}_sleep"]
    else:
        names = [f"bm_{topo}_de10_{arm}", f"bm_{topo}_de10_{LEGACY.get(arm, arm)}"]
    out = []
    for s in SEEDS:
        d = next((ROOT / "results" / s / n for n in names
                  if (ROOT / "results" / s / n).exists()), None)
        if not d:
            continue
        cs = sorted(d.glob("curve_*.json"))
        if cs:
            out.append([e["reward"] for e in json.load(open(cs[0], encoding="utf-8"))
                        ["episodes"] if e][:N])
    return out


def smooth(y, w=3):
    h = w // 2
    return [statistics.mean(y[max(0, i - h):min(len(y), i + h + 1)]) for i in range(len(y))]


def band(runs):
    m, lo, hi = [], [], []
    for i in range(N):
        v = [r[i] for r in runs if i < len(r)]
        m.append(statistics.mean(v))
        lo.append(min(v))
        hi.append(max(v))
    return m, lo, hi


ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "Paper" / "fig_primaite_convergence.png"))
args = ap.parse_args()

fig, ax = plt.subplots(figsize=(3.6, 2.6), dpi=400)
x = range(1, N + 1)

for topo, style in (("multi", "-"), ("single", "--")):
    for arm, colour, lw, z in (("sleep", SLEEP, 1.2, 1), ("tree", CONTROL, 1.5, 2)):
        runs = curves(topo, arm)
        if runs:
            ax.plot(x, smooth(band(runs)[0]), style, color=colour, lw=lw, zorder=z)
    for key, label, colour in MODELS:
        runs = curves(topo, key)
        if runs:
            ax.plot(x, smooth(band(runs)[0]), style, color=colour, lw=1.5, zorder=4)

ax.set_xlabel("episode, from an empty tree", fontsize=8.5, color=INK)
ax.set_ylabel("episode reward", fontsize=8.5, color=INK)
ax.set_xticks([1, 4, 7, 10])
ax.grid(axis="y", color=RULE, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("bottom", "left"):
    ax.spines[side].set_color(RULE)
ax.tick_params(colors=MUTED, labelsize=8, length=3)
ax.annotate("tree-only", (10, -8300), fontsize=7, color=CONTROL, ha="right")
ax.annotate("Sleep", (10, -12800), fontsize=7, color=SLEEP, ha="right")

handles = [plt.Line2D([], [], color=c, lw=1.5, label=lbl) for _, lbl, c in MODELS]
handles += [plt.Line2D([], [], color=MUTED, lw=1.2, ls=st, label=lb)
            for st, lb in (("-", "multi-network"), ("--", "single-network"))]
fig.legend(handles=handles, frameon=False, fontsize=6.5, ncol=4,
           loc="lower center", bbox_to_anchor=(0.53, -0.17), handlelength=1.3,
           columnspacing=0.9, labelspacing=0.25)

fig.tight_layout(pad=0.4)
fig.savefig(args.out, facecolor="white", bbox_inches="tight")
print(f"wrote {args.out}")

for topo in ("multi", "single"):
    for key, label, _ in MODELS:
        m = band(curves(topo, key))[0]
        print(f"  {topo:>7} {label:<15} ep1 {m[0]:>8.0f}   ep10 {m[-1]:>8.0f}   "
              f"climb {m[-1] - m[0]:>+7.0f}")
