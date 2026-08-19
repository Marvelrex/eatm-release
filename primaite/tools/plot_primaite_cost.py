import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ["seed42", "seed2025", "seed818"]
MODELS = [("claude-opus-5", "Claude Opus 5"), ("gpt-5.6-terra", "GPT-5.6 Terra"),
          ("glm-5.2", "GLM-5.2"), ("deepseek-v4-flash-0731", "DeepSeek-v4"),
          ("qwen3.7-flash", "Qwen3.7")]
LEGACY = {"qwen3.7-flash": "info_qwen3.7", "gpt-5.6-terra": "terra"}
ACD, EATM, CONTROL = "#c8781a", "#2a78d6", "#6b6a66"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d8d6d2"


def cell(topo, arm):
    if arm == "tree":
        names = [f"bm_{topo}_baseline", f"bm_baseline_{topo}"]
    elif arm.startswith("acd:"):
        m = arm[4:]
        names = [f"bm_{topo}_de10_acd_{m}", f"bm_{topo}_de10_acd_{LEGACY.get(m, m)}"]
    else:
        names = [f"bm_{topo}_de10_{arm}", f"bm_{topo}_de10_{LEGACY.get(arm, arm)}"]
    for s in SEEDS:
        d = next((ROOT / "results" / s / n for n in names
                  if (ROOT / "results" / s / n).exists()), None)
        if not d:
            continue
        cs = sorted(d.glob("curve_*.json"))
        if cs:
            yield json.load(open(cs[0], encoding="utf-8"))


def field(topo, arm, key, n):
    return [[e[key] for e in c["episodes"] if e][:n] for c in cell(topo, arm)]


ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "Paper" / "fig_primaite_cost.png"))
args = ap.parse_args()

fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=400)

ys = range(len(MODELS))
for i, (key, label) in enumerate(MODELS):
    for arm, colour, off in ((f"acd:{key}", ACD, 0.17), (key, EATM, -0.17)):
        n = 5 if arm.startswith("acd:") else 10
        v = [x for run in field("multi", arm, "cost_usd", n) for x in run]
        if not v:
            continue
        m = statistics.mean(v)
        ax.plot([m], [i + off], "o", color=colour, ms=5, zorder=3)
        ax.plot([min(v), max(v)], [i + off] * 2, "-", color=colour, lw=1.6,
                alpha=0.45, zorder=2, solid_capstyle="butt")
        ax.annotate(f"{m:.3f}", (m, i + off), textcoords="offset points",
                    xytext=(0, 5 if off < 0 else -11), ha="center",
                    fontsize=7, color=MUTED)
ax.set_yticks(list(ys))
ax.set_yticklabels([lbl for _, lbl in MODELS], fontsize=8.5, color=INK)
ax.set_xlabel("US dollars per episode (multi-network)", fontsize=8.5, color=INK)
ax.set_xscale("log")
ax.set_ylim(len(MODELS) - 0.35, -0.55)

ax.grid(axis="x", color=RULE, lw=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(RULE)
ax.tick_params(colors=MUTED, labelsize=8, length=3)

ax.plot([], [], "o-", color=EATM, label="EATM", ms=5)
ax.plot([], [], "o-", color=ACD, label="ACD", ms=5)
ax.legend(frameon=False, fontsize=8, loc="lower right", handlelength=1.4)

ax.set_xlim(ax.get_xlim()[0]*0.8, ax.get_xlim()[1]*1.9)
fig.tight_layout(pad=0.4)
fig.savefig(args.out, facecolor="white", bbox_inches="tight")
print(f"wrote {args.out}")

for topo in ("multi", "single"):
    for key, label in MODELS:
        a = [x for r in field(topo, f"acd:{key}", "cost_usd", 5) for x in r]
        e = [x for r in field(topo, key, "cost_usd", 10) for x in r]
        print(f"  {topo:>7} {label:<15} ACD ${statistics.mean(a):.4f}  "
              f"EATM ${statistics.mean(e):.4f}  ratio {statistics.mean(a) / statistics.mean(e):.2f}x")
