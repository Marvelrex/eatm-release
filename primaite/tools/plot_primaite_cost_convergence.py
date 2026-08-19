import argparse
import json
import statistics
import sys
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
ACD, EATM = "#c8781a", "#2a78d6"
CONTROL, SLEEP = "#6b6a66", "#b4b2ae"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d8d6d2"
N = 10


def cells(topo, arm):
    if arm == "tree":
        names = [f"bm_{topo}_baseline", f"bm_baseline_{topo}"]
    elif arm == "sleep":
        names = [f"bm_{topo}_sleep"]
    elif arm.startswith("acd:"):
        m = arm[4:]
        names = [f"bm_{topo}_de10_acd_{m}", f"bm_{topo}_de10_acd_{LEGACY.get(m, m)}"]
    else:
        names = [f"bm_{topo}_de10_{arm}", f"bm_{topo}_de10_{LEGACY.get(arm, arm)}"]
    out = []
    for s in SEEDS:
        d = next((ROOT / "results" / s / n for n in names
                  if (ROOT / "results" / s / n).exists()), None)
        if d:
            cs = sorted(d.glob("curve_*.json"))
            if cs:
                out.append(json.load(open(cs[0], encoding="utf-8")))
    return out


def field(topo, arm, key, n):
    return [[e[key] for e in c["episodes"] if e][:n] for c in cells(topo, arm)]


def per_episode_mean(runs):
    return [statistics.mean(r[i] for r in runs if i < len(r)) for i in range(N)]


def smooth(y, w=3):
    h = w // 2
    return [statistics.mean(y[max(0, i - h):min(len(y), i + h + 1)]) for i in range(len(y))]


ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "Paper" / "fig_primaite_cost_convergence.png"))
args = ap.parse_args()

fig, (axc, axk) = plt.subplots(1, 2, figsize=(7.1, 2.45), dpi=400,
                               gridspec_kw={"width_ratios": [1, 1.15]})

for i, (key, label, _) in enumerate(MODELS):
    means = {}
    for arm, colour in ((key, EATM), (f"acd:{key}", ACD)):
        n = 5 if arm.startswith("acd:") else N
        v = [x for run in field("multi", arm, "cost_usd", n) for x in run]
        if v:
            means[colour] = statistics.mean(v)
    if len(means) != 2:
        continue
    lo, hi = means[EATM], means[ACD]
    axc.plot([lo, hi], [i, i], "-", color=RULE, lw=2.2, zorder=2,
             solid_capstyle="round")
    axc.plot([lo], [i], "o", color=EATM, ms=5, zorder=3)
    axc.plot([hi], [i], "o", color=ACD, ms=5, zorder=3)
    axc.annotate(f"{lo:.3f}", (lo, i), textcoords="offset points",
                 xytext=(-6, -2.5), ha="right", fontsize=6.5, color=MUTED)
    axc.annotate(f"{hi:.3f}", (hi, i), textcoords="offset points",
                 xytext=(6, -2.5), ha="left", fontsize=6.5, color=MUTED)
axc.set_yticks(range(len(MODELS)))
axc.set_yticklabels([lbl for _, lbl, _ in MODELS], fontsize=8, color=INK)
axc.set_xlabel("US dollars per episode (multi-network)", fontsize=8, color=INK)
axc.set_xscale("log")
axc.set_xlim(0.0065, 26)
axc.set_ylim(len(MODELS) - 0.5, -0.5)
axc.plot([], [], "o", color=EATM, label="EATM", ms=5)
axc.plot([], [], "o", color=ACD, label="ACD", ms=5)
axc.legend(frameon=False, fontsize=7.5, loc="lower right", handlelength=1.3)
axc.grid(axis="x", color=RULE, lw=0.6, zorder=0)
for side in ("top", "right", "left"):
    axc.spines[side].set_visible(False)

x = range(1, N + 1)
for topo, style in (("multi", "-"), ("single", "--")):
    for arm, colour, lw, z in (("sleep", SLEEP, 1.2, 1), ("tree", CONTROL, 1.5, 2)):
        runs = field(topo, arm, "reward", N)
        if runs:
            axk.plot(x, smooth(per_episode_mean(runs)), style, color=colour,
                     lw=lw, zorder=z)
    for key, _, colour in MODELS:
        runs = field(topo, key, "reward", N)
        if runs:
            axk.plot(x, smooth(per_episode_mean(runs)), style, color=colour,
                     lw=1.4, zorder=4)
axk.set_xlabel("episode, from an empty tree", fontsize=8, color=INK)
axk.set_ylabel("episode reward", fontsize=8, color=INK)
axk.set_xticks([1, 4, 7, 10])
axk.grid(axis="y", color=RULE, lw=0.6, zorder=0)
axk.annotate("tree-only", (10, -8300), fontsize=7, color=CONTROL, ha="right")
axk.annotate("Sleep", (10, -12800), fontsize=7, color=SLEEP, ha="right")
for side in ("top", "right"):
    axk.spines[side].set_visible(False)
axk.spines["left"].set_color(RULE)

for ax in (axc, axk):
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=7.5, length=3)

handles = [plt.Line2D([], [], color=c, lw=1.5, label=lbl) for _, lbl, c in MODELS]
handles += [plt.Line2D([], [], color=MUTED, lw=1.2, ls=st, label=lb)
            for st, lb in (("-", "multi-network"), ("--", "single-network"))]
fig.legend(handles=handles, frameon=False, fontsize=7, ncol=7, loc="lower center",
           bbox_to_anchor=(0.5, -0.09), handlelength=1.3, columnspacing=1.0)

fig.tight_layout(pad=0.5)
fig.savefig(args.out, facecolor="white", bbox_inches="tight")
print(f"wrote {args.out}")

for topo in ("multi", "single"):
    r = []
    for key, label, _ in MODELS:
        a = statistics.mean(x for run in field(topo, f"acd:{key}", "cost_usd", 5) for x in run)
        e = statistics.mean(x for run in field(topo, key, "cost_usd", N) for x in run)
        r.append(a / e)
    print(f"  {topo:>7}  ACD/EATM cost ratio {min(r):.2f}x to {max(r):.2f}x")
