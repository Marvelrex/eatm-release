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
MODELS = [("claude-opus-5", "Claude Opus 5"), ("gpt-5.6-terra", "GPT-5.6 Terra"),
          ("glm-5.2", "GLM-5.2"), ("deepseek-v4-flash-0731", "DeepSeek-v4"),
          ("qwen3.7-flash", "Qwen3.7")]
TOPO = "multi"
BEFORE, AFTER = "#b8b6b2", "#2a78d6"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d8d6d2"


def cells(topo, arm):
    if arm.startswith("acd:"):
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


def pooled(topo, arm, key, n):
    return statistics.mean(e[key] for c in cells(topo, arm)
                           for e in [x for x in c["episodes"] if x][:n])


def rows_for(topo):
    out = []
    for k, label in MODELS:
        out.append((label,
                    pooled(topo, f"acd:{k}", "reward", 5),
                    pooled(topo, k, "reward", 10),
                    pooled(topo, f"acd:{k}", "cost_usd", 5),
                    pooled(topo, k, "cost_usd", 10)))
    return out


def panel(ax, rows, xa, xb, title, xlabel, log=False):
    y = range(len(rows))
    for i, r in enumerate(rows):
        ax.plot([r[xa], r[xb]], [i, i], color=RULE, lw=2.4, zorder=1,
                solid_capstyle="round")
        ax.plot(r[xa], i, "o", ms=6, color=BEFORE, mec="white", mew=1.0, zorder=3)
        ax.plot(r[xb], i, "o", ms=6.5, color=AFTER, mec="white", mew=1.0, zorder=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8, color=INK)
    ax.invert_yaxis()
    if log:
        ax.set_xscale("log")
    else:
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_title(title, fontsize=9, color=INK, pad=4)
    ax.set_xlabel(xlabel, fontsize=8, color=INK, labelpad=2)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main():
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--out", default="Paper/fig_primaite_main.png")
    ap.add_argument("--topo", default=TOPO, choices=["multi", "single"])
    args = ap.parse_args()

    rows = sorted(rows_for(args.topo), key=lambda r: r[2])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.0),
                             gridspec_kw={"hspace": 0.70})
    panel(a1, rows, 1, 2, "Episode reward   (0 is best)", "reward")
    panel(a2, rows, 3, 4, "Cost per episode", "USD, log scale", log=True)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=BEFORE, label="ACD baseline"),
               plt.Line2D([], [], marker="o", ls="", ms=6, color=AFTER, label="EATM")]
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=2,
               loc="lower center", bbox_to_anchor=(0.58, -0.10),
               handletextpad=0.4, columnspacing=1.6)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}   ({args.topo}-network topology)\n")
    for n, acd, eatm, ca, ce in rows:
        print(f"  {n:<16}reward {acd:>8,.0f} -> {eatm:>8,.0f} ({acd / eatm:4.2f}x)"
              f"   cost ${ca:>6.3f} -> ${ce:<6.3f} ({ca / ce:4.2f}x)")


if __name__ == "__main__":
    main()
