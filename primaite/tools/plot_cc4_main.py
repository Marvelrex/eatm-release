import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ROWS = [
    ("GPT-5.4",        -3828, -1606, 14.69, 3.24),
    ("GPT-5.4 mini",   -3033,  -672,  2.15, 0.51),
    ("o3-mini",        -2708,  -771, 36.82, 3.79),
    ("Haiku 4.5",      -1542, -1077,  7.74, 1.19),
    ("DeepSeek-v3.2",  -2925, -1241,  1.67, 0.37),
    ("Gemini 3 Flash", -2623,  -953,  0.84, 0.19),
]
BEFORE, AFTER = "#b8b6b2", "#2a78d6"
INK, MUTED, RULE = "#0b0b0b", "#52514e", "#d8d6d2"


def panel(ax, rows, xa, xb, title, xlabel, log=False, fmt="{:,.0f}"):
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
    ap.add_argument("--out", default="Paper/fig_cc4_main.png")
    args = ap.parse_args()

    rows = sorted(ROWS, key=lambda r: r[2])
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
    print(f"wrote {out}\n")
    for n, acd, eatm, ca, ce in rows:
        print(f"  {n:<16}reward {acd:>7,} -> {eatm:>7,} ({acd / eatm:4.1f}x)"
              f"   cost ${ca:>5.2f} -> ${ce:<5.2f} ({ca / ce:4.1f}x)")


if __name__ == "__main__":
    main()
