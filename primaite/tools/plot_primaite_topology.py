import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
RESTRICTED_A, OPERATIONAL, RESTRICTED_B = "#6bb8e8", "#b5e0a8", "#f5c09b"
HQ, NEUTRAL, OUTER = "#c5e3f5", "#e6e6e6", "#ffffff"
LINK, RED, INK, MUTED = "#2a6f97", "#cc2200", "#0b0b0b", "#52514e"

BW, BH = 1.44, 0.84
LEFT, MID, RIGHT = -1.15, 1.03, 3.21
TOP, ROW, FLOOR = 2.14, 1.12, -0.10


def block(ax, x, y, name, fill, agent=None, step=None):
    ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=0.6, edgecolor="#8a8a8a", facecolor=fill, zorder=3))
    ax.text(x + BW / 2, y + BH * 0.38, name, ha="center", va="center", fontsize=6.4,
            color=INK, zorder=4, linespacing=1.3)
    if agent is not None:
        ax.text(x + BW - 0.07, y + BH - 0.12, f"$\\mathbb{{D}}_{agent}$ Agent {agent}",
                ha="right", va="center", fontsize=5.0, color=INK, weight="bold", zorder=4)
    if step:
        ax.text(x + 0.09, y + BH - 0.12, str(step), ha="left", va="center", fontsize=5.4,
                color=RED, weight="bold", zorder=4)


def outer(ax, x0, y0, x1, y1):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.0, edgecolor="#5a5a5a", facecolor=OUTER, zorder=2))


def link(ax, a, b, colour=LINK, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=6, color=colour,
                                 lw=0.9, shrinkA=1, shrinkB=2, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--out", default="Paper/.build/fig_primaite_topology_generated.png")
    args = ap.parse_args()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.16, 2.62))

    outer(axL, -1.28, -0.22, 4.78, 0.86)
    for i, n in enumerate(("Public\nAccess", "Admin\nNetwork", "Office\nNetwork")):
        block(axL, LEFT + i * 2.18, FLOOR, n, HQ, step=6 if i == 0 else None)
    axL.text(-1.28, 0.99, "$\\mathbb{D}_4$ Agent 4,  three zones", ha="left",
             va="center", fontsize=5.8, color=INK, weight="bold", zorder=6)

    block(axL, MID, ROW, "Contractor\nNetwork", NEUTRAL, step=1)
    block(axL, LEFT, TOP, "Restricted\nZone A", RESTRICTED_A, agent=0, step=2)
    block(axL, LEFT, ROW, "Operational\nZone A", OPERATIONAL, agent=1, step=3)
    block(axL, RIGHT, TOP, "Restricted\nZone B", RESTRICTED_B, agent=2, step=4)
    block(axL, RIGHT, ROW, "Operational\nZone B", OPERATIONAL, agent=3, step=5)

    for tgt, rad in (((LEFT + BW, TOP + 0.30), 0.16), ((LEFT + BW, ROW + 0.42), 0.0)):
        link(axL, (MID, ROW + 0.55), tgt, rad=rad)
    for tgt, rad in (((RIGHT, TOP + 0.30), -0.16), ((RIGHT, ROW + 0.42), 0.0)):
        link(axL, (MID + BW, ROW + 0.55), tgt, rad=rad)
    link(axL, (MID + BW / 2, ROW), (MID + BW / 2, 0.86))
    link(axL, (MID + BW / 2, 2.98), (MID + BW / 2, ROW + BH + 0.04), colour=RED)
    axL.text(MID + BW / 2, 3.08, "attacker entry", ha="center", fontsize=5.8, color=RED)
    axL.set_title("PrimAITE multi-network topology\n7 zones, 5 defenders",
                  fontsize=8.0, color=INK, pad=3)

    outer(axR, -1.28, -0.22, 4.78, 2.10)
    for i, n in enumerate(("Zone\nCharlie", "Zone\nDelta", "Zone\nEcho")):
        block(axR, LEFT + i * 2.18, FLOOR, n, HQ, step=i + 3)
    for i, n in enumerate(("Zone\nAlpha", "Zone\nBravo")):
        block(axR, LEFT + 1.09 + i * 2.18, ROW, n, HQ, step=i + 1)
    axR.text(4.70, 2.22, "$\mathbb{D}_0$ Agent 0,  all five zones",
             ha="right", fontsize=5.8, color=INK, weight="bold")
    entry = LEFT + 1.09 + BW / 2
    link(axR, (entry, 2.98), (entry, ROW + BH + 0.04), colour=RED)
    axR.text(entry, 3.08, "attacker entry", ha="center", fontsize=5.8, color=RED)
    axR.set_title("PrimAITE single-network topology\n5 zones, 1 defender",
                  fontsize=8.0, color=INK, pad=3)

    for ax in (axL, axR):
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.42, 4.92)
        ax.set_ylim(-0.36, 3.26)
    fig.subplots_adjust(wspace=0.04)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
