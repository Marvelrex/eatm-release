import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SEEDS = ["seed42", "seed2025", "seed818"]
ARMS = [("tree", "tree-only"), ("claude-opus-5", "Claude Opus 5"),
        ("gpt-5.6-terra", "GPT-5.6 Terra"), ("glm-5.2", "GLM-5.2"),
        ("deepseek-v4-flash-0731", "DeepSeek-v4"), ("qwen3.7-flash", "Qwen3.7")]
LEGACY = {"qwen3.7-flash": "info_qwen3.7", "gpt-5.6-terra": "terra"}
FAMILIES = ["Sleep", "Restore", "Analyse", "Remove"]
COLOUR = {"Sleep": "#b8b6b2", "Restore": "#2a78d6", "Analyse": "#1baf7a", "Remove": "#eda100"}
INK, MUTED = "#0b0b0b", "#52514e"


def mix(topo, arm):
    tot = {}
    for s in SEEDS:
        if arm == "tree":
            names = [f"bm_{topo}_baseline", f"bm_baseline_{topo}"]
        else:
            names = [f"bm_{topo}_de10_{arm}", f"bm_{topo}_de10_{LEGACY.get(arm, arm)}"]
        d = next((ROOT / "results" / s / n for n in names
                  if (ROOT / "results" / s / n).exists()), None)
        if not d:
            continue
        f = sorted(d.glob("*_ep9.json"))
        if not f:
            continue
        g = json.load(open(f[0], encoding="utf-8")).get("eatm_action_graph", {})
        for info in g.values():
            for fam, n in (info.get("family_visits") or {}).items():
                tot[fam] = tot.get(fam, 0) + n
    n = sum(tot.values())
    return [100 * tot.get(f, 0) / n for f in FAMILIES] if n else None


def main():
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--out", default="Paper/fig_action_mix.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0), sharey=True)
    for ax, topo in zip(axes, ("multi", "single")):
        rows = [(nice, mix(topo, key)) for key, nice in ARMS]
        rows = [(n, v) for n, v in rows if v]
        y = range(len(rows))
        left = [0.0] * len(rows)
        for fi, fam in enumerate(FAMILIES):
            vals = [r[1][fi] for r in rows]
            ax.barh(y, vals, left=left, height=0.66, color=COLOUR[fam],
                    edgecolor="white", linewidth=1.2,
                    label=fam if ax is axes[0] else None)
            for i, (v, l) in enumerate(zip(vals, left)):
                if v >= 12:
                    ax.text(l + v / 2, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=8.5, color="white" if fam != "Sleep" else INK,
                            weight="bold")
            left = [a + b for a, b in zip(left, vals)]
        ax.set_yticks(list(y))
        ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("share of the episode's steps (%)", fontsize=9, color=INK)
        ax.set_title(f"{topo}-network topology", fontsize=10.5, color=INK)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].legend(fontsize=9, frameon=False, ncol=4, loc="upper center",
                   bbox_to_anchor=(1.02, -0.28))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}\n")
    for topo in ("multi", "single"):
        print(f"-- {topo}")
        for key, nice in ARMS:
            v = mix(topo, key)
            if v:
                print(f"   {nice:<16}" + "  ".join(f"{f}={x:.0f}%"
                                                   for f, x in zip(FAMILIES, v)))


if __name__ == "__main__":
    main()
