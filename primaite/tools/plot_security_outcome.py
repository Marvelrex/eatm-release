import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
CONTROL = "#6b6a66"
INK, MUTED = "#0b0b0b", "#52514e"
MODELS = ["gpt-5.6-terra", "claude-opus-5", "glm-5.2", "deepseek-v4-flash-0731", "qwen3.7-flash"]
LEGACY = {"qwen3.7-flash": "info_qwen3.7", "gpt-5.6-terra": "terra"}
NICE = {"gpt-5.6-terra": "GPT-5.6 Terra", "claude-opus-5": "Claude Opus 5", "glm-5.2": "GLM-5.2",
        "deepseek-v4-flash-0731": "DeepSeek-v4", "qwen3.7-flash": "Qwen3.7"}


def episodes(d: Path):
    out = []
    for f in sorted(d.glob("*_ep*.json"), key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1))):
        try:
            j = json.load(open(f, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        m = j.get("meta", {})
        r, tot, mx = m.get("native_blue_reward"), m.get("total_corrupt_steps"), m.get("max_corrupt_steps")
        if r is not None and tot is not None and mx:
            out.append((r, 100.0 * tot / mx))
    return out


def find(root: Path, topo: str, model):
    if model is None:
        names = [f"bm_{topo}_baseline", f"bm_baseline_{topo}"]
    else:
        names = [f"bm_{topo}_de10_{model}", f"bm_{topo}_de10_{LEGACY.get(model, model)}"]
    return next((root / n for n in names if (root / n).exists()), None)


def fit(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    slope = sxy / sxx
    return slope, my - slope * mx, sxy / (sxx * syy) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--rounds", nargs="+", default=["seed42", "seed2025", "seed818"])
    ap.add_argument("--out", default="Paper/fig_security_outcome.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    summary = []
    for ax, topo in zip(axes, ("multi", "single")):
        all_x, all_y = [], []
        for i, model in enumerate(MODELS):
            xs, ys = [], []
            for rnd in args.rounds:
                d = find(ROOT / "results" / rnd, topo, model)
                if d:
                    for r, c in episodes(d):
                        xs.append(r)
                        ys.append(c)
            if xs:
                ax.scatter(xs, ys, s=34, color=SERIES[i], edgecolor="white", linewidth=0.6,
                           label=NICE[model], zorder=3)
                all_x += xs
                all_y += ys
        cx, cy = [], []
        for rnd in args.rounds:
            d = find(ROOT / "results" / rnd, topo, None)
            if d:
                for r, c in episodes(d):
                    cx.append(r)
                    cy.append(c)
        if cx:
            ax.scatter(cx, cy, s=30, marker="x", color=CONTROL, linewidth=1.3,
                       label="tree-only", zorder=2)
            all_x += cx
            all_y += cy

        f = fit(all_x, all_y)
        if f:
            slope, b, r = f
            lo, hi = min(all_x), max(all_x)
            ax.plot([lo, hi], [slope * lo + b, slope * hi + b], color=MUTED, lw=1.4, ls="--",
                    zorder=1)
            ax.annotate(f"$r = {r:.2f}$   $n = {len(all_x)}$", xy=(0.03, 0.06),
                        xycoords="axes fraction", fontsize=9.5, color=INK)
            summary.append((topo, r, len(all_x), slope))
        ax.set_title(f"{topo}-network topology", fontsize=11, color=INK)
        ax.set_xlabel("episode reward   (penalty-only, 0 is best)", color=INK, fontsize=9.5)
        ax.grid(alpha=0.22)
        ax.margins(0.04)
        ax.set_ylim(top=min(ax.get_ylim()[1], 101))
    axes[0].set_ylabel("zone-steps with a corrupted\ncrown jewel   (% of worst case)",
                       color=INK, fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, frameon=False, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.06))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}\n")
    for topo, r, n, slope in summary:
        print(f"{topo:<8} pearson r = {r:+.3f}   n = {n:>3} episodes   "
              f"slope = {slope * 1000:+.2f} pct-points per 1000 reward")


if __name__ == "__main__":
    main()
