import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CONTROL = "#6b6a66"
INK, MUTED = "#0b0b0b", "#52514e"


def load(dirs, labels):
    out = []
    for d, lab in zip(dirs, labels):
        for f in sorted(Path(d).glob("curve_*.json")):
            c = json.load(open(f))
            eps = [e for e in (c.get("episodes") or []) if e]
            if not eps:
                continue
            tag = c.get("tag") or f.stem.replace("curve_", "")
            ctrl = tag == "tree-only"
            name = (c.get("model") or tag).split("/")[-1]
            out.append((f"{name}" if lab is None else f"{name} [{lab}]", ctrl, c, eps))
    return sorted(out, key=lambda r: not r[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None,
                    help="one short label per directory, used when overlaying (e.g. de10 de5)")
    ap.add_argument("--out", default=None,
                    help="output PNG. Defaults to the LAST --dirs directory, beside the curve files "
                         "it was drawn from, named after every source directory so plotting a second "
                         "run cannot silently overwrite the first.")
    args = ap.parse_args()

    labels = args.labels or [None] * len(args.dirs)
    if len(labels) != len(args.dirs):
        raise SystemExit(f"--labels needs one entry per --dirs ({len(args.dirs)} given)")
    arms = load(args.dirs, labels)
    if not arms:
        raise SystemExit(f"no curve_*.json with completed episodes under {args.dirs}")

    meta = arms[0][2]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.5, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1], "hspace": 0.09})
    hue = 0
    rows = []
    for name, ctrl, c, eps in arms:
        color = CONTROL if ctrl else SERIES[hue % len(SERIES)]
        if not ctrl:
            hue += 1
        style = dict(color=color, lw=2.0, marker="o", ms=5, zorder=3)
        if ctrl:
            style.update(ls="--", lw=1.8, marker="", zorder=2)
        xs = [e["episode"] for e in eps]
        ax.plot(xs, [e["reward"] for e in eps], label=name, **style)
        ax2.step(xs, [e["hq_covered"] for e in eps], where="mid",
                 color=color, lw=1.8, ls="--" if ctrl else "-", zorder=style["zorder"])
        last = eps[-1]
        rows.append((name, last["reward"], last["hq_reward"], last["hq_covered"],
                     "/".join(map(str, last["hq_restores"])), c.get("spend_usd") or 0.0))

    ncov = max(len(e[3][-1]["hq_restores"]) for e in arms)
    ax.set_ylabel("episode reward   (penalty-only, 0 is best)", color=INK)
    llm = f", LLM every {meta.get('decide_every','?')} steps (T={meta.get('temperature','?')})" \
        if any(not ctrl for _n, ctrl, _c, _e in arms) else ""
    ax.set_title(f"EATM on PrimAITE: {meta.get('topo','?')} topology, "
                 f"{meta.get('steps','?')} steps/episode{llm}\n"
                 f"each arm trains its own action tree from scratch",
                 fontsize=11, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9, frameon=True, framealpha=0.92, edgecolor="0.8", loc="lower right")
    ax.margins(x=0.02)

    ax2.set_ylabel(f"zones attended\n(of {ncov})", color=INK)
    ax2.set_xlabel("episode", color=INK)
    ax2.set_ylim(-0.3, ncov + 0.3)
    ax2.set_yticks(range(ncov + 1))
    ax2.grid(alpha=0.22)

    if args.out:
        out = Path(args.out)
    else:
        out = Path(args.dirs[-1]) / ("benchmark_" + "_".join(Path(d).name for d in args.dirs) + ".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {out}\n")
    print(f"{'arm':<34}{'final':>10}{'zones':>10}{'cover':>8}{'restores/zone':>20}{'spend':>10}")
    for name, rew, hq, cov, spread, spend in rows:
        print(f"{name:<34}{rew:>10,.0f}{hq:>10,.0f}{cov:>6}/{ncov}{spread:>20}${spend:>9.2f}")


if __name__ == "__main__":
    main()
