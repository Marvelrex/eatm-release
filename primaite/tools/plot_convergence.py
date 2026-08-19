import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DO_NOTHING = -13191
SCRIPTED = -5228
LABELS = {"free": "Restore free (the exploit)",
          "cost1": "Restore priced -1, mission phase pinned (as ported)",
          "phase": "Restore priced -1, mission phase advancing (fixed)"}
COLORS = ["#c1440e", "#1f6f8b", "#3f7d20", "#6b4c9a"]
FLAT_FRACTION = 0.10


def load(path):
    d = json.load(open(path))
    return d.get("tag", Path(path).stem), d.get("episodes") or []


def tail_stats(rewards, n):
    tail = [r for r in rewards[-n:] if r is not None]
    if len(tail) < 2:
        return None, None, False
    m, span = statistics.mean(tail), max(tail) - min(tail)
    return m, span, span <= abs(m) * FLAT_FRACTION


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curves", nargs="+",
                    default=[str(ROOT / f"results/smoke/curve_{t}.json")
                             for t in ("free", "cost1", "phase")])
    ap.add_argument("--out", default=None,
                    help="output PNG. Defaults to beside the last --curves file, so a run's "
                         "figure travels with its data.")
    ap.add_argument("--tail", type=int, default=10, help="episodes averaged to report the endpoint")
    args = ap.parse_args()

    curves = [load(p) for p in args.curves if Path(p).exists()]
    if not curves:
        raise SystemExit(f"no curve files among {args.curves}\n"
                         f"run tools/backfill_curves.py, or a fresh tools/smoke_graphonly.py sweep")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.08})
    summary = []
    for i, (tag, eps) in enumerate(curves):
        c = COLORS[i % len(COLORS)]
        xs = [e["episode"] for e in eps]
        rewards = [e["reward"] for e in eps]
        m, span, flat = tail_stats(rewards, args.tail)
        note = "" if flat else "  (still descending)"
        ax.plot(xs, rewards, "-o", ms=3.2, lw=1.6, color=c,
                label=f"{LABELS.get(tag, tag)} — ends {rewards[-1]:,}{note}")
        if flat:
            ax.axhline(m, color=c, ls=":", lw=1.2, alpha=0.9)
            ax.annotate(f"converged {m:,.0f}", xy=(xs[-1], m), xytext=(-6, 9),
                        textcoords="offset points", ha="right", fontsize=9, color=c, weight="bold")
        ax2.plot(xs, [e.get("sleep_pct") for e in eps], "-", lw=1.6, color=c)
        ax2.plot(xs, [e.get("top_bucket_pct") for e in eps], "--", lw=1.2, color=c, alpha=0.6)
        summary.append((tag, len(eps), rewards[-1], m, span, flat))

    for y, lab in ((DO_NOTHING, "do nothing"), (SCRIPTED, "scripted defender")):
        ax.axhline(y, color="0.45", ls="--", lw=1.1, zorder=0)
        ax.annotate(lab, xy=(1, y), xytext=(0, -12), textcoords="offset points",
                    fontsize=9, color="0.35")

    ax.set_ylabel("episode reward   (penalty-only, 0 is best)")
    ax.set_title("EATM tree-only on the PrimAITE CC4 port: what the game's rules do to the policy\n"
                 "500 steps per episode; each curve changes one rule from the one above it",
                 fontsize=11.5)
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.92, edgecolor="0.8")
    ax.grid(alpha=0.25)
    ax.margins(x=0.02)

    ax2.set_xlabel("episode")
    ax2.set_ylabel("percent")
    ax2.set_ylim(-3, 108)
    ax2.grid(alpha=0.25)
    ax2.plot([], [], "-", color="0.3", label="Sleep share")
    ax2.plot([], [], "--", color="0.3", alpha=0.7, label="top context bucket occupancy")
    ax2.legend(loc="upper right", fontsize=9, ncol=2, frameon=True, framealpha=0.92, edgecolor="0.8")

    out = Path(args.out) if args.out else (
        Path(args.curves[-1]).parent / ("convergence_" + "_".join(t for t, _e in curves) + ".png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}\n")
    print(f"{'curve':<8}{'eps':>5}{'final':>10}{'tail mean':>12}{'tail span':>11}  status")
    for tag, n, final, m, span, flat in summary:
        print(f"{tag:<8}{n:>5}{final:>10,}{m:>12,.0f}{span:>11,.0f}  "
              f"{'converged' if flat else 'STILL DESCENDING'}")


if __name__ == "__main__":
    main()
