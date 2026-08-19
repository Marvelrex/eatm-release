from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENTS = {
    0: ("agent 0 — restricted_a (DB)", True),
    1: ("agent 1 — operational_a (web)", False),
    2: ("agent 2 — restricted_b (DB)", True),
    3: ("agent 3 — operational_b (web)", False),
    4: ("agent 4 — HQ ×3 (DB)", True),
}
COLORS = {0: "#e6194B", 1: "#3cb44b", 2: "#4363d8", 3: "#f58231", 4: "#911eb4"}


def episode_files(d: Path):
    fs = [p for p in sorted(d.glob("ep*.json")) if not p.name.endswith(".steps.jsonl")]
    return fs


def per_agent_exposure(epfile: Path):
    d = json.loads(epfile.read_text(encoding="utf-8"))
    out = {}
    for ref, log in (d.get("blue_decision_logs") or {}).items():
        try:
            a = int(ref.rsplit("_", 1)[1])
        except (ValueError, IndexError):
            continue
        log = log or []
        if not log:
            out[a] = 0.0
            continue
        corrupted = sum(1 for e in log if (e.get("threat") or 0) >= 3)
        out[a] = 100.0 * corrupted / len(log)
    return out


def main(dpath: str):
    d = Path(dpath)
    eps = episode_files(d)
    if not eps:
        print("no episode json found in", d)
        return
    series = {a: [] for a in AGENTS}
    xs = list(range(1, len(eps) + 1))
    for ep in eps:
        exp = per_agent_exposure(ep)
        for a in AGENTS:
            series[a].append(exp.get(a, 0.0))

    fig, ax = plt.subplots(figsize=(9, 5))
    for a in AGENTS:
        label, is_db = AGENTS[a]
        ax.plot(xs, series[a], marker="o", ms=5, lw=2, color=COLORS[a], label=label,
                linestyle="-" if is_db else "--")
    ax.set_xlabel("episode")
    ax.set_ylabel("compromise exposure  (% of decision steps with DB corrupted)\nlower = better defence")
    ax.set_title(f"Per-agent defensive performance across {len(eps)} episodes (EATM, CC4)\n"
                 "lower curve = that agent kept its zone(s) cleaner", fontsize=11)
    ax.set_xticks(xs)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(title="blue agent (solid = DB defender, dashed = web zone, no DB)", fontsize=8, loc="best")
    ax.text(0.99, 0.02, "web agents (1,3) own no crown-jewel DB → exposure ≈ 0 by construction",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, style="italic", color="#666")
    out = d / "per_agent_curves.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("\nepisode " + " ".join(f"{x:>5}" for x in xs))
    for a in AGENTS:
        print(f"agent{a}  " + " ".join(f"{v:5.0f}" for v in series[a]))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/cc4_eatm_vs_acd_20260620_214758/eatm")
