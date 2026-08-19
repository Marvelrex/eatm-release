import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODELS = ["qwen3.7-flash", "deepseek-v4-flash-0731", "glm-5.2", "gpt-5.6-terra", "claude-opus-5"]

LEGACY_DIRS = {"qwen3.7-flash": "info_qwen3.7", "deepseek-v4-flash-0731": "deepseek-v4-flash-0731",
               "glm-5.2": "glm-5.2", "gpt-5.6-terra": "terra", "claude-opus-5": "claude-opus-5"}

ARM_LABELS = {"eatm": "EATM (Action-Graph tree + LLM chooser)",
              "acd": "ACD pure-LLM (native llms-are-acd interface)"}


def load_curve(dirpath: Path):
    curves = sorted(dirpath.glob("curve_*.json"))
    if not curves:
        return None
    d = json.load(open(curves[0]))
    eps = [e for e in d.get("episodes", []) if e]
    return {"rewards": [e["reward"] for e in eps], "spend": float(d.get("spend_usd") or 0.0),
            "seed": d.get("seed")}


def find(root: Path, topo: str, model: str | None, arm: str = "eatm"):
    if model is None:
        names = [f"bm_{topo}_baseline", f"bm_baseline_{topo}"]
    elif model == "sleep":
        names = [f"bm_{topo}_sleep"]
    elif arm == "acd":
        names = [f"bm_{topo}_de10_acd_{model}"]
    else:
        names = [f"bm_{topo}_de10_{model}", f"bm_{topo}_de10_{LEGACY_DIRS[model]}"]
    return next((root / n for n in names if (root / n).exists()), None)


def arm_tables(args, topo: str, arm: str, deltas: dict, lines: list) -> None:
    lines += [f"### {ARM_LABELS[arm]}", "",
              "| round | sleep (worst case) | baseline mean | " + " | ".join(MODELS) + " |",
              "|---|---|---|" + "---|" * len(MODELS)]
    for rnd in args.rounds:
        root = ROOT / "results" / rnd
        base = load_curve(find(root, topo, None) or root)
        if not base:
            lines.append(f"| {rnd} | – | (no baseline) |" + " – |" * len(MODELS))
            continue
        sleep_dir = find(root, topo, "sleep")
        sleep = load_curve(sleep_dir) if sleep_dir else None
        sleep_cell = f"{statistics.mean(sleep['rewards'][:args.episodes]):,.0f}" if sleep else "–"
        bmean = statistics.mean(base["rewards"][:args.episodes])
        row = [rnd, sleep_cell, f"{bmean:,.0f}"]
        for m in MODELS:
            d = find(root, topo, m, arm)
            c = load_curve(d) if d else None
            if not c or len(c["rewards"]) < args.episodes:
                row.append("–")
                continue
            delta = statistics.mean(c["rewards"][:args.episodes]) - bmean
            deltas.setdefault((arm, topo, m), {})[rnd] = delta
            row.append(f"{delta:+,.0f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "| model | mean delta | min..max | seeds |", "|---|---|---|---|"]
    ranked = sorted(((statistics.mean(v.values()), m, v) for m, v in
                     ((m, deltas.get((arm, topo, m), {})) for m in MODELS) if v), reverse=True)
    for mean_d, m, v in ranked:
        lines.append(f"| {m} | {mean_d:+,.0f} | {min(v.values()):+,.0f}..{max(v.values()):+,.0f}"
                     f" | {len(v)} |")
    lines.append("")


def main():
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument("--rounds", nargs="+", default=["seed2025", "seed42", "seed818"],
                    help="subdirectories of results/ holding one seed round each")
    ap.add_argument("--episodes", type=int, default=5,
                    help="episodes averaged per arm (all arms truncated to the same count; 5 is "
                         "the largest window every arm has on every seed)")
    ap.add_argument("--out", default="results/seed_summary.md")
    args = ap.parse_args()

    lines = ["# Benchmark summary across seed rounds", "",
             f"Rounds: {', '.join(args.rounds)}. Delta = arm mean minus same-round baseline mean, "
             f"all over the first {args.episodes} from-scratch episodes (penalty-only reward, 0 is "
             "best, higher delta = better). Raw rewards are not comparable across rounds.", ""]
    deltas: dict = {}
    for topo in ("multi", "single"):
        lines += [f"## {topo}", ""]
        for arm in ("eatm", "acd"):
            arm_tables(args, topo, arm, deltas, lines)
        lines += ["### EATM vs ACD (cross-seed mean delta, same baseline)", "",
                  "| model | EATM | ACD | EATM advantage |", "|---|---|---|---|"]
        for m in MODELS:
            e, a = deltas.get(("eatm", topo, m)), deltas.get(("acd", topo, m))
            if e and a:
                me, ma = statistics.mean(e.values()), statistics.mean(a.values())
                lines.append(f"| {m} | {me:+,.0f} | {ma:+,.0f} | {me - ma:+,.0f} |")
        lines.append("")

    text = "\n".join(lines)
    out = ROOT / args.out
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
