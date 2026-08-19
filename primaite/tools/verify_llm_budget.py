import json
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = [sys.argv[0]]
from summarize_acd import MODELS, SEEDS, TOPOS, LABEL, cell_dir

fails = []


def episode_budget(seed, topo, model, acd):
    d = cell_dir(seed, topo, f"acd_{model}" if acd else model)
    fs = sorted(d.glob("*_ep0.steps.jsonl"))
    if not fs:
        return None
    n = calls = tokens = 0
    for line in fs[0].open(encoding="utf-8"):
        for b in json.loads(line)["blue"]:
            dec = b.get("decision") or {}
            used = (dec.get("usage") or {}).get("total_tokens", 0)
            n += 1
            if used:
                calls += 1
                tokens += used
    return n, calls, tokens


print("=== share of blue decisions backed by an LLM call, episode 0 of every cell ===")
print(f"{'backbone':>16}{'topo':>9}{'seed':>7}{'EATM %':>9}{'ACD %':>8}"
      f"{'EATM tok':>10}{'ACD tok':>10}{'tok ratio':>11}")
shares, ratios = [], []
for m in MODELS:
    for t in TOPOS:
        for s in SEEDS:
            e, a = episode_budget(s, t, m, False), episode_budget(s, t, m, True)
            if not e or not a:
                print(f"{LABEL[m]:>16}{t:>9}{s.replace('seed', ''):>7}   MISSING")
                fails.append(f"{m}/{t}/{s}")
                continue
            es, as_ = 100 * e[1] / e[0], 100 * a[1] / a[0]
            shares += [es, as_]
            ratios.append(a[2] / e[2])
            print(f"{LABEL[m]:>16}{t:>9}{s.replace('seed', ''):>7}{es:>9.1f}{as_:>8.1f}"
                  f"{e[2]:>10}{a[2]:>10}{a[2] / e[2]:>11.2f}")

print(f"\n  LLM-backed share across all 60 episodes: "
      f"{min(shares):.1f} to {max(shares):.1f} percent")
if not 9.5 <= min(shares) <= max(shares) <= 10.0:
    fails.append(f"call share outside 9.5-10.0 percent: {min(shares):.1f}-{max(shares):.1f}")
worst = max(abs(shares[i] - shares[i + 1]) for i in range(0, len(shares), 2))
print(f"  largest EATM-vs-ACD difference in call share within a cell: {worst:.1f} points")
if worst > 0.5:
    fails.append("the two arms do not share a call budget")
print(f"  tokens per episode, ACD over EATM: {min(ratios):.2f} to {max(ratios):.2f}x, "
      f"mean {statistics.mean(ratios):.2f}x")

print("\n" + ("ALL CHECKS PASS" if not fails else f"{len(fails)} PROBLEM: " + "; ".join(fails)))
