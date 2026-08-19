import json
import re
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1] / "results"
SEEDS = ["seed42", "seed2025", "seed818"]
MODELS = ["claude-opus-5", "gpt-5.6-terra", "glm-5.2", "deepseek-v4-flash-0731", "qwen3.7-flash"]
NICE = dict(zip(MODELS, ["Claude Opus 5", "GPT-5.6 Terra", "GLM-5.2", "DeepSeek-v4", "Qwen3.7"]))
LEGACY = {"qwen3.7-flash": "info_qwen3.7", "gpt-5.6-terra": "terra"}
N = 10

fails = []


def check(name, got, want, tol=0.51):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<52} paper {want:>10,.2f}   data {got:>10,.2f}")
    if not ok:
        fails.append(name)


def curve_dir(root, topo, arm):
    if arm == "tree":
        names = [f"bm_{topo}_baseline", f"bm_baseline_{topo}"]
    elif arm == "sleep":
        names = [f"bm_{topo}_sleep"]
    else:
        names = [f"bm_{topo}_de10_{arm}", f"bm_{topo}_de10_{LEGACY.get(arm, arm)}"]
    return next((root / n for n in names if (root / n).exists()), None)


def rewards(seed, topo, arm):
    d = curve_dir(ROOT / seed, topo, arm)
    if not d:
        return []
    cs = sorted(d.glob("curve_*.json"))
    if not cs:
        return []
    return [e["reward"] for e in json.load(open(cs[0]))["episodes"] if e][:N]


R = {(s, t, a): rewards(s, t, a)
     for s in SEEDS for t in ("multi", "single") for a in MODELS + ["tree", "sleep"]}
for k, v in R.items():
    assert len(v) == N, f"{k} has {len(v)} episodes, expected {N}"
print(f"loaded {len(R)} cells x {N} episodes, none short\n")

M = {k: statistics.mean(v) for k, v in R.items()}
POOL = {(t, a): [x for s in SEEDS for x in R[(s, t, a)]] for t in ("multi", "single")
        for a in MODELS + ["tree", "sleep"]}

TABLE = {
    ("multi", "claude-opus-5"): (-3928, 1047), ("multi", "gpt-5.6-terra"): (-4412, 2331),
    ("multi", "glm-5.2"): (-4767, 2366), ("multi", "deepseek-v4-flash-0731"): (-5047, 2090),
    ("multi", "qwen3.7-flash"): (-5252, 1974), ("multi", "tree"): (-9026, 2354),
    ("multi", "sleep"): (-13147, 73),
    ("single", "claude-opus-5"): (-5713, 940), ("single", "gpt-5.6-terra"): (-5774, 1005),
    ("single", "glm-5.2"): (-6110, 1156), ("single", "deepseek-v4-flash-0731"): (-7179, 1009),
    ("single", "qwen3.7-flash"): (-6385, 1186), ("single", "tree"): (-9437, 1209),
    ("single", "sleep"): (-10421, 109),
}
print("=== Table V: reward mean and std over 30 pooled episodes ===")
for (t, a), (wm, wsd) in TABLE.items():
    check(f"{t}/{NICE.get(a, a)} mean", statistics.mean(POOL[(t, a)]), wm)
    check(f"{t}/{NICE.get(a, a)} std", statistics.stdev(POOL[(t, a)]), wsd)

print("\n=== Table V: Tree/LLM % ===")
SHARE = {("multi", "claude-opus-5"): 45, ("multi", "gpt-5.6-terra"): 47,
         ("multi", "glm-5.2"): 49, ("multi", "deepseek-v4-flash-0731"): 51,
         ("multi", "qwen3.7-flash"): 52,
         ("single", "claude-opus-5"): 21, ("single", "gpt-5.6-terra"): 21,
         ("single", "glm-5.2"): 23, ("single", "deepseek-v4-flash-0731"): 30,
         ("single", "qwen3.7-flash"): 24}
gaps = {}
for (t, a), want in SHARE.items():
    over_sleep = statistics.mean(M[(s, t, a)] - M[(s, t, "sleep")] for s in SEEDS)
    over_tree = statistics.mean(M[(s, t, a)] - M[(s, t, "tree")] for s in SEEDS)
    gaps[(t, a)] = (over_sleep, over_tree)
    check(f"{t}/{NICE[a]} tree share", 100 * (over_sleep - over_tree) / over_sleep, want, tol=0.5)

print("\n=== section V-C prose ===")
tree_gain = {t: statistics.mean(M[(s, t, "tree")] - M[(s, t, "sleep")] for s in SEEDS)
             for t in ("multi", "single")}
check("multi tree gain over sleep", tree_gain["multi"], 4120)
check("single tree gain over sleep", tree_gain["single"], 984)
check("multi LLM agent min (Qwen3.7 vs tree)", gaps[("multi", "qwen3.7-flash")][1], 3774)
check("multi LLM agent max (Opus 5 vs tree)", gaps[("multi", "claude-opus-5")][1], 5099)
check("single LLM agent min (DeepSeek-v4 vs tree)", gaps[("single", "deepseek-v4-flash-0731")][1], 2258)
check("single LLM agent max (Opus 5 vs tree)", gaps[("single", "claude-opus-5")][1], 3724)

per_seed_tree_single = [M[(s, "single", "tree")] - M[(s, "single", "sleep")] for s in SEEDS]
for s, want in zip(SEEDS, (375, 2272, 304)):
    check(f"single tree gain, {s}", M[(s, "single", "tree")] - M[(s, "single", "sleep")], want)

drops = sorted(gaps[("multi", a)][1] - gaps[("single", a)][1] for a in MODELS)
check("multi-minus-single LLM agent gap, min", drops[0], 722)
check("multi-minus-single LLM agent gap, max", drops[-1], 1721)

mg = {a: gaps[("multi", a)][1] for a in MODELS}
sg = {a: gaps[("single", a)][1] for a in MODELS}
check("family spread factor, multi", max(mg.values()) / min(mg.values()), 1.35, tol=0.005)
check("family spread factor, single", max(sg.values()) / min(sg.values()), 1.65, tol=0.005)

print("\n  all thirty (model, topology, seed) gaps vs the same-seed control positive? ", end="")
allpos = all(M[(s, t, a)] - M[(s, t, "tree")] > 0
             for s in SEEDS for t in ("multi", "single") for a in MODELS)
print("YES" if allpos else "NO")
if not allpos:
    fails.append("thirty positive cells")

print("  ordering by LLM agent gain, multi :", [NICE[a] for a in sorted(mg, key=mg.get, reverse=True)])
print("  ordering by LLM agent gain, single:", [NICE[a] for a in sorted(sg, key=sg.get, reverse=True)])

print("\n=== zone coverage (hq_covered) ===")
COV = {("multi", "claude-opus-5"): 3.00, ("multi", "gpt-5.6-terra"): 3.00,
       ("multi", "glm-5.2"): 2.83, ("multi", "deepseek-v4-flash-0731"): 2.77,
       ("multi", "qwen3.7-flash"): 2.97, ("multi", "tree"): 0.80, ("multi", "sleep"): 0.00,
       ("single", "claude-opus-5"): 4.40, ("single", "gpt-5.6-terra"): 4.33,
       ("single", "glm-5.2"): 3.17, ("single", "deepseek-v4-flash-0731"): 2.40,
       ("single", "qwen3.7-flash"): 3.80, ("single", "tree"): 0.57, ("single", "sleep"): 0.00}


def covers(topo, arm):
    out = []
    for s in SEEDS:
        d = curve_dir(ROOT / s, topo, arm)
        if d:
            cs = sorted(d.glob("curve_*.json"))
            if cs:
                out += [e["hq_covered"] for e in json.load(open(cs[0]))["episodes"] if e][:N]
    return out


for (t, a), want in COV.items():
    check(f"{t}/{NICE.get(a, a)} zones covered", statistics.mean(covers(t, a)), want, tol=0.005)

ctrl_eps = covers("multi", "tree") + covers("single", "tree")
check("control episodes", len(ctrl_eps), 60, tol=0)
check("control max zones covered in any episode", max(ctrl_eps), 1, tol=0)


def restores(topo, arm):
    rows = []
    for s in SEEDS:
        d = curve_dir(ROOT / s, topo, arm)
        if d:
            cs = sorted(d.glob("curve_*.json"))
            if cs:
                rows += [e["hq_restores"] for e in json.load(open(cs[0]))["episodes"] if e][:N]
    return [statistics.mean(r[i] for r in rows) for i in range(len(rows[0]))]


mr = restores("multi", "tree")
check("control busiest multi zone, mean Restores", max(mr), 237, tol=0.5)
check("control idlest multi zone, mean Restores", min(mr), 0, tol=0)

print("\n=== figure: reward vs share of compromised zone-steps ===")


def episodes(d):
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


for topo, want_r, want_slope, want_n in (("multi", -0.94, -8.9, 210), ("single", -0.82, -4.0, 210)):
    xs, ys = [], []
    for s in SEEDS:
        for a in MODELS + ["tree"]:
            d = curve_dir(ROOT / s, topo, a)
            if d:
                for x, y in episodes(d):
                    xs.append(x)
                    ys.append(y)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    check(f"{topo} n episodes", len(xs), want_n, tol=0)
    check(f"{topo} pearson r", sxy / (sxx * syy) ** 0.5, want_r, tol=0.005)
    check(f"{topo} slope per 1000 reward", 1000 * sxy / sxx, want_slope, tol=0.05)

def acd(seed, topo, model):
    d = ROOT / seed / f"bm_{topo}_de10_acd_{model}"
    cs = sorted(d.glob("curve_*.json"))
    assert cs, d
    return [e["reward"] for e in json.load(open(cs[0], encoding="utf-8"))["episodes"] if e][:5]


ACD = {(t, a): [x for s in SEEDS for x in acd(s, t, a)]
       for t in ("multi", "single") for a in MODELS}
for k, v in ACD.items():
    assert len(v) == 15, f"{k} has {len(v)} ACD episodes, expected 15"

print("\n=== Table V: ACD arm, 15 episodes per cell ===")
ACD_TABLE = {
    ("multi", "claude-opus-5"): (-6979, 150), ("multi", "gpt-5.6-terra"): (-6921, 189),
    ("multi", "glm-5.2"): (-7252, 206), ("multi", "deepseek-v4-flash-0731"): (-7938, 302),
    ("multi", "qwen3.7-flash"): (-8125, 244),
    ("single", "claude-opus-5"): (-9648, 141), ("single", "gpt-5.6-terra"): (-9591, 155),
    ("single", "glm-5.2"): (-9561, 153), ("single", "deepseek-v4-flash-0731"): (-9812, 150),
    ("single", "qwen3.7-flash"): (-9757, 146),
}
for (t, a), (wm, ws) in ACD_TABLE.items():
    check(f"{t}/{NICE[a]} ACD mean", statistics.mean(ACD[(t, a)]), wm)
    check(f"{t}/{NICE[a]} ACD std", statistics.stdev(ACD[(t, a)]), ws)

print("\n=== Table V interaction column, and the section V-C ranges ===")
INTER = {("multi", "claude-opus-5"): -12, ("multi", "gpt-5.6-terra"): -18,
         ("multi", "glm-5.2"): -20, ("multi", "deepseek-v4-flash-0731"): -15,
         ("multi", "qwen3.7-flash"): -16,
         ("single", "claude-opus-5"): 63, ("single", "gpt-5.6-terra"): 61,
         ("single", "glm-5.2"): 57, ("single", "deepseek-v4-flash-0731"): 51,
         ("single", "qwen3.7-flash"): 59}
comp = {}
for t in ("multi", "single"):
    base = statistics.mean(POOL[(t, "sleep")])
    tr = statistics.mean(POOL[(t, "tree")]) - base
    for a in MODELS:
        llm = statistics.mean(ACD[(t, a)]) - base
        both = statistics.mean(POOL[(t, a)]) - base
        comp[(t, a)] = (tr, llm, both, both - tr - llm)
        check(f"{t}/{NICE[a]} interaction %", 100 * (both - tr - llm) / both, INTER[(t, a)])

for t, wt, wlmin, wlmax, wbmin, wbmax, wimin, wimax in (
        ("multi", 4120, 5021, 6225, 7895, 9219, -20, -12),
        ("single", 984, 609, 860, 3242, 4708, 51, 63)):
    col = [c for k, c in comp.items() if k[0] == t]
    check(f"{t} tree alone", col[0][0], wt)
    check(f"{t} LLM alone, min", min(c[1] for c in col), wlmin)
    check(f"{t} LLM alone, max", max(c[1] for c in col), wlmax)
    check(f"{t} both components, min", min(c[2] for c in col), wbmin)
    check(f"{t} both components, max", max(c[2] for c in col), wbmax)
    check(f"{t} interaction %, min", min(100 * c[3] / c[2] for c in col), wimin)
    check(f"{t} interaction %, max", max(100 * c[3] / c[2] for c in col), wimax)

worst = max(max(100 * c[0] / c[2], 100 * c[1] / c[2])
            for k, c in comp.items() if k[0] == "single")
print(f"\n  largest single-component share, single-network: {worst:.1f} percent"
      "   (abstract claims under a third)")
if worst >= 33.3:
    fails.append("abstract: under a third")


print("\n=== section V-C: what the tree adds on top of the LLM ===")
for t, wmin, wmax in (("multi", 2484, 3052), ("single", 2633, 3935)):
    d = [statistics.mean(POOL[(t, a)]) - statistics.mean(ACD[(t, a)]) for a in MODELS]
    check(f"{t} tree over ACD, min", min(d), wmin)
    check(f"{t} tree over ACD, max", max(d), wmax)
allpos = all(M[(s, t, a)] > statistics.mean(acd(s, t, a))
             for s in SEEDS for t in ("multi", "single") for a in MODELS)
print("  EATM over ACD positive in all thirty cells? ", "YES" if allpos else "NO")
if not allpos:
    fails.append("thirty positive ACD cells")

sd_acd = [statistics.stdev(ACD[(t, a)]) for t in ("multi", "single") for a in MODELS]
sd_eatm = [statistics.stdev(POOL[(t, a)]) for t in ("multi", "single") for a in MODELS]
check("ACD std, min", min(sd_acd), 141)
check("ACD std, max", max(sd_acd), 302)
check("EATM std, min", min(sd_eatm), 940)
check("EATM std, max", max(sd_eatm), 2366)

print("\n" + ("ALL CHECKS PASS" if not fails else f"{len(fails)} MISMATCH: " + "; ".join(fails)))
