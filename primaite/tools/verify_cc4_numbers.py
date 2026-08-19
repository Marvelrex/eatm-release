import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
fails = []


def check(name, got, want, tol=0.51):
    ok = abs(got - want) <= tol
    print(f"  {'ok ' if ok else 'FAIL'}  {name:<46} paper {want:>9,.2f}   derived {got:>9,.2f}")
    if not ok:
        fails.append(name)


A = {
    "GPT-5.4":        (-3828, -1606, 135, 12.5, 14.69, 3.24),
    "GPT-5.4 mini":   (-3033,  -672,  93,  0.9,  2.15, 0.51),
    "o3-mini":        (-2708,  -771,  97, 28.8, 36.82, 3.79),
    "Haiku 4.5":      (-1542, -1077, 152,  2.2,  7.74, 1.19),
    "DeepSeek-v3.2":  (-2925, -1241, 124, 17.3,  1.67, 0.37),
    "Gemini 3 Flash": (-2623,  -953, 107,  1.9,  0.84, 0.19),
}
R_TREE = -837

print("=== Table II: Delta_LLM = EATM - tree-only, recomputed from the reward column ===")
DELTA = {"GPT-5.4": -769, "GPT-5.4 mini": 165, "o3-mini": 66,
         "Haiku 4.5": -240, "DeepSeek-v3.2": -404, "Gemini 3 Flash": -116}
for m, (acd, eatm, _rho, inv, c_acd, c_eatm) in A.items():
    check(f"{m} Delta_LLM", eatm - R_TREE, DELTA[m], tol=0)
check("GPT-5.4 mini tree gain over ACD", R_TREE - A["GPT-5.4 mini"][0], 2196, tol=0)
check("o3-mini tree gain over ACD", R_TREE - A["o3-mini"][0], 1871, tol=0)

print("\n=== headline multipliers quoted in the abstract and body ===")
check("o3-mini reward multiple", A["o3-mini"][0] / A["o3-mini"][1], 3.5, tol=0.05)
check("GPT-5.4 mini reward multiple", A["GPT-5.4 mini"][0] / A["GPT-5.4 mini"][1], 4.5, tol=0.05)
check("Haiku 4.5 reward multiple", A["Haiku 4.5"][0] / A["Haiku 4.5"][1], 1.4, tol=0.05)
check("GPT-5.4 reward multiple", A["GPT-5.4"][0] / A["GPT-5.4"][1], 2.4, tol=0.05)
check("o3-mini cost ratio", A["o3-mini"][4] / A["o3-mini"][5], 9.70, tol=0.02)

print("\n=== cost decomposition: call-count factor = 1/(1 - invalid rate) ===")
for m, want in (("o3-mini", 1.40), ("DeepSeek-v3.2", 1.21), ("GPT-5.4", 1.14)):
    check(f"{m} call-count factor", 1 / (1 - A[m][3] / 100), want, tol=0.005)
for m in ("Haiku 4.5", "GPT-5.4 mini", "Gemini 3 Flash"):
    f = 1 / (1 - A[m][3] / 100)
    ok = f < 1.05
    print(f"  {'ok ' if ok else 'FAIL'}  {m:<46} paper  < 1.05    derived {f:>9.3f}")
    if not ok:
        fails.append(m)
check("o3-mini per-call factor (ratio / call-count)",
      (A["o3-mini"][4] / A["o3-mini"][5]) / (1 / (1 - A["o3-mini"][3] / 100)), 6.93, tol=0.02)

print("\n=== Table IV reference points versus the Scenario A ACD column ===")
FIVE_RANDOM = -1334.5
worse = [m for m, v in A.items() if v[0] < FIVE_RANDOM]
print(f"  ACD backbones scoring below the 5random floor: {len(worse)} of {len(A)}  {worse}")

print("\n=== Q2: the GPT tiers reorder rather than compress ===")
GPT = ["GPT-5.4 mini", "GPT-5.4", "o3-mini"]
acd = {m: A[m][0] for m in GPT}
eatm = {m: A[m][1] for m in GPT}
check("GPT tier spread, ACD", max(acd.values()) - min(acd.values()), 1120, tol=0)
check("GPT tier spread, EATM", max(eatm.values()) - min(eatm.values()), 934, tol=0)
best = max(eatm, key=eatm.get)
print(f"  best GPT cell after EATM: {best} at {eatm[best]}", end="")
print("   (the paper says the weak backbone ends ahead)")
if best != "GPT-5.4 mini":
    fails.append("Q2: the weak backbone is no longer ahead")
check("GPT-5.4 mini ahead of o3-mini", eatm["GPT-5.4 mini"] - eatm["o3-mini"], 99, tol=0)
check("GPT-5.4 mini ahead of GPT-5.4", eatm["GPT-5.4 mini"] - eatm["GPT-5.4"], 934, tol=0)

RL = -190.0
check("5rl over the best EATM cell", max(eatm.values()) / RL, 3.5, tol=0.05)

print("\n" + ("ALL CHECKS PASS" if not fails else f"{len(fails)} MISMATCH: " + "; ".join(fails)))
