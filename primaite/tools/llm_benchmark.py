import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
HQ_BY_TOPO = {
    "multi": ("public_access_zone", "admin_network", "office_network"),
    "single": ("zone_alpha", "zone_bravo", "zone_charlie", "zone_delta", "zone_echo"),
}
PRICES = {"openai/gpt-5-nano": (0.05, 0.40), "openai/gpt-5.5": (5.00, 30.00),
          "openai/gpt-4o-mini": (0.15, 0.60),
          "qwen/qwen3.7-flash": (0.03, 0.13), "qwen/qwen3.8-max": (2.00, 6.00),
          "deepseek/deepseek-v4-flash-0731": (0.09, 0.18), "z-ai/glm-5.2": (0.76, 2.42),
          "openai/gpt-5.6-luna": (0.10, 0.60), "openai/gpt-5.6-terra": (1.00, 6.00),
          "google/gemini-3.6-flash": (1.50, 7.50),
          "anthropic/claude-haiku-4-5": (1.00, 5.00), "anthropic/claude-sonnet-4-6": (3.00, 15.00),
          "anthropic/claude-opus-5": (5.00, 25.00), "anthropic/claude-fable-5": (10.00, 50.00)}
SEC_PER_CALL = 4.46
TOKENS_ACD = {"multi": (4100, 125), "single": (6400, 185)}


def estimate(model, episodes, steps, decide_every, topo="multi"):
    n_agents = 5 if topo == "multi" else 1
    calls = episodes * steps * n_agents / max(1, decide_every)
    pin, pout = PRICES.get(model or "", (0.0, 0.0))
    tin, tout = TOKENS_ACD[topo]
    usd = calls * tin / 1e6 * pin + calls * tout / 1e6 * pout
    return calls, usd, calls * SEC_PER_CALL / 3600


def episode(model, out, steps, decide_every, temperature, max_tokens, topo, seed, blue):
    cmd = [PY, str(ROOT / "run_cage4.py"), "--blue", blue, "--topo", topo, "--steps", str(steps),
           "--restore-cost", "-1", "--seed", str(seed), "--out", str(out)]
    if model is not None:
        cmd += ["--model", model, "--temperature", str(temperature),
                "--max-tokens", str(max_tokens), "--decide-every", str(decide_every)]
    r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode != 0 or not out.exists():
        print(f"      ! failed: {r.stderr.decode(errors='replace')[-400:]}", flush=True)
        return None
    return json.load(open(out))


def check_reached(model, log, ep):
    if model is None:
        return
    recs = [r for dl in (log.get("blue_decision_logs") or {}).values() if dl for r in dl]
    if not any(r.get("usage") for r in recs):
        errs = {str(r.get("error")) for r in recs if r.get("error")}
        raise SystemExit(f"\nABORT: '{model}' never reached the model on episode {ep}.\n"
                         f"  error: {errs.pop() if errs else 'unknown'}\n"
                         f"  Set OPENROUTER_API_KEY, or put it in ./.env (run_cage4.py loads it).")


def load_progress(curve_path, args, name):
    if not curve_path.exists():
        return [], 0.0
    try:
        d = json.load(open(curve_path))
    except (OSError, json.JSONDecodeError):
        return [], 0.0
    for field, want in (("topo", args.topo), ("decide_every", args.decide_every),
                        ("steps", args.steps), ("temperature", args.temperature),
                        ("seed", args.seed), ("blue", args.blue)):
        if d.get(field) is not None and d[field] != want:
            raise SystemExit(f"\nrefusing to resume {name}: it was run with {field}={d[field]}, "
                             f"you asked for {want}. Use a different --outdir.")
    done = [e for e in d.get("episodes") or [] if e]
    return done, float(d.get("spend_usd") or 0.0)


def run_model(model, name, args, outdir):
    hq = HQ_BY_TOPO[args.topo]
    curve_path = outdir / f"curve_{name}.json"
    curve, spend = ([], 0.0)
    if args.resume:
        curve, spend = load_progress(curve_path, args, name)
        if curve:
            print(f"      resuming after {len(curve)} episodes (${spend:.2f} spent so far)", flush=True)
    t0 = time.time()
    for ep in range(len(curve), args.episodes):
        out = outdir / f"{name}_ep{ep}.json"
        out.unlink(missing_ok=True)
        log = episode(model, out, args.steps, args.decide_every,
                      args.temperature, args.max_tokens, args.topo, args.seed, args.blue)
        if log is None:
            curve.append(None)
            continue
        check_reached(model, log, ep)
        meta, pz = log["meta"], log["meta"]["per_zone"]
        usage = meta.get("llm_usage") or {}
        spend += float(usage.get("est_cost_usd") or 0.0)
        curve.append({"episode": ep, "reward": meta["native_blue_reward"],
                      "hq_reward": sum(pz[z]["native_reward"] for z in hq),
                      "hq_restores": [pz[z]["restores"] for z in hq],
                      "hq_covered": sum(1 for z in hq if pz[z]["restores"] > 0),
                      "llm_answered": usage.get("llm_answered"),
                      "fell_back": usage.get("fell_back_to_tree"),
                      "cost_usd": usage.get("est_cost_usd")})
        c = curve[-1]
        print(f"      ep{ep}: {c['reward']:>8,.0f}   zones {c['hq_reward']:>7,.0f}   "
              f"cover {c['hq_covered']}/{len(hq)}  {'/'.join(map(str, c['hq_restores']))}"
              f"   ${spend:.2f}   {(time.time()-t0)/60:.0f} min", flush=True)
        json.dump({"tag": name, "model": model, "steps": args.steps, "blue": args.blue,
                   "decide_every": args.decide_every, "restore_cost": -1.0,
                   "topo": args.topo, "temperature": args.temperature, "seed": args.seed,
                   "episodes": curve, "spend_usd": spend},
                  open(curve_path, "w"), indent=2)
    return curve, spend


def default_outdir(args) -> str:
    if args.blue == "sleep":
        return f"bm_{args.topo}_sleep"
    who = models_slug(args.models[0]) if len(args.models) == 1 else f"{len(args.models)}models"
    return f"bm_{args.topo}_de{args.decide_every}_acd_{who}"


def models_slug(model: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "-" for c in model.split("/")[-1])


def main():
    ap = argparse.ArgumentParser(description=None, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=[], help="OpenRouter model ids to run, one arm each")
    ap.add_argument("--blue", choices=["llm", "sleep"], default="llm",
                    help="llm = the ACD-style pure-LLM defender; sleep = the free do-nothing worst case "
                         "(no --models; one arm, red runs unopposed).")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2025,
                    help="Base seed handed to run_cage4.py for the scripted reds (red i gets seed+i).")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--decide-every", type=int, default=10)
    ap.add_argument("--topo", choices=["multi", "single"], default="multi",
                    help="multi = the CC4 port (5 agents over 3 networks); single = one network of 5 zones under one agent")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--results-root", default=None,
                    help="parent directory the auto-named run directory goes under (default results/)")
    ap.add_argument("--outdir", default=None,
                    help="where the per-episode logs and curves go. Defaults to a name built from the "
                         "topology, the cadence and the model(s).")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run in --outdir instead of restarting; refuses if the "
                         "stored run used different settings.")
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()

    if args.blue == "sleep" and args.models:
        sys.exit("--blue sleep is a single free arm: drop --models")
    if args.blue == "llm" and not args.models:
        sys.exit("--blue llm needs --models")
    print(f"\n{args.episodes} episodes x {args.steps} steps, topo={args.topo}")
    total = hours = 0.0
    for m in args.models:
        calls, usd, h = estimate(m, args.episodes, args.steps, args.decide_every, topo=args.topo)
        total, hours = total + usd, hours + h
        flag = "" if m in PRICES else "   (unpriced -- $0 assumed)"
        print(f"  {m:<34}{calls:>7.0f} calls  ~${usd:8.2f}  ~{h:5.1f} h{flag}")
    print(f"  {'TOTAL':<34}{'':>7}         ~${total:8.2f}  ~{hours:5.1f} h")
    if args.estimate_only:
        return

    root = Path(args.results_root) if args.results_root else ROOT / "results"
    if not root.is_absolute():
        root = ROOT / root
    outdir = Path(args.outdir) if args.outdir else root / default_outdir(args)
    outdir.mkdir(parents=True, exist_ok=True)
    if args.blue == "sleep":
        arms = [(None, "sleep")]
    else:
        arms = [(m, m.replace("/", "_")) for m in args.models]

    results = []
    for model, name in arms:
        print(f"\n=== {name} ===", flush=True)
        curve, spend = run_model(model, name, args, outdir)
        done = [c for c in curve if c]
        if done:
            results.append((name, done[-1], spend, done))

    print("\n" + "=" * 92)
    ncov = len(HQ_BY_TOPO[args.topo])
    owner = "agent4" if args.topo == "multi" else "agent0"
    print(f"{'arm':<34}{'final':>10}{owner:>10}{'cover':>8}{'restores/zone':>18}{'spend':>10}")
    for name, last, spend, _ in results:
        print(f"{name:<34}{last['reward']:>10,.0f}{last['hq_reward']:>10,.0f}"
              f"{last['hq_covered']:>6}/{ncov}{'/'.join(map(str, last['hq_restores'])):>18}${spend:>9.2f}")
    print(f"\ncurves and logs -> {outdir}")


if __name__ == "__main__":
    main()
