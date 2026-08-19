import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "results" / "smoke"

FREE = dict(
    reward=[-13192, -11094, -10995, -6984, -6972, -6971, -6999, -6917, -4889, -4937,
            -4907, -5008, -3782, -3411, -3493, -3406, -2240, -2286, -2267, -2258],
    sleep=[100.0, 55.0, 51.5, 54.2, 50.2, 50.2, 30.6, 31.5, 29.6, 29.4,
           30.6, 10.3, 23.9, 10.4, 9.5, 0.0, 0.7, 0.0, 0.0, 0.7],
    top=[98, 84, 85, 65, 65, 65, 65, 65, 46, 46, 46, 45, 57, 64, 63, 63, 80, 80, 80, 80])

COST1 = dict(
    reward=[-13133, -13150, -11082, -9128, -7807, -7216, -5909, -5651, -5686, -5678,
            -4841, -4552, -4546, -4604, -4604, -4623, -4576, -5142, -5091, -4594],
    sleep=[100.0, 41.6, 54.8, 65.2, 40.7, 20.2, 34.8, 20.0, 20.7, 20.7,
           13.3, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 10.6, 10.2, 0.0],
    top=[97, 98, 84, 71, 65, 59, 54, 60, 61, 60, 73, 80, 80, 80, 80, 80, 80, 57, 57, 80])

PHASE = dict(
    reward=[-13191, -11561, -11454, -8653, -8754, -7558, -7498, -7582, -6587, -6650,
            -6359, -6827, -6009, -6000, -5772, -5494, -5126, -4798, -4861, -4597,
            -4459, -4334, -4372, -4330, -4388, -4360, -4309, -4330, -4347, -4120,
            -4268, -4175, -4210, -4120, -4144, -4236, -4126, -4165, -4204, -4168],
    sleep=[100.0, 71.8, 70.9, 71.8, 67.4, 64.5, 58.4, 56.6, 53.8, 53.5,
           52.4, 52.6, 54.9, 50.2, 53.0, 46.0, 33.3, 27.8, 14.9, 8.1,
           3.5, 9.2, 7.2, 3.6, 6.6, 0.0, 4.4, 4.3, 0.0, 3.6,
           0.0, 3.0, 0.0, 0.0, 0.0, 3.6, 0.0, 0.0, 4.3, 4.0],
    top=[98, 93, 93, 87, 87, 80, 80, 75, 71, 71, 67, 67, 63, 62, 57, 52, 50, 41, 37, 44,
         54, 49, 42, 48, 42, 45, 45, 39, 41, 48, 41, 41, 44, 48, 47, 49, 47, 41, 40, 44])

RUNS = [("free", FREE, 0.0, "Restore free, mission phase pinned -- the free-Restore exploit"),
        ("cost1", COST1, -1.0, "Restore priced at the native -1, mission phase pinned"),
        ("phase", PHASE, -1.0, "Restore priced, mission phase advancing (20 + 20 resumed)")]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, d, cost, note in RUNS:
        n = len(d["reward"])
        assert len(d["sleep"]) == n and len(d["top"]) == n, f"{tag}: series lengths disagree"
        eps = [{"episode": i, "reward": r, "sleep_pct": s, "ctx_buckets": None, "top_bucket_pct": t}
               for i, (r, s, t) in enumerate(zip(d["reward"], d["sleep"], d["top"]))]
        path = OUT / f"curve_{tag}.json"
        path.write_text(json.dumps(
            {"tag": tag, "steps": 500, "restore_cost": cost, "note": note, "episodes": eps}, indent=2))
        tail = d["reward"][-10:]
        print(f"{path.name:<18} {n:>3} episodes   last10 mean {sum(tail)/len(tail):>8,.0f}")


if __name__ == "__main__":
    main()
