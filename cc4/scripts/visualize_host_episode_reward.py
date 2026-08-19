#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ZONE_LABELS = {
    "restricted_zone_a_subnet":   "Restricted Zone A (Agent 0)",
    "operational_zone_a_subnet":  "Operational Zone A (Agent 1)",
    "restricted_zone_b_subnet":   "Restricted Zone B (Agent 2)",
    "operational_zone_b_subnet":  "Operational Zone B (Agent 3)",
    "public_access_zone_subnet":  "Public Access Zone (Agent 4)",
    "admin_network_subnet":       "Admin Network (Agent 4)",
    "office_network_subnet":      "Office Network (Agent 4)",
    "contractor_network_subnet":  "Contractor Network (no agent)",
}

RUN_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def short_name(hostname: str, subnet: str) -> str:
    prefix = subnet + "_"
    return hostname[len(prefix):] if hostname.startswith(prefix) else hostname


def load_run_host_data(run_dir: Path) -> dict:
    step_file = run_dir / "step_log.jsonl"
    if not step_file.exists():
        return {}

    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for line in step_file.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        ep = row["episode"]
        for host, info in row["host_status"].items():
            subnet = info["subnet"]
            if info["status"] == "compromised":
                counts[subnet][host][ep] += 1

    step_file_handle = step_file.open()
    first_line = step_file_handle.readline().strip()
    step_file_handle.close()
    if first_line:
        first = json.loads(first_line)
        for host, info in first["host_status"].items():
            counts[info["subnet"]].setdefault(host, defaultdict(int))

    return {s: dict(h) for s, h in counts.items()}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    run_dirs = sorted(exp_dir.glob("run_*"))
    if not run_dirs:
        print(f"No run_* dirs in {exp_dir}")
        sys.exit(1)

    print(f"Processing {len(run_dirs)} runs...")

    merged = defaultdict(lambda: defaultdict(list))
    run_boundaries = []
    global_offset = 0

    for rd in run_dirs:
        print(f"  {rd.name}...")
        data = load_run_host_data(rd)

        max_ep = 0
        for subnet in data:
            for host in data[subnet]:
                for ep in data[subnet][host]:
                    max_ep = max(max_ep, ep)

        ep_file = rd / "episodes.jsonl"
        n_eps = 0
        if ep_file.exists():
            for line in ep_file.read_text().splitlines():
                if line.strip():
                    n_eps += 1

        if n_eps == 0:
            n_eps = max_ep + 1

        for subnet in data:
            for host in data[subnet]:
                for ep in range(n_eps):
                    comp = data[subnet][host].get(ep, 0)
                    merged[subnet][host].append((global_offset + ep, comp))

        run_boundaries.append(global_offset + n_eps)
        global_offset += n_eps

    out_dir = exp_dir / "host_figures"
    out_dir.mkdir(exist_ok=True)

    for subnet in sorted(merged.keys()):
        if subnet == "internet_subnet":
            continue

        label = ZONE_LABELS.get(subnet, subnet)
        hosts = sorted(merged[subnet].keys())
        n_hosts = len(hosts)

        if n_hosts == 0:
            continue

        ncols = 3
        nrows = (n_hosts + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows),
                                 sharex=True, sharey=True)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for hi, host in enumerate(hosts):
            r, c = divmod(hi, ncols)
            ax = axes[r, c]

            eps_list = [x[0] for x in merged[subnet][host]]
            comp_list = [x[1] for x in merged[subnet][host]]

            run_idx = 0
            for i, (ep, comp) in enumerate(zip(eps_list, comp_list)):
                while run_idx < len(run_boundaries) - 1 and ep >= run_boundaries[run_idx]:
                    run_idx += 1
                color = RUN_COLORS[run_idx % len(RUN_COLORS)]
                ax.bar(ep, comp, width=1.0, color=color, alpha=0.6, linewidth=0)

            arr = np.array(comp_list, dtype=float)
            window = 5
            if len(arr) >= window:
                rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
                rolling_x = np.array(eps_list[window - 1:])[:len(rolling)]
                ax.plot(rolling_x, rolling, color="black", linewidth=1.2)

            for b in run_boundaries[:-1]:
                ax.axvline(b - 0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)

            sn = short_name(host, subnet)
            mean_val = np.mean(comp_list)
            ax.set_title(f"{sn}  (mean={mean_val:.1f})", fontsize=9, fontweight="bold")
            ax.grid(True, alpha=0.15, axis="y")

        for hi in range(n_hosts, nrows * ncols):
            r, c = divmod(hi, ncols)
            axes[r, c].set_visible(False)

        fig.supxlabel("Global Episode Index", fontsize=11)
        fig.supylabel("Compromised Steps in Episode", fontsize=11)
        fig.suptitle(f"{label} — Per-Host Compromised Steps per Episode",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])

        safe = subnet.replace("_subnet", "")
        out_path = out_dir / f"host_episode_detail_{safe}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
