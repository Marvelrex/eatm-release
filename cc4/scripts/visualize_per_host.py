#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ZONE_INFO = {
    "restricted_zone_a_subnet":   {"agent": "blue_agent_0", "label": "Restricted Zone A"},
    "operational_zone_a_subnet":  {"agent": "blue_agent_1", "label": "Operational Zone A"},
    "restricted_zone_b_subnet":   {"agent": "blue_agent_2", "label": "Restricted Zone B"},
    "operational_zone_b_subnet":  {"agent": "blue_agent_3", "label": "Operational Zone B"},
    "public_access_zone_subnet":  {"agent": "blue_agent_4", "label": "Public Access Zone"},
    "admin_network_subnet":       {"agent": "blue_agent_4", "label": "Admin Network"},
    "office_network_subnet":      {"agent": "blue_agent_4", "label": "Office Network"},
    "contractor_network_subnet":  {"agent": None,           "label": "Contractor Network"},
}

HOST_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94",
]


def short_host_name(hostname: str, subnet: str) -> str:
    prefix = subnet + "_"
    if hostname.startswith(prefix):
        return hostname[len(prefix):]
    return hostname


def process_run(run_dir: Path) -> dict:
    step_file = run_dir / "step_log.jsonl"
    if not step_file.exists():
        return {}

    counts = defaultdict(list)

    for line in step_file.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        ep = row["episode"]
        for host, info in row["host_status"].items():
            subnet = info["subnet"]
            is_comp = 1 if info["status"] == "compromised" else 0
            counts[(ep, subnet, host)].append(is_comp)

    result = defaultdict(lambda: defaultdict(list))
    for (ep, subnet, host), vals in counts.items():
        frac = sum(vals) / len(vals)
        result[subnet][host].append((ep, frac))

    for subnet in result:
        for host in result[subnet]:
            result[subnet][host].sort(key=lambda x: x[0])

    return dict(result)


def merge_runs(all_runs_data: list[dict]) -> dict:
    merged = defaultdict(lambda: defaultdict(list))
    global_offset = 0

    for run_data in all_runs_data:
        max_ep = 0
        for subnet in run_data:
            for host in run_data[subnet]:
                for ep, _ in run_data[subnet][host]:
                    max_ep = max(max_ep, ep)

        for subnet in run_data:
            for host in run_data[subnet]:
                for ep, frac in run_data[subnet][host]:
                    merged[subnet][host].append((global_offset + ep, frac))

        global_offset += max_ep + 1

    return dict(merged)


def plot_zone(subnet: str, host_data: dict, zone_label: str, n_global_eps: int,
              run_boundaries: list[int], out_dir: Path):
    hosts = sorted(host_data.keys())
    short_names = [short_host_name(h, subnet) for h in hosts]
    n_hosts = len(hosts)

    fig, (ax_heat, ax_lines, ax_bar) = plt.subplots(
        3, 1, figsize=(16, 4 + n_hosts * 0.35 + 5),
        gridspec_kw={"height_ratios": [max(n_hosts * 0.5, 3), 4, 3]},
    )

    matrix = np.full((n_hosts, n_global_eps), np.nan)
    for hi, host in enumerate(hosts):
        for ep_idx, frac in host_data[host]:
            if ep_idx < n_global_eps:
                matrix[hi, ep_idx] = frac

    im = ax_heat.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1,
                        interpolation="nearest")
    ax_heat.set_yticks(range(n_hosts))
    ax_heat.set_yticklabels(short_names, fontsize=8)
    ax_heat.set_xlabel("Global Episode Index", fontsize=9)
    ax_heat.set_title(f"{zone_label} — Host Compromise Fraction per Episode", fontsize=12,
                      fontweight="bold")

    for b in run_boundaries:
        ax_heat.axvline(b - 0.5, color="white", linewidth=0.8, alpha=0.7)

    plt.colorbar(im, ax=ax_heat, label="Fraction of Steps Compromised", shrink=0.8)

    window = max(3, n_global_eps // (len(run_boundaries) + 1) // 2)
    for hi, host in enumerate(hosts):
        color = HOST_COLORS[hi % len(HOST_COLORS)]
        eps = [x[0] for x in host_data[host]]
        fracs = [x[1] for x in host_data[host]]

        ax_lines.scatter(eps, fracs, color=color, s=8, alpha=0.3)

        arr = np.array(fracs)
        if len(arr) >= window:
            rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
            rolling_x = np.array(eps[window - 1:])[:len(rolling)]
            ax_lines.plot(rolling_x, rolling, color=color, linewidth=1.2,
                          label=short_names[hi])

    for b in run_boundaries:
        ax_lines.axvline(b - 0.5, color="gray", linestyle=":", alpha=0.3)

    ax_lines.set_xlabel("Global Episode Index", fontsize=9)
    ax_lines.set_ylabel("Compromise Fraction", fontsize=9)
    ax_lines.set_title(f"{zone_label} — Per-Host Compromise Rate (rolling mean w={window})",
                       fontsize=11, fontweight="bold")
    ax_lines.set_ylim(-0.05, 1.05)
    ax_lines.legend(fontsize=7, ncol=min(4, n_hosts), loc="upper right")
    ax_lines.grid(True, alpha=0.2)

    overall_rates = []
    for host in hosts:
        fracs = [x[1] for x in host_data[host]]
        overall_rates.append(np.mean(fracs))

    bar_colors = [HOST_COLORS[i % len(HOST_COLORS)] for i in range(n_hosts)]
    bars = ax_bar.barh(range(n_hosts), overall_rates, color=bar_colors, alpha=0.7, height=0.6)
    ax_bar.set_yticks(range(n_hosts))
    ax_bar.set_yticklabels(short_names, fontsize=8)
    ax_bar.set_xlabel("Mean Compromise Fraction", fontsize=9)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_title(f"{zone_label} — Overall Host Compromise Rate", fontsize=11,
                     fontweight="bold")
    ax_bar.grid(True, alpha=0.2, axis="x")

    for i, rate in enumerate(overall_rates):
        ax_bar.text(rate + 0.01, i, f"{rate:.2%}", va="center", fontsize=8)

    fig.tight_layout()
    safe_name = subnet.replace("_subnet", "")
    out_path = out_dir / f"host_performance_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    run_dirs = sorted(exp_dir.glob("run_*"))
    if not run_dirs:
        print(f"No run_* directories found in {exp_dir}")
        sys.exit(1)

    print(f"Processing {len(run_dirs)} runs...")

    all_runs_data = []
    eps_per_run = []
    for rd in run_dirs:
        print(f"  Loading {rd.name}...")
        data = process_run(rd)
        all_runs_data.append(data)
        max_ep = 0
        for subnet in data:
            for host in data[subnet]:
                for ep, _ in data[subnet][host]:
                    max_ep = max(max_ep, ep)
        eps_per_run.append(max_ep + 1)

    merged = merge_runs(all_runs_data)

    run_boundaries = []
    cumsum = 0
    for n in eps_per_run:
        cumsum += n
        run_boundaries.append(cumsum)
    n_global_eps = cumsum

    out_dir = exp_dir / "host_figures"
    out_dir.mkdir(exist_ok=True)

    for subnet in sorted(merged.keys()):
        if subnet == "internet_subnet":
            continue
        info = ZONE_INFO.get(subnet, {"label": subnet, "agent": None})
        print(f"Plotting {info['label']}...")
        plot_zone(subnet, merged[subnet], info["label"], n_global_eps, run_boundaries[:-1],
                  out_dir)

    print(f"\nAll figures saved to: {out_dir}")


if __name__ == "__main__":
    main()
