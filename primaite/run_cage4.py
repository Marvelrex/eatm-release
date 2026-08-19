import argparse
import importlib.resources
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

from primaite.game.game import PrimaiteGame
from primaite.game.agent.scripted_agents.cage4 import comms
from primaite.game.agent.scripted_agents.cage4.cage4_red import reset_activation as reset_red_activation
from primaite.game.agent.scripted_agents.cage4 import families as host_selector

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def _load_dotenv() -> None:
    import os
    env = Path(__file__).resolve().parent / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

TOPO = "cage4_fixed_topology"
CHANNEL = "cage4"
RED_CHANNEL = "cage4_red"
BEACHHEAD, BEACHHEAD_IP = "red_beachhead", "10.0.10.30"
BEACHHEAD_GW, BEACHHEAD_SWITCH, BEACHHEAD_PORT, DNS_IP = "10.0.10.1", "uav_switch", 4, "10.1.2.10"
DB_FOLDER, DB_FILE, DB_SERVICE = "database", "database.db", "database-service"
BAD_HEALTH = (2, 3)

PRICES = {
    "openai/gpt-5-nano": (0.05, 0.40), "openai/gpt-5.5": (5.00, 30.00),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "qwen/qwen3.7-flash": (0.03, 0.13), "qwen/qwen3.8-max": (2.00, 6.00),
    "deepseek/deepseek-v4-flash-0731": (0.09, 0.18),
    "z-ai/glm-5.2": (0.76, 2.42),
    "openai/gpt-5.6-luna": (0.10, 0.60), "openai/gpt-5.6-terra": (1.00, 6.00),
    "google/gemini-3.6-flash": (1.50, 7.50),
    "anthropic/claude-haiku-4-5": (1.00, 5.00), "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "anthropic/claude-opus-5": (5.00, 25.00), "anthropic/claude-fable-5": (10.00, 50.00),
}


def aggregate_usage(decision_logs, model: str) -> dict:
    pin, pout = PRICES.get(model or "", (0.0, 0.0))
    calls = answered = fell_back = ptok = ctok = 0
    errors: dict = {}
    for dl in decision_logs:
        for d in dl or []:
            if not isinstance(d, dict):
                continue
            calls += 1
            u = d.get("usage")
            if isinstance(u, dict):
                answered += 1
                ptok += int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
                ctok += int(u.get("completion_tokens") or u.get("output_tokens") or 0)
            if str(d.get("reasoning", "")).endswith("_tree_top"):
                fell_back += 1
            if d.get("error"):
                errors[str(d["error"])[:120]] = errors.get(str(d["error"])[:120], 0) + 1
    return {"decisions": calls, "llm_answered": answered, "fell_back_to_tree": fell_back,
            "prompt_tokens": ptok, "completion_tokens": ctok, "total_tokens": ptok + ctok,
            "est_cost_usd": round(ptok / 1e6 * pin + ctok / 1e6 * pout, 4),
            "price_per_1m": {"input": pin, "output": pout},
            "price_known": (model or "") in PRICES, "errors": errors}

ZONES = [
    dict(name="restricted_zone_a",  agent=0, server="restricted_a_server",  ip="10.2.1.10",
         client="restricted_a_host_1", kind="db", service=DB_SERVICE),
    dict(name="operational_zone_a", agent=1, server="operational_a_server", ip="10.2.2.10",
         client="operational_a_host_1", kind="db", service=DB_SERVICE),
    dict(name="restricted_zone_b",  agent=2, server="restricted_b_server",  ip="10.3.1.10",
         client="restricted_b_host_1", kind="db", service=DB_SERVICE),
    dict(name="operational_zone_b", agent=3, server="operational_b_server", ip="10.3.2.10",
         client="operational_b_host_1", kind="db", service=DB_SERVICE),
    dict(name="public_access_zone", agent=4, server="public_web_server",    ip="10.1.1.10",
         client="public_access_host_1", kind="db", service=DB_SERVICE,
         inject_host=dict(hostname="public_access_host_1", ip="10.1.1.21", gw="10.1.1.1",
                          switch="public_access_switch", port=2)),
    dict(name="admin_network",      agent=4, server="admin_server",         ip="10.1.2.10",
         client="admin_host_1",        kind="db", service=DB_SERVICE),
    dict(name="office_network",     agent=4, server="office_server",        ip="10.1.3.10",
         client="office_host_1",       kind="db", service=DB_SERVICE),
]
AGENT_IDS = sorted({z["agent"] for z in ZONES})


SINGLE_TOPO = "cage4_single_network"
SINGLE_BEACHHEAD_IP = "192.168.10.30"
SINGLE_ZONES = [
    dict(name="zone_alpha",   agent=0, server="alpha_server",   ip="192.168.10.11",
         client="alpha_host_1",   kind="db", service=DB_SERVICE,
         inject_host=dict(hostname="alpha_host_1", ip="192.168.10.21", gw="192.168.10.1",
                          switch="single_switch", port=6)),
    dict(name="zone_bravo",   agent=0, server="bravo_server",   ip="192.168.10.12",
         client="bravo_host_1",   kind="db", service=DB_SERVICE,
         inject_host=dict(hostname="bravo_host_1", ip="192.168.10.22", gw="192.168.10.1",
                          switch="single_switch", port=7)),
    dict(name="zone_charlie", agent=0, server="charlie_server", ip="192.168.10.13",
         client="charlie_host_1", kind="db", service=DB_SERVICE,
         inject_host=dict(hostname="charlie_host_1", ip="192.168.10.23", gw="192.168.10.1",
                          switch="single_switch", port=8)),
    dict(name="zone_delta",   agent=0, server="delta_server",   ip="192.168.10.14",
         client="delta_host_1",   kind="db", service=DB_SERVICE,
         inject_host=dict(hostname="delta_host_1", ip="192.168.10.24", gw="192.168.10.1",
                          switch="single_switch", port=9)),
    dict(name="zone_echo",    agent=0, server="echo_server",    ip="192.168.10.15",
         client="echo_host_1",    kind="db", service=DB_SERVICE,
         inject_host=dict(hostname="echo_host_1", ip="192.168.10.25", gw="192.168.10.1",
                          switch="single_switch", port=10)),
]
SINGLE_RED_GROUPS = [[z["name"]] for z in SINGLE_ZONES]
SINGLE_PHASE_W = {
    0: {z["name"]: {"RIA": -1, "ASF": -3 if i < 2 else -1, "LWF": -1}
        for i, z in enumerate(SINGLE_ZONES)},
    1: {z["name"]: ({"RIA": -10, "ASF": 0, "LWF": -10} if z["name"] == "zone_bravo"
                    else {"RIA": -3 if i == 0 else -1, "ASF": -1, "LWF": -1})
        for i, z in enumerate(SINGLE_ZONES)},
    2: {z["name"]: ({"RIA": -10, "ASF": 0, "LWF": -10} if z["name"] == "zone_delta"
                    else {"RIA": -3 if i == 2 else -1, "ASF": -1, "LWF": -1})
        for i, z in enumerate(SINGLE_ZONES)},
}


_CC4: dict = {}


def apply_topology(name: str) -> None:
    global TOPO, ZONES, AGENT_IDS, RED_GROUPS, PHASE_W, CONTRACTOR_ZONE
    global BEACHHEAD_IP, BEACHHEAD_GW, BEACHHEAD_SWITCH, BEACHHEAD_PORT, DNS_IP
    if not _CC4:
        _CC4.update(topo=TOPO, zones=ZONES, red_groups=RED_GROUPS, phase_w=PHASE_W,
                    contractor=CONTRACTOR_ZONE, beachhead_ip=BEACHHEAD_IP)
    if name == "single":
        TOPO, ZONES, RED_GROUPS, PHASE_W = SINGLE_TOPO, SINGLE_ZONES, SINGLE_RED_GROUPS, SINGLE_PHASE_W
        CONTRACTOR_ZONE = None
        BEACHHEAD_IP, BEACHHEAD_GW = SINGLE_BEACHHEAD_IP, "192.168.10.1"
        BEACHHEAD_SWITCH, BEACHHEAD_PORT, DNS_IP = "single_switch", 11, "192.168.10.11"
    else:
        TOPO, ZONES = _CC4["topo"], _CC4["zones"]
        RED_GROUPS, PHASE_W = _CC4["red_groups"], _CC4["phase_w"]
        CONTRACTOR_ZONE = _CC4["contractor"]
        BEACHHEAD_IP, BEACHHEAD_GW = _CC4["beachhead_ip"], "10.0.10.1"
        BEACHHEAD_SWITCH, BEACHHEAD_PORT, DNS_IP = "uav_switch", 4, "10.1.2.10"
    AGENT_IDS = sorted({z["agent"] for z in ZONES})


def agent_zones(a: int) -> list:
    return [z for z in ZONES if z["agent"] == a]


def _zone_acl_pos(z: dict) -> int:
    return 1 + ZONES.index(z)


def load_topology() -> dict:
    text = importlib.resources.files("primaite.config._package_data").joinpath(f"{TOPO}.yaml").read_text()
    cfg = yaml.safe_load(text)
    cfg.setdefault("game", {}).setdefault("thresholds", {"nmne": {"high": 10, "medium": 5, "low": 0}})
    return cfg


def _node(cfg: dict, hostname: str) -> dict:
    return next(n for n in cfg["simulation"]["network"]["nodes"] if n["hostname"] == hostname)


def inject_red_infra(cfg: dict) -> None:
    nodes = cfg["simulation"]["network"]["nodes"]
    links = cfg["simulation"]["network"]["links"]
    nodes.append({
        "hostname": BEACHHEAD, "type": "computer", "ip_address": BEACHHEAD_IP,
        "subnet_mask": "255.255.255.0", "default_gateway": BEACHHEAD_GW, "dns_server": DNS_IP,
        "applications": [{"type": "database-client", "options": {"db_server_ip": ZONES[-1]["ip"]}}],
        "services": [{"type": "dns-client"}]})
    links.append({"endpoint_a_hostname": BEACHHEAD, "endpoint_a_port": 1,
                  "endpoint_b_hostname": BEACHHEAD_SWITCH, "endpoint_b_port": BEACHHEAD_PORT})
    for z in ZONES:
        if z["kind"] != "db":
            continue
        srv = _node(cfg, z["server"])
        svcs = srv.setdefault("services", [])
        if not any(s.get("type") == "database-service" for s in svcs):
            svcs.append({"type": "database-service"})
        ih = z.get("inject_host")
        if ih and not any(n.get("hostname") == ih["hostname"] for n in nodes):
            nodes.append({"hostname": ih["hostname"], "type": "computer", "ip_address": ih["ip"],
                          "subnet_mask": "255.255.255.0", "default_gateway": ih["gw"],
                          "dns_server": "10.1.2.10", "applications": []})
            links.append({"endpoint_a_hostname": ih["hostname"], "endpoint_a_port": 1,
                          "endpoint_b_hostname": ih["switch"], "endpoint_b_port": ih["port"]})
        apps = _node(cfg, z["client"]).setdefault("applications", [])
        if not any(ap.get("type") == "database-client" for ap in apps):
            apps.append({"type": "database-client", "options": {"db_server_ip": z["ip"]}})
        apps.append({"type": "data-manipulation-bot", "options": {
            "port_scan_p_of_success": 0.8, "data_manipulation_p_of_success": 0.8,
            "payload": "ENCRYPT", "server_ip": z["ip"]}})


def green_agents() -> list:
    greens = []
    for z in ZONES:
        if z["kind"] != "db" or not z["client"]:
            continue
        greens.append({
            "ref": f"green_{z['name']}", "team": "GREEN", "type": "probabilistic-agent",
            "agent_settings": {"action_probabilities": {0: 0.34, 1: 0.66}},
            "action_space": {"action_map": {
                0: {"action": "do-nothing", "options": {}},
                1: {"action": "node-application-execute",
                    "options": {"node_name": z["client"], "application_name": "database-client"}}}},
            "reward_function": {"reward_components": [
                {"type": "green-admin-database-unreachable-penalty", "weight": 1.0,
                 "options": {"node_hostname": z["client"]}}]}})
    return greens


def _agent_reward(a: int) -> dict:
    comps = [{"type": "database-file-integrity", "weight": 1.0,
              "options": {"node_hostname": z["server"], "folder_name": DB_FOLDER, "file_name": DB_FILE}}
             for z in agent_zones(a) if z["kind"] == "db"]
    return {"reward_components": comps or [{"type": "dummy", "weight": 1.0, "options": {}}]}


def _agent_obs(a: int) -> dict:
    hosts, has_db = [], any(z["kind"] == "db" for z in agent_zones(a))
    for z in agent_zones(a):
        if z["kind"] == "db":
            hosts.append({"hostname": z["server"], "services": [{"service_name": z["service"]}],
                          "folders": [{"folder_name": DB_FOLDER, "files": [{"file_name": DB_FILE}]}]})
            if z["client"]:
                hosts.append({"hostname": z["client"]})
        else:
            hosts.append({"hostname": z["server"], "services": [{"service_name": z["service"]}]})
    return {"type": "custom", "options": {"components": [
        {"type": "nodes", "label": "NODES", "options": {
            "hosts": hosts, "num_services": 1, "num_applications": 2,
            "num_folders": 1 if has_db else 0, "num_files": 1 if has_db else 0,
            "num_nics": 1, "include_num_access": False, "include_nmne": False,
            "routers": [{"hostname": "hq_router"}], "num_ports": 0,
            "ip_list": [z["ip"] for z in ZONES] + [BEACHHEAD_IP], "wildcard_list": ["0.0.0.1"],
            "port_list": ["POSTGRES_SERVER"], "protocol_list": ["TCP"], "num_rules": 10}},
        {"type": "none", "label": "ICS", "options": {}}]}}


def _zone_actions(z: dict, start: int) -> tuple:
    i, server = start, z["server"]
    m: dict = {}
    if z["kind"] == "db" and z["client"]:
        m[i] = {"action": "node-application-close",
                "options": {"node_name": z["client"], "application_name": "data-manipulation-bot"}}; i += 1
        m[i] = {"action": "node-folder-restore",
                "options": {"node_name": server, "folder_name": DB_FOLDER}}; i += 1
        m[i] = {"action": "node-folder-scan",
                "options": {"node_name": server, "folder_name": DB_FOLDER}}; i += 1
    else:
        m[i] = {"action": "node-service-restart",
                "options": {"node_name": server, "service_name": z["service"]}}; i += 1
        m[i] = {"action": "node-service-scan",
                "options": {"node_name": server, "service_name": z["service"]}}; i += 1
    for e in m.values():
        e["zone"] = z["name"]
    return m, i


def _agent_action_map(a: int) -> dict:
    amap, i = {}, 0
    for z in agent_zones(a):
        zm, i = _zone_actions(z, i)
        amap.update(zm)
    return amap


def _analyse_index(amap: dict) -> int:
    return next((i for i, e in amap.items()
                 if e["action"] in ("node-folder-scan", "node-service-scan")), min(amap))


def _agent_zone_cfg(z: dict) -> dict:
    return {"name": z["name"], "server": z["server"], "ip": z["ip"], "kind": z["kind"],
            "service": z["service"], "client": z["client"] or "", "block_position": _zone_acl_pos(z)}


def zone_priorities() -> dict:
    out = {}
    for phase, w in PHASE_W.items():
        names = [z["name"] for z in ZONES if z["name"] in w]
        ria = {n: abs(w[n]["RIA"]) for n in names}
        base = min(ria.values()) or 1
        out[phase] = {n: round(ria[n] / base, 1) for n in names}
    return out


def blue_agents() -> list:
    agents = []
    for a in AGENT_IDS:
        agents.append({
            "ref": f"blue_agent_{a}", "team": "BLUE", "type": "cage4-blue-agent",
            "observation_space": {"type": "none"},
            "reward_function": _agent_reward(a),
            "agent_settings": {
                "zone_index": a, "comms_channel": CHANNEL,
                "zones": [_agent_zone_cfg(z) for z in agent_zones(a)],
                "bot_app": "data-manipulation-bot",
                "decoy_service": "", "beachhead_ip": "", "block_router": "hq_router",
                "folder": DB_FOLDER, "file": DB_FILE, "scan_interval": 5}})
    return agents


def sleep_blue_agents() -> list:
    return [{"ref": f"blue_agent_{a}", "team": "BLUE", "type": "cage4-sleep-agent",
             "observation_space": {"type": "none"}, "reward_function": _agent_reward(a)}
            for a in AGENT_IDS]


def llm_blue_agents(model: str, temperature: float, max_tokens: int, decide_every: int = 1) -> list:
    agents = []
    for a in AGENT_IDS:
        azs = agent_zones(a)
        amap = _agent_action_map(a)
        amap_strict = {i: {k: v for k, v in e.items() if k != "zone"} for i, e in amap.items()}
        brief = (f"You are CAGE-4 BLUE AGENT {a}, defending: {', '.join(z['name'] for z in azs)}. "
                 f"Protect the crown-jewel servers ({', '.join(z['server'] for z in azs)}).")
        agents.append({
            "ref": f"blue_agent_{a}", "team": "BLUE", "type": "cage4-blue-llm-agent",
            "observation_space": _agent_obs(a),
            "reward_function": _agent_reward(a),
            "action_space": {"action_map": amap_strict},
            "agent_settings": {
                "model": model, "temperature": temperature, "max_tokens": max_tokens,
                "mission_brief": brief, "fallback_action": _analyse_index(amap), "decide_every": decide_every,
                "native_actions": True,
                "action_zones": {i: e["zone"] for i, e in amap.items() if e.get("zone")},
                "zone_index": a, "comms_channel": CHANNEL,
                "zones": [{"server": z["server"], "folder": DB_FOLDER, "file": DB_FILE}
                          for z in azs if z["kind"] == "db"]}})
    return agents


CONTRACTOR_ZONE = "contractor_network"
RED_GROUPS = [
    [CONTRACTOR_ZONE],
    ["restricted_zone_a"],
    ["operational_zone_a"],
    ["restricted_zone_b"],
    ["operational_zone_b"],
    ["public_access_zone", "admin_network", "office_network"],
]


def red_agents(base_seed: int = 2025) -> list:
    by_name = {z["name"]: z for z in ZONES}

    def _target(name: str) -> dict:
        z = by_name[name]
        return {"zone_index": z["agent"], "server": z["server"], "server_ip": z["ip"],
                "client": z["client"], "kind": z["kind"], "service": z["service"],
                "folder": DB_FOLDER, "file": DB_FILE, "bot_app": "data-manipulation-bot",
                "client_app": "database-client", "admin_user": "admin", "admin_password": "admin"}

    agents, prev = [], None
    for i, group in enumerate(RED_GROUPS):
        home = group[0]
        targets = [] if home == CONTRACTOR_ZONE else [_target(n) for n in group]
        ref = "red_0_contractor" if home == CONTRACTOR_ZONE else ("red_hq" if len(group) > 1 else f"red_{home}")
        agents.append({"ref": ref, "team": "RED", "type": "cage4-red-agent",
                       "observation_space": {"type": "none"},
                       "agent_settings": {"beachhead": BEACHHEAD, "targets": targets,
                                          "seed": base_seed + i, "home_zone": home,
                                          "activation_channel": RED_CHANNEL, "activates_after": prev}})
        prev = home
    return agents


PHASE_W = {
    0: {
        "restricted_zone_a":  {"RIA": -1, "ASF": -3, "LWF": -1},
        "operational_zone_a": {"RIA": -1, "ASF": -1, "LWF": -1},
        "restricted_zone_b":  {"RIA": -1, "ASF": -3, "LWF": -1},
        "operational_zone_b": {"RIA": -1, "ASF": -1, "LWF": -1},
        "public_access_zone": {"RIA": -3, "ASF": -1, "LWF": -1},
        "admin_network":      {"RIA": -3, "ASF": -1, "LWF": -1},
        "office_network":     {"RIA": -3, "ASF": -1, "LWF": -1},
    },
    1: {
        "restricted_zone_a":  {"RIA": -3, "ASF": -1, "LWF": -2},
        "operational_zone_a": {"RIA": -10, "ASF": 0, "LWF": -10},
        "restricted_zone_b":  {"RIA": -1, "ASF": -1, "LWF": -1},
        "operational_zone_b": {"RIA": -1, "ASF": -1, "LWF": -1},
        "public_access_zone": {"RIA": -3, "ASF": -1, "LWF": -1},
        "admin_network":      {"RIA": -3, "ASF": -1, "LWF": -1},
        "office_network":     {"RIA": -3, "ASF": -1, "LWF": -1},
    },
    2: {
        "restricted_zone_a":  {"RIA": -3, "ASF": -3, "LWF": -1},
        "operational_zone_a": {"RIA": -1, "ASF": -1, "LWF": -1},
        "restricted_zone_b":  {"RIA": -3, "ASF": -1, "LWF": -2},
        "operational_zone_b": {"RIA": -10, "ASF": 0, "LWF": -10},
        "public_access_zone": {"RIA": -3, "ASF": -1, "LWF": -1},
        "admin_network":      {"RIA": -3, "ASF": -1, "LWF": -1},
        "office_network":     {"RIA": -3, "ASF": -1, "LWF": -1},
    },
}


def mission_phase(step: int, total_steps: int) -> int:
    q, r = divmod(total_steps, 3)
    sizes = [q + (1 if r >= 1 else 0), q + (1 if r >= 2 else 0), q]
    if step < sizes[0]:
        return 0
    if step < sizes[0] + sizes[1]:
        return 1
    return 2


def _corrupt_db_zones(game) -> set:
    st = game.simulation.describe_state()
    out = set()
    for z in ZONES:
        if z["kind"] != "db":
            continue
        f = (((((st.get("network", {}) or {}).get("nodes", {}) or {}).get(z["server"], {}) or {}
               ).get("file_system", {}) or {}).get("folders", {}) or {}).get(DB_FOLDER, {}) or {}
        if (f.get("files", {}) or {}).get(DB_FILE, {}).get("health_status") in BAD_HEALTH:
            out.add(z["name"])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Run the decentralised CAGE-4 scenario.")
    p.add_argument("--blue", choices=["scripted", "llm", "sleep"], default="scripted",
                   help="scripted = heuristic zone defenders; llm = one ACD-style LLM defender per zone (OpenRouter); "
                        "sleep = do-nothing defenders.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--model", default="openai/gpt-5-nano", help="OpenRouter model id for LLM blue.")
    p.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for the LLM blue.")
    p.add_argument("--max-tokens", type=int, default=2048,
                   help="Reply budget. A truncated reply parses as no integer and falls back to menu[0]; "
                        "meta.llm_usage.fell_back_to_tree counts those.")
    p.add_argument("--decide-every", type=int, default=1,
                   help="LLM blues re-decide every N sim steps, persisting the established "
                        "action in between. N>1 cuts LLM calls ~Nx (the lever for affordable long runs); "
                        "1 = decide every step. The FSM red always acts every step.")
    p.add_argument("--topo", choices=["multi", "single"], default="multi",
                   help="multi = the CC4 port (5 zone-local agents over 3 routed networks, commvector "
                        "active); single = ONE network of 5 zones defended by ONE agent, which also "
                        "leaves the commvector with no peers. Everything else is held identical.")
    p.add_argument("--restore-cost", type=float, default=-1.0,
                   help="Per-Restore action cost charged to the issuing agent, native CC4's flat -1 for "
                        "host downtime (the only blue action CybORG charges for). Pass 0 to reproduce "
                        "the free-Restore game, in which the best policy is to Restore every step.")
    p.add_argument("--seed", type=int, default=2025,
                   help="Base seed for the scripted red FSMs (red i gets seed+i). Only the reds are "
                        "seeded — green traffic and LLM sampling stay stochastic. The default matches "
                        "the hard-coded value every benchmark so far ran under, so omitting it keeps "
                        "new runs red-comparable with the existing result directories.")
    p.add_argument("--out", default="match_cage4.json")
    args = p.parse_args()
    apply_topology(args.topo)

    comms.reset(CHANNEL)
    reset_red_activation(RED_CHANNEL)
    cfg = load_topology()
    inject_red_infra(cfg)
    if args.blue == "llm":
        blue = llm_blue_agents(args.model, args.temperature, args.max_tokens, args.decide_every)
    elif args.blue == "sleep":
        blue = sleep_blue_agents()
    else:
        blue = blue_agents()
    cfg["agents"] = blue + green_agents() + red_agents(args.seed)

    game = PrimaiteGame.from_config(cfg)
    blues = [a for a in game.agents.values() if a.config.team == "BLUE"]
    reds = [a for a in game.agents.values() if a.config.team == "RED"]
    ip_to_red = {t["server_ip"]: a for a in reds
                 for t in a.config.agent_settings.targets}
    ip_to_zone = {z["ip"]: z["name"] for z in ZONES}

    def _agent_of(b):
        try:
            return int(str(b.config.ref).rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return -1

    db_blues = [b for b in blues if any(z["kind"] == "db" for z in agent_zones(_agent_of(b)))]

    print(f"CAGE-4 decentralised scenario [{args.blue} blue]: {len(blues)} agents over {len(ZONES)} "
          f"zones (agent 4 owns 3 HQ zones), 1 FSM red, {args.steps} steps")
    if args.blue == "llm":
        print(f"  model={args.model}  temp={args.temperature}  max_tokens={args.max_tokens}")
    print(f"\n{'t':>3} | {'red target:action':<30} | {'corrupt':>7} | {'native R':>9} | "
          "blue agents (idx:CC4->CAOS, ALL incl. Sleep)")
    print("-" * 120)
    trajectory = []
    greens = [a for a in game.agents.values() if a.config.team == "GREEN"]
    green_zone = {f"green_{z['name']}": z["name"] for z in ZONES if z["kind"] == "db"}
    native_reward = 0.0
    zone_owner = {z["name"]: z["agent"] for z in ZONES}
    agent_native = {a: 0.0 for a in AGENT_IDS}
    zone_native = {z["name"]: 0.0 for z in ZONES}
    zone_restores = {z["name"]: 0 for z in ZONES}
    restore_node_ip = {z["server"]: z["ip"] for z in ZONES if z["kind"] == "db"}
    remove_node_ip = {z["client"]: z["ip"] for z in ZONES if z["kind"] == "db" and z["client"]}
    stopped_services: set = set()
    step_penalty_template = {a: 0.0 for a in AGENT_IDS}
    out = Path(args.out)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    steps_path = out.with_name(out.stem + ".steps.jsonl")
    steps_f = steps_path.open("w", encoding="utf-8")

    def _blue_step(b):
        rec = {"agent": _agent_of(b), "action": b.history[-1].action,
               "cc4_to_caos": f"{host_selector.family_of(b.history[-1].action)} -> {b.history[-1].action}"}
        dl = getattr(b, "decision_log", None)
        if dl and dl[-1].get("timestep") == b.history[-1].timestep:
            rec["decision"] = dl[-1]
        return rec

    for step_idx in range(args.steps):
        game.step()
        def _evict(zone_ip, user_only=False):
            r_ = ip_to_red.get(zone_ip)
            if r_ is not None and hasattr(r_, "notify_blue_recovery"):
                r_.notify_blue_recovery(zone_ip, user_only=user_only)
        restores_by_agent: dict = {}
        for b in blues:
            h = b.history[-1]
            if getattr(h.response, "status", None) != "success":
                continue
            p = h.parameters or {}
            node = p.get("node_name")
            if h.action == "node-folder-restore" and node in restore_node_ip:
                _evict(restore_node_ip[node])
                a_ = _agent_of(b)
                if a_ in step_penalty_template:
                    restores_by_agent[a_] = restores_by_agent.get(a_, 0) + 1
                zname_ = ip_to_zone.get(restore_node_ip[node])
                if zname_ in zone_restores:
                    zone_restores[zname_] += 1
                    zone_native[zname_] += args.restore_cost
            elif h.action == "node-application-close" and node in remove_node_ip:
                _evict(remove_node_ip[node], user_only=True)
        active_reds = [(a, a.decision_log[-1]) for a in reds if a.decision_log
                       and (a.decision_log[-1].get("action_name") or "").startswith(("Discover", "Aggressive",
                       "Stealth", "Exploit", "Privilege", "Impact", "Degrade"))]
        rrec = active_reds[0][1] if active_reds else (reds[0].decision_log[-1] if reds and reds[0].decision_log else {})
        rdesc = f"{len(active_reds)}red {(rrec.get('target') or '-')}:{rrec.get('action_name') or '-'}"
        tstep = reds[0].history[-1].timestep if reds else 0
        acts = [f"{_agent_of(b)}:{host_selector.family_of(b.history[-1].action)}->{b.history[-1].action}"
                for b in sorted(blues, key=_agent_of)]
        corrupt_zones = _corrupt_db_zones(game)
        corrupt = len(corrupt_zones)
        for z in ZONES:
            if z["kind"] != "db":
                continue
            if z["name"] in corrupt_zones and z["name"] not in stopped_services:
                game.simulation.apply_request(["network", "node", z["server"], "service", z["service"], "stop"])
                stopped_services.add(z["name"])
            elif z["name"] not in corrupt_zones and z["name"] in stopped_services:
                game.simulation.apply_request(["network", "node", z["server"], "service", z["service"], "start"])
                stopped_services.discard(z["name"])
        W = PHASE_W[mission_phase(step_idx, args.steps)]
        step_penalty = {a: 0.0 for a in AGENT_IDS}
        unowned_total = 0.0
        for a_, n_ in restores_by_agent.items():
            pen = args.restore_cost * n_
            native_reward += pen
            agent_native[a_] += pen
            step_penalty[a_] += pen
        ria_zones = set()
        for r in reds:
            dl = r.decision_log[-1] if r.decision_log else None
            if dl and dl.get("action_name") == "Impact":
                zname = ip_to_zone.get(dl.get("target"))
                if zname in corrupt_zones:
                    ria_zones.add(zname)
        for zi in ria_zones:
            pen = W[zi]["RIA"]
            native_reward += pen
            zone_native[zi] += pen
            agent_native[zone_owner[zi]] += pen
            step_penalty[zone_owner[zi]] += pen
        for zi in stopped_services:
            pen = W[zi]["LWF"]
            native_reward += pen
            zone_native[zi] += pen
            agent_native[zone_owner[zi]] += pen
            step_penalty[zone_owner[zi]] += pen
        for g in greens:
            h = g.history[-1]
            if h.action != "do-nothing" and getattr(h.response, "status", None) == "failure":
                zname = green_zone.get(g.config.ref)
                pen = W.get(zname, {"ASF": 0})["ASF"]
                native_reward += pen
                if zname in zone_native:
                    zone_native[zname] += pen
                if zname in zone_owner:
                    agent_native[zone_owner[zname]] += pen
                    step_penalty[zone_owner[zname]] += pen
        if unowned_total:
            for a in AGENT_IDS:
                step_penalty[a] += unowned_total / len(AGENT_IDS)
        team = sum(b.reward_function.total_reward for b in db_blues)
        trajectory.append({"t": tstep, "corrupt": corrupt, "team_reward": round(team, 3),
                           "native_reward": round(native_reward, 1),
                           "red_action": rrec.get("action_name") or r.action, "red_target": rrec.get("target")})
        print(f"{tstep:>3} | {rdesc:<30.30} | {corrupt:>7} | {native_reward:>+9.1f} | {', '.join(acts)}")
        steps_f.write(json.dumps({
            "t": tstep, "corrupt": corrupt, "native_reward": round(native_reward, 1),
            "red_action": rrec.get("action_name") or r.action, "red_target": rrec.get("target"),
            "blue": [_blue_step(b) for b in sorted(blues, key=_agent_of)]}, default=str) + "\n")
        steps_f.flush()

    steps_f.close()

    out = Path(args.out)
    log = {"meta": {"scenario": "cage4-decentralised", "blue": args.blue, "steps": args.steps,
                    "model": args.model if args.blue == "llm" else None,
                    "decide_every": args.decide_every,
                    "restore_cost": args.restore_cost,
                    "topo": args.topo,
                    "seed": args.seed,
                    "llm_usage": aggregate_usage(
                        [getattr(b, "decision_log", None) for b in blues], args.model),
                    "zones": [z["name"] for z in ZONES],
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "team_db_integrity_reward": round(sum(b.reward_function.total_reward for b in db_blues), 3),
                    "native_blue_reward": round(native_reward, 1),
                    "total_corrupt_steps": sum(t["corrupt"] for t in trajectory),
                    "max_corrupt_steps": len(trajectory) * sum(1 for z in ZONES if z["kind"] == "db"),
                    "per_agent": {_agent_of(b): {
                        "db_integrity_reward": round(b.reward_function.total_reward, 3),
                        "native_reward": round(agent_native.get(_agent_of(b), 0.0), 1),
                        "zones": [z["name"] for z in agent_zones(_agent_of(b))]}
                        for b in sorted(blues, key=_agent_of)},
                    "per_zone": {z["name"]: {
                        "owner": z["agent"],
                        "native_reward": round(zone_native.get(z["name"], 0.0), 1),
                        "restores": zone_restores.get(z["name"], 0),
                        "damage_reward": round(zone_native.get(z["name"], 0.0)
                                               - args.restore_cost * zone_restores.get(z["name"], 0), 1)}
                        for z in ZONES}},
           "trajectory": trajectory,
           "blue_decision_logs": {b.config.ref: getattr(b, "decision_log", None) for b in blues},
           "red_decision_logs": {a.config.ref: getattr(a, "decision_log", None) for a in reds}}
    out.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print("-" * 108)
    print(f"TEAM DB-integrity reward (positive, +1/healthy DB/step): {log['meta']['team_db_integrity_reward']:+.3f}")
    print(f"NATIVE CC4 blue reward   (penalty-only, best=0)        : {log['meta']['native_blue_reward']:+.1f}"
          f"   |   corrupt-steps {log['meta']['total_corrupt_steps']}/{log['meta']['max_corrupt_steps']}   |   log: {out}")


if __name__ == "__main__":
    main()
