from __future__ import annotations

from typing import Dict, Mapping, Optional

FAMILY_BY_ACTION: Dict[str, str] = {
    "do-nothing": "Sleep",
    "node-os-scan": "Analyse",
    "node-service-scan": "Analyse",
    "node-file-scan": "Analyse",
    "node-folder-scan": "Analyse",
    "node-folder-checkhash": "Analyse",
    "node-file-checkhash": "Analyse",
    "node-network-service-recon": "Analyse",
    "node-nmap-ping-scan": "Analyse",
    "node-nmap-port-scan": "Analyse",
    "node-application-close": "Remove",
    "node-folder-restore": "Restore",
    "node-file-repair": "Restore",
    "node-service-restart": "Restore",
    "node-service-stop": "Restore",
    "node-reset": "Restore",
    "node-service-start": "DeployDecoy",
    "node-application-execute": "DeployDecoy",
    "host-nic-disable": "BlockTrafficZone",
    "router-acl-add-rule": "BlockTrafficZone",
    "firewall-acl-add-rule": "BlockTrafficZone",
    "host-nic-enable": "AllowTrafficZone",
    "router-acl-remove-rule": "AllowTrafficZone",
    "firewall-acl-remove-rule": "AllowTrafficZone",
}
_DEFAULT_FAMILY = "Restore"


def family_of(action_name: str, overrides: Optional[Mapping[str, str]] = None) -> str:
    if overrides and action_name in overrides:
        return overrides[action_name]
    return FAMILY_BY_ACTION.get(action_name, _DEFAULT_FAMILY)
