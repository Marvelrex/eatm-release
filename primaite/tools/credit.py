import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def api_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if line.startswith("OPENROUTER_API_KEY=") and "=" in line:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("no OPENROUTER_API_KEY in the environment or ./.env")


def get(url, key):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(req, timeout=30))["data"]


def main():
    key = api_key()
    k = get("https://openrouter.ai/api/v1/auth/key", key)
    print(f"REMAINING  ${k['limit_remaining']:,.2f}   of a ${k['limit']:,.0f} key limit "
          f"(used ${k['usage']:,.2f})")
    print(f"spent      today ${k['usage_daily']:,.2f} | week ${k['usage_weekly']:,.2f} | "
          f"month ${k['usage_monthly']:,.2f}")
    try:
        c = get("https://openrouter.ai/api/v1/credits", key)
        left = c["total_credits"] - c["total_usage"]
        print(f"account    ${left:,.2f} left (${c['total_credits']:,.0f} bought, "
              f"${c['total_usage']:,.2f} used) -- the key limit above is the binding one")
    except Exception as e:
        print(f"account    unavailable ({type(e).__name__})")


if __name__ == "__main__":
    main()
