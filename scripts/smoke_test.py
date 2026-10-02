
import argparse
import json
import sys
import time
import urllib.request

RISKY = {
    "amount": 950.0,
    "hour": 3,
    "merchant_risk": 0.8,
    "distance_from_home_km": 400.0,
    "txn_count_24h": 9,
    "account_age_days": 20,
    "is_foreign": 1,
}
SAFE = {
    "amount": 12.0,
    "hour": 14,
    "merchant_risk": 0.02,
    "distance_from_home_km": 2.0,
    "txn_count_24h": 1,
    "account_age_days": 3000,
    "is_foreign": 0,
}


def call(url, payload=None, timeout=10):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--wait", type=int, default=300, help="seconds to wait for /ready")
    a = ap.parse_args()
    base = a.url.rstrip("/")

    deadline = time.time() + a.wait
    while True:
        try:
            info = call(f"{base}/ready")
            print("ready:", info)
            break
        except Exception as e:  # noqa: BLE001
            if time.time() > deadline:
                sys.exit(f"FAIL: /ready never became healthy: {e}")
            time.sleep(5)

    out = call(f"{base}/predict", {"transactions": [RISKY, SAFE]})
    risky, safe = (p["fraud_probability"] for p in out["predictions"])
    print(f"risky={risky:.3f} safe={safe:.3f} model_version={out['model_version']}")
    if not (0 <= safe <= 1 and 0 <= risky <= 1 and risky > safe):
        sys.exit("FAIL: predictions look wrong (risky must score above safe)")
    print("OK")


if __name__ == "__main__":
    main()
