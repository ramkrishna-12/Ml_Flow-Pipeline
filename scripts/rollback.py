
import argparse
import json
import os
import urllib.request

import mlflow
from mlflow import MlflowClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--to-version", required=True)
    ap.add_argument("--model-name", default=os.getenv("MODEL_NAME", "fraud-detector"))
    ap.add_argument("--alias", default=os.getenv("CHAMPION_ALIAS", "champion"))
    ap.add_argument("--api-url")
    a = ap.parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    c = MlflowClient()
    before = c.get_model_version_by_alias(a.model_name, a.alias).version
    c.get_model_version(a.model_name, a.to_version)  # raises if the version does not exist
    c.set_registered_model_alias(a.model_name, a.alias, a.to_version)
    c.set_model_version_tag(a.model_name, a.to_version, "rolled_back_from", str(before))
    print(f"{a.model_name}@{a.alias}: v{before} -> v{a.to_version}")

    if a.api_url:
        req = urllib.request.Request(a.api_url.rstrip("/") + "/reload", method="POST")
        print(json.loads(urllib.request.urlopen(req, timeout=30).read()))


if __name__ == "__main__":
    main()
