# MLOps with MLflow on AWS — Fraud Detection

A small but *complete* MLOps loop, built to be understood end-to-end for an interview:

```
 data (S3/parquet) ──► validate ──► train sweep ──► quality gate ──► register ──► challenger vs champion
                                     (MLflow nested runs)                          │ promote = move @champion alias
 GitHub push/cron ──► CI (lint, tests, pipeline smoke, docker, tf fmt/validate)    ▼
                  └─► CD: train ─► build image ─► [approval] ─► ECS rolling deploy ─► smoke test
                                                                    │
 Client ─► ALB:80 ─► ECS Fargate "api"  ── loads models:/fraud-detector@champion ──► ECS "mlflow" ─► RDS Postgres
                                                                                        (proxy)     └► S3 artifacts
```

| Concern | Local | AWS |
|---|---|---|
| Backend store (runs, params, metrics, **registry**) | sqlite / Postgres container | **RDS Postgres** |
| Artifact store (models, plots, data lineage) | `./mlartifacts` / MinIO | **S3** (via MLflow `--serve-artifacts` proxy) |
| Tracking server + registry | `mlflow ui` / compose | **ECS Fargate** behind **ALB** |
| Inference | `uvicorn` | **ECS Fargate** service, image in **ECR** |
| CI/CD auth | – | **GitHub OIDC → IAM role** (no static keys) |
| Secrets | – | **Secrets Manager** (DB password) |
| Logs | stdout | **CloudWatch Logs** |

## Repo map (and which MLflow concept each file teaches)

| File | What to learn from it |
|---|---|
| `src/data.py` | Data validation gate, S3 I/O, why accuracy is useless at 3% fraud |
| `src/train.py` | Experiments/runs, **nested runs**, params/metrics/tags, artifacts (`log_figure`, `log_dict`), **dataset lineage** (`log_input`), **signature + input example**, `log_model` (sklearn flavor, `pyfunc_predict_fn`), skops vs pickle, train/val/test discipline |
| `src/register.py` | **Model Registry**: register, **aliases** (`@challenger`, `@champion`), quality gate, champion-vs-challenger on the *same* holdout |
| `src/pipeline.py` | Orchestration, exit codes (0 ok, 2 gate failed), result JSON consumed by CI/CD |
| `MLproject`, `python_env.yaml` | **MLflow Projects** (`MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow run . -e quick --env-manager local --experiment-name fraud-detection`) |
| `app/main.py` | Serving via generic **pyfunc**, liveness vs readiness, alias-based loading, `/reload` |
| `scripts/rollback.py` | Rollback = move the alias, no rebuild |
| `docker-compose.yml` | Prod-shaped local stack (Postgres + MinIO + MLflow + API) |
| `.github/workflows/ci.yml` | Integration: lint, tests, real pipeline run, docker build, terraform fmt/validate |
| `.github/workflows/cd.yml` | Continuous training + gated deployment, OIDC, environments, smoke test |
| `infra/terraform/` | S3, ECR, RDS, ECS/Fargate, ALB, Cloud Map, IAM (least privilege), OIDC |

## 1. Run it locally (5 minutes, no Docker, no AWS)

```bash
python -m venv .venv && source .venv/bin/activate
make install
make test                    # 10 tests: data, pipeline, gate, registry, API
make train                   # full sweep -> registers v1 and sets @champion
make ui                      # http://localhost:5000  -> Experiments, nested runs, Models tab
make train                   # run again: v2 is registered but NOT promoted (no improvement)
make serve                   # http://localhost:8000/docs
curl -s localhost:8000/ready
make smoke
```

## 2. Prod-shaped local stack

```bash
docker compose up -d --build          # postgres + minio + mlflow + api
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.pipeline                # artifacts go through the server; you need no S3 creds
# S3-style data source against MinIO:
AWS_ENDPOINT_URL_S3=http://localhost:9000 AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
  DATA_SOURCE=s3://mlflow/data/transactions.parquet python -m src.pipeline
```

## 3. Deploy to AWS (order matters)

```bash
cd infra/terraform
terraform init
terraform apply -var github_repo=<owner/repo> -var api_desired_count=0   # API can't be ready: no champion yet
cd ../.. && make push-mlflow                                              # MLflow image -> ECR; ECS service turns healthy
terraform -chdir=infra/terraform output github_actions_variables         # -> GitHub repo *Variables*
```
1. Create GitHub **environments** `staging` and `production` (add *required reviewers* on production).
2. Bootstrap the first champion: `MLFLOW_TRACKING_URI=<mlflow_tracking_uri output> python -m src.pipeline`
3. `terraform apply` again (default `api_desired_count=2`) → API tasks load `@champion` and go healthy.
4. From now on everything is the pipeline: push to `main` (or run *cd* manually).

**Cost/safety caveats (read these):** the lab uses the default VPC with public subnets (no NAT), and MLflow has **no
authentication** — `mlflow_allowed_cidrs` defaults to `0.0.0.0/0` so GitHub-hosted runners can reach it. Before real use:
restrict CIDRs, put HTTPS (ACM) on the ALB, enable MLflow basic-auth (`--app-name basic-auth`) or move to private subnets
with a self-hosted runner, use remote Terraform state (the DB password is in state), and set `deletion_protection` on RDS.
Run `terraform destroy` when done.

## 4. How the CI/CD works

**ci.yml** (PRs / non-main pushes): `ruff` → `pytest` → run the *actual* pipeline against throwaway sqlite → build the image
and hit `/health` → `terraform fmt -check` + `validate`.

**cd.yml** (push to `main`, weekly cron, manual):
1. **train** (env `staging`): assume AWS role via OIDC → `python -m src.pipeline` against the remote MLflow and S3 data.
   Gate fails ⇒ exit code 2 ⇒ job fails ⇒ nothing registered, nothing deployed.
2. **deploy** (env `production` ⇒ manual approval): build/push image (tag = git SHA) → fetch live task definition →
   swap the image → ECS rolling deploy (`wait-for-service-stability`) → smoke test the ALB URL.
   Cron runs deploy only when a new champion was promoted.

Why the image contains no model: the API resolves `models:/fraud-detector@champion` at start-up. Code changes ship a new
image; model changes ship by moving an alias. Two independent release trains.

## 5. Operations runbook

| Situation | Action |
|---|---|
| Bad model reached production | `MLFLOW_TRACKING_URI=… python scripts/rollback.py --to-version N --api-url <ALB>` (moves `@champion`, hot-reloads) |
| Bad image/code | ECS circuit breaker auto-rolls back if new tasks never pass `/ready`; otherwise redeploy previous SHA |
| Gate keeps failing | Open the parent `sweep` run → tag `promotion_decision`; compare child runs in the UI |
| API up but 503 on `/ready` | Can't reach MLflow, or `@champion` missing → CloudWatch logs `/ecs/mlops-fraud/api` |

## 6. Exercises that make it stick

1. Set `MIN_ROC_AUC=0.99`, run the pipeline → watch the gate fail with exit code 2, and confirm nothing was registered.
2. Lower `MIN_IMPROVEMENT` to `-1` → a worse model gets promoted. Explain why gates need a *positive* margin.
3. Run twice with different `--rows` → why is comparing metrics from *different* datasets misleading? (see `evaluate_uri`).
4. In the UI: select the sweep's children → **Compare** → parallel-coordinates plot.
5. `scripts/rollback.py` to v1, then `curl /ready` → confirm the version changed without a restart.
6. Replace `log_model(..., serialization_format=…)` with `cloudpickle` and read why skops is the safe default.
7. Add `mlflow.models.evaluate(...)` with `validation_thresholds` in `register.py` as an alternative gate.

## 7. Interview answers (≈20 seconds each)

**What are MLflow's components?** Tracking (runs, params, metrics, artifacts), Models (a packaging format with flavors and
a signature), the Model Registry (versions, aliases, lineage), and Projects (reproducible entry points). Newer versions add
evaluation and GenAI tracing.

**Backend store vs artifact store?** Backend store is a database with metadata — params, metrics, tags, registry — so RDS
Postgres. Artifact store holds large files — models, plots — so S3. File-based backends have no registry, so use a DB.

**Why use the artifact proxy (`--serve-artifacts`)?** Clients talk only to the tracking server; only the server needs S3
permissions. Fewer credentials, tighter IAM. Trade-off: all artifact traffic flows through the server.

**Stages vs aliases?** Stages (Staging/Production) are deprecated. Aliases are named, movable pointers to a version —
`@champion`. Consumers load `models:/name@champion`, so promotion and rollback are alias moves, not code changes.

**How do you promote a model?** Train, then a quality gate on a holdout. Register the version, tag it `@challenger`,
evaluate it against the current champion on the same data, and move `@champion` only if it wins by a margin. All of it is in code, with an audit trail in tags.

**What's a model signature and why care?** It's the input/output schema stored with the model. MLflow enforces it at
inference, so a bad payload fails fast instead of producing silent garbage. I cast features to float64 so the schema stays strict and simple.

**How do you avoid train/serve skew?** The scaler is inside the sklearn Pipeline, so the logged artifact is preprocessing
plus model. Also the same `requirements.txt` builds the training env and the serving image.

**Why nested runs?** A hyper-parameter sweep is one parent run with a child per candidate. It keeps the UI clean and
lets me compare children and tag the parent with the decision.

**Why PR-AUC and not accuracy?** Fraud is about 3% of rows, so predicting "not fraud" gives 97% accuracy. PR-AUC and recall
measure what we care about. I pick the best candidate on validation and gate on an untouched test set.

**How does CI/CD authenticate to AWS?** GitHub OIDC. The job gets a short-lived token, and the IAM trust policy pins the
audience and the repo/environment `sub`. No access keys are stored. The role can push to one ECR repo and update one ECS service.

**How do you deploy safely?** Approval gate on the `production` environment, rolling deploy at 100% min healthy, ALB
health check on `/ready` (readiness, not liveness), ECS circuit breaker with automatic rollback, and a post-deploy smoke test.

**Model in the image or pulled at start-up?** Pulled from the registry by alias: code and model release independently, and
rollback needs no rebuild. Trade-off: start-up depends on MLflow, which is why readiness gates traffic. Alternative: bake the model in for immutable, air-gapped releases.

**How would you monitor it?** CloudWatch for latency and errors, plus data/prediction drift (feature distributions vs
training, PSI) and delayed-label performance. A drift alarm triggers the retrain workflow. *(Not implemented here — the natural next step.)*

## 8. Known gaps (deliberate, to keep this learnable)

Synthetic data; no drift monitoring; no autoscaling; no HTTPS/auth (see caveats); Terraform is syntax-checked and
formatted but you should `terraform plan` before applying; one region, one environment.
