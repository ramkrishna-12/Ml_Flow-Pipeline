.PHONY: install lint test up down train train-quick ui serve smoke docker-api push-mlflow tf-init tf-plan tf-apply
PY ?= python
AWS_REGION ?= ap-south-1

install:            ## deps
	$(PY) -m pip install -r requirements-dev.txt
lint:
	ruff check . && ruff format --check .
test:
	$(PY) -m pytest -q

up:                 ## local mlflow + postgres + minio (+ api)
	docker compose up -d --build
down:
	docker compose down -v

train:              ## full sweep (uses MLFLOW_TRACKING_URI, default sqlite:///mlflow.db)
	$(PY) -m src.pipeline
train-quick:
	$(PY) -m src.pipeline --no-sweep
ui:                 ## browse local sqlite tracking store
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
serve:
	uvicorn app.main:app --reload --port 8000
smoke:
	$(PY) scripts/smoke_test.py --url http://localhost:8000

docker-api:
	docker build -t fraud-api:local .

push-mlflow:        ## build the MLflow server image for linux/amd64 (Fargate) and push to ECR
	@REPO=$$(cd infra/terraform && terraform output -raw ecr_mlflow_repo_url); \
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $${REPO%%/*}; \
	docker buildx build --platform linux/amd64 -f docker/mlflow-server.Dockerfile -t $$REPO:latest --push .

tf-init:
	cd infra/terraform && terraform init
tf-plan:
	cd infra/terraform && terraform plan
tf-apply:
	cd infra/terraform && terraform apply
