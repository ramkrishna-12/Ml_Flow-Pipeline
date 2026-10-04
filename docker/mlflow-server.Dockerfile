# Tracking server + registry. Backend store = Postgres (RDS), artifacts = S3 (proxied by the server).
FROM python:3.11-slim
ARG MLFLOW_VERSION=3.16.1
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" boto3 psycopg2-binary
RUN useradd --create-home --uid 10001 mlflow
USER mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
