# Inference image = CODE ONLY. The model is pulled from the MLflow Registry (models:/<name>@champion)
# at container start, so a new champion never requires an image rebuild - only a task restart / /reload.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /srv

# Same requirements.txt as training => same sklearn/pandas versions => no train/serve skew.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY app ./app

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /srv
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
