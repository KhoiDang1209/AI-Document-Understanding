# syntax=docker/dockerfile:1
# CPU-only runtime image for the DocIntel API (run-time environment).
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# OpenCV (non-headless) runtime libs — libglib2.0-0 for GLib, libgl1 for libGL.so.1
# (docTR pulls non-headless opencv-python as a transitive dep; libgl1 is required).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
# Pin a CPU-only torch/torchvision (PyPI's default Linux wheel is CUDA — huge),
# then install the project with the runtime extras the serving path needs. The
# app factory wires every router unconditionally (api/main.py), so the image must
# carry the deps those routers import at module load:
#   serve (onnxruntime, transformers, prometheus instrumentator), kie (mlflow, boto3),
#   contracts (pymupdf/liteparse), rag (langchain-core, fastembed, qdrant),
#   graph (neo4j), agent (langgraph, langfuse).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install ".[serve,kie,contracts,rag,graph,agent]"
# Bake docTR weights into the image so the running container needs no network.
RUN python -c "from doctr.models import ocr_predictor; ocr_predictor(pretrained=True)"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "docintel.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
