FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install CPU-only PyTorch first.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

# Install application dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face downloader.
RUN pip install --no-cache-dir huggingface_hub

# Copy application source.
COPY backend/ ./backend/
COPY Layer-1/ ./Layer-1/
COPY Layer-2/ ./Layer-2/
COPY Layer-3/ ./Layer-3/
COPY pipeline.py ./

# Download the trained 3-class model from Hugging Face.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hehehehe84737/PhishShield-AI-3class', local_dir='/app/Layer-2/models/phishing-model-3class')"
# Render passes the port to use in $PORT; fall back to 8000 locally.
ENV PORT=8000
EXPOSE 8000

# Container health check.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/api/health')" || exit 1

# Start FastAPI (shell form so $PORT is expanded).
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]