# PhishShield AI — Container image
#
# Packages the detection pipeline + Streamlit SOC dashboard into one
# reproducible image. Anyone can run the whole app with:
#
#   docker build -t phishshield .
#   docker run -p 8501:8501 phishshield
#
# then open http://localhost:8501
#
# NOTES
# - The trained model is git-ignored, so it is NOT baked into the image by
#   default. Mount it at run time (recommended, keeps the image small):
#     docker run -p 8501:8501 -v ${PWD}/Layer-2/models:/app/Layer-2/models phishshield
#   ...or uncomment the COPY line below to bake it in.
# - Layer 3 (Ollama) runs as a SEPARATE service on the host. Inside a
#   container it isn't reachable at localhost, so Layer 3 is optional here.
#   To use it, run Ollama on the host and point the container at it via
#   host.docker.internal (see the OLLAMA_HOST env below).

FROM python:3.12-slim

# System deps kept minimal; slim + no build tools needed for the wheels we use
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install CPU-only torch first (smaller, no CUDA) then the rest.
# CPU build is deliberate: the image runs anywhere, no GPU required.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

# Copy application code (models/data are excluded via .dockerignore)
COPY Layer-1/ ./Layer-1/
COPY Layer-2/ ./Layer-2/
COPY Layer-3/ ./Layer-3/
COPY pipeline.py soc_dashboard.py ./

# To bake the trained model into the image instead of mounting it, uncomment:
# COPY Layer-2/models/ ./Layer-2/models/

# If you run Ollama on the host, this lets Layer 3 reach it from the container
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8501

# Healthcheck so orchestrators know when the app is ready
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["python", "-m", "streamlit", "run", "soc_dashboard.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]