# ─────────────────────────────────────────────────────────────────
# Smart Classroom Edge AI — Dockerfile
# Optimised for Linux / WSL edge deployment
# ─────────────────────────────────────────────────────────────────

# --- Stage 1: Slim Python base ---
FROM python:3.10-slim

# Metadata
LABEL maintainer="Smart Classroom Edge AI"
LABEL description="Occupancy detection dashboard using ONNX Runtime + Streamlit"

# --- System dependencies ---
# libgl1 + libglib2.0-0 are required by OpenCV even in headless mode
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python environment ---
WORKDIR /app

# Copy requirements first (layer caching: only re-installs if requirements change)
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# --- Copy application source ---
COPY app.py .
COPY utils.py .
COPY model.onnx .
COPY labels.txt .

# --- Streamlit configuration ---
# Disable telemetry, set server settings for container environment
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Expose the Streamlit port
EXPOSE 8501

# Health check — ensures container is actually serving before routing traffic
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# --- Entry point ---
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
