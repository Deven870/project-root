# VoiceBot Trading System - Docker Container
FROM python:3.10-slim

LABEL maintainer=\"VoiceBot Team\"
LABEL description=\"Integrated Trading Bot + API + Dashboard\"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    wget \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 voicebot

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \\
    && pip install --no-cache-dir \\
    gunicorn==20.1.0 \\
    psycopg2-binary==2.9.6

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data \\
    && chown -R voicebot:voicebot /app

# Switch to non-root user
USER voicebot

# Expose ports
EXPOSE 5000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:5000/api/health || exit 1

# Entrypoint
ENTRYPOINT [\"python\", \"system_launcher.py\"]

# Default command - start all services
CMD [\"--no-api\"]
