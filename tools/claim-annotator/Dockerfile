# Claim Annotation Tool - Docker Image for EC2 Deployment
#
# Build:
#   docker build -t claim-annotator .
#
# Run with file picker (recommended):
#   docker run -p 8080:8080 -v /path/to/data:/data claim-annotator
#   Then open browser and select files via web interface
#
# Run with PASSWORD PROTECTION (recommended for EC2):
#   docker run -p 8080:8080 -v /path/to/data:/data claim-annotator --password YOUR_SECRET
#
# DATA PERSISTENCE:
#   The -v flag mounts your host directory to /data inside the container.
#   All files are stored on your HOST machine, not inside the container.
#   Your annotations are preserved when the container stops or is deleted.
#
# Run with pre-selected file:
#   docker run -p 8080:8080 -v /path/to/data:/data claim-annotator \
#     /data/conversations.jsonl --output /data/annotations.jsonl
#
# Run with HTTPS:
#   docker run -p 443:443 -v /path/to/data:/data -v /path/to/certs:/certs \
#     claim-annotator --ssl-cert /certs/cert.pem --ssl-key /certs/key.pem --port 443

FROM python:3.11-slim

LABEL maintainer="Claim Annotator" \
      description="Web-based claim extraction and annotation tool for EC2 deployment"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash annotator

WORKDIR /app

# Copy application files
COPY annotate_responses.py .
COPY claim_annotator/ ./claim_annotator/

# No external dependencies needed - uses only Python standard library

# Create data directory
RUN mkdir -p /data && chown annotator:annotator /data

# Switch to non-root user
USER annotator

# Default port
EXPOSE 8080

# Healthcheck - checks root which works for both file picker and annotation pages
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

# Default entrypoint - server mode with file picker
ENTRYPOINT ["python", "annotate_responses.py", "--server-mode", "--data-dir", "/data", "--port", "8080"]

# Default: no additional args (uses file picker)
CMD []
