FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Optional but helpful for scientific Python wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Work under a simple app root
WORKDIR /workspace

# Install Python dependencies first (cacheable layer)
COPY frontend_client/requirements.txt frontend_client/requirements.txt
RUN pip install --no-cache-dir -r frontend_client/requirements.txt

# Copy only what the Gradio frontend needs at runtime:
# - frontend_client/ (app.py, images, etc.)
# - process_data.py (for REFERENCE_DATE and EDUCATION_MAPPING)
COPY frontend_client frontend_client
COPY process_data.py process_data.py

# Gradio will listen on 7860 by default
EXPOSE 7860

# Run the Gradio app; process_data.py is imported from repo root (/workspace)
CMD ["python", "frontend_client/app.py"]
