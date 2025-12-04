# Dockerfile for PM Surya Ghar Rooftop PV Verifier
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (add more if your code needs them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Default entrypoint is the inference pipeline.
# You can pass CLI args after the image name:
#   docker run <image> --input_csv ... --img_root ... --output_dir ...
ENTRYPOINT ["python", "src/inference.py"]
