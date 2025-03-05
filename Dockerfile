FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install specific versions of transformers, PEFT, and other libraries
RUN pip install --no-cache-dir \
    accelerate \
    transformers==4.44.1 \
    peft==0.12.0 \
    bitsandbytes==0.41.3 \
    trl==0.8.3 \
    flash-attention

# Copy application code
COPY datasets.py finetune.py results.py evaluation.py ./
COPY data/ ./data/
RUN mkdir -p output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV HF_HOME=/app/.cache/huggingface

# Set default command
CMD ["bash", "/app/run.sh"]