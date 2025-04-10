FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies including PyTorch
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command (runs classification model training by default)
CMD ["python", "-m", "angle_estimation_framework.scripts.train", "--config", "angle_estimation_framework/configs/swin_classification_18deg.yaml", "--seed", "42"]