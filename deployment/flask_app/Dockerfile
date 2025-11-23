# Start from official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy app
COPY . /app

# Install system deps (for shap/xgboost build if needed)
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Default command (use gunicorn for production)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app:app"]
