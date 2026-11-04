FROM python:3.12-slim

# Install Tesseract OCR and OpenCV system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY . .

# Create directories for data and reports
RUN mkdir -p data reports public

# Non-root security user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV OCR_ENGINE=tesseract
ENV DEPLOY_TARGET=local
ENV LOG_LEVEL=INFO

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

