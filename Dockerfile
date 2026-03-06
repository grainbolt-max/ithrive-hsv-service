FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# - poppler-utils: required by pdf2image
# - tesseract-ocr: required by pytesseract
# - libgl1: required by opencv
RUN apt-get update && \
    apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Production server configuration
CMD ["python", "app.py"]
