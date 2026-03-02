FROM python:3.10-slim

WORKDIR /app

# Install Poppler (required by pdf2image)
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Production server (memory safe config)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "2"]
