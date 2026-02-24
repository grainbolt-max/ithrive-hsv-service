FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (REQUIRED for pdf2image)
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    apt-get clean

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
