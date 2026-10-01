FROM python:3.11-slim

WORKDIR /app

# tesseract para OCR MVP + poppler no necesario (usamos pypdfium2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
ENV PIP_ROOT_USER_ACTION=ignore PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8000
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
