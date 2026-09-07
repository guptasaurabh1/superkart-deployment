# ---- SuperKart Sales Prediction: Flask Backend ----
FROM python:3.10-slim

# Keep Python output unbuffered so logs show up immediately in Codespace
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the serialized model
COPY app.py .
COPY superkart_model.joblib .

# Flask API is served on port 7860 (forwarded publicly from the Codespace)
EXPOSE 7860

CMD ["python", "app.py"]
