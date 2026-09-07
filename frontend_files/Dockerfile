# ---- SuperKart Sales Prediction: Streamlit Frontend ----
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app.py .

# Streamlit's default port
EXPOSE 8501

# BACKEND_URL is overridden at `docker run` time with --env so the
# frontend can reach the backend container over the shared Docker network
ENV BACKEND_URL=http://superkart-backend:7860

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
