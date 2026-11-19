FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# install dependencies
COPY requirements.txt .
COPY deployment/model/requirements.txt deployment/model/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

#copy application and model
COPY src/ ./src/
COPY deployment/model/ ./deployment/model/

# Run as a non-root user
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

ENV MODEL_URI=/app/deployment/model

EXPOSE 8000

# start fastAPI server
CMD ["python", "-m", "uvicorn", "src.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]