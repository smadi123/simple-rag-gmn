# rag_service/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy *only* requirements.txt first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]