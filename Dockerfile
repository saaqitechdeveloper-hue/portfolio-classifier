FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

# Install dependencies
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision pillow && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Hugging Face default port is 7860, Render uses PORT env var
EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
