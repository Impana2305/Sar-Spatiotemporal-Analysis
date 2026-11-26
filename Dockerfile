# Stage 1: Build React + Vite frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

# Stage 2: Python + PyTorch + Flask production runtime
FROM python:3.10-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PORT=5000

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py ./
COPY server/ ./server/
COPY models/ ./models/
COPY --from=frontend-builder /app/web/dist ./web/dist

EXPOSE 5000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 server.app:app"]
