# Stage 1: Frontend builder
FROM node:20-alpine AS frontend-builder
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit
COPY frontend/ ./
RUN npm run build

# Stage 2: Production
FROM python:3.11-slim AS production

# Install system dependencies
# - libgomp1: required by LightGBM for OpenMP parallelism
# - cmake, gcc, g++: required to build pyqlib from source
# - postgresql-client: for alembic migrations
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl nginx supervisor dumb-init postgresql-client \
    libgomp1 cmake gcc g++ gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/backend

# Install Python dependencies first (layer caching)
COPY backend/pyproject.toml ./backend/
RUN pip install --no-cache-dir \
    $(python3 -c "import tomllib; print(' '.join(tomllib.load(open('backend/pyproject.toml','rb'))['project']['dependencies']))")

# Remove build tools to reduce image size
RUN apt-get purge -y cmake gcc g++ gfortran \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend to nginx html
COPY --from=frontend-builder /build/dist /usr/share/nginx/html

# Copy Docker config
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY docker/supervisord.conf /etc/supervisor/conf.d/alphaforge.conf
COPY docker/entrypoint.sh /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# Create data directories
RUN mkdir -p /app/data/qlib /app/data/predictions

# Remove default nginx config
RUN rm -f /etc/nginx/sites-enabled/default

# Create non-root user
RUN groupadd -g 1000 alphaforge \
    && useradd -u 1000 -g alphaforge -s /bin/bash -m alphaforge \
    && chown -R alphaforge:alphaforge /app /var/log/nginx /var/lib/nginx /run

USER alphaforge

EXPOSE 80 8015

ENTRYPOINT ["/usr/bin/dumb-init", "--"]
CMD ["/app/docker/entrypoint.sh"]
