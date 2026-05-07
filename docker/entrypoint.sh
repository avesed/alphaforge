#!/bin/bash
set -e

echo "Running database migrations..."
cd /app/backend && alembic upgrade head

echo "Starting AlphaForge..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/alphaforge.conf
