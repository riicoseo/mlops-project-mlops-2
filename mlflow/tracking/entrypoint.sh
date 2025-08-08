#!/bin/sh
set -e

if [ -n "${GUNICORN_OPTS:-}" ]; then
  exec mlflow server \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --artifacts-destination "${ARTIFACTS_DESTINATION}" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --gunicorn-opts "${GUNICORN_OPTS}"
else
  exec mlflow server \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --artifacts-destination "${ARTIFACTS_DESTINATION}" \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000
fi