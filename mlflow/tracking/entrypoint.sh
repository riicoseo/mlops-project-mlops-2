#!/bin/bash
set -e

mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --artifacts-destination ${ARTIFACTS_DESTINATION} \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000

    
