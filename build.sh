#!/bin/bash
set -e

echo "==> Building from mood_detector_web directory"
cd mood_detector_web

echo "==> Clearing pip cache"
pip cache purge || true

echo "==> Installing dependencies from mood_detector_web/requirements.txt"
pip install --no-cache-dir -r requirements.txt

echo "==> Downloading ONNX model"
cd ..
python download_model.py

echo "==> Build complete!"
