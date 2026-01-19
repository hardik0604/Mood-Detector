#!/usr/bin/env python3
"""Download the ONNX model if it doesn't exist."""
import os
import sys
import urllib.request

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mood_detector_web", "emotion-ferplus-8.onnx")

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"✓ Model already exists at {MODEL_PATH}")
        return True
    
    print(f"Downloading ONNX model from {MODEL_URL}...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✓ Downloaded successfully ({size_mb:.1f}MB)")
        return True
    except Exception as e:
        print(f"✗ Failed to download model: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
