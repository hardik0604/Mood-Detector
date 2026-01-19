import requests
import os
import sys

# The specific raw URL that usually works for LFS files
URLS = [
    "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "https://media.githubusercontent.com/media/onnx/models/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "https://raw.githubusercontent.com/microsoft/FERPlus/master/src/models/emotion-ferplus-8.onnx",
]

TARGET = os.path.join("mood_detector_web", "emotion-ferplus-8.onnx")

def check_is_valid_onnx(path):
    if not os.path.exists(path):
        return False
    # Check size (should be > 100KB)
    size = os.path.getsize(path)
    if size < 10000: # 10KB is too small
        return False
    
    # Check magic bytes (ONNX usually starts with specific bytes, but ensuring it's not HTML is good enough)
    with open(path, 'rb') as f:
        header = f.read(100)
        if b'<!DOCTYPE' in header or b'<html' in header or b'version https://git-lfs' in header:
            return False
    return True

print(f"Attempting to fix {TARGET}...")

for url in URLS:
    print(f"Downloading from {url}...")
    try:
        r = requests.get(url, allow_redirects=True)
        if r.status_code == 200:
            with open(TARGET, 'wb') as f:
                f.write(r.content)
            
            if check_is_valid_onnx(TARGET):
                print(f"SUCCESS! Valid model downloaded ({os.path.getsize(TARGET)} bytes).")
                sys.exit(0)
            else:
                print("Downloaded file was invalid (HTML or LFS pointer).")
        else:
            print(f"HTTP Error: {r.status_code}")
    except Exception as e:
        print(f"Exception: {e}")

print("NON-CRITICAL FAILURE: Could not automatically download. Please download 'emotion-ferplus-8.onnx' manually.")
sys.exit(1)
