import requests
import os

urls = [
    "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "https://media.githubusercontent.com/media/onnx/models/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "https://github.com/microsoft/FERPlus/raw/master/src/models/emotion-ferplus-8.onnx",
    "https://huggingface.co/onnx/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx" # Hypothetical mirror
]

target_path = "mood_detector_web/emotion-ferplus-8.onnx"

for url in urls:
    print(f"Trying {url}...")
    try:
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200:
            content = response.content
            if b'<!DOCTYPE html>' in content[:100] or b'<html' in content[:100]:
                print("  -> Failed: Content is HTML.")
            elif len(content) < 1000:
                print(f"  -> Failed: Content too small ({len(content)} bytes).")
            else:
                with open(target_path, 'wb') as f:
                    f.write(content)
                print(f"SUCCESS! Downloaded {len(content)} bytes to {target_path}.")
                break
        else:
            print(f"  -> Failed: Status {response.status_code}")
    except Exception as e:
        print(f"  -> Error: {e}")
