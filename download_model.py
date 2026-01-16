import requests

url = "https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

print(f"Downloading from {url}...")
try:
    response = requests.get(url, headers=headers, allow_redirects=True)
    if response.status_code == 200:
        # Check if it looks like HTML
        if b'<!DOCTYPE html>' in response.content[:100]:
            print("Failed: Downloaded content is HTML (likely 404 or auth page).")
        else:
            with open('emotion-ferplus-8.onnx', 'wb') as f:
                f.write(response.content)
            print(f"Success! Downloaded {len(response.content)} bytes.")
    else:
        print(f"Failed with status code: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
