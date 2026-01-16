import os
import cv2
from app import app, process_image

# Set up the app context/config
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs('uploads', exist_ok=True)

from deepface import DeepFace

def test_image(image_path):
    print(f"\n--- Testing {image_path} ---")
    
    # Test App Logic (Current, with CLAHE)
    print("1. App Logic (CLAHE + RGB conversion):")
    try:
        results, output_path = process_image(image_path)
        if results:
            for i, face in enumerate(results):
                emotions = face['emotions']
                dominant_emotion = max(emotions, key=emotions.get)
                print(f"   Face {i+1}: Dominant = {dominant_emotion} ({emotions[dominant_emotion]*100:.2f}%)")
        else:
            print(f"   Failed: {output_path}")
    except Exception as e:
        print(f"   Error: {e}")

    # Test Direct DeepFace (Standard, no extra preprocessing)
    print("2. Direct DeepFace (Standard):")
    try:
        objs = DeepFace.analyze(img_path=image_path, actions=['emotion'], detector_backend='retinaface', enforce_detection=False)
        for i, obj in enumerate(objs):
            emotions = obj['emotion']
            dominant_emotion = obj['dominant_emotion']
            print(f"   Face {i+1}: Dominant = {dominant_emotion} ({emotions[dominant_emotion]:.2f}%)")
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    if os.path.exists('happy.jpg'):
        test_image('happy.jpg')
    else:
        print("happy.jpg not found")

    if os.path.exists('sad.jpg'):
        test_image('sad.jpg')
    else:
        print("sad.jpg not found")
