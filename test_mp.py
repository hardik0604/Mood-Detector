import mediapipe as mp
print("MediaPipe imported")
try:
    print(mp.solutions.face_detection)
    print("Solutions available")
except Exception as e:
    print(f"Error: {e}")
