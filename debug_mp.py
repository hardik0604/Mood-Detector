import mediapipe as mp
try:
    print(f"mp.solutions: {mp.solutions}")
    print("Success")
except AttributeError:
    print("Failed: No attribute solutions")

try:
    from mediapipe import solutions
    print(f"from mediapipe import solutions: {solutions}")
except ImportError as e:
    print(f"ImportError: {e}")
