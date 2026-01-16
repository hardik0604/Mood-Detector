
try:
    with open('emotion-ferplus-8.onnx', 'rb') as f:
        header = f.read(100)
    print(f"Header: {header}")
except Exception as e:
    print(f"Error: {e}")
