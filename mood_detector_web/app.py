from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
import onnxruntime as ort
import base64
import io
import uuid
from PIL import Image

app = Flask(__name__)

# Ensure upload folder is absolute path relative to this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ORT_SESSION = None
INPUT_NAME = None

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

EMOTION_KEYS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

EMOTION_TABLE = {
    "neutral": "neutral",
    "happiness": "happy",
    "surprise": "surprise",
    "sadness": "sad",
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "contempt": "disgust",
}

def get_onnx_session():
    global ORT_SESSION, INPUT_NAME
    if ORT_SESSION is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "emotion-ferplus-8.onnx")
        ORT_SESSION = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        INPUT_NAME = ORT_SESSION.get_inputs()[0].name
    return ORT_SESSION, INPUT_NAME

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

def process_image(image_path):
    try:
        print(f"[DEBUG] Starting process_image for: {image_path}")
        session, input_name = get_onnx_session()
        print("[DEBUG] ONNX session loaded successfully")
        
        # Use OpenCV Haar Cascade (lightweight, reliable on Render)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("[DEBUG] Haar cascade loaded")
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"[ERROR] Failed to read image: {image_path}")
            return None, "Invalid image file"
        print(f"[DEBUG] Image loaded: shape={image.shape}")

        h, w, _ = image.shape
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        print(f"[DEBUG] Detected {len(faces)} faces")

        if len(faces) == 0:
            return None, "No face detected"

        output = []

        for (x, y, w_box, h_box) in faces:
            pad_w = int(w_box * 0.20)
            pad_h = int(h_box * 0.20)
            
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(w, x + w_box + pad_w)
            y2 = min(h, y + h_box + pad_h)

            if x2 <= x1 or y2 <= y1:
                continue
            
            face = image[y1:y2, x1:x2]
            
            if face.size == 0: 
                continue

            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray_face, (64, 64))
            normalized = resized.astype(np.float32)
            
            input_tensor = normalized[np.newaxis, np.newaxis, :, :]
            scores = session.run(None, {input_name: input_tensor})[0][0]
            probs = softmax(scores)

            emotions = {}
            for i, score in enumerate(probs):
                key = EMOTION_TABLE[EMOTION_KEYS[i]]
                emotions[key] = max(emotions.get(key, 0), float(score))

            dominant = max(emotions, key=emotions.get)
            confidence = emotions[dominant]

            output.append({
                "box": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                "emotions": emotions
            })

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            label = f"{dominant}: {confidence:.2f}"
            cv2.putText(image, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        output_path = os.path.join(app.config["UPLOAD_FOLDER"], f"processed_{os.path.basename(image_path)}")

        if not cv2.imwrite(output_path, image):
            return None, "Failed to save image"

        return output, output_path

    except Exception as e:
        import traceback
        print(f"[ERROR] Exception in process_image: {str(e)}")
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        return None, str(e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename.replace(' ', '_')}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    data, result = process_image(filepath)
    if data is None:
        return jsonify({"error": result}), 400

    return jsonify({
        "success": True,
        "original_image": f"/uploads/{filename}",
        "processed_image": f"/uploads/{os.path.basename(result)}",
        "emotions": data,
    })

@app.route("/webcam", methods=["POST"])
def webcam():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Invalid payload"}), 400

    image_data = data["image"].split(",")[-1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    filename = f"webcam_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    image.save(filepath)

    data, result = process_image(filepath)
    if data is None:
        return jsonify({"error": result}), 400

    return jsonify({
        "success": True,
        "processed_image": f"/uploads/{os.path.basename(result)}",
        "emotions": data,
    })

@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/favicon.ico")
def favicon():
    return "", 204  # No Content - prevents console errors


@app.route("/health")
def health():
    try:
        get_onnx_session()
        return jsonify({"status": "ok", "model": "loaded"}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route("/warmup")
def warmup():
    get_onnx_session()
    return "Warmed up"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
