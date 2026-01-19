from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import base64
import io
import uuid
from PIL import Image

# -------------------- APP --------------------

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# -------------------- GLOBALS (LAZY LOADED) --------------------

ORT_SESSION = None
INPUT_NAME = None
FACE_DETECTOR = None

# -------------------- CONSTANTS --------------------

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

# -------------------- LAZY LOADERS --------------------

def get_onnx_session():
    global ORT_SESSION, INPUT_NAME
    if ORT_SESSION is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "emotion-ferplus-8.onnx")
        ORT_SESSION = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        INPUT_NAME = ORT_SESSION.get_inputs()[0].name
    return ORT_SESSION, INPUT_NAME




# -------------------- UTILS --------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


# -------------------- CORE LOGIC --------------------

def process_image(image_path):
    try:
        session, input_name = get_onnx_session()
        
        # Use OpenCV Haar Cascade (Robust Fallback)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        image = cv2.imread(image_path)
        if image is None:
            return None, "Invalid image file"

        h, w, _ = image.shape
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return None, "No face detected"

        output = []

        for (x, y, w_box, h_box) in faces:
            # Convert to coordinates
            x1, y1 = x, y
            x2, y2 = x + w_box, y + h_box

            # Padding (optional but good for emotion model)
            face = image[y1:y2, x1:x2]
            
            if face.size == 0: 
                continue

            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray_face, (64, 64))
            normalized = resized.astype(np.float32) / 255.0

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
                "box": [int(x1), int(y1), int(w_box), int(h_box)],
                "emotions": emotions
            })

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{dominant}: {confidence:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        output_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"processed_{os.path.basename(image_path)}"
        )

        if not cv2.imwrite(output_path, image):
            return None, "Failed to save image"

        return output, output_path

    except Exception as e:
        return None, str(e)

# -------------------- ROUTES --------------------

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

    filename = f"{uuid.uuid4().hex}_{file.filename}"
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


@app.route("/warmup")
def warmup():
    get_onnx_session()
    return "Warmed up"


# -------------------- ENTRY --------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
