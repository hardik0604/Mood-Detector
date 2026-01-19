from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import base64
import io
from PIL import Image

app = Flask(__name__)

# ------------------ CONFIG ------------------

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB

# ------------------ GLOBALS (LAZY LOADED) ------------------

ORT_SESSION = None
INPUT_NAME = None
face_detector = None

# ------------------ CONSTANTS ------------------

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

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# ------------------ LAZY LOADERS ------------------

def get_onnx_session():
    global ORT_SESSION, INPUT_NAME
    if ORT_SESSION is None:
        ORT_SESSION = ort.InferenceSession(
            "emotion-ferplus-8.onnx",
            providers=["CPUExecutionProvider"],
        )
        INPUT_NAME = ORT_SESSION.get_inputs()[0].name
    return ORT_SESSION, INPUT_NAME


def get_face_detector():
    global face_detector
    if face_detector is None:
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
    return face_detector


# ------------------ UTILS ------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


# ------------------ CORE LOGIC ------------------

def process_image(image_path):
    try:
        session, input_name = get_onnx_session()
        detector = get_face_detector()

        image = cv2.imread(image_path)
        if image is None:
            return None, "Invalid image"

        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = detector.process(image_rgb)
        if not results.detections:
            return None, "No face detected"

        formatted_results = []

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            x, y = max(0, x), max(0, y)
            bw, bh = min(w - x, bw), min(h - y, bh)

            if bw <= 0 or bh <= 0:
                continue

            face = image[y:y+bh, x:x+bw]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            normalized = resized.astype(np.float32) / 255.0

            input_tensor = normalized[np.newaxis, np.newaxis, :, :]
            scores = session.run(None, {input_name: input_tensor})[0][0]
            probs = softmax(scores)

            emotions = {}
            for i, score in enumerate(probs):
                key = EMOTION_TABLE.get(EMOTION_KEYS[i], EMOTION_KEYS[i])
                emotions[key] = max(emotions.get(key, 0), float(score))

            dominant = max(emotions, key=emotions.get)
            confidence = emotions[dominant]

            formatted_results.append({
                "box": [x, y, bw, bh],
                "emotions": emotions,
            })

            cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{dominant}: {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        output_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            "processed_" + os.path.basename(image_path),
        )
        cv2.imwrite(output_path, image)

        return formatted_results, output_path

    except Exception as e:
        return None, str(e)


# ------------------ ROUTES ------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    data, result = process_image(filepath)
    if data is None:
        return jsonify({"error": result}), 400

    return jsonify({
        "success": True,
        "original_image": f"/uploads/{file.filename}",
        "processed_image": f"/uploads/{os.path.basename(result)}",
        "emotions": data,
    })


@app.route("/webcam", methods=["POST"])
def webcam():
    data = request.get_json()
    image_data = data["image"].split(",")[1]
    image_bytes = base64.b64decode(image_data)

    image = Image.open(io.BytesIO(image_bytes))
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], "webcam.jpg")
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
    get_face_detector()
    return "Warmed"


# ------------------ ENTRY ------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
