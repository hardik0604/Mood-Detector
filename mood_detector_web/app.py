from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
import onnxruntime as ort
import base64
import io
import uuid
from PIL import Image

# =========================
# BASIC APP SETUP
# =========================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# =========================
# PERFORMANCE / MEMORY FIXES
# =========================

# Prevent OpenCV from spawning threads (VERY IMPORTANT on Render)
cv2.setNumThreads(1)

# =========================
# LOAD FACE DETECTOR ONCE
# =========================

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if FACE_CASCADE.empty():
    raise RuntimeError("Failed to load Haar Cascade")

# =========================
# LOAD ONNX MODEL ONCE (PRELOAD)
# =========================

print("[BOOT] Loading ONNX emotion model...")

MODEL_PATH = os.path.join(BASE_DIR, "emotion-ferplus-8.onnx")

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1

ORT_SESSION = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
    sess_options=sess_options
)

INPUT_NAME = ORT_SESSION.get_inputs()[0].name

print("[BOOT] ONNX model loaded successfully")

# =========================
# CONSTANTS
# =========================

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

# Song recommendations database
SONG_RECOMMENDATIONS = {
    'happy': [
        {'title': 'Happy', 'artist': 'Pharrell Williams', 'genre': 'Pop'},
        {'title': 'Good as Hell', 'artist': 'Lizzo', 'genre': 'Pop/R&B'},
        {'title': "Can't Stop the Feeling!", 'artist': 'Justin Timberlake', 'genre': 'Pop'},
        {'title': 'Walking on Sunshine', 'artist': 'Katrina and the Waves', 'genre': 'Rock'},
        {'title': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars', 'genre': 'Funk/Pop'},
    ],
    'sad': [
        {'title': 'Someone Like You', 'artist': 'Adele', 'genre': 'Pop/Soul'},
        {'title': 'The Scientist', 'artist': 'Coldplay', 'genre': 'Alternative'},
        {'title': 'Hurt', 'artist': 'Johnny Cash', 'genre': 'Country'},
        {'title': 'The Sound of Silence', 'artist': 'Simon & Garfunkel', 'genre': 'Folk'},
        {'title': 'Everybody Hurts', 'artist': 'R.E.M.', 'genre': 'Alternative Rock'},
    ],
    'angry': [
        {'title': 'Break Stuff', 'artist': 'Limp Bizkit', 'genre': 'Nu Metal'},
        {'title': 'Killing in the Name', 'artist': 'Rage Against the Machine', 'genre': 'Alternative Metal'},
        {'title': 'Chop Suey!', 'artist': 'System of a Down', 'genre': 'Alternative Metal'},
        {'title': 'Numb', 'artist': 'Linkin Park', 'genre': 'Nu Metal'},
        {'title': 'Stronger', 'artist': 'Kelly Clarkson', 'genre': 'Pop Rock'},
    ],
    'fear': [
        {'title': 'Brave', 'artist': 'Sara Bareilles', 'genre': 'Pop'},
        {'title': 'Stronger (What Doesn\'t Kill You)', 'artist': 'Kelly Clarkson', 'genre': 'Pop'},
        {'title': 'Fight Song', 'artist': 'Rachel Platten', 'genre': 'Pop'},
        {'title': 'Titanium', 'artist': 'David Guetta ft. Sia', 'genre': 'Electronic/Pop'},
        {'title': 'Eye of the Tiger', 'artist': 'Survivor', 'genre': 'Rock'},
    ],
    'surprise': [
        {'title': 'Levitating', 'artist': 'Dua Lipa', 'genre': 'Pop'},
        {'title': 'Blinding Lights', 'artist': 'The Weeknd', 'genre': 'Synth-pop'},
        {'title': 'Shake It Off', 'artist': 'Taylor Swift', 'genre': 'Pop'},
        {'title': 'Electric Feel', 'artist': 'MGMT', 'genre': 'Indie Pop'},
        {'title': 'Mr. Blue Sky', 'artist': 'Electric Light Orchestra', 'genre': 'Rock'},
    ],
    'disgust': [
        {'title': 'Confident', 'artist': 'Demi Lovato', 'genre': 'Pop'},
        {'title': 'Roar', 'artist': 'Katy Perry', 'genre': 'Pop'},
        {'title': 'Stronger', 'artist': 'Kelly Clarkson', 'genre': 'Pop Rock'},
        {'title': 'Unstoppable', 'artist': 'Sia', 'genre': 'Pop'},
        {'title': 'Rise Up', 'artist': 'Andra Day', 'genre': 'Soul'},
    ],
    'neutral': [
        {'title': 'Weightless', 'artist': 'Marconi Union', 'genre': 'Ambient'},
        {'title': 'Claire de Lune', 'artist': 'Claude Debussy', 'genre': 'Classical'},
        {'title': 'River', 'artist': 'Joni Mitchell', 'genre': 'Folk'},
        {'title': 'Holocene', 'artist': 'Bon Iver', 'genre': 'Indie Folk'},
        {'title': 'The Night We Met', 'artist': 'Lord Huron', 'genre': 'Indie Folk'},
    ]
}

def get_song_recommendations(emotion, num_songs=5):
    """Get song recommendations based on detected emotion"""
    import random
    if emotion.lower() in SONG_RECOMMENDATIONS:
        songs = SONG_RECOMMENDATIONS[emotion.lower()]
        return random.sample(songs, min(num_songs, len(songs)))
    return SONG_RECOMMENDATIONS.get('neutral', [])[:num_songs]

def get_mood_description(emotion):
    """Get a friendly description of the detected mood"""
    descriptions = {
        'happy': "You're radiating positivity! Here are some upbeat songs to keep your energy high.",
        'sad': "Feeling a bit down? These songs might help you process your emotions or lift your spirits.",
        'angry': "Feeling intense? These powerful songs can help you channel that energy.",
        'fear': "Feeling anxious? These empowering songs can help you feel stronger and more confident.",
        'surprise': "What a moment! Here are some energetic songs to match your surprise.",
        'disgust': "Not feeling it? These empowering songs might help shift your mood.",
        'neutral': "In a calm state? Here are some peaceful songs for your contemplative mood."
    }
    return descriptions.get(emotion.lower(), "Here are some songs that might match your current vibe.")

# =========================
# HELPERS
# =========================

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Invalid image"

    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, "No face detected"

    results = []

    for (x, y, fw, fh) in faces:
        pad_w = int(fw * 0.2)
        pad_h = int(fh * 0.2)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + fw + pad_w)
        y2 = min(h, y + fh + pad_h)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            continue

        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_face, (64, 64))
        normalized = resized.astype(np.float32)

        input_tensor = normalized[np.newaxis, np.newaxis, :, :]
        scores = ORT_SESSION.run(None, {INPUT_NAME: input_tensor})[0][0]
        probs = softmax(scores)

        emotions = {}
        for i, score in enumerate(probs):
            key = EMOTION_TABLE[EMOTION_KEYS[i]]
            emotions[key] = max(emotions.get(key, 0), float(score))

        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]

        results.append({
            "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "emotions": emotions
        })

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{dominant}: {confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

    output_path = os.path.join(
        UPLOAD_FOLDER, f"processed_{os.path.basename(image_path)}"
    )
    cv2.imwrite(output_path, image)

    return results, output_path

# =========================
# ROUTES
# =========================

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
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    data, result = process_image(filepath)
    if data is None:
        return jsonify({"error": result}), 400

    # Get dominant emotion for song recommendations
    dominant_emotion = None
    mood_description = ""
    recommended_songs = []
    
    if data and len(data) > 0:
        first_face_emotions = data[0]['emotions']
        dominant_emotion = max(first_face_emotions, key=first_face_emotions.get)
        mood_description = get_mood_description(dominant_emotion)
        recommended_songs = get_song_recommendations(dominant_emotion)

    return jsonify({
        "success": True,
        "original_image": f"/uploads/{filename}",
        "processed_image": f"/uploads/{os.path.basename(result)}",
        "emotions": data,
        "dominant_emotion": dominant_emotion,
        "mood_description": mood_description,
        "recommended_songs": recommended_songs,
    })


@app.route("/webcam", methods=["POST"])
def webcam():
    payload = request.get_json()
    if not payload or "image" not in payload:
        return jsonify({"error": "Invalid payload"}), 400

    image_data = payload["image"].split(",")[-1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    filename = f"webcam_{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    data, result = process_image(filepath)
    if data is None:
        return jsonify({"error": result}), 400

    # Get dominant emotion for song recommendations
    dominant_emotion = None
    mood_description = ""
    recommended_songs = []
    
    if data and len(data) > 0:
        first_face_emotions = data[0]['emotions']
        dominant_emotion = max(first_face_emotions, key=first_face_emotions.get)
        mood_description = get_mood_description(dominant_emotion)
        recommended_songs = get_song_recommendations(dominant_emotion)

    return jsonify({
        "success": True,
        "processed_image": f"/uploads/{os.path.basename(result)}",
        "emotions": data,
        "dominant_emotion": dominant_emotion,
        "mood_description": mood_description,
        "recommended_songs": recommended_songs,
    })


@app.route("/uploads/<filename>")
def uploads(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


# =========================
# LOCAL RUN
# =========================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
