from flask import Flask, request, render_template, jsonify, send_from_directory
import os, cv2, uuid, base64, io
import numpy as np
import onnxruntime as ort
from PIL import Image

app = Flask(__name__)

# Prevent browser caching
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

cv2.setNumThreads(1)

# ================= FACE DETECTOR =================
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if FACE_CASCADE.empty():
    raise RuntimeError("Face detector failed")

# ================= ONNX MODEL =================
MODEL_PATH = os.path.join(BASE_DIR, "emotion-ferplus-8.onnx")

opts = ort.SessionOptions()
opts.intra_op_num_threads = 1
opts.inter_op_num_threads = 1

ORT_SESSION = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
    sess_options=opts
)

INPUT_NAME = ORT_SESSION.get_inputs()[0].name

# ================= EMOTIONS =================
EMOTION_KEYS = [
    "neutral","happiness","surprise",
    "sadness","anger","disgust",
    "fear","contempt"
]

EMOTION_MAP = {
    "neutral":"neutral",
    "happiness":"happy",
    "surprise":"surprise",
    "sadness":"sad",
    "anger":"angry",
    "disgust":"disgust",
    "fear":"fear",
    "contempt":"disgust"
}

# ================= MUSIC =================
SONG_RECOMMENDATIONS = {
    "happy": [
        {"title": "Happy", "artist": "Pharrell Williams"},
        {"title": "Can't Stop the Feeling!", "artist": "Justin Timberlake"}
    ],
    "sad": [
        {"title": "Someone Like You", "artist": "Adele"},
        {"title": "Fix You", "artist": "Coldplay"}
    ],
    "angry": [
        {"title": "Numb", "artist": "Linkin Park"},
        {"title": "Break Stuff", "artist": "Limp Bizkit"}
    ],
    "neutral": [
        {"title": "Weightless", "artist": "Marconi Union"},
        {"title": "Holocene", "artist": "Bon Iver"}
    ],
    "fear": [
        {"title": "Brave", "artist": "Sara Bareilles"}
    ],
    "surprise": [
        {"title": "Blinding Lights", "artist": "The Weeknd"}
    ],
    "disgust": [
        {"title": "Confident", "artist": "Demi Lovato"}
    ]
}

# ================= HELPERS =================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def confidence_level(v):
    return "high" if v >= 0.85 else "medium" if v >= 0.65 else "low"

def mood_description(e):
    return {
        "happy":"You’re showing strong positive facial cues.",
        "sad":"Your expression suggests low mood or fatigue.",
        "angry":"Your expression shows tension or frustration.",
        "fear":"Your expression suggests uncertainty or concern.",
        "surprise":"Your expression reflects heightened alertness.",
        "disgust":"Your expression suggests discomfort.",
        "neutral":"Your expression appears calm and neutral."
    }.get(e,"")

# ================= IMAGE PROCESSING =================
def process_image(path):
    img = cv2.imread(path)
    if img is None:
        return None, "Invalid image"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray,1.1,5)

    if len(faces) == 0:
        return None, "No face detected"

    results = []

    for (x,y,w,h) in faces:
        face = gray[y:y+h,x:x+w]
        face = cv2.resize(face,(64,64)).astype(np.float32)

        tensor = face[np.newaxis,np.newaxis,:,:]
        scores = ORT_SESSION.run(None,{INPUT_NAME:tensor})[0][0]
        probs = softmax(scores)

        emotions={}
        for i,p in enumerate(probs):
            k = EMOTION_MAP[EMOTION_KEYS[i]]
            emotions[k] = max(emotions.get(k,0),float(p))

        emotion = max(emotions,key=emotions.get)
        score = emotions[emotion]

        results.append({
            "emotion":emotion,
            "confidence":confidence_level(score)
        })

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    out = os.path.join(UPLOAD_FOLDER,"processed_"+os.path.basename(path))
    cv2.imwrite(out,img)

    return results,out

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload",methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error":"No file"}),400

    name = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_FOLDER,name)
    f.save(path)

    faces,out = process_image(path)
    if faces is None:
        return jsonify({"error":out}),400

    face = faces[0]
    e = face["emotion"]

    return jsonify({
        "image":f"/uploads/{os.path.basename(out)}",
        "insight":{
            "emotion":e,
            "confidence":face["confidence"],
            "description":mood_description(e)
        },
        "songs": SONG_RECOMMENDATIONS.get(e,[])
    })

@app.route("/webcam",methods=["POST"])
def webcam():
    data = request.get_json()
    img = Image.open(io.BytesIO(
        base64.b64decode(data["image"].split(",")[1])
    )).convert("RGB")

    name = f"webcam_{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_FOLDER,name)
    img.save(path)

    faces,out = process_image(path)
    if faces is None:
        return jsonify({"error":out}),400

    face = faces[0]
    e = face["emotion"]

    return jsonify({
        "image":f"/uploads/{os.path.basename(out)}",
        "insight":{
            "emotion":e,
            "confidence":face["confidence"],
            "description":mood_description(e)
        },
        "songs": SONG_RECOMMENDATIONS.get(e,[])
    })

@app.route("/uploads/<f>")
def uploads(f):
    return send_from_directory(UPLOAD_FOLDER,f)

@app.route("/favicon.ico")
def favicon():
    return "",204

if __name__=="__main__":
    app.run(debug=False)
