from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import random

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = load_model('model.h5')

# Load emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the song database
song_database = {
    "Happy": [
        "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
        "https://www.youtube.com/watch?v=y6Sxv-sUYtM"
    ],
    "Sad": [
        "https://www.youtube.com/watch?v=4N3N1MlvVc4",
        "https://www.youtube.com/watch?v=ho9rZjlsyYY"
    ],
    "Angry": [
        "https://www.youtube.com/watch?v=YlUKcNNmywk",
        "https://www.youtube.com/watch?v=rYEDA3JcQqw"
    ],
    "Surprise": [
        "https://www.youtube.com/watch?v=0KSOMA3QBU0",
        "https://www.youtube.com/watch?v=3tmd-ClpJxA"
    ],
    "Neutral": [
        "https://www.youtube.com/watch?v=kXYiU_JCYtU",
        "https://www.youtube.com/watch?v=fJ9rUzIMcZQ"
    ],
    "Fear": [
        "https://www.youtube.com/watch?v=RB-RcX5DS5A",
        "https://www.youtube.com/watch?v=QH2-TGUlwu4"
    ],
    "Disgust": [
        "https://www.youtube.com/watch?v=2vjPBrBU-TM",
        "https://www.youtube.com/watch?v=lp-EO5I60KA"
    ]
}

def get_song_recommendations(emotion, num_songs=5):
    """Return a list of song URLs based on detected emotion"""
    return random.sample(song_database.get(emotion, []), min(num_songs, len(song_database.get(emotion, []))))

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict emotion from uploaded image"""
    try:
        file = request.files['image']
        file_path = 'temp.jpg'
        file.save(file_path)

        # Read and preprocess the image
        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return jsonify({'error': 'No face detected'})

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(face, axis=0)
            face = np.expand_dims(face, axis=-1)

            prediction = model.predict(face)
            emotion_index = np.argmax(prediction)
            emotion = emotion_labels[emotion_index]
            break

        os.remove(file_path)
        return jsonify({'emotion': emotion})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get-songs/<emotion>')
def get_songs_by_emotion(emotion):
    """API endpoint to get songs for a specific emotion"""
    try:
        songs = get_song_recommendations(emotion, num_songs=8)
        return jsonify({'songs': songs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
