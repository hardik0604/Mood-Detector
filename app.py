from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
from fer import FER
import base64
import io
from PIL import Image
import json
import random

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize FER detector
detector = FER(mtcnn=True)

# Song recommendations database organized by emotion
SONG_RECOMMENDATIONS = {
    'happy': [
        {'title': 'Happy', 'artist': 'Pharrell Williams', 'genre': 'Pop'},
        {'title': 'Good as Hell', 'artist': 'Lizzo', 'genre': 'Pop/R&B'},
        {'title': 'Can\'t Stop the Feeling!', 'artist': 'Justin Timberlake', 'genre': 'Pop'},
        {'title': 'Walking on Sunshine', 'artist': 'Katrina and the Waves', 'genre': 'Rock'},
        {'title': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars', 'genre': 'Funk/Pop'},
        {'title': 'I Gotta Feeling', 'artist': 'The Black Eyed Peas', 'genre': 'Pop'},
        {'title': 'Don\'t Worry Be Happy', 'artist': 'Bobby McFerrin', 'genre': 'Reggae'},
        {'title': 'September', 'artist': 'Earth, Wind & Fire', 'genre': 'Funk/Soul'},
    ],
    'sad': [
        {'title': 'Someone Like You', 'artist': 'Adele', 'genre': 'Pop/Soul'},
        {'title': 'Mad World', 'artist': 'Gary Jules', 'genre': 'Alternative'},
        {'title': 'Hurt', 'artist': 'Johnny Cash', 'genre': 'Country'},
        {'title': 'The Sound of Silence', 'artist': 'Simon & Garfunkel', 'genre': 'Folk'},
        {'title': 'Everybody Hurts', 'artist': 'R.E.M.', 'genre': 'Alternative Rock'},
        {'title': 'Black', 'artist': 'Pearl Jam', 'genre': 'Grunge'},
        {'title': 'Tears in Heaven', 'artist': 'Eric Clapton', 'genre': 'Blues/Rock'},
        {'title': 'Hallelujah', 'artist': 'Jeff Buckley', 'genre': 'Alternative Rock'},
    ],
    'angry': [
        {'title': 'Break Stuff', 'artist': 'Limp Bizkit', 'genre': 'Nu Metal'},
        {'title': 'Killing in the Name', 'artist': 'Rage Against the Machine', 'genre': 'Alternative Metal'},
        {'title': 'Bodies', 'artist': 'Drowning Pool', 'genre': 'Nu Metal'},
        {'title': 'Chop Suey!', 'artist': 'System of a Down', 'genre': 'Alternative Metal'},
        {'title': 'Boulevard of Broken Dreams', 'artist': 'Green Day', 'genre': 'Punk Rock'},
        {'title': 'Breathe Me', 'artist': 'Sia', 'genre': 'Pop'},
        {'title': 'Numb', 'artist': 'Linkin Park', 'genre': 'Nu Metal'},
        {'title': 'Stronger', 'artist': 'Kelly Clarkson', 'genre': 'Pop Rock'},
    ],
    'fear': [
        {'title': 'Breathe Me', 'artist': 'Sia', 'genre': 'Pop'},
        {'title': 'Heavy', 'artist': 'Linkin Park ft. Kiiara', 'genre': 'Alternative'},
        {'title': 'Anxiety', 'artist': 'Julia Michaels', 'genre': 'Pop'},
        {'title': 'Scars to Your Beautiful', 'artist': 'Alessia Cara', 'genre': 'Pop'},
        {'title': 'Brave', 'artist': 'Sara Bareilles', 'genre': 'Pop'},
        {'title': 'Stronger (What Doesn\'t Kill You)', 'artist': 'Kelly Clarkson', 'genre': 'Pop'},
        {'title': 'Fight Song', 'artist': 'Rachel Platten', 'genre': 'Pop'},
        {'title': 'Titanium', 'artist': 'David Guetta ft. Sia', 'genre': 'Electronic/Pop'},
    ],
    'surprise': [
        {'title': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars', 'genre': 'Funk/Pop'},
        {'title': 'Levitating', 'artist': 'Dua Lipa', 'genre': 'Pop'},
        {'title': 'Blinding Lights', 'artist': 'The Weeknd', 'genre': 'Synth-pop'},
        {'title': 'Good 4 U', 'artist': 'Olivia Rodrigo', 'genre': 'Pop Rock'},
        {'title': 'Shake It Off', 'artist': 'Taylor Swift', 'genre': 'Pop'},
        {'title': 'Can\'t Stop the Feeling!', 'artist': 'Justin Timberlake', 'genre': 'Pop'},
        {'title': 'Electric Feel', 'artist': 'MGMT', 'genre': 'Indie Pop'},
        {'title': 'Mr. Blue Sky', 'artist': 'Electric Light Orchestra', 'genre': 'Rock'},
    ],
    'disgust': [
        {'title': 'Confident', 'artist': 'Demi Lovato', 'genre': 'Pop'},
        {'title': 'Stronger', 'artist': 'Kelly Clarkson', 'genre': 'Pop Rock'},
        {'title': 'Roar', 'artist': 'Katy Perry', 'genre': 'Pop'},
        {'title': 'Fight Song', 'artist': 'Rachel Platten', 'genre': 'Pop'},
        {'title': 'Stronger (What Doesn\'t Kill You)', 'artist': 'Kelly Clarkson', 'genre': 'Pop'},
        {'title': 'Titanium', 'artist': 'David Guetta ft. Sia', 'genre': 'Electronic/Pop'},
        {'title': 'Rise Up', 'artist': 'Andra Day', 'genre': 'Soul'},
        {'title': 'Unstoppable', 'artist': 'Sia', 'genre': 'Pop'},
    ],
    'neutral': [
        {'title': 'Weightless', 'artist': 'Marconi Union', 'genre': 'Ambient'},
        {'title': 'Claire de Lune', 'artist': 'Claude Debussy', 'genre': 'Classical'},
        {'title': 'Gymnopédie No. 1', 'artist': 'Erik Satie', 'genre': 'Classical'},
        {'title': 'Mindfulness Meditation', 'artist': 'Various Artists', 'genre': 'Ambient'},
        {'title': 'River', 'artist': 'Joni Mitchell', 'genre': 'Folk'},
        {'title': 'The Night We Met', 'artist': 'Lord Huron', 'genre': 'Indie Folk'},
        {'title': 'Holocene', 'artist': 'Bon Iver', 'genre': 'Indie Folk'},
        {'title': 'Mad About You', 'artist': 'Sting', 'genre': 'Pop/Rock'},
    ]
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_song_recommendations(emotion, num_songs=5):
    """Get song recommendations based on detected emotion"""
    if emotion.lower() in SONG_RECOMMENDATIONS:
        songs = SONG_RECOMMENDATIONS[emotion.lower()]
        # Return random selection of songs
        return random.sample(songs, min(num_songs, len(songs)))
    else:
        # Default to neutral if emotion not found
        songs = SONG_RECOMMENDATIONS['neutral']
        return random.sample(songs, min(num_songs, len(songs)))


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


def process_image(image_path):
    """Process image and detect emotions"""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not read image file"

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect emotions
        result = detector.detect_emotions(image_rgb)

        if not result:
            return None, "No faces detected in the image"

        # Draw bounding boxes and labels
        for face in result:
            (x, y, w, h) = face["box"]
            emotions = face["emotions"]

            # Find dominant emotion
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion]

            # Draw rectangle around face
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add emotion label
            label = f"{dominant_emotion}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        # Save processed image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
        cv2.imwrite(output_path, image)

        return result, output_path

    except Exception as e:
        return None, f"Error processing image: {str(e)}"


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and emotion detection"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Process image
        emotions_data, processed_path = process_image(filepath)

        if emotions_data is None:
            return jsonify({'error': processed_path}), 400

        # Get dominant emotion from first detected face
        if emotions_data:
            dominant_emotion = max(emotions_data[0]["emotions"], key=emotions_data[0]["emotions"].get)

            # Get song recommendations
            recommended_songs = get_song_recommendations(dominant_emotion)
            mood_description = get_mood_description(dominant_emotion)

            # Prepare response data
            response_data = {
                'success': True,
                'original_image': f'/uploads/{filename}',
                'processed_image': f'/uploads/{os.path.basename(processed_path)}',
                'emotions': emotions_data,
                'dominant_emotion': dominant_emotion,
                'mood_description': mood_description,
                'recommended_songs': recommended_songs
            }
        else:
            response_data = {
                'success': True,
                'original_image': f'/uploads/{filename}',
                'processed_image': f'/uploads/{os.path.basename(processed_path)}',
                'emotions': emotions_data
            }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/webcam', methods=['POST'])
def webcam_capture():
    """Handle webcam image capture"""
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64, prefix

        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Save image
        filename = 'webcam_capture.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(filepath)

        # Process image
        emotions_data, processed_path = process_image(filepath)

        if emotions_data is None:
            return jsonify({'error': processed_path}), 400

        # Get dominant emotion from first detected face
        if emotions_data:
            dominant_emotion = max(emotions_data[0]["emotions"], key=emotions_data[0]["emotions"].get)

            # Get song recommendations
            recommended_songs = get_song_recommendations(dominant_emotion)
            mood_description = get_mood_description(dominant_emotion)

            response_data = {
                'success': True,
                'processed_image': f'/uploads/{os.path.basename(processed_path)}',
                'emotions': emotions_data,
                'dominant_emotion': dominant_emotion,
                'mood_description': mood_description,
                'recommended_songs': recommended_songs
            }
        else:
            response_data = {
                'success': True,
                'processed_image': f'/uploads/{os.path.basename(processed_path)}',
                'emotions': emotions_data
            }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/get-songs/<emotion>')
def get_songs_by_emotion(emotion):
    """API endpoint to get songs for a specific emotion"""
    try:
        songs = get_song_recommendations(emotion, num_songs=8)
        return jsonify({
            'emotion': emotion,
            'songs': songs,
            'description': get_mood_description(emotion)
        })
    except Exception as e:
        return jsonify({'error': f'Error getting songs: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)