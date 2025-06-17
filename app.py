from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import cv2
import numpy as np
from fer import FER
import base64
import io
from PIL import Image
import json

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


def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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

        # Prepare response data
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

        response_data = {
            'success': True,
            'processed_image': f'/uploads/{os.path.basename(processed_path)}',
            'emotions': emotions_data
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)