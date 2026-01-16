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

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Loading the ONNX model
try:
    ORT_SESSION = ort.InferenceSession("emotion-ferplus-8.onnx")
    INPUT_NAME = ORT_SESSION.get_inputs()[0].name
except Exception as e:
    print(f"Error loading ONNX model: {e}")
    ORT_SESSION = None

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Emotion labels for FERPlus
EMOTION_TABLE = {
    'neutral': 'neutral',
    'happiness': 'happy',
    'surprise': 'surprise',
    'sadness': 'sad',
    'anger': 'angry',
    'disgust': 'disgust',
    'fear': 'fear',
    'contempt': 'disgust' # Map contempt which is not in standard simple list
}
EMOTION_KEYS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image_path):
    """Process image and detect emotions using ONNX FERPlus + MediaPipe"""
    if ORT_SESSION is None:
        return None, "Model not loaded properly."

    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Could not read image file"

        h, w, c = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces using MediaPipe
        results = face_detection.process(image_rgb)

        if not results.detections:
            return None, "No faces detected in the image"

        formatted_results = []

        for detection in results.detections:
            # MediaPipe returns relative bounding box (0-1)
            bboxC = detection.location_data.relative_bounding_box
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            # Ensure box is within image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(w - x, width)
            height = min(h - y, height)

            if width <= 0 or height <= 0:
                continue

            # Extract face ROI
            face_roi = image[y:y+height, x:x+width]

            if face_roi.size == 0:
                continue

            # Preprocess for FERPlus
            # 1. Grayscale
            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            # 2. Resize to 64x64
            resized_face = cv2.resize(gray_face, (64, 64))
            
            # 3. Normalize to [0, 1]
            processed_face = resized_face.astype(np.float32) / 255.0
            
            # Add batch and channel dimensions [1, 1, 64, 64]
            input_tensor = np.expand_dims(processed_face, axis=0)
            input_tensor = np.expand_dims(input_tensor, axis=0)

            # Run inference
            ort_inputs = {INPUT_NAME: input_tensor}
            ort_outs = ORT_SESSION.run(None, ort_inputs)
            scores = ort_outs[0][0] # First batch
            
            # Softmax
            probs = softmax(scores)
            
            # Map to emotion dict
            emotions_dict = {}
            for i, score in enumerate(probs):
                original_key = EMOTION_KEYS[i]
                mapped_key = EMOTION_TABLE.get(original_key, original_key)
                if mapped_key in emotions_dict:
                     emotions_dict[mapped_key] = float(max(emotions_dict[mapped_key], score))
                else:
                     emotions_dict[mapped_key] = float(score)

            # Find dominant
            dominant_emotion = max(emotions_dict, key=emotions_dict.get)
            confidence = emotions_dict[dominant_emotion]

            # Append to results
            formatted_face = {
                "box": [x, y, width, height],
                "emotions": emotions_dict # Values are 0-1
            }
            formatted_results.append(formatted_face)

            # Draw on image
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
            label = f"{dominant_emotion}: {confidence:.2f}"
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

        # Save processed image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
        cv2.imwrite(output_path, image)

        return formatted_results, output_path

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
            # If processed_path contains error message
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


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)