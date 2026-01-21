# MOOD.OS - AI Emotion Detection

Modern web application for real-time facial emotion detection using AI.

## Features

- 📸 Upload photos or use webcam for live capture
- 🎭 AI-powered emotion detection using FER+ ONNX model
- 🎵 Music recommendations based on detected mood
- 🎨 Beautiful, modern UI with smooth animations
- 📱 Fully responsive design

## Tech Stack

- **Backend**: Flask, ONNX Runtime, OpenCV
- **Frontend**: Vanilla JavaScript, Modern CSS
- **AI Model**: FER+ Emotion Recognition

## Deployment

This app is configured for deployment on Render.

### Deploy to Render

1. Push code to GitHub
2. Connect your GitHub repo to Render
3. Render will automatically detect `render.yaml` and deploy

### Environment

- Python 3.11
- Gunicorn WSGI server
- 1 worker, 1 thread (optimized for free tier)

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000`

## License

MIT
