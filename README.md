# Face Recognition API

A FastAPI-based service for face comparison and recognition.

## Features

- Compare faces between two images
- Calculate face similarity with confidence scores
- Detect multiple faces in images
- Secure API endpoints for face verification

## Requirements

- Python 3.6+
- FastAPI
- face_recognition
- numpy
- PIL
- uvicorn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Nabeel-Shehzad/face-recognition.git
cd face-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Additional installation steps:
```bash
pip install wheel setuptools pip --upgrade 
pip install git+https://github.com/ageitgey/face_recognition_models --verbose
```

## Usage

Run the FastAPI server:
```bash
python main_fastapi.py
```

The API will be available at http://localhost:8000

## API Endpoints

### POST /compare_faces
Compare a face in an uploaded image with a face from a URL.

Parameters:
- `image`: The image file to upload
- `id`: ID of the image to compare against (used to construct URL)

Returns:
- Success status
- Match result
- Distance between faces
- Confidence score
- Number of faces detected in each image

### GET /
Index route that returns a welcome message.
