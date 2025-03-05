from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
import numpy as np
import os
import shutil
from typing import Optional
import uvicorn
from pydantic import BaseModel
import requests
from io import BytesIO
from PIL import Image

app = FastAPI(title="Face Comparison API")

IMAGES_URL = 'https://zfhkmedpzkfnhxoefnql.supabase.co/storage/v1/object/public/face_images//'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename: str) -> bool:
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_face_from_path(image_path: str):
    """Detect face from an image file path using DeepFace"""
    try:
        # DeepFace will automatically detect faces in the image
        face_detection = DeepFace.detectFace(image_path)
        return True if face_detection is not None else False
    except Exception as e:
        print(f"Error detecting face: {e}")
        return False

class ResponseModel(BaseModel):
    success: bool
    message: Optional[str] = None
    match: Optional[bool] = None
    distance: Optional[float] = None
    tolerance: Optional[float] = None
    confidence: Optional[float] = None
    verification_match: Optional[bool] = None
    faces_detected: Optional[dict] = None
    filename: Optional[str] = None

@app.post("/compare_faces", response_model=ResponseModel)
async def compare_faces(
    image: UploadFile = File(...),
    id: str = Form(...)
):

    """Compare uploaded faces with enhanced accuracy"""
    try:
        teacher_image = IMAGES_URL + id

        # Validate file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No selected file")
        
        if not allowed_file(image.filename):
            raise HTTPException(status_code=400, detail="File type not allowed")
        
        # Save the uploaded file
        # Download the image from the provided URL
        response = requests.get(teacher_image)

        if not response.ok:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # save the image from response using the BytesIO
        teacher_image = Image.open(BytesIO(response.content))
        teacher_image_path = os.path.join(UPLOAD_FOLDER, "teacher_image.jpg")
        teacher_image.save(teacher_image_path)

        # save request image as request_image.jpg
        filename = os.path.basename(image.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"request_{filename}")
        
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Detect face
        face_detected = detect_face_from_path(filepath)
        
        if not face_detected:
            os.remove(filepath)
            raise HTTPException(status_code=400, detail="No face detected in uploaded image")

        # Check if both images exist
        request_image_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith('request_')]
        
        if not request_image_files or not teacher_image:
            raise HTTPException(status_code=400, detail="Both images must be uploaded first")
        
        request_image_path = os.path.join(UPLOAD_FOLDER, request_image_files[0])
        
        # Compare faces using DeepFace
        try:
            # DeepFace.verify returns a dictionary with verification result
            result = DeepFace.verify(teacher_image_path, request_image_path, 
                                     model_name='VGG-Face', # You can choose different models: 'VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib', 'ArcFace'
                                     distance_metric='cosine') # You can choose different metrics: 'cosine', 'euclidean', 'euclidean_l2'
            
            # Extract results
            is_match = result['verified']
            face_distance = result['distance']
            verification_match = is_match
            
            # Use a threshold for better accuracy
            tolerance = 0.4
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error comparing faces: {str(e)}")
        
        # Confidence score (0-100%): higher means more confident that faces are the same
        # Convert distance to confidence score (inverse relationship)
        # For cosine distance, lower is better (0 is perfect match)
        confidence = max(0, min(100, (1 - face_distance) * 100))

        # Delete the uploaded image
        os.remove(request_image_path)
        os.remove(teacher_image_path)
        
        return {
            'success': True,
            'match': bool(is_match),
            'distance': float(face_distance),
            'tolerance': tolerance,
            'confidence': round(confidence, 2),  # Confidence as percentage
            'verification_match': bool(verification_match),
            'faces_detected': {
                'first_image': 1,  # DeepFace.verify only works with one face per image
                'second_image': 1
            }
        }

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error in compare_faces: {e}")
        raise HTTPException(status_code=500, detail=f"Error comparing faces: {str(e)}")

@app.get("/", response_model=ResponseModel)
async def index():
    """Index route"""
    return {"success": True, "message": "Welcome to the Face Comparison API"}

if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)