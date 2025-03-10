from fastapi import FastAPI, HTTPException, Body
from database import supabase_client
from models import Attendance, Person, Profile, RegisterFaceRequest, DetectionResponse
from typing import List
from datetime import datetime
import base64
import numpy as np
import cv2
import face_recognition
from face_recognition_utils import process_image  # Ensure this module exists
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API is running!"}

@app.post("/detect-faces", response_model=DetectionResponse)
async def detect_faces(image_data: str = Body(..., embed=True)):
    """Detect and recognize faces in an image"""
    try:
        results = process_image(image_data)
        return DetectionResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/register-face")
async def register_face(request: RegisterFaceRequest):
    """Register a new face in the system"""
    try:
        # Clean up base64 data
        image_data = request.image_data
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
            
        # Decode image
        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            raise HTTPException(status_code=400, detail="No face detected in image")
            
        face_encoding = face_recognition.face_encodings(rgb_image, [face_locations[0]])[0]
        
        # Convert face encoding to base64
        encoded_face = base64.b64encode(face_encoding.tobytes()).decode('utf-8')
        
        # Prepare person data
        person_data = {
            "name": request.name,
            "employee_id": request.employee_id,
            "department": request.department,
            "position": request.position,
            "face_embedding": encoded_face,
            "active": True
        }
        
        # Insert into Supabase
        response = supabase_client.from_("people").insert(person_data).execute()
        
        if hasattr(response, "error") and response.error:
            raise HTTPException(status_code=500, detail=f"Database Error: {response.error}")
            
        return {"message": "Face registered successfully"}
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error registering face: {str(e)}")

@app.get("/attendance/", response_model=List[Attendance])
async def get_attendance():
    """Fetch attendance records from Supabase"""
    try:
        # Query the base table with joins instead of the view
        response = supabase_client.from_('attendance')\
            .select('*, people!inner(*)')\
            .order('date.desc,time.asc')\
            .execute()
        
        if hasattr(response, "error") and response.error:
            raise HTTPException(status_code=500, detail=f"Supabase Error: {response.error}")

        attendance_records = [
            Attendance(
                **{
                    **record,
                    'date': datetime.strptime(record['date'], '%Y-%m-%d').date() if 'date' in record else None,
                    'time': datetime.strptime(record['time'], '%H:%M:%S').time() if 'time' in record else None,
                    'created_at': datetime.fromisoformat(record['created_at'].replace('Z', '+00:00')) if 'created_at' in record else None
                }
            ) 
            for record in response.data
        ]
        
        return attendance_records

    except Exception as e:
        print(f"Debug - Error details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching attendance: {str(e)}")
