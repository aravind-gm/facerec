from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from database import supabase_client
from models import (
    AttendanceResponse, 
    Person, 
    RegisterFaceRequest, 
    DetectionResponse,
    Attendance  # Add this import
)
from typing import List, Optional
from datetime import datetime, date
import base64
import numpy as np
import cv2
import face_recognition
from face_recognition_utils import process_image
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Jain University AIML-A Attendance System")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",  # If your frontend is served from port 8000
        "http://localhost:8000",
        "http://127.0.0.1:8080",  # If your frontend is served from port 8080 (like uvicorn's default)
        "http://localhost:8080",  # Or any other ports your frontend might use
        "http://localhost:5173"  # Common Vite port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "message": "Jain University AIML-A Attendance System API is running!"}

@app.post("/api/detect-faces", response_model=DetectionResponse)
async def detect_faces(image_data: str = Body(..., embed=True)):
    """Detect and recognize faces in an image"""
    try:
        results = process_image(image_data)
        return DetectionResponse(results=results)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Table name verification constants
PEOPLE_TABLE = "people"
ATTENDANCE_TABLE = "attendance"

# Update table references in the register_face endpoint
@app.post("/api/register-face")
async def register_face(request: RegisterFaceRequest):
    try:
        logger.info(f"Starting registration for {request.name}")
        
        # Validate input data
        if not request.image_data:
            raise HTTPException(status_code=400, detail="No image data provided")
            
        if not request.name or not request.employee_id:
            raise HTTPException(status_code=400, detail="Name and employee ID are required")
        
        # Create temp directory if it doesn't exist
        image_path = f"temp/{request.name}_{uuid.uuid4()}.jpg"
        os.makedirs("temp", exist_ok=True)
        
        try:
            # Clean up base64 data
            image_data = request.image_data
            if "base64," in image_data:
                image_data = image_data.split("base64,")[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            
            # Save image temporarily
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            # Load and process image
            image = cv2.imread(image_path)
            if image is None:
                raise HTTPException(status_code=400, detail="Could not read image data")
            
            # Convert to RGB for face_recognition
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_image)
            if not face_locations:
                raise HTTPException(status_code=400, detail="No face detected in image")
            
            # Get face encoding
            face_encoding = face_recognition.face_encodings(rgb_image, [face_locations[0]])[0]
            encoded_face = base64.b64encode(face_encoding.tobytes()).decode('utf-8')
            
            # Prepare person data
            person_data = {
                "name": request.name,
                "employee_id": request.employee_id,
                "department": request.department or "AIML-A",
                "position": request.position or "Student",
                "face_embedding": encoded_face,
                "active": True
            }
            
            # Insert into database
            response = supabase_client.table(PEOPLE_TABLE).insert(person_data).execute()
            
            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to insert data into database")
            
            logger.info(f"Successfully registered {request.name}")
            return {"message": "Face registered successfully", "name": request.name}
            
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Error processing registration: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
        finally:
            # Cleanup temporary file
            if os.path.exists(image_path):
                os.remove(image_path)
                logger.info(f"Cleaned up temporary file: {image_path}")
                
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in registration process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/attendance/", response_model=List[Attendance])
async def get_attendance():
    """Fetch today's attendance records from Supabase"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Fetching attendance for today: {today}")
        
        # Build the query for today only
        query = supabase_client.table("attendance")\
            .select("*, people(*)")\
            .eq('date', today)\
            .order('created_at', desc=True)
            
        response = query.execute()
        
        if not response.data:
            logger.info("No attendance records found for today")
            return []
            
        logger.info(f"Found {len(response.data)} attendance records for today")
        logger.debug(f"Response data: {response.data}")
        
        # Format the response data
        attendance_records = []
        for record in response.data:
            try:
                person_data = record.get('people', {})
                attendance_record = {
                    "id": record.get("id"),
                    "person_id": record.get("person_id"),
                    "date": record.get("date"),
                    "time": record.get("time"),
                    "status": record.get("status"),
                    "confidence": record.get("confidence"),
                    "marked_by": record.get("marked_by", "system"),
                    "notes": record.get("notes"),
                    "created_at": record.get("created_at"),
                    "person": Person(
                        id=person_data.get("id"),
                        name=person_data.get("name"),
                        employee_id=person_data.get("employee_id"),
                        department=person_data.get("department"),
                        position=person_data.get("position")
                    ) if person_data else None
                }
                attendance_records.append(Attendance(**attendance_record))
            except Exception as e:
                logger.error(f"Error processing attendance record: {str(e)}")
                continue

        return attendance_records

    except Exception as e:
        logger.error(f"Error fetching attendance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching attendance: {str(e)}")

@app.get("/api/students/")
async def get_students():
    """Fetch all students from the people table"""
    try:
        response = supabase_client.from_("people")\
            .select("id, name, employee_id, department, position, active")\
            .eq("department", "AIML-A")\
            .execute()
            
        if hasattr(response, "error") and response.error:
            raise HTTPException(status_code=500, detail=f"Supabase Error: {response.error}")
            
        return response.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching students: {str(e)}")

@app.post("/api/attendance/manual")
async def mark_manual_attendance(
    student_id: str = Body(...),
    status: str = Body(...),
    notes: Optional[str] = Body(None)
):
    """Manually mark attendance for a student"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Check if attendance already marked today
        check_response = supabase_client.table("attendance") \
            .select("id") \
            .eq("person_id", student_id) \
            .eq("date", today) \
            .execute()
        
        if check_response.data:
            # Update existing record
            attendance_data = {
                "status": status,
                "time": current_time,
                "marked_by": "manual",
                "notes": notes
            }
            
            response = supabase_client.table("attendance")\
                .update(attendance_data)\
                .eq("id", check_response.data[0]["id"])\
                .execute()
                
            return {"message": "Attendance updated successfully"}
        else:
            # Create new attendance record
            attendance_data = {
                "person_id": student_id,
                "date": today,
                "time": current_time,
                "status": status,
                "marked_by": "manual",
                "notes": notes
            }
            
            response = supabase_client.table("attendance").insert(attendance_data).execute()
            
            return {"message": "Attendance marked successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error marking attendance: {str(e)}")

@app.get("/api/attendance/stats")
async def get_attendance_stats(start_date: Optional[str] = None, end_date: Optional[str] = None):
    """Get attendance statistics for the class"""
    try:
        # Compute date range
        if not start_date:
            start_date = (datetime.now().replace(day=1)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"Calculating attendance stats for period: {start_date} to {end_date}")
        
        # Get all students
        students_response = supabase_client.from_("people")\
            .select("id, name")\
            .eq("department", "AIML-A")\
            .eq("active", True)\
            .execute()
            
        if not students_response.data:
            logger.warning("No active students found in AIML-A department")
            return {"total_students": 0, "daily_stats": [], "student_stats": []}
            
        # Get attendance records in date range
        attendance_response = supabase_client.from_("attendance")\
            .select("person_id, date, status")\
            .gte("date", start_date)\
            .lte("date", end_date)\
            .execute()
            
        # Calculate stats
        total_students = len(students_response.data)
        attendance_by_date = {}
        student_stats = {student["id"]: {"name": student["name"], "present": 0, "absent": 0, "late": 0} 
                        for student in students_response.data}
        
        # Initialize stats dictionary
        stats = {
            "total_students": total_students,
            "date_range": {
                "start": start_date,
                "end": end_date
            },
            "daily_stats": [],
            "student_stats": []
        }
        
        # Process attendance data with error handling
        for record in attendance_response.data:
            try:
                date_str = record["date"]
                if date_str not in attendance_by_date:
                    attendance_by_date[date_str] = {
                        "present": 0,
                        "absent": 0,
                        "late": 0
                    }
                
                status = record.get("status", "absent")
                person_id = record.get("person_id")
                
                if status not in ["present", "absent", "late"]:
                    logger.warning(f"Invalid status '{status}' found for date {date_str}")
                    status = "absent"
                    
                attendance_by_date[date_str][status] += 1
                student_stats[person_id][status] += 1
            except KeyError as ke:
                logger.error(f"Missing key in attendance record: {ke}")
                continue
            except Exception as e:
                logger.error(f"Error processing attendance record: {e}")
                continue
        
        # Prepare daily stats
        for date_str, counts in attendance_by_date.items():
            stats["daily_stats"].append({
                "date": date_str,
                "present": counts["present"],
                "absent": counts["absent"],
                "late": counts["late"]
            })
        
        # Prepare student stats
        for student_id, counts in student_stats.items():
            stats["student_stats"].append({
                "student_id": student_id,
                "name": counts["name"],
                "present": counts["present"],
                "absent": counts["absent"],
                "late": counts["late"]
            })
        
        logger.info(f"Processed attendance records for {len(attendance_by_date)} days")
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating attendance stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating attendance stats: {str(e)}")