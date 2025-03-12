import base64
import numpy as np
import cv2
import face_recognition
from datetime import datetime
from database import supabase_client
from models import DetectedFace, FaceLocation
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_face_embeddings():
    """Fetch stored face embeddings from Supabase"""
    try:
        response = supabase_client.from_("people").select("id, name, face_embedding, department").execute()
        
        if hasattr(response, "error") and response.error:
            logger.error(f"Supabase error: {response.error}")
            raise Exception(f"Supabase error: {response.error}")
        
        known_face_encodings = []
        known_face_names = []
        known_face_ids = []
        known_departments = []
        
        for person in response.data:
            if person.get("face_embedding"):
                try:
                    face_embedding = np.frombuffer(base64.b64decode(person["face_embedding"]), dtype=np.float64)
                    known_face_encodings.append(face_embedding)
                    known_face_names.append(person["name"])
                    known_face_ids.append(person["id"])
                    known_departments.append(person.get("department", ""))
                except Exception as e:
                    logger.error(f"Error decoding face embedding for {person['name']}: {str(e)}")
        
        logger.info(f"Loaded {len(known_face_encodings)} face embeddings from database")
        return known_face_encodings, known_face_names, known_face_ids, known_departments
    except Exception as e:
        logger.error(f"Error in get_face_embeddings: {str(e)}")
        raise e

def decode_image(image_data):
    """Decode base64 image data to OpenCV image"""
    try:
        if isinstance(image_data, str) and image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        
        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Error decoding image: {str(e)}")
        return None

def mark_attendance(person_id, confidence, name):
    """Mark attendance for a person"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        current_hour = datetime.now().hour
        
        status = "present" if current_hour < 9 else "late"
        
        # Check if attendance already exists
        check_response = supabase_client.table("attendance") \
            .select("id") \
            .eq("person_id", person_id) \
            .eq("date", today) \
            .execute()
        
        if not check_response.data:
            attendance_data = {
                "person_id": person_id,
                "date": today,
                "time": current_time,
                "status": status,
                "confidence": float(confidence),
                "marked_by": "system"
            }
            
            response = supabase_client.table("attendance").insert(attendance_data).execute()
            logger.info(f"Attendance response: {response.data}")
            
            if response.data:
                logger.info(f"Marked attendance for {name} as {status}")
                return True
            else:
                logger.error("Failed to insert attendance record")
                return False
        else:
            logger.info(f"Attendance already marked for {name} today")
            return True
    except Exception as e:
        logger.error(f"Error marking attendance: {str(e)}")
        return False

def process_face_encoding(face_encoding, known_face_encodings, known_face_names, known_face_ids, known_departments, tolerance, class_filter, face_location):
    """Process a single face encoding"""
    try:
        name = "Unknown"
        person_id = None
        confidence = 0.0
        department = ""
        attendance_marked = False
        top, right, bottom, left = face_location
        
        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if face_distances[best_match_index] <= tolerance:
                    confidence = 1 - face_distances[best_match_index]
                    name = known_face_names[best_match_index]
                    person_id = known_face_ids[best_match_index]
                    department = known_departments[best_match_index]
                    
                    if class_filter and department != class_filter:
                        logger.info(f"Skipping {name} - not in {class_filter} class")
                        return None
                    
                    if person_id:
                        attendance_marked = mark_attendance(person_id, confidence, name)
        
        return DetectedFace(
            name=name,
            confidence=float(confidence),
            location=FaceLocation(top=top, right=right, bottom=bottom, left=left),
            attendance_marked=attendance_marked
        )
    except Exception as e:
        logger.error(f"Error processing face encoding: {str(e)}")
        return None

def process_image(image_data, tolerance=0.6, model="hog", class_filter="AIML-A"):
    """Process image for face detection and recognition"""
    try:
        known_face_encodings, known_face_names, known_face_ids, known_departments = get_face_embeddings()
        
        image = decode_image(image_data)
        if image is None:
            logger.error("Failed to decode image")
            raise ValueError("Failed to decode image")
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_image, model=model)
        if not face_locations:
            logger.info("No faces detected in the image")
            return []
            
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            result = process_face_encoding(face_encoding, known_face_encodings, known_face_names, known_face_ids, known_departments, tolerance, class_filter, (top, right, bottom, left))
            if result:
                results.append(result)
        
        return results
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        raise 