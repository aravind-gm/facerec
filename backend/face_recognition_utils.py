import base64
import numpy as np
import cv2
import face_recognition
from datetime import datetime
from database import supabase_client
from models import DetectedFace, FaceLocation

def get_face_embeddings():
    """Fetch stored face embeddings from Supabase"""
    response = supabase_client.from_("people").select("id, name, face_embedding").execute()
    
    if hasattr(response, "error") and response.error:
        raise Exception(f"Supabase error: {response.error}")
    
    known_face_encodings = []
    known_face_names = []
    known_face_ids = []
    
    for person in response.data:
        if person.get("face_embedding"):
            face_embedding = np.frombuffer(base64.b64decode(person["face_embedding"]), dtype=np.float64)
            known_face_encodings.append(face_embedding)
            known_face_names.append(person["name"])
            known_face_ids.append(person["id"])
    
    return known_face_encodings, known_face_names, known_face_ids

def process_image(image_data, tolerance=0.6, model="hog"):
    """Process image for face detection and recognition"""
    try:
        known_face_encodings, known_face_names, known_face_ids = get_face_embeddings()
        
        # Clean up base64 data
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        
        # Convert image
        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
            
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image, model=model)
        if not face_locations:
            return []
            
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        results = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            person_id = None
            confidence = 0.0
            attendance_marked = False
            
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if face_distances[best_match_index] <= tolerance:
                    confidence = 1 - face_distances[best_match_index]
                    name = known_face_names[best_match_index]
                    person_id = known_face_ids[best_match_index]
                    
                    if person_id:
                        today = datetime.now().strftime("%Y-%m-%d")
                        current_time = datetime.now().strftime("%H:%M:%S")
                        status = "present" if datetime.now().hour < 9 else "late"
                        
                        # Check if attendance already marked today
                        check_response = supabase_client.table("attendance") \
                            .select("id") \
                            .eq("person_id", person_id) \
                            .eq("date", today) \
                            .execute()
                        
                        if not check_response.data:
                            # Add attendance record
                            attendance_data = {
                                "person_id": person_id,
                                "date": today,
                                "time": current_time,
                                "status": status,
                                "confidence": float(confidence),
                                "marked_by": "system"
                            }
                            
                            # Insert into attendance table directly
                            supabase_client.table("attendance").insert(attendance_data).execute()
                            attendance_marked = True
            
            results.append(DetectedFace(
                name=name,
                confidence=float(confidence),
                location=FaceLocation(top=top, right=right, bottom=bottom, left=left),
                attendance_marked=attendance_marked
            ))
        
        return results
    except Exception as e:
        raise e
