from pydantic import BaseModel, constr
from typing import Optional, Literal, List
from datetime import date, time, datetime
from uuid import UUID

class Attendance(BaseModel):
    id: UUID
    person_id: UUID
    date: date
    time: time
    status: Literal['present', 'absent', 'late']
    confidence: Optional[float] = None
    marked_by: str = 'system'
    notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            date: lambda v: v.isoformat(),
            time: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class Person(BaseModel):
    id: UUID
    name: str
    employee_id: str
    department: Optional[str] = None
    position: Optional[str] = None
    face_embedding: Optional[bytes] = None
    face_image_path: Optional[str] = None
    created_at: datetime
    created_by: Optional[UUID] = None
    active: bool = True

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
            bytes: lambda v: v.hex() if v else None
        }

class Profile(BaseModel):
    id: UUID
    username: str
    role: Literal['admin', 'user'] = 'user'
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }

class FaceLocation(BaseModel):
    top: int
    right: int
    bottom: int
    left: int

class DetectedFace(BaseModel):
    name: str
    confidence: float
    location: FaceLocation
    attendance_marked: bool

class DetectionResponse(BaseModel):
    results: List[DetectedFace]

class RegisterFaceRequest(BaseModel):
    name: str
    employee_id: str
    department: Optional[str] = None
    position: Optional[str] = None
    image_data: str  # Base64 encoded image