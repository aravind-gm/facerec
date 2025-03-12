from pydantic import BaseModel, constr
from typing import Optional, Literal, List
from datetime import date, time, datetime
from uuid import UUID

class Person(BaseModel):
    id: Optional[UUID] = None
    name: str
    employee_id: str
    department: Optional[str] = None
    position: Optional[str] = None
    image: Optional[str] = None  # Add this line

    class Config:
        from_attributes = True

class Attendance(BaseModel):
    id: Optional[UUID] = None
    person_id: UUID
    date: str  # Keep as str to match database format
    time: str  # Keep as str to match database format
    status: str
    confidence: Optional[float] = None
    marked_by: str = 'system'
    notes: Optional[str] = None
    created_at: Optional[str] = None  # Keep as str to match database format
    person: Optional[Person] = None

    class Config:
        from_attributes = True

class AttendanceResponse(BaseModel):
    id: Optional[UUID] = None
    person_id: UUID
    date: date
    time: time
    status: str
    confidence: Optional[float] = None
    marked_by: str = 'system'
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    person: Person

    class Config:
        from_attributes = True

class profile(BaseModel):
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
    attendance_marked: bool = False

class DetectionResponse(BaseModel):
    results: List[DetectedFace]

class RegisterFaceRequest(BaseModel):
    name: str
    employee_id: str
    department: Optional[str] = None
    position: Optional[str] = None
    image_data: str  # Base64 encoded image