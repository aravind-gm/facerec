from flask import Flask, render_template, request, jsonify
import base64
import numpy as np
import cv2
import face_recognition
from database import supabase_client
from models import RegisterFaceRequest

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/register-face", methods=["POST"])
def register_face_flask():
    try:
        data = request.get_json()
        name = data.get("name")
        employee_id = data.get("employee_id")
        department = data.get("department")
        position = data.get("position")
        image_data = data.get("image_data")

        # Decode image
        image_bytes = base64.b64decode(image_data)
        np_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)
        if not face_locations:
            return jsonify({"error": "No face detected in image"}), 400

        face_encoding = face_recognition.face_encodings(rgb_image, [face_locations[0]])[0]

        # Convert face encoding to base64
        encoded_face = base64.b64encode(face_encoding.tobytes()).decode('utf-8')

        # Prepare person data
        person_data = {
            "name": name,
            "employee_id": employee_id,
            "department": department,
            "position": position,
            "face_embedding": encoded_face,
            "active": True
        }

        # Insert into Supabase
        response = supabase_client.from_("people").insert(person_data).execute()

        if hasattr(response, "error") and response.error:
            return jsonify({"error": f"Database Error: {response.error}"}), 500

        return jsonify({"message": "Student registered successfully!"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)