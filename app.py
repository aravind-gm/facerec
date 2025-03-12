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


if __name__ == "__main__":
    app.run(debug=True)