import os
import platform
import threading
import time
import cv2
import dlib
import numpy as np
import pyttsx3  # For voice alerts
from flask import Flask, Response, render_template, jsonify, request, redirect, session, url_for
from scipy.spatial import distance as dist

app = Flask(__name__)
app.secret_key = "super_secret_key"  # Change this for security

# Dummy user credentials
USER_CREDENTIALS = {
    "admin": "password123"
}

# Load face detector & landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file exists

params = {
    "EAR_THRESHOLD": 0.25,  # Adjust this threshold for EAR
    "CLOSED_EYE_TIME_THRESHOLD": 2,  # Seconds eyes must be closed for drowsiness
}

camera_lock = threading.Lock()
cap = None
is_running = False
start_time = None  # Track time eyes are closed
engine = pyttsx3.init()  # Text-to-speech engine
alert_active = False  # Flag to prevent multiple alerts


# Helper Function: EAR Calculation
def eye_aspect_ratio(eye):
    """Calculate Eye Aspect Ratio (EAR)."""
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    return (A + B) / (2.0 * C)


# Alert Mechanisms
def sound_alarm():
    """Plays a beep alert repeatedly while the driver is drowsy."""
    global alert_active
    while alert_active:
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 700)  # 700ms beep at 1000Hz
            elif platform.system() == "Darwin":  # macOS
                os.system("afplay /System/Library/Sounds/Glass.aiff")
            else:  # Linux
                os.system("beep -f 1000 -l 700")
        except Exception as e:
            print(f"Error in playing beep alert: {e}")
        time.sleep(1)  # Beep every second while drowsiness is detected


def voice_alert():
    """Continuously speaks an alert message while eyes are closed."""
    global alert_active
    while alert_active:
        try:
            engine.say("Wake up! You are drowsy!")
            engine.runAndWait()
        except Exception as e:
            print(f"Error in voice alert: {e}")
        time.sleep(1)  # Speak every second while drowsiness is detected


# Define the login_required decorator
def login_required(func):
    """Decorator to enforce login requirement."""
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return func(*args, **kwargs)
    wrapper.__name__ = func.__name__
    return wrapper


# Flask Routes
@app.route("/", methods=["GET", "POST"])
def login():
    """Handles user login."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        driver_reg = request.form.get("driver_reg")
        car_brand = request.form.get("car_brand")
        car_number = request.form.get("car_number")
        driving_date = request.form.get("driving_date")

        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            session["user"] = username
            session["driver"] = driver_reg
            session["car"] = car_brand
            session["car_number"] = car_number
            session["date"] = driving_date
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")


@app.route("/logout")
def logout():
    """Logs out the user."""
    session.clear()
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    """Handles user registration."""
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in USER_CREDENTIALS:
            return render_template("register.html", error="User already exists!")

        USER_CREDENTIALS[username] = password
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/index")
@login_required
def index():
    """Renders the dashboard with user details."""
    user_details = {
        "user": session.get("user"),
        "driver": session.get("driver"),
        "car": session.get("car"),
        "car_number": session.get("car_number"),
        "date": session.get("date"),
    }
    return render_template("index.html", **user_details)


@app.route("/start_video")
@login_required
def start_video():
    """Starts the video stream."""
    global is_running, cap
    with camera_lock:
        if not is_running:
            is_running = True
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                is_running = False
                cap = None
                return jsonify({"status": "error", "message": "Unable to access the camera."})

            threading.Thread(target=generate_frames, daemon=True).start()

    return jsonify({"status": "started", "running": is_running})


@app.route("/stop_video")
@login_required
def stop_video():
    """Stops the video stream."""
    global is_running, cap
    with camera_lock:
        is_running = False
        if cap and cap.isOpened():
            cap.release()
            cap = None
    return jsonify({"status": "stopped", "running": is_running})


@app.route("/video_feed")
@login_required
def video_feed():
    """Provides the video feed for the frontend."""
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


def generate_frames():
    """Generates video frames with drowsiness detection."""
    global is_running, cap, start_time, alert_active
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    while is_running:
        with camera_lock:
            success, frame = cap.read()
            if not success:
                print("Warning: Failed to capture frame.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
                right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
                EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                # Detect drowsiness
                if EAR < params["EAR_THRESHOLD"]:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= params["CLOSED_EYE_TIME_THRESHOLD"]:
                        if not alert_active:
                            alert_active = True
                            threading.Thread(target=sound_alarm, daemon=True).start()
                            threading.Thread(target=voice_alert, daemon=True).start()
                else:
                    start_time = None
                    alert_active = False

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()


if __name__ == "__main__":
    is_running = True  # Set stream to running
    app.run(debug=True)
