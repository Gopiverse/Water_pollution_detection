from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import time
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


APPSCRIPT_URL = os.getenv("APPSCRIPT_URL")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
SUBJECT_PREFIX = os.getenv("SUBJECT_PREFIX")

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)


# Load trained YOLO model
model = YOLO("best.pt")

@app.route("/", methods=["GET", "POST"])
def index():
    original_img = None
    result_img = None
    status_message = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            # Save uploaded image
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            original_img = f"/{filepath}"

            # Run YOLO prediction
            results = model.predict(filepath, conf=0.10)

            # Create unique filename
            filename = f"{int(time.time()*1000)}_{file.filename}"
            save_path = os.path.join("static", filename)

            # Save detection result image
            results[0].save(save_path)

            result_img = f"/static/{filename}"

            # Count detections
            total_objects = len(results[0].boxes)

            # if total_objects > 0:
            #     alert_message = f"Plastic detected! Count: {total_objects}"

            # Log every detection (even 0)
            total_objects = len(results[0].boxes)
            status_message = log_and_compare(total_objects)
            print(status_message)

    return render_template(
        "index.html",
        original_img=original_img,
        result_img=result_img,
        status_message=status_message
    )

def log_and_compare(count):
    log_file = "log.json"

    # Create file if not exists
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump([], f)

    # Read previous data
    with open(log_file, "r") as f:
        data = json.load(f)

    previous_count = data[-1]["plastic_count"] if data else None
    message = None

    # Compare counts
    if previous_count is not None:
        if count > previous_count:
            message = f"⚠ Plastic count increased! Previous: {previous_count}, Now: {count}"
            send_email_notification("Alert !!", message)
            # print("Email sent successfully")

        elif count < previous_count:
            message = f"✅ Plastic count decreased! Previous: {previous_count}, Now: {count}"
            send_email_notification("Good job !", message)
            # print("Email sent successfully")

        # If equal → DO NOTHING (no message)
    else:
        message = f"First detection recorded. Count: {count}"

    # Save new entry
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plastic_count": count
    }

    data.append(entry)

    with open(log_file, "w") as f:
        json.dump(data, f, indent=4)

    return message


def send_email_notification(subject, message_body):
    data = {
        'to': RECEIVER_EMAIL,
        'subject': f"{SUBJECT_PREFIX} - {subject}",
        'body': message_body
    }
    response = requests.post(
        APPSCRIPT_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    print(response.text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8040, debug=True)