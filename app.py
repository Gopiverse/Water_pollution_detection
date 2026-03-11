from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import glob
import time
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
from flask import url_for
from roboflow import Roboflow
import base64
import cv2
import numpy as np


APPSCRIPT_URL = os.getenv("APPSCRIPT_URL")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
SUBJECT_PREFIX = os.getenv("SUBJECT_PREFIX")
API_KEY = os.getenv("API_KEY")

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)


# Load trained YOLO model
plastic_model = YOLO("best_plastic.pt")
leaf_model = YOLO("best_leaf.pt")
oil_model = YOLO("best_oil.pt")


# Roboflow water segmentation model
rf = Roboflow(api_key="API_KEY")
project = rf.workspace().project("water-segmentation-n6ecd")
segmentation_model = project.version(1).model



@app.route("/", methods=["GET", "POST"])
def index():
    clean_up_static_folder()
    original_img = None
    plastic_img = None
    leaf_img = None
    status_message = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            # Save uploaded image
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            original_img = f"/{filepath}"

            # run YOLO model
            plastic_results = plastic_model.predict(filepath, conf=0.39)

            plastic_filename = f"plastic_{int(time.time()*1000)}.jpg"
            plastic_path = os.path.join("static", plastic_filename)

            plastic_results[0].save(plastic_path)
            plastic_img = url_for('static', filename=plastic_filename)


            leaf_results = leaf_model.predict(filepath, conf=0.0127)

            leaf_filename = f"leaf_{int(time.time()*1000)}.jpg"
            leaf_path = os.path.join("static", leaf_filename)

            leaf_results[0].save(leaf_path)
            leaf_img = url_for('static', filename=leaf_filename)

           # -------- WATER SEGMENTATION --------
            water_only = extract_water_region(filepath)
            segmented_path, segmented_name = save_segmented_image(water_only, file.filename)
            segmented_img = f"/static/{segmented_name}"

            # -------- OIL DETECTION --------
            oil_path, oil_filename, oil_count = detect_oil(segmented_path)
            oil_img = f"/static/{oil_filename}"



            # Count detections of plastic and leaf
            plastic_count = len(plastic_results[0].boxes)
            leaf_count = len(leaf_results[0].boxes)

            # if total_objects > 0:
            #     alert_message = f"Plastic detected! Count: {total_objects}"

            # Log every detection (even 0)
            status_message = log_and_compare(plastic_count, leaf_count)
            print(status_message)
            print("Plastic saved:", plastic_path)
            print("Leaf saved:", leaf_path)


    return render_template(
        "index.html",
        original_img=original_img,
        plastic_img=plastic_img,
        leaf_img=leaf_img,
        segmented_img=segmented_img,
        oil_img=oil_img,
        status_message=status_message
    )


def extract_water_region(image_path):
    result = segmentation_model.predict(image_path).json()
    mask_base64 = result["predictions"][0]["segmentation_mask"]
    mask_bytes = base64.b64decode(mask_base64)
    mask_np = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    water_mask = (mask_resized == 1)
    water_only = image.copy()
    water_only[~water_mask] = 255
    return water_only

def save_segmented_image(water_only, filename):

    segmented_name = f"segmented_{filename}"
    segmented_path = os.path.join("static", segmented_name)

    water_bgr = cv2.cvtColor(water_only, cv2.COLOR_RGB2BGR)

    cv2.imwrite(segmented_path, water_bgr)

    return segmented_path, segmented_name


def detect_oil(segmented_path):

    results = oil_model.predict(segmented_path, conf=0.01)

    oil_filename = f"oil_{int(time.time()*1000)}.jpg"
    oil_path = os.path.join("static", oil_filename)

    results[0].save(oil_path)

    oil_count = len(results[0].boxes)

    return oil_path, oil_filename, oil_count


def log_and_compare(plastic_count, leaf_count):
    log_file = "log.json"

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump([], f)

    with open(log_file, "r") as f:
        data = json.load(f)

    previous_count = data[-1]["plastic_count"] if data else None
    message = None

    if previous_count is not None:
        if plastic_count > previous_count:
            message = f"⚠ Plastic count increased! Previous: {previous_count}, Now: {plastic_count}"
            send_email_notification("Alert !!", message)
            # print("Email sent successfully")

        elif plastic_count < previous_count:
            message = f" Plastic count decreased! Previous: {previous_count}, Now: {plastic_count}"
            send_email_notification("Good job !", message)
            # print("Email sent successfully")

    else:
        message = f"First detection recorded. Count: {plastic_count}"

    # Save new entry
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plastic_count": plastic_count,
        "leaf_count" : leaf_count
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


def clean_up_static_folder():
    for file in glob.glob("static/plastic_*") + glob.glob("static/leaf_*"):
        os.remove(file)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8040))
    app.run(host="0.0.0.0", port=port)
