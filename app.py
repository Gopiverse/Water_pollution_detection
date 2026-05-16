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
# from roboflow import Roboflow
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


plastic_model = YOLO("best_plastic.pt")
leaf_model = YOLO("best_leaf.pt")
oil_model = YOLO("best.pt")


@app.route("/", methods=["GET", "POST"])
def index():
    clean_up_static_folder()
    original_img = None
    plastic_img = None
    leaf_img = None
    status_message = None
    oil_img = None

    if request.method == "POST":
        file = request.files.get("file")

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            original_img = f"/{filepath}"


            plastic_results = plastic_model.predict(filepath, conf=0.20)
            plastic_filename = f"plastic_{int(time.time()*1000)}.jpg"
            plastic_path = os.path.join("static", plastic_filename)
            plastic_results[0].save(plastic_path)
            plastic_img = url_for('static', filename=plastic_filename)


            leaf_results = leaf_model.predict(filepath)
            leaf_filename = f"leaf_{int(time.time()*1000)}.jpg"
            leaf_path = os.path.join("static", leaf_filename)
            leaf_results[0].save(leaf_path)
            leaf_img = url_for('static', filename=leaf_filename)

           

            oil_results = oil_model.predict(filepath, conf=0.09)
            oil_filename = f"oil_{int(time.time()*1000)}.jpg"
            oil_path = os.path.join("static", oil_filename)
            oil_results[0].save(oil_path)
            oil_img = url_for('static', filename=oil_filename)


            #count plastic, leaf and oil detections
            plastic_count = len(plastic_results[0].boxes)
            leaf_count = len(leaf_results[0].boxes)
            oil_count = len(oil_results[0].boxes)

           
            status_message = log_and_compare(
                    plastic_count, leaf_count, oil_count,
                    
                )
            print(status_message)
            print("Plastic saved:", plastic_path)
            print("Leaf saved:", leaf_path)


    return render_template(
        "index.html",
        original_img=original_img,
        plastic_img=plastic_img,
        #leaf_img=leaf_img,
        oil_img=oil_img,
        status_message=status_message
    )



import json
from flask import jsonify

@app.route("/detections")
def get_detections():
    log_file = "log.json"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            data = json.load(f)
    else:
        data = []
    return jsonify(data)





def log_and_compare(plastic_count, leaf_count, oil_count):
    log_file = "log.json"

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            json.dump([], f)

    with open(log_file, "r") as f:
        data = json.load(f)

    previous_count = data[-1]["plastic_count"] if data else None
    message = None

    if previous_count is not None:
        if (plastic_count > previous_count):
            if(oil_count):
                message = f"⚠ Plastic count increased! Previous : {previous_count}, Now: {plastic_count} \n ⚠ Oil spills detected !! "
            else:
                message = f"⚠ Plastic count increased! Previous: {previous_count}, Now: {plastic_count}"
            send_email_notification("Alert !!", message)
            
            # print("Email sent successfully")

        elif plastic_count < previous_count:
            if(oil_count):
                message = f"⚠ Oil spills detected! \n Plastic count decreased! Previous : {previous_count}, Now: {plastic_count} "
            message = f" Plastic count decreased! Previous: {previous_count}, Now: {plastic_count}"
            send_email_notification("Alert : ", message)
            # print("Email sent successfully")

    else:
        message = f"First detection recorded. Count: {plastic_count}"

    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "plastic_count": plastic_count,
        "leaf_count": leaf_count,
        "oil_count": oil_count
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
    for file in glob.glob("static/plastic_*") + \
            glob.glob("static/leaf_*") + \
            glob.glob("static/oil_*") + \
            glob.glob("static/seg_water_*"):
                os.remove(file)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8040))
    app.run(host="0.0.0.0", port=port)
