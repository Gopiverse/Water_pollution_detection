from flask import Flask, request, render_template
import os
import base64
import cv2
import numpy as np
from roboflow import Roboflow
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Roboflow project info
API_KEY = os.getenv("ROBOFLOW_API_KEY")
PROJECT = "water-segmentation-n6ecd"
VERSION = 1

# Load Roboflow model
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project(PROJECT)
model = project.version(VERSION).model

@app.route('/', methods=['GET', 'POST'])
def index():
    original_image_path = None
    water_only_base64 = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            # Save uploaded image
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            original_image_path = path

            # Predict segmentation mask
            result = model.predict(path).json()

            if 'predictions' in result and len(result['predictions']) > 0:
                # Decode Roboflow mask
                mask_base64 = result['predictions'][0]['segmentation_mask']
                mask_bytes = base64.b64decode(mask_base64)
                mask_np = np.frombuffer(mask_bytes, np.uint8)
                mask_img = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)

                # Read original image
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Resize mask to match image
                h, w, _ = image.shape
                mask_resized = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_NEAREST)

                # Apply mask: only water region visible
                water_mask = (mask_resized == 1)
                water_only = image.copy()
                water_only[~water_mask] = 0

                # Encode water-only image as base64 for HTML
                _, buffer = cv2.imencode('.png', cv2.cvtColor(water_only, cv2.COLOR_RGB2BGR))
                water_only_base64 = base64.b64encode(buffer).decode('utf-8')

    return render_template(
        'index.html',
        original_image=original_image_path,
        water_only_image=water_only_base64
    )

if __name__ == '__main__':
    app.run(debug=True)
