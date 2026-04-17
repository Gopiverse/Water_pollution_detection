# Water_pollution_detection

# Water Pollution Detection System 🌊🔍

An AI-powered web application that detects and classifies water contaminants using Computer Vision. This project utilizes multiple **YOLOv8** models to identify pollutants like plastic, oil spills, and organic waste (leaves) in real-time or from uploaded images.

## 🚀 Live Demo
The application is deployed and can be accessed here:  
[Water Pollution Detection App](https://water-pollution-detection-ds5d.onrender.com)

## ✨ Features
- **Multi-Class Detection:** Specialized models for different types of pollutants:
  - 🧴 **Plastic:** Detects bottles and floating plastic waste.
  - 🛢️ **Oil Spills:** Identifies oil slicks on water surfaces.
  - 🍃 **Organic Waste:** Detects excessive leaves or biological debris.
- **Image Upload & Real-time Processing:** Users can upload images for instant analysis.
- **Web Interface:** Built with Flask for a smooth, user-friendly experience.
- **Detailed Logging:** System logs results for historical analysis.

## 🛠️ Tech Stack
- **AI Framework:** Ultralytics YOLOv8
- **Backend:** Python, Flask
- **Frontend:** HTML5, CSS3, JavaScript
- **Libraries:** OpenCV, PyTorch, JSON

## 📂 Project Structure
- `app.py`: The main Flask application handling routing and model inference.
- `best_*.pt`: Pre-trained YOLOv8 weights for specific contaminant detection.
- `static/` & `templates/`: Frontend assets and HTML layouts.
- `CAM/`: Logic for camera-based detection/processing.
- `test.py`: Script for testing model performance locally.

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Gopiverse/Water_pollution_detection.git](https://github.com/Gopiverse/Water_pollution_detection.git)
   cd Water_pollution_detection
   
