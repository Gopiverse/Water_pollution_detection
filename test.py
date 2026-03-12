from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best_seg.pt")

image_path = r"C:\Users\user\projects\MINIPROJECT\water_pollution_detection\Water_pollution_detection\CAM\WEB_CAM_1\day_1.jpg"

results = model(image_path)

img = cv2.imread(image_path)

if results[0].masks is not None:

    masks = results[0].masks.data.cpu().numpy()
    mask = np.max(masks, axis=0)

    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    water_mask = mask > 0.5

    output = img.copy()
    output[~water_mask] = [255,255,255]

    cv2.imwrite("result1.jpg", output)

print("Done")