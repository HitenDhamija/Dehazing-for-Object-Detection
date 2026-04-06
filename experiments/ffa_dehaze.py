import os
import cv2
import numpy as np

input_folder = "datasets/SIR_IMAGES/hazy"
output_folder = "outputs/SIR/FFA"

os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    path = os.path.join(input_folder, img_name)
    img = cv2.imread(path)

    if img is None:
        continue

    # Simple contrast enhancement (placeholder for FFA-Net)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    l = cv2.equalizeHist(l)

    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    save_path = os.path.join(output_folder, img_name)
    cv2.imwrite(save_path, enhanced)

print("Dehazing completed (contrast enhanced).")