import cv2
import os
import numpy as np

input_folder = "datasets/SIR_IMAGES/hazy"
output_folder = "output/SIR"
os.makedirs(output_folder, exist_ok=True)

def aod_dehaze(img):
    img = img.astype(np.float32) / 255.0

    # Estimate atmospheric light
    dark = np.min(img, axis=2)
    A = np.percentile(img.reshape(-1, 3), 99.9, axis=0)

    # Transmission (AOD-Net style approximation)
    t = 1 - 0.85 * dark
    t = np.clip(t, 0.2, 1)

    # Recover scene radiance
    J = np.empty_like(img)
    for i in range(3):
        J[:, :, i] = (img[:, :, i] - A[i]) / t + A[i]

    J = np.clip(J, 0, 1)

    # Mild contrast enhancement
    J = cv2.convertScaleAbs(J * 255, alpha=1.1, beta=5)

    return J


for name in os.listdir(input_folder):
    path = os.path.join(input_folder, name)
    img = cv2.imread(path)

    if img is None:
        continue

    result = aod_dehaze(img)

    save_path = os.path.join(output_folder, name)
    cv2.imwrite(save_path, result)

print("✅ Final AOD-style fog removal completed.")