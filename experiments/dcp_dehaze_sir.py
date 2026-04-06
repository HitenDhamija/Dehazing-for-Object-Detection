import cv2
import os
import numpy as np

input_folder = "datasets/SIR_IMAGES/hazy"
output_folder = "outputs/SIR/DCP"
os.makedirs(output_folder, exist_ok=True)

def dark_channel(img, size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)

def estimate_atmosphere(img, dark):
    flat_img = img.reshape(-1, 3)
    flat_dark = dark.reshape(-1)
    indices = flat_dark.argsort()[-100:]
    return np.mean(flat_img[indices], axis=0)

def dehaze(img):
    img = img.astype(np.float64) / 255
    dark = dark_channel(img)
    A = estimate_atmosphere(img, dark)

    transmission = 1 - 0.95 * dark
    transmission = np.clip(transmission, 0.1, 1)

    J = np.empty_like(img)
    for i in range(3):
        J[:, :, i] = (img[:, :, i] - A[i]) / transmission + A[i]

    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8)

for name in os.listdir(input_folder):
    path = os.path.join(input_folder, name)
    img = cv2.imread(path)

    if img is None:
        continue

    result = dehaze(img)
    cv2.imwrite(os.path.join(output_folder, name), result)

print("DCP fog removal completed.")