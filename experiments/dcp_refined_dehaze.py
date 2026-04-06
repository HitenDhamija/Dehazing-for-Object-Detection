import cv2
import os
import numpy as np

input_folder = "datasets/SIR_IMAGES/hazy"
output_folder = "outputs/SIR/DCP_Refined"
os.makedirs(output_folder, exist_ok=True)

def dehaze(img):
    img = img.astype(np.float64) / 255

    # Dark channel
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(min_channel, kernel)

    # Atmospheric light
    flat_img = img.reshape(-1, 3)
    flat_dark = dark.reshape(-1)
    indices = flat_dark.argsort()[-100:]
    A = np.mean(flat_img[indices], axis=0)

    # Transmission
    t = 1 - 0.95 * dark
    t = cv2.GaussianBlur(t, (15, 15), 0)
    t = np.clip(t, 0.2, 1)

    # Recover image
    J = np.empty_like(img)
    for i in range(3):
        J[:, :, i] = (img[:, :, i] - A[i]) / t + A[i]

    J = np.clip(J, 0, 1)

    # Contrast restoration
    J = cv2.convertScaleAbs(J * 255, alpha=1.2, beta=10)

    return J

for name in os.listdir(input_folder):
    path = os.path.join(input_folder, name)
    img = cv2.imread(path)

    if img is None:
        continue

    result = dehaze(img)
    cv2.imwrite(os.path.join(output_folder, name), result)

print("Refined DCP dehazing completed.")