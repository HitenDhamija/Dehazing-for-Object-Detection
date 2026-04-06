import cv2
import os
import numpy as np
from ultralytics import YOLO

# Paths
input_folder = "datasets/RTTS/hazy"
dehazed_folder = "outputs/SIR/dcp_RTTS_FOG"
yolo_output_folder = "runs/new/dcp_RTTS_result"
yolo_model_path = "yolo11m.pt"

os.makedirs(dehazed_folder, exist_ok=True)
os.makedirs(yolo_output_folder, exist_ok=True)
os.makedirs(os.path.join(yolo_output_folder, "labels"), exist_ok=True)

# Load YOLO model
print("Loading YOLO model...")
yolo_model = YOLO(yolo_model_path)

def dehaze_dcp_refined(img):
    """
    DCP Refined dehazing algorithm
    """
    img = img.astype(np.float64) / 255

    # Dark channel prior
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark = cv2.erode(min_channel, kernel)

    # Estimate atmospheric light
    flat_img = img.reshape(-1, 3)
    flat_dark = dark.reshape(-1)
    indices = flat_dark.argsort()[-100:]
    A = np.mean(flat_img[indices], axis=0)

    # Estimate transmission map
    t = 1 - 0.95 * dark
    t = cv2.GaussianBlur(t, (15, 15), 0)
    t = np.clip(t, 0.2, 1)

    # Recover image
    J = np.empty_like(img)
    for i in range(3):
        J[:, :, i] = (img[:, :, i] - A[i]) / t + A[i]

    J = np.clip(J, 0, 1)

    # Contrast restoration and enhancement
    J = cv2.convertScaleAbs(J * 255, alpha=1.2, beta=10)

    return J

# Process images
print("\nProcessing RTTS hazy images with DCP Refined...")
image_count = 0
for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"❌ Warning: cannot open {img_path}")
        continue

    print(f"Processing {filename}...", end=" ")
    
    try:
        # Dehaze with DCP Refined
        dehazed_img = dehaze_dcp_refined(img)
        dehazed_path = os.path.join(dehazed_folder, filename)
        cv2.imwrite(dehazed_path, dehazed_img)
        print("✓ Dehazed", end=" ")

        # Run YOLO on dehazed image
        results = yolo_model(dehazed_path, conf=0.25)

        # Draw boxes on image and save
        annotated_img = None
        for result in results:
            annotated_img = result.plot()
        if annotated_img is None:
            annotated_img = dehazed_img

        output_path = os.path.join(yolo_output_folder, filename)
        cv2.imwrite(output_path, annotated_img)
        print("✓ YOLO Detection", end=" ")

        # Save labels in YOLO format
        label_path = os.path.join(yolo_output_folder, "labels", filename.rsplit('.', 1)[0] + '.txt')
        with open(label_path, 'w') as f:
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    img_h, img_w = annotated_img.shape[:2]
                    center_x = ((x1 + x2) / 2) / img_w
                    center_y = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    f.write(f"{cls} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

        print("✓ Labels saved")
        image_count += 1

        # Print detections summary
        detection_count = 0
        for result in results:
            detection_count += len(result.boxes)
        print(f"  → Detections: {detection_count}")

    except Exception as e:
        print(f"❌ Error: {e}")

print(f"\n✅ DCP_RTTS: Pipeline complete! Processed {image_count} images.")
print(f"   Dehazed images: {dehazed_folder}")
print(f"   YOLO results: {yolo_output_folder}")
