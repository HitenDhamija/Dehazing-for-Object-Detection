import torch
import cv2
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.aod_net import AODNet
import requests
from ultralytics import YOLO

# Paths
input_folder = "datasets/SIR_IMAGES/hazy"
dehazed_folder = "outputs/SIR_AOD_Dehazed"
yolo_model_path = "yolo11m.pt"
weights_url = "https://github.com/MayankSingal/PyTorch-Image-Dehazing/raw/master/snapshots/dehazer.pth"
weights_path = "models/aod_net.pth"

os.makedirs(dehazed_folder, exist_ok=True)

# Download weights if not present
if not os.path.exists(weights_path):
    print("Downloading AOD-Net weights...")
    response = requests.get(weights_url)
    with open(weights_path, 'wb') as f:
        f.write(response.content)
    print("Weights downloaded.")

# Load AOD-Net model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AODNet().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# Load YOLO model
yolo_model = YOLO(yolo_model_path)

def dehaze_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        dehazed = model(img).squeeze(0).permute(1, 2, 0).cpu().numpy()

    dehazed = np.clip(dehazed * 255, 0, 255).astype(np.uint8)
    dehazed = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
    return dehazed

# Process images
for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Dehaze
    dehazed_img = dehaze_image(img)
    dehazed_path = os.path.join(dehazed_folder, filename)
    cv2.imwrite(dehazed_path, dehazed_img)

    # Run YOLO on dehazed image
    results = yolo_model(dehazed_path, conf=0.25)

    # Save YOLO results with visualizations
    os.makedirs("runs/SIR_AOD_Dehazed_yolo", exist_ok=True)
    os.makedirs("runs/SIR_AOD_Dehazed_yolo/labels", exist_ok=True)
    
    # Draw boxes on image and save
    for result in results:
        annotated_img = result.plot()
        output_path = os.path.join("runs/SIR_AOD_Dehazed_yolo", filename)
        cv2.imwrite(output_path, annotated_img)
        
        # Save labels
        boxes = result.boxes
        if boxes is not None:
            label_path = os.path.join("runs/SIR_AOD_Dehazed_yolo/labels", filename.rsplit('.', 1)[0] + '.txt')
            with open(label_path, 'w') as f:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    x1, y1, x2, y2 = box.xyxy[0]
                    # Convert to YOLO format (normalized center x, center y, width, height)
                    img_h, img_w = annotated_img.shape[:2]
                    center_x = ((x1 + x2) / 2) / img_w
                    center_y = ((y1 + y2) / 2) / img_h
                    width = (x2 - x1) / img_w
                    height = (y2 - y1) / img_h
                    f.write(f"{cls} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")

    # Print detections
    print(f"\nDetections for {filename}:")
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                coords = box.xyxy[0]
                print(f"  Class {cls}: {yolo_model.names[cls]} with confidence {conf:.2f} at {coords}")

print("\n✅ AOD-Net dehazing and YOLO detection completed.")