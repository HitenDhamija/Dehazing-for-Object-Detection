import torch
import cv2
import os
import sys
import numpy as np
import torchvision.transforms as tfs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models/FFA-Net/net'))
from models.FFA import FFA
from ultralytics import YOLO
import requests
import torch.nn as nn

# Paths
input_folder = "datasets/SIR_IMAGES/hazy"
dehazed_folder = "outputs/SIR/FFA-Net"
yolo_output_folder = "runs/SIR_FFA_Dehazed_yolo"
yolo_model_path = "yolo11m.pt"
weights_path = "models/FFA-Net/net/trained_models/its_train_ffa_3_19.pk"

os.makedirs(dehazed_folder, exist_ok=True)
os.makedirs(yolo_output_folder, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load FFA-Net model
print("Loading FFA-Net model...")
gps = 3
blocks = 19
net = FFA(gps=gps, blocks=blocks)

# Try to load pretrained weights
weights_filename = f"its_train_ffa_{gps}_{blocks}.pk"
if os.path.exists(weights_path):
    print(f"Loading weights from {weights_path}...")
    ckp = torch.load(weights_path, map_location=device)
    if isinstance(ckp, dict) and 'model' in ckp:
        net = nn.DataParallel(net)
        net.load_state_dict(ckp['model'])
    else:
        net = nn.DataParallel(net)
        net.load_state_dict(ckp)
else:
    # Search for alternative weight file names
    alt_files = [
        os.path.join("models/FFA-Net/net/trained_models", f"its_train_ffa_{gps}_{blocks}.pth"),
        os.path.join("models/FFA-Net/net/trained_models", f"its_ffa.pth"),
        os.path.join("models/FFA-Net/net/trained_models", f"its_train_ffa_{gps}_{blocks}.pk"),
    ]
    
    found = False
    for alt_path in alt_files:
        if os.path.exists(alt_path):
            print(f"Found weights at {alt_path}...")
            try:
                ckp = torch.load(alt_path, map_location=device)
                if isinstance(ckp, dict) and 'model' in ckp:
                    net = nn.DataParallel(net)
                    net.load_state_dict(ckp['model'])
                else:
                    net = nn.DataParallel(net)
                    net.load_state_dict(ckp)
                found = True
                print(f"✅ Weights loaded successfully from {alt_path}")
                break
            except Exception as e:
                print(f"Failed to load from {alt_path}: {e}")
    
    if not found:
        print(f"Warning: Pretrained weights not found.")
        print(f"Expected locations: {weights_path}")
        print(f"Please download FFA-Net weights from https://drive.google.com/drive/folders/19_lSUPrpLDZl9AyewhHBsHidZEpTMIV5")
        print("Using model without pre-trained weights. Results will not be optimal.")
        net = nn.DataParallel(net)

net = net.to(device)
net.eval()

# Load YOLO model
print("Loading YOLO model...")
yolo_model = YOLO(yolo_model_path)

# Preprocessing for FFA-Net
def dehaze_image(img_path):
    haze_img = cv2.imread(img_path)
    haze_img_rgb = cv2.cvtColor(haze_img, cv2.COLOR_BGR2RGB)
    haze_pil = torch.from_numpy(np.array(haze_img_rgb)).permute(2, 0, 1).float()
    
    # Normalize
    normalize = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    haze_normalized = normalize(haze_pil.div(255.0))
    haze_input = haze_normalized.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        dehazed = net(haze_input)
    
    # Denormalize and convert back
    dehazed = dehazed.squeeze(0).cpu()
    dehazed = torch.clamp(dehazed, 0, 1)
    dehazed_np = dehazed.permute(1, 2, 0).numpy()
    dehazed_np = (dehazed_np * 255).astype(np.uint8)
    dehazed_bgr = cv2.cvtColor(dehazed_np, cv2.COLOR_RGB2BGR)
    
    return dehazed_bgr

# Process images
print("\nProcessing images...")
for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, filename)
    print(f"Processing {filename}...", end=" ")
    
    try:
        # Dehaze
        dehazed_img = dehaze_image(img_path)
        dehazed_path = os.path.join(dehazed_folder, filename)
        cv2.imwrite(dehazed_path, dehazed_img)
        print("✓ Dehazed", end=" ")

        # Run YOLO on dehazed image
        results = yolo_model(dehazed_path, conf=0.25)
        
        # Save YOLO results with visualizations
        os.makedirs(os.path.join(yolo_output_folder, "labels"), exist_ok=True)
        
        # Draw boxes on image and save
        for result in results:
            annotated_img = result.plot()
            output_path = os.path.join(yolo_output_folder, filename)
            cv2.imwrite(output_path, annotated_img)
            
            # Save labels
            boxes = result.boxes
            if boxes is not None:
                label_path = os.path.join(yolo_output_folder, "labels", filename.rsplit('.', 1)[0] + '.txt')
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
                print(f"✓ YOLO ({len(boxes)} objects)")
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    print(f"    - {yolo_model.names[cls]}: {conf:.2f}")
            else:
                print("✓ YOLO (no objects)")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n✅ FFA-Net dehazing and YOLO detection completed.")