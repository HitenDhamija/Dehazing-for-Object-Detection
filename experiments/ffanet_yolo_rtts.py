import torch
import cv2
import os
import sys
import numpy as np
import torchvision.transforms as tfs
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../models/FFA-Net/net/models'))
from FFA import FFA
from ultralytics import YOLO
import torch.nn as nn

# Paths
input_folder = "datasets/RTTS/hazy"
dehazed_folder = "outputs/SIR/FFANet_RTTS_FOG"
yolo_output_folder = "runs/new/FFANet_RTTS_result"
yolo_model_path = "yolo11m.pt"
# Try OTS (outdoor) model first since RTTS is outdoor dataset
weights_path = "models/FFA-Net/net/trained_models/ots_train_ffa_3_19.pk"
weights_path_fallback = "models/FFA-Net/net/trained_models/its_train_ffa_3_19.pk"

os.makedirs(dehazed_folder, exist_ok=True)
os.makedirs(yolo_output_folder, exist_ok=True)
os.makedirs(os.path.join(yolo_output_folder, "labels"), exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load FFA-Net model
print("Loading FFA-Net model...")
gps = 3
blocks = 19
net = FFA(gps=gps, blocks=blocks)

# Try to load pretrained weights
if os.path.exists(weights_path):
    print(f"Loading weights from {weights_path}...")
    ckp = torch.load(weights_path, map_location=device, weights_only=False)
    if isinstance(ckp, dict) and 'model' in ckp:
        net = nn.DataParallel(net)
        net.load_state_dict(ckp['model'])
        print(f"✅ OTS Weights loaded successfully")
    else:
        net = nn.DataParallel(net)
        net.load_state_dict(ckp)
        print(f"✅ OTS Weights loaded successfully")
elif os.path.exists(weights_path_fallback):
    print(f"OTS weights not found, trying ITS model...")
    ckp = torch.load(weights_path_fallback, map_location=device, weights_only=False)
    if isinstance(ckp, dict) and 'model' in ckp:
        net = nn.DataParallel(net)
        net.load_state_dict(ckp['model'])
        print(f"✅ ITS Weights loaded successfully")
    else:
        net = nn.DataParallel(net)
        net.load_state_dict(ckp)
        print(f"✅ ITS Weights loaded successfully")
else:
    print(f"Warning: Neither OTS nor ITS weights found")
    print(f"Expected: {weights_path} or {weights_path_fallback}")
    net = nn.DataParallel(net)

net = net.to(device)
net.eval()

# Load YOLO model
print("Loading YOLO model...")
yolo_model = YOLO(yolo_model_path)

# Preprocessing for FFA-Net - Using PIL like the original test.py
def dehaze_image(img_path, enhance=1.2):
    try:
        # Load image using PIL for proper preprocessing
        haze_pil = Image.open(img_path).convert('RGB')
        
        # Apply same transformations as FFA-Net test.py
        img_tensor = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
        ])(haze_pil)
        
        haze_input = img_tensor.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            dehazed = net(haze_input)
        
        # Clamp and convert back to image
        dehazed = dehazed.squeeze(0).cpu()
        dehazed = torch.clamp(dehazed, 0, 1)
        
        # Optional: Enhance contrast slightly to improve visibility
        # This boosts the dehazing effect
        if enhance > 1.0:
            dehazed = torch.clamp(dehazed * enhance, 0, 1)
        
        # Convert tensor to numpy and then to uint8
        dehazed_np = dehazed.permute(1, 2, 0).numpy()
        dehazed_np = (dehazed_np * 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV/cv2
        dehazed_bgr = cv2.cvtColor(dehazed_np, cv2.COLOR_RGB2BGR)
        
        # Optional: Apply histogram equalization to boost local contrast
        # Convert to HSV for histogram equalization on V channel
        hsv = cv2.cvtColor(dehazed_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.equalizeHist(v)
        hsv_eq = cv2.merge([h, s, v])
        dehazed_bgr = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
        
        return dehazed_bgr
    except Exception as e:
        print(f"\n❌ Error processing {img_path}: {e}")
        return None

# Process images
print("\nProcessing RTTS hazy images with FFA-Net...")
image_count = 0
for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, filename)
    print(f"Processing {filename}...", end=" ")
    
    try:
        # Dehaze
        dehazed_img = dehaze_image(img_path)
        if dehazed_img is None:
            print("❌ Failed to read image")
            continue
            
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
        
        # Save labels
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

print(f"\n✅ FFANet_RTTS: Pipeline complete! Processed {image_count} images.")
print(f"   Dehazed images: {dehazed_folder}")
print(f"   YOLO results: {yolo_output_folder}")
