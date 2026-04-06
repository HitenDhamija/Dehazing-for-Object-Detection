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
import torch.nn as nn

# Paths
input_folder = "datasets/SIR_IMAGES/hazy"
dehazed_folder = "outputs/SIR/FFA-Net-Enhanced"
yolo_output_folder = "runs/SIR_FFA_Enhanced_yolo"
yolo_model_path = "yolo11m.pt"
weights_path = "models/FFA-Net/net/trained_models/its_train_ffa_3_19.pk"

os.makedirs(dehazed_folder, exist_ok=True)
os.makedirs(yolo_output_folder, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load FFA-Net model
print("Loading FFA-Net model with pre-trained weights...")
gps = 3
blocks = 19
net = FFA(gps=gps, blocks=blocks)

if os.path.exists(weights_path):
    print(f"✓ Loading weights from {weights_path}")
    ckp = torch.load(weights_path, map_location=device)
    if isinstance(ckp, dict) and 'model' in ckp:
        net = nn.DataParallel(net)
        net.load_state_dict(ckp['model'])
    else:
        net = nn.DataParallel(net)
        net.load_state_dict(ckp)
else:
    net = nn.DataParallel(net)

net = net.to(device)
net.eval()

# Load YOLO model
print("Loading YOLO model...\n")
yolo_model = YOLO(yolo_model_path)

def enhance_image_quality(img):
    """Advanced post-processing for better image quality"""
    
    # 1. Convert to LAB for better contrast enhancement
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    
    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) - stronger
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
    l = clahe.apply(l)
    
    img_lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    
    # 3. Color correction - boost midtones
    img = img.astype(np.float32) / 255.0
    img = np.power(img, 0.95)  # Slight gamma correction
    img = (img * 255).astype(np.uint8)
    
    # 4. Multi-scale Unsharp Masking for sharpening
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 1.0)
    img_sharp = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)
    img_sharp = np.clip(img_sharp, 0, 255).astype(np.uint8)
    
    # 5. Slight bilateral filtering to reduce noise while preserving edges
    img_smooth = cv2.bilateralFilter(img_sharp, 9, 75, 75)
    
    # 6. Blend sharp and smooth versions
    result = cv2.addWeighted(img_smooth, 0.7, img_sharp, 0.3, 0)
    
    return result

def dehaze_image(img_path):
    """Dehaze using FFA-Net with enhancement"""
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
    
    # Convert back
    dehazed = dehazed.squeeze(0).cpu()
    dehazed = torch.clamp(dehazed, 0, 1)
    dehazed_np = dehazed.permute(1, 2, 0).numpy()
    dehazed_np = (dehazed_np * 255).astype(np.uint8)
    dehazed_bgr = cv2.cvtColor(dehazed_np, cv2.COLOR_RGB2BGR)
    
    # Apply quality enhancement
    dehazed_bgr = enhance_image_quality(dehazed_bgr)
    
    return dehazed_bgr

# Process images
print("="*70)
print("Processing images with Enhanced FFA-Net...")
print("="*70 + "\n")

for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, filename)
    print(f"📷 {filename:<15}", end=" | ")
    
    try:
        # Dehaze with enhancement
        dehazed_img = dehaze_image(img_path)
        dehazed_path = os.path.join(dehazed_folder, filename)
        cv2.imwrite(dehazed_path, dehazed_img)
        print("✓ Enhanced", end=" | ")

        # Run YOLO
        results = yolo_model(dehazed_path, conf=0.25, verbose=False)
        
        # Save YOLO results
        os.makedirs(os.path.join(yolo_output_folder, "labels"), exist_ok=True)
        
        detection_count = 0
        for result in results:
            annotated_img = result.plot()
            output_path = os.path.join(yolo_output_folder, filename)
            cv2.imwrite(output_path, annotated_img)
            
            # Save labels
            boxes = result.boxes
            if boxes is not None:
                detection_count = len(boxes)
                label_path = os.path.join(yolo_output_folder, "labels", filename.rsplit('.', 1)[0] + '.txt')
                with open(label_path, 'w') as f:
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = box.conf[0]
                        x1, y1, x2, y2 = box.xyxy[0]
                        img_h, img_w = annotated_img.shape[:2]
                        center_x = ((x1 + x2) / 2) / img_w
                        center_y = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        f.write(f"{cls} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
        
        print(f"✓ YOLO ({detection_count} objects)")
                
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n" + "="*70)
print("✅ Enhanced FFA-Net dehazing completed!")
print(f"📁 Enhanced images: {dehazed_folder}")
print(f"📁 Detection results: {yolo_output_folder}")