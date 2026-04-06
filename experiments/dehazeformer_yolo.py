import os
import torch
import numpy as np
import cv2
from ultralytics import YOLO
from models.DehazeFormer.models import dehazeformer_b
from models.DehazeFormer.utils import hwc_to_chw, chw_to_hwc
from collections import OrderedDict

def read_img(path):
    return cv2.imread(path)[:, :, ::-1] / 255.0

def preprocess(img):
    img = img * 2 - 1
    return torch.from_numpy(hwc_to_chw(img)).unsqueeze(0).float()

def postprocess(tensor):
    img = chw_to_hwc(tensor.squeeze(0).cpu().numpy())
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

# Load model
model = dehazeformer_b()
model_path = 'models/DehazeFormer/saved_models/dehazeformer-b.pth'  # Note: Download from https://drive.google.com/drive/folders/1zfkx4iR2GNqT8mO8l7vZGZfGj8V8w7w
try:
    state_dict = torch.load(model_path)['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    model.cuda()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please download the pretrained weights from the DehazeFormer repository.")
    exit(1)

# Directories
hazy_dir = 'datasets/SIR_IMAGES/hazy'
output_dir = 'outputs/SIR/DehazeFormer'
os.makedirs(output_dir, exist_ok=True)

yolo_model = YOLO('yolo11m.pt')
run_dir = 'runs/dehazeformer_yolo'
os.makedirs(run_dir, exist_ok=True)

total_detections = 0
image_count = 0

for img_name in os.listdir(hazy_dir):
    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img_path = os.path.join(hazy_dir, img_name)
    img = read_img(img_path)
    img_tensor = preprocess(img).cuda()
    
    with torch.no_grad():
        dehazed_tensor = model(img_tensor)
    
    dehazed_np = postprocess(dehazed_tensor)
    dehazed_img = cv2.cvtColor(dehazed_np, cv2.COLOR_RGB2BGR)
    dehazed_path = os.path.join(output_dir, img_name)
    cv2.imwrite(dehazed_path, dehazed_img)
    
    # Run YOLO
    results = yolo_model(dehazed_path, save=True, project=run_dir, name='predict', verbose=False)
    detections = len(results[0].boxes)
    total_detections += detections
    image_count += 1
    print(f'{img_name}: {detections} detections')

print(f"Total images processed: {image_count}")
print(f"Total detections: {total_detections}")