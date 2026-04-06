import os
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import sys

# Ensure GridDehazeNet module is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models' / 'GridDehazeNet'))

from model import GridDehazeNet
from ultralytics import YOLO


def dehaze_dataset(input_dir, output_dir, category='indoor', device=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    network_height = 3
    network_width = 6

    model = GridDehazeNet(height=network_height, width=network_width)
    model_path = ROOT / 'models' / 'GridDehazeNet' / f"{category}_haze_best_{network_height}_{network_width}"
    if not model_path.exists():
        raise FileNotFoundError(f"GridDehazeNet weights not found: {model_path}")

    model = model.to(device)
    loaded = torch.load(str(model_path), map_location=device)

    from collections import OrderedDict
    # Remove DataParallel prefix if present
    if isinstance(loaded, dict) and list(loaded.keys())[0].startswith('module.'):
        loaded = OrderedDict((k.replace('module.', ''), v) for k, v in loaded.items())

    # Debug: print keys
    model_keys = list(model.state_dict().keys())
    loaded_keys = list(loaded.keys())
    print(f"Model keys sample: {model_keys[:5]}")
    print(f"Loaded keys sample: {loaded_keys[:5]}")
    print(f"Model has {len(model_keys)} keys, loaded has {len(loaded_keys)} keys")

    # Simple mapping
    mapped = OrderedDict()
    for k, v in loaded.items():
        new_key = k.replace('rdb_', 'rdb_module.').replace('rdb_module.in', 'rdb_in').replace('rdb_module.out', 'rdb_out').replace('upsample_', 'upsample_module.').replace('downsample_', 'downsample_module.')
        mapped[new_key] = v

    model.load_state_dict(mapped)
    model.eval()

    normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    unnormalize = lambda x: torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)

    files = sorted([x for x in input_dir.iterdir() if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    if not files:
        raise FileNotFoundError(f'No image files found in {input_dir}')

    for path in files:
        img = Image.open(path).convert('RGB')
        inp = T.ToTensor()(img)
        inp = normalize(inp).unsqueeze(0).to(device)

        with torch.no_grad():
            # 3-pass dehazing for stronger fog removal
            out = model(inp)
            out = torch.clamp(out, 0.0, 1.0)
            out = out.squeeze(0)  # [3,H,W]
            out = normalize(out).unsqueeze(0).to(device)

            out = model(out)
            out = torch.clamp(out, 0.0, 1.0)
            out = out.squeeze(0)  # [3,H,W]
            out = normalize(out).unsqueeze(0).to(device)

            out = model(out).squeeze(0)

        out = out.detach().cpu()
        out = unnormalize(out)
        out = torch.clamp(out, 0.0, 1.0)
        out_img = (out.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        # Post-processing to enhance dehazing
        import cv2
        # Convert to LAB for CLAHE
        lab = cv2.cvtColor(out_img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Gamma correction for brightness
        gamma = 1.1
        enhanced = np.power(enhanced / 255.0, gamma) * 255.0
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Unsharp masking for sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        # Color enhancement: increase saturation slightly
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.1, 0, 255)  # Saturation
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        out_img = enhanced

        out_path = output_dir / path.name
        Image.fromarray(out_img).save(out_path)
        print(f"Dehazed: {path.name} -> {out_path}")

    return str(output_dir)


if __name__ == '__main__':
    source = ROOT / 'datasets' / 'SIR_IMAGES' / 'hazy'
    dehazed_output = ROOT / 'outputs' / 'SIR' / 'GridDehazeNet'
    dehazed_output.mkdir(parents=True, exist_ok=True)

    print('Running GridDehazeNet dehaze...')
    dehazed_output = ROOT / 'outputs' / 'SIR' / 'GridDehazeNet1'
    dehazed_output.mkdir(parents=True, exist_ok=True)
    source_dir = dehaze_dataset(source, dehazed_output, category='outdoor')

    print('Running YOLO on dehazed images...')
    yolo = YOLO('yolo11m.pt')
    yolo.predict(source=source_dir,
                 save=True,
                 project=str(ROOT / 'runs'),
                 name='SIR_GridDehazeNet1_yolo',
                 exist_ok=True,
                 save_txt=True,
                 save_conf=True)

    print('Done. Results in runs/SIR_GridDehazeNet_yolo')
