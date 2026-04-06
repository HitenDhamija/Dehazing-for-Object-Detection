import os
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import sys
import cv2

# Ensure GridDehazeNet module is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models' / 'GridDehazeNet'))

from model import GridDehazeNet
from ultralytics import YOLO


def dark_channel(img, size=15):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    return cv2.erode(min_channel, kernel)


def estimate_atmosphere(img, dark):
    flat_img = img.reshape(-1, 3)
    flat_dark = dark.reshape(-1)
    indices = flat_dark.argsort()[-100:]
    return np.mean(flat_img[indices], axis=0)


def dcp_dehaze(img):
    img = img.astype(np.float32) / 255.0
    dark = dark_channel(img)
    A = estimate_atmosphere(img, dark)
    transmission = np.clip(1 - 0.95 * dark, 0.1, 1)
    J = np.empty_like(img)
    for i in range(3):
        J[:, :, i] = (img[:, :, i] - A[i]) / transmission + A[i]
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8)


def dehaze_dataset(input_dir, output_dir, category='outdoor', device=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pre_dcp_dir = ROOT / 'outputs' / 'SIR' / 'GridDehazeNet1_pre_dcp'
    pre_dcp_dir.mkdir(parents=True, exist_ok=True)

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
    if isinstance(loaded, dict) and list(loaded.keys())[0].startswith('module.'):
        loaded = OrderedDict((k.replace('module.', ''), v) for k, v in loaded.items())

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
        img = cv2.imread(str(path))
        if img is None:
            continue

        img_dcp = dcp_dehaze(img)
        cv2.imwrite(str(pre_dcp_dir / path.name), img_dcp)

        inp = Image.fromarray(cv2.cvtColor(img_dcp, cv2.COLOR_BGR2RGB))
        inp = T.ToTensor()(inp)
        inp = normalize(inp).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(inp)
            out = torch.clamp(out, 0.0, 1.0)
            out = out.squeeze(0)
            out = normalize(out).unsqueeze(0).to(device)
            out = model(out)
            out = torch.clamp(out, 0.0, 1.0)
            out = out.squeeze(0)
            out = normalize(out).unsqueeze(0).to(device)
            out = model(out).squeeze(0)

        out = out.detach().cpu()
        out = unnormalize(out)
        out = torch.clamp(out, 0.0, 1.0)
        out_img = (out.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        # Enhancement pipeline
        lab = cv2.cvtColor(out_img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        gamma = 1.15
        enhanced = np.power(enhanced / 255.0, gamma) * 255.0
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
        enhanced = cv2.addWeighted(enhanced, 1.6, gaussian, -0.6, 0)

        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        final_out = enhanced
        Image.fromarray(final_out).save(output_dir / path.name)
        print(f"Dehazed: {path.name} -> {output_dir / path.name}")

    return str(output_dir)


if __name__ == '__main__':
    source = ROOT / 'datasets' / 'SIR_IMAGES' / 'hazy'
    dehazed_output = ROOT / 'outputs' / 'SIR' / 'GridDehazeNet2'
    dehazed_output.mkdir(parents=True, exist_ok=True)

    print('Running GridDehazeNet + DCP dehaze...')
    source_dir = dehaze_dataset(source, dehazed_output, category='outdoor')

    print('Running YOLO on dehazed images...')
    yolo = YOLO('yolo11m.pt')
    yolo.predict(source=source_dir,
                 save=True,
                 project=str(ROOT / 'runs'),
                 name='SIR_GridDehazeNet2_yolo',
                 exist_ok=True,
                 save_txt=True,
                 save_conf=True)

    print('Done. Results in runs/SIR_GridDehazeNet2_yolo')