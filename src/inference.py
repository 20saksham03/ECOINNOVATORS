import os
import json
import argparse
from typing import List, Dict

import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, models


# ------------------------
# CLASSIFIER MODEL (same as train_classifier.py)
# ------------------------

def get_classifier_model() -> nn.Module:
    """
    ResNet18 binary classifier (same architecture as train_classifier.py).
    Output: single logit -> use sigmoid for probability.
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


# ------------------------
# SEGMENTATION MODEL (same UNetSmall as in train_segmenter.py)
# ------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv1 = DoubleConv(64, 32)

        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)       # -> 32
        x2 = self.pool(x1)

        x2 = self.down2(x2)      # -> 64
        x3 = self.pool(x2)

        x3 = self.down3(x3)      # -> 128

        # Decoder
        x = self.up2(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv1(x)

        logits = self.out_conv(x)
        return logits


class DummySegmenter(nn.Module):
    """
    Fallback dummy segmenter if no seg_path is provided.
    """
    def forward(self, x: torch.Tensor) -> np.ndarray:
        _, _, h, w = x.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        h1, h2 = int(h * 0.2), int(h * 0.8)
        w1, w2 = int(w * 0.2), int(w * 0.8)
        cv2.rectangle(mask, (w1, h1), (w2, h2), 255, -1)
        return mask


# ------------------------
# IMAGE & PIPELINE UTILS
# ------------------------

def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("RGB")
    return img


def build_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])


def estimate_area_from_mask(mask: np.ndarray, sqm_per_pixel: float = 0.05) -> float:
    """
    Simple area estimation: count white pixels and multiply by a constant.
    """
    if mask is None:
        return 0.0
    panel_pixels = (mask > 0).sum()
    return float(panel_pixels) * sqm_per_pixel


def load_segmenter(seg_path: str, device: torch.device):
    """
    Load trained UNetSmall segmentation model if weights provided.
    """
    if seg_path and os.path.exists(seg_path):
        model = UNetSmall(in_channels=3, out_channels=1).to(device)
        state_dict = torch.load(seg_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"[INFO] Loaded segmenter weights from: {seg_path}")
        return model, True
    else:
        print("[WARN] No segmentation weights found or path not provided. Using DummySegmenter.")
        return DummySegmenter(), False


# ------------------------
# MAIN INFERENCE FUNCTION
# ------------------------

def run_inference(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load CSV
    df = pd.read_csv(args.input_csv)
    required_cols = ["id", "lat", "long"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column '{col}'")

    # Detect image column name
    img_col = None
    for candidate in ["rooftop_image", "image", "image_path"]:
        if candidate in df.columns:
            img_col = candidate
            break
    if img_col is None:
        raise ValueError("CSV must contain one of: 'rooftop_image', 'image', or 'image_path'")

    # ----- CLASSIFIER MODEL -----
    classifier = get_classifier_model().to(device)
    using_trained_cls = False

    if args.cls_path and os.path.exists(args.cls_path):
        state_dict = torch.load(args.cls_path, map_location=device)
        classifier.load_state_dict(state_dict)
        using_trained_cls = True
        print(f"[INFO] Loaded classifier weights from: {args.cls_path}")
    else:
        print("[WARN] No classifier weights found or path not provided. Using untrained classifier.")

    classifier.eval()

    # ----- SEGMENTER MODEL -----
    segmenter, using_trained_seg = load_segmenter(args.seg_path, device)

    # ----- OUTPUT SETUP -----
    os.makedirs(args.output_dir, exist_ok=True)
    mask_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    transform = build_transform()
    results: List[Dict] = []

    for idx, row in df.iterrows():
        _id = row["id"]
        lat = row["lat"]
        lon = row["long"]
        img_name = row[img_col]

        img_path = os.path.join(args.img_root, str(img_name))
        print(f"[INFO] Processing ID={_id}, image={img_path}")

        try:
            img = load_image(img_path)
        except Exception as e:
            print(f"[ERROR] Failed to load image for ID={_id}: {e}")
            result = {
                "id": _id,
                "lat": lat,
                "lon": lon,
                "has_solar": 0,
                "confidence": 0.0,
                "panel_count": 0,
                "area_sqm": 0.0,
                "capacity_kw": 0.0,
                "qc_status": "NOT_VERIFIABLE",
                "reason_codes": ["image_load_failed"],
                "mask_path": None,
            }
            results.append(result)
            continue

        # Preprocess
        tensor = transform(img).unsqueeze(0).to(device)

        # ---- CLASSIFICATION ----
        with torch.no_grad():
            logit = classifier(tensor)  # (1,1)
            prob = torch.sigmoid(logit).item()

        has_solar = int(prob >= args.threshold)

        # ---- SEGMENTATION ----
        if isinstance(segmenter, DummySegmenter):
            mask = segmenter(tensor)
        else:
            with torch.no_grad():
                logits = segmenter(tensor)          # (1,1,H,W)
                prob_mask = torch.sigmoid(logits)   # (1,1,H,W)
                bin_mask = (prob_mask > 0.5).float()
                mask = bin_mask.squeeze().cpu().numpy().astype(np.uint8) * 255

        mask_filename = f"{_id}_mask.png"
        mask_path = os.path.join(mask_dir, mask_filename)
        cv2.imwrite(mask_path, mask)

        # ---- QUANTIFICATION ----
        area_sqm = estimate_area_from_mask(mask, sqm_per_pixel=args.sqm_per_pixel) if has_solar else 0.0
        capacity_kw = area_sqm * args.wp_per_sqm if has_solar else 0.0
        panel_count = int(area_sqm / args.avg_panel_area_sqm) if has_solar and args.avg_panel_area_sqm > 0 else 0

        # ---- QC & REASON CODES ----
        reason_codes = []
        if using_trained_cls:
            reason_codes.append("using_trained_classifier")
        else:
            reason_codes.append("untrained_classifier")

        if using_trained_seg:
            reason_codes.append("using_trained_segmenter")
        else:
            reason_codes.append("dummy_segmenter")

        if has_solar:
            reason_codes.append("positive_prediction")
        else:
            reason_codes.append("negative_prediction")

        if 0.4 < prob < 0.6:
            qc_status = "NOT_VERIFIABLE"
            reason_codes.append("low_confidence")
        else:
            qc_status = "VERIFIABLE"
            reason_codes.append("confidence_ok")

        result = {
            "id": _id,
            "lat": float(lat),
            "lon": float(lon),
            "has_solar": has_solar,
            "confidence": float(prob),
            "panel_count": panel_count,
            "area_sqm": float(area_sqm),
            "capacity_kw": float(capacity_kw),
            "qc_status": qc_status,
            "reason_codes": reason_codes,
            "mask_path": mask_path,
        }
        results.append(result)

    # Save JSON
    output_json_path = os.path.join(args.output_dir, "results.json")
    with open(output_json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[✔] Inference completed. Results saved to: {output_json_path}")


# ------------------------
# CLI ENTRYPOINT
# ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rooftop Solar Verification Inference Pipeline")
    parser.add_argument("--input_csv", required=True, help="Path to test CSV (with id, lat, long, rooftop_image)")
    parser.add_argument("--img_root", required=True, help="Root folder where images are stored")
    parser.add_argument("--output_dir", default="outputs", help="Output folder for JSON + masks")

    parser.add_argument("--cls_path", default=None, help="Path to trained classifier weights (.pth)")
    parser.add_argument("--seg_path", default=None, help="Path to trained segmenter weights (.pth)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for has_solar=1")

    # Quantification hyperparameters
    parser.add_argument("--sqm_per_pixel", type=float, default=0.05,
                        help="Estimated square meters represented by one mask pixel")
    parser.add_argument("--wp_per_sqm", type=float, default=0.18,
                        help="Capacity in kW per m^2 (e.g., 0.18 kW/m² ~ 180 Wp/m²)")
    parser.add_argument("--avg_panel_area_sqm", type=float, default=1.6,
                        help="Average area of a single panel in m² (for estimating panel_count)")

    args = parser.parse_args()
    run_inference(args)
