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
# MODEL DEFINITIONS
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


class DummySegmenter(nn.Module):
    """
    Very simple dummy segmenter.
    You can later replace this with a trained segmentation model.
    """
    def forward(self, x: torch.Tensor) -> np.ndarray:
        # Input x: (1, C, H, W)
        _, _, h, w = x.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        # Draw a central rectangle as a fake PV region
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
    sqm_per_pixel is a placeholder; with geo-referenced imagery you can improve this.
    """
    if mask is None:
        return 0.0
    panel_pixels = (mask > 0).sum()
    return float(panel_pixels) * sqm_per_pixel


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
    using_trained_weights = False

    if args.cls_path and os.path.exists(args.cls_path):
        state_dict = torch.load(args.cls_path, map_location=device)
        classifier.load_state_dict(state_dict)
        using_trained_weights = True
        print(f"[INFO] Loaded classifier weights from: {args.cls_path}")
    else:
        print("[WARN] No classifier weights found or path not provided. Using untrained model.")

    classifier.eval()

    # ----- SEGMENTER (currently dummy) -----
    segmenter = DummySegmenter()

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

        # ---- SEGMENTATION (dummy) ----
        mask = segmenter(tensor)
        mask_filename = f"{_id}_mask.png"
        mask_path = os.path.join(mask_dir, mask_filename)
        cv2.imwrite(mask_path, mask)

        # ---- QUANTIFICATION ----
        area_sqm = estimate_area_from_mask(mask, sqm_per_pixel=args.sqm_per_pixel) if has_solar else 0.0
        capacity_kw = area_sqm * args.wp_per_sqm if has_solar else 0.0
        panel_count = int(area_sqm / args.avg_panel_area_sqm) if has_solar and args.avg_panel_area_sqm > 0 else 0

        # ---- QC & REASON CODES ----
        reason_codes = []
        if using_trained_weights:
            reason_codes.append("using_trained_classifier")
        else:
            reason_codes.append("untrained_classifier")

        if has_solar:
            reason_codes.append("positive_prediction")
        else:
            reason_codes.append("negative_prediction")

        if prob < 0.6 and prob > 0.4:
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
