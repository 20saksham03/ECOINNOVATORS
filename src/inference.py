import os
import json
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# ------------------------
# DUMMY CLASSIFICATION MODEL (Replace later with trained model)
# ------------------------
class DummyClassifier(torch.nn.Module):
    def forward(self, x):
        # Fake output: random probability
        prob = torch.rand(1).item()
        return prob

# ------------------------
# DUMMY SEGMENTATION MODEL
# ------------------------
class DummySegmenter(torch.nn.Module):
    def forward(self, x):
        # Returns a fake segmentation mask
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Put a fake rectangle for demo
        cv2.rectangle(mask, (50,50), (200,200), 255, -1)
        return mask

# ------------------------
# RUN INFERENCE
# ------------------------
def run_inference(args):
    df = np.genfromtxt(args.input_csv, delimiter=',', dtype=str, skip_header=1)

    classifier = DummyClassifier()
    segmenter = DummySegmenter()

    os.makedirs(args.output_dir, exist_ok=True)
    mask_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)

    results = []

    for row in df:
        _id, lat, lon, img_name = row[0], row[1], row[2], row[3]
        img_path = os.path.join(args.img_root, img_name)

        # load image
        img = Image.open(img_path).convert("RGB")
        tensor = transforms.ToTensor()(img).unsqueeze(0)

        # classification
        prob = classifier(tensor)
        has_solar = int(prob > 0.5)

        # segmentation
        mask = segmenter(tensor)
        mask_path = os.path.join(mask_dir, f"{_id}_mask.png")
        cv2.imwrite(mask_path, mask)

        # estimation
        area_sqm = float(mask.sum() / 255) * 0.05   # dummy scaling
        capacity = area_sqm * 0.18

        result = {
            "id": _id,
            "lat": lat,
            "lon": lon,
            "has_solar": has_solar,
            "confidence": float(prob),
            "panel_count": int(area_sqm / 1.6) if has_solar else 0,
            "area_sqm": area_sqm,
            "capacity_kw": capacity,
            "qc_status": "VERIFIABLE",
            "reason_codes": ["dummy_model_output"],
            "mask_path": mask_path
        }
        results.append(result)

    # save json
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("[✔] Inference Completed!")
    print(f"[✔] Output saved to: {args.output_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--img_root", required=True)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    run_inference(args)
