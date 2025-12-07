# ECOINNOVATORS – Rooftop Solar Verification System

This repository contains the prototype submission for the  
**Global Learning Council – EcoInnovators Ideathon 2026**,  
tasked with verifying rooftop solar installations under the  
**PM Surya Ghar: Muft Bijli Yojana**.

---

## 🚀 Features

- AI-powered classification of rooftop solar presence  
- Segmentation-based estimation of area (m²) & capacity (kW)  
- Explainable outputs: masks, reason codes, QC flags  
- Complete Docker-based reproducible pipeline  
- JSON output ready for audit & governance workflows  

---

## 🔧 How to Run (Local)

```bash
python src/inference.py \
  --input_csv data/test_rooftop_data.csv \
  --img_root data/images \
  --output_dir outputs \
  --cls_path weights/trained/classifier_best.pth \
  --seg_path weights/trained/segmenter_best.pth
```

---

## 🐳 How to Run (Docker)

```bash
docker run --rm \
  -v /path/to/data:/app/data \
  -v /path/to/outputs:/app/outputs \
  20saksham03/solar-verifier:latest \
  --input_csv data/test_rooftop_data.csv \
  --img_root data/images \
  --output_dir outputs
```

---

## 📦 Contents

- `src/` → inference + model code  
- `tools/` → validator + synthetic demo generator  
- `weights/` → trained model weights (placeholder)  
- `model_card.md` → responsible AI documentation  
- `Dockerfile` → reproducible environment  

---

## 📝 Model Card

See `model_card.md` (required for submission).

---

## 👤 Author

**Saksham Bansal**  
Team ECOINNOVATORS
