Model Card – PM Surya Ghar Rooftop Solar Verification System
1. Model Overview

This AI system is designed to verify rooftop solar installations under the PM Surya Ghar: Muft Bijli Yojana.
It supports government agencies and DISCOMs by providing a remote, low-cost, auditable verification pipeline.

Core Tasks
Task	Description
Solar Presence Classification	Predict whether a rooftop solar PV system exists at a given (lat, lon). Binary: 0/1
PV Quantification	If solar is present → estimate panel count, PV area (m²), and possible capacity (kW)
Explainability	Generate mask/bounding region, confidence score, and reason codes
QC Tagging	Mark result as VERIFIABLE or NOT_VERIFIABLE, supporting audit compliance
2. Model Architecture Summary
2.1 Classifier – Has Solar (0/1)

Backbone: ResNet-18

Input: 256×256 RGB rooftop patch

Output: single probability (sigmoid output)

Loss: Binary Cross Entropy With Logits

Purpose: Detect whether solar panels are visible on the roof.

2.2 Segmentation Model

Architecture: Lightweight UNet (or placeholder dummy model if trained segmentation unavailable)

Purpose: Estimate the approximate PV region → used for:

Area estimation

Panel count heuristics

Explainability masks

3. Training Details
3.1 Datasets Used

This prototype currently uses synthetic rooftop samples for pipeline testing.
Once organisers release official training data on 17 Nov 2025, classifier and segmenter will be trained on:

High-resolution rooftop images

Provided labels:

has_solar

panel_count

area_sqm

3.2 Preprocessing

Resize images to 256×256

Normalize pixel values

Handle missing/invalid images gracefully

Coarse roof search around GPS coordinates (±10–20 m) when needed

3.3 Training Setup

Optimizer: Adam

LR: 1e-4

Epochs: 10–20

Train/Val split: 80/20

Metrics tracked:

F1 Score for solar detection

MAE for area estimation

RMSE for capacity estimation

4. Evaluation Strategy

The following metrics will be used (as required by the ideathon):

1. Solar Detection Accuracy

Metric: F1 Score

Meaning: Harmonic mean of precision and recall

Why: Balanced measure for imbalanced classes

2. Quantification Quality

MAE for area (m²)

RMSE for capacity (kW)

3. Generalization & Robustness

Model is evaluated across:

Various rooftop types (sloped, flat, tiled, concrete)

Different lighting/seasonal conditions

Occlusions (tanks, trees, shadows)

4. Explainability & Auditability

Each prediction includes:

Confidence score

Reason codes

QC tag

Mask PNG file

5. Intended Use
Primary Use Case

To assist DISCOM engineers, auditors, and government agencies in verifying rooftop solar installations remotely.

The model should be used to:

Check whether solar PV exists at a location

Assist subsidy approval workflows

Support audits and fraud detection

Speed up verification where field visits are slow/costly

6. Limitations of the Model

Satellite image quality varies across regions and vendors

Dense shadows, clouds, tree cover may cause NOT_VERIFIABLE statuses

Synthetic training data used for prototype — real performance depends on actual dataset

Segmenter performance limited if no annotated masks are available

Older imagery may not match real-time installation status

Small installations (<2 panels) may be harder to detect

7. Ethical Considerations
✔ Privacy

No personally identifiable information (PII) is used

Only rooftop imagery is analyzed

Respect for image licensing (only permissible sources allowed)

✔ Bias Transparency

Possible performance variations across:

Urban vs rural roof types

Different camera perspectives

Minority roof materials (tin, blue sheet, slanted tile)

Documented mitigations include:

Training on diverse samples

QC flags when imagery is insufficient

✔ Responsible AI

System outputs VERIFIABLE / NOT_VERIFIABLE instead of forcing a prediction

Reason codes provide transparency to auditors

The system avoids high-risk autonomous decision making

8. How to Run the Model
Local Execution
python src/inference.py \
  --input_csv data/test_rooftop_data.csv \
  --img_root data/images \
  --output_dir outputs \
  --cls_path weights/trained/classifier_best.pth \
  --seg_path weights/trained/segmenter_best.pth

Docker Execution
docker run --rm \
  -v /path/to/data:/app/data \
  -v /path/to/outputs:/app/outputs \
  <dockerhub-username>/solar-verifier:latest \
  --input_csv data/test_rooftop_data.csv \
  --img_root data/images \
  --output_dir outputs

9. Outputs

Each row produces a JSON object:

{
  "id": "1234",
  "lat": 28.561,
  "lon": 77.234,
  "has_solar": 1,
  "confidence": 0.92,
  "panel_count": 12,
  "area_sqm": 18.5,
  "capacity_kw": 3.0,
  "qc_status": "VERIFIABLE",
  "reason_codes": ["grid_pattern", "high_confidence"],
  "mask_path": "outputs/masks/1234_mask.png"
}


This JSON is included in results.json.

10. Future Improvements

Train with full-scale real dataset

Improve segmentation using roof-outline detection

Add transformer-based rooftop classifiers

Remove GIS noise via coordinate snapping

Integrate temporal checks using historical imagery
