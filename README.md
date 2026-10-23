# 🌊 AquaYOLO: High-Accuracy Underwater Marine Object Detection

AquaYOLO is a deep learning object detection framework tailored for underwater marine environments. It addresses low-contrast, chromatic distortion, and turbidity challenges in underwater vision through specialized physical-domain preprocessing and spectral-contrast attention mechanisms.

---

## 📌 Project Highlights
- **Specialized Attention Modules**:
  - **SAC (Spectral Attenuation Compensator)**: Channel-wise attention mitigating wavelength-dependent underwater light absorption.
  - **TCG (Turbidity Contrast Gate)**: Spatial gating network suppressing background water haze and turbidity.
- **High-Accuracy YOLO Pipeline (`train_yolo.py`)**:
  - Marine and COCO transfer learning with mosaic augmentation and cosine learning rate scheduling.
  - Generates comprehensive diagnostic and publication-ready metric graphs (`results.png`, `confusion_matrix.png`, `PR_curve.png`, `F1_curve.png`).
- **Google Colab Ready**:
  - Run end-to-end training and evaluation with GPU acceleration using `AquaYOLO_Colab_Training_and_Evaluation.ipynb`.
  - Connects directly to Google Drive for large dataset storage.

---

## 🚀 Quick Start in Google Colab

1. Open [`AquaYOLO_Colab_Training_and_Evaluation.ipynb`](AquaYOLO_Colab_Training_and_Evaluation.ipynb) in Google Colab.
2. Select **Runtime -> Change runtime type -> T4 GPU**.
3. Mount Google Drive and point `DRIVE_DATASET_SOURCE` to your dataset folder or `.zip` file in Google Drive.
4. Run all cells:
   - Clones repository
   - Prepares dataset
   - Trains high-accuracy detector
   - **Displays all metric graphs directly inside the notebook output**:
     - `results.png` (Training & Validation Losses, Precision, Recall, mAP@0.50, mAP@0.50:0.95)
     - `confusion_matrix.png` & `confusion_matrix_normalized.png`
     - `PR_curve.png` (Precision-Recall curve)
     - `F1_curve.png` (F1 score vs confidence)
     - `val_batch0_pred.jpg` (Visual prediction samples)

---

## 💻 Local Setup & Training

### 1. Requirements
```bash
pip install torch torchvision ultralytics opencv-python matplotlib seaborn pyyaml pandas
```

### 2. High-Accuracy Training
```bash
python train_yolo.py \
    --data ./data/AquaYOLO_26class_balanced/data.yaml \
    --model yolov8s.pt \
    --epochs 50 \
    --batch 16 \
    --imgsz 640 \
    --project ./runs/yolo_train \
    --name aquayolo_high_acc
```

### 3. Custom AquaYOLO Architecture Training
To train the custom PyTorch AquaYOLO model (with SAC and TCG attention):
```bash
python src/aquayolo/train.py \
    --data ./data/AquaYOLO_26class_balanced \
    --epochs 30 \
    --batch 4 \
    --size 640 \
    --lr 0.0001 \
    --out ./runs/aquayolo_custom
```

### 4. Evaluation
```bash
python src/aquayolo/evaluate.py \
    --data ./data/AquaYOLO_26class_balanced \
    --weights ./runs/aquayolo_custom/best.pt \
    --split val \
    --conf 0.001 \
    --out ./runs/evaluation
```

---

## 📈 Accuracy Optimizations & Fixes Applied

In earlier baseline attempts, results suffered from low accuracy / mAP due to:
1. **Excessive Hard-Negative Mining**: Previous loss applied a 1.5x penalty to the top 10% background cells (~640 cells per image), drowning out positive object gradients. This was corrected with an **OHEM 3:1 negative-to-positive ratio** to maintain strong positive recall.
2. **Evaluation Confidence Threshold**: Default evaluation previously used `conf=0.10` on softmax-distributed multi-class probabilities, filtering out valid detections prior to PR integration. Standard `conf=0.001` is now used for standard mAP integration.
3. **Transfer Learning**: Backbone initialization from pretrained weights (`yolov8s.pt` / `FishInv.pt`) prevents underfitting on small underwater datasets.
4. **Learning Rate & Augmentation**: Linear warmup with Cosine Annealing decay and mosaic/color jitter augmentations ensure robust generalization.

---

## 🐠 Supported Marine Classes (26 Categories)
1. `jellyfish`
2. `penguin`
3. `puffin`
4. `starfish`
5. `stingray`
6. `basking_shark`
7. `blacktip_reef_shark`
8. `great_white_shark`
9. `hammerhead_shark`
10. `nurse_shark`
11. `oceanic_whitetip_shark`
12. `thresher_shark`
13. `whale_shark`
14. `whitetip_reef_shark`
15. `ballan_wrasse`
16. `cuckoo_wrasse`
17. `corkwing_wrasse`
18. `goldsinny_wrasse`
19. `rock_cook`
20. `atlantic_cod`
21. `pollack`
22. `whiting`
23. `spiny_dogfish`
24. `brown_crab`
25. `two_spot_goby`
26. `human`
