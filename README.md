# 🍎 FruitVision – Deep Learning Fruit Quality Grading System

> **Deep Learning-Based Fruit Quality Grading & Chemical Adulteration Awareness**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

**FruitVision** is an end-to-end computer vision application for fruit quality grading and chemical adulteration awareness. It classifies fruit images into **15 classes** (5 fruit types × 3 conditions: Fresh, Rotten, Formalin-Mixed) using three trained deep learning models with Grad-CAM explainability.

### Key Highlights
- 🔬 **15-class classification** across 5 fruits (Apple, Banana, Grape, Mango, Orange) × 3 conditions
- 🧠 **Three model architectures** compared under one unified inference framework
- 🗺️ **Grad-CAM explainability** — visual attention maps per prediction
- ⚡ **FastAPI backend** with real-time `/api/predict` endpoint
- 🌐 **Browser UI** supporting file upload and live webcam capture
- 🔍 **Path W attention loss** — models learn to focus on the fruit, not the background

---

## 🧑‍💻 Team

| Name | Roll Number |
|---|---|
| Brahmajosyula Gowtham | 21BCE8851 |
| Ramana Boina Venkata Uday Kumar | 22BCE20026 |
| Mudduluru Charith Varma | 22BCE7339 |

**Institution:** SCOPE, VIT-AP University, Amravati, India  
**SDP ID:** 20250770

---

## 🏗️ System Architecture

```
User Browser (Upload / Webcam)
        ↓
Frontend UI (HTML + JS + TailwindCSS)
        ↓
FastAPI Backend — POST /api/predict
        ↓
Preprocess (256×256 + ImageNet Norm)
        ↓
┌──────────────────────────────────┐
│  Model A: EffNet+SwinT (96.85%)  │
│  Model B: DBA-ViNet   (88.96%)   │
│  Model C: MobileNetV4 (94.98%)   │
└──────────────────────────────────┘
        ↓
Grad-CAM Generation → Confidence Rules
        ↓
JSON Response (predictions + heatmaps)
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Status check |
| `/api/models` | GET | Model metadata |
| `/api/predict` | POST | Multipart image → JSON + Grad-CAM |
| `/` | Static | Frontend HTML/JS/CSS |

---

## 🤖 Model Architectures

### Model A — EffNet+SwinT *(Highest Accuracy)*
- **Params:** 48.8M | **Test Accuracy:** 96.85% | **Focus Lift:** 1.46×
- Parallel late-fusion of EfficientNetV2-S (CNN) and SwinV2-Tiny (Transformer)
- EfficientNet captures local texture; Swin captures global shape & spatial relations
- Best for: **cloud / server deployment**

### Model B — DBA-ViNet *(Best Explainability)*
- **Params:** 12.0M | **Test Accuracy:** 88.96% | **Focus Lift:** 2.82×
- Dual-branch attention architecture with cross-attention fusion
- Smallest model; attention gate directly learns to focus on the fruit region
- Best for: **audit / explainability scenarios and edge deployment**

### Model C — MobileNetV4 *(Recommended — Best Pareto Balance)*
- **Params:** 37.0M | **Test Accuracy:** 94.98% | **Focus Lift:** 4.62×
- UIB + Mobile MQA hybrid backbone (NAS-discovered)
- Highest focus lift of all 3 models; best accuracy × interpretability × size trade-off
- Best for: **mobile / edge deployment**

---

## 📊 Results Summary

| Model | Params (M) | Test Accuracy | Focus Lift | Inference Role |
|---|---|---|---|---|
| EffNet+SwinT (A) | 48.8 | **96.85%** | 1.46× | Highest accuracy server model |
| DBA-ViNet (B) | 12.0 | 88.96% | **2.82×** | Compact explainability-oriented |
| MobileNetV4 (C) | 37.0 | 94.98% | **4.62×** | Best Pareto balance ✅ recommended |

---

## 🗂️ Dataset

| Fruit | Formalin-Mixed | Fresh | Rotten | Total |
|---|---|---|---|---|
| Apple | 643 | 765 | 630 | 2,038 |
| Banana | 660 | 749 | 632 | 2,041 |
| Grape | 610 | 770 | 630 | 2,010 |
| Mango | 616 | 763 | 630 | 2,009 |
| Orange | 647 | 753 | 656 | 2,056 |
| **Total** | **3,176** | **3,800** | **3,178** | **10,154** |

- **After mask-quality filter (5–80% area):** 9,727 images
- **Resolution:** 1536×2040 px (raw JPEG, smartphone-captured)
- **Split:** Train 6,632 (68.2%) | Val 1,601 (16.5%) | Test 1,494 (15.4%)
- **Group-aware split** using pHash16 perceptual hashing + Union-Find clustering to prevent data leakage

---

## 🔧 Data Pipeline

1. **Stage 1 — MobileSAM Fruit Segmentation:** 10,154 raw images → binary masks (512×512 PNG) → 9,727 passed area filter
2. **Stage 2 — Indoor Background Dataset:** 882 background images from HuggingFace (kitchen, pantry, restaurant, etc.) for augmentation
3. **Stage 3 — Group-Aware Split:** pHash16 → Union-Find clustering → GroupShuffleSplit (zero group overlap across train/val/test confirmed)

---

## ⚙️ Training Details

| Hyperparameter | Value |
|---|---|
| Epochs | 30 |
| Learning Rate | 3 × 10⁻⁴ |
| Optimizer | AdamW |
| Weight Decay | 0.05 |
| Label Smoothing | 0.1 |
| Scheduler | CosineAnnealingLR |
| Resolution | 256 × 256 |
| Precision | Mixed (AMP + GradScaler) |
| Seed | 42 |

### Path W Attention Loss
```
L = CE + λ × Σ(cam_normalized × bg_mask)
```
- λ (ATTN_LAMBDA) = 0.05; penalizes CAM energy outside MobileSAM mask
- Models are guided to focus attention on the fruit region during training

### Augmentation Pipeline
- Background replacement (p=0.95), Blackout (p=0.20), Mask erosion (p=0.30)
- Localized CLAHE (p=0.70), Spatial flip/rotate/crop (Kornia), Color jitter
- CutMix + MixUp (50/50 per batch), Random erasing, ImageNet normalization

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Core** | Python 3.x, PyTorch 2.5.1, timm 1.0.26, Kornia 0.8.2 |
| **Backend** | FastAPI + Uvicorn, pytorch-grad-cam |
| **Data** | MobileSAM (vit_t), HuggingFace Datasets 2.21, pandas |
| **Training** | AdamW + CosineAnnealingLR, torch.amp (mixed precision), matplotlib + seaborn |
| **Frontend** | HTML5 / CSS3 / JS, TailwindCSS (CDN), Browser MediaDevices API |
| **Hardware** | NVIDIA RTX 3070 Laptop (8.6 GB VRAM) |

---

## 📁 Project Structure

```
FruitVision/
├── main.py                    # FastAPI app lifecycle & model loading
├── inference.py               # Preprocessing, multi-model inference, Grad-CAM
├── models.py                  # EffNetSwinT, DBAViNet, MobileNetV4 definitions
├── checkpoints/
│   ├── model_a_best.pt
│   ├── model_b_best.pt
│   └── model_c_best.pt
├── frontend/
│   ├── predict.html           # Upload/webcam prediction dashboard
│   ├── analysis.html          # Metric summaries & chart analysis
│   ├── methodology.html       # Methodology narrative
│   └── assets/
│       └── analysis/          # Analysis figures
├── splits/
│   └── label_map.json
└── start_demo.bat             # Windows batch startup
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.x
CUDA-compatible GPU (recommended) or CPU
```

### Installation
```bash
git clone https://github.com/your-username/FruitVision.git
cd FruitVision
pip install -r requirements.txt
```

### Run the Server
```bash
# Windows
start_demo.bat

# Or directly
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Access the App
Open your browser at: `http://localhost:8000`

---

## 🔍 Confidence Display Logic

| Confidence | Display |
|---|---|
| top-1 < 0.35 | "Object Not Detected" |
| 0.35 ≤ top-1 < 0.75 | Show top-1 and top-2 predictions |
| top-1 ≥ 0.75 | Show only top-1 prediction |

---

## ⚠️ Limitations

- Single-node, local-first deployment only
- Dataset acquired in controlled indoor conditions only
- Explainability quality depends on chosen CAM target layers
- DBA-ViNet attention gate is architecture-specific, not generalizable

---

## 🔭 Future Work

- Model ensembling / accuracy-weighted stacking
- Uncertainty quantification (temperature scaling, conformal prediction)
- Expand dataset with outdoor / varied lighting conditions
- MLOps pipeline: experiment tracking, model registry, CI validation
- Quantization / pruning for stronger edge-device deployment
- Multi-task learning: ripeness + defect + size jointly
- Active-learning loop from low-confidence real-world samples

---

## 📚 Key References

1. Zhang et al., "Vision Transformer for automated mango classification," *Pattern Recognition*, 2024
2. Moretti et al., "MobileNetV4-based fruit ripeness assessment," *IEEE Trans. Instrum. Meas.*, 2024
3. Selvaraju et al., "Grad-CAM: Visual explanations via gradient-based localization," *ICCV 2017*
4. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," *ICCV 2021*
5. Li et al. (Google), "MobileNetV4 – Universal models for the mobile ecosystem," *ECCV 2024*
6. Parmar et al., "Formalin-treated produce detection using neural networks," *J. Food Quality & Safety*, 2023

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ at <strong>VIT-AP University, Amravati, India</strong>
</p>
