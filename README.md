# Leakage-Aware, Class-Balanced, and Explainable EfficientNet-B0-CBAM Framework for Multi-Class Skin Lesion Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Status](https://img.shields.io/badge/Status-Completed-success.svg)](#)

An end-to-end, reproducible deep learning framework for seven-class dermatological screening on the **HAM10000** dataset. This project addresses critical clinical challenges in medical computer vision: **data leakage prevention** via strict lesion-level stratification, **severe class imbalance mitigation** via effective-number weighted Focal Loss, **spatial/channel salience** via Convolutional Block Attention Modules (CBAM), **visual explainability** via Grad-CAM, and **out-of-distribution filtering** via a 3-way neural gatekeeper and clinical ABCDE feature analysis.

> **Academic Safety Notice:** This software is an academic research prototype and decision-support demonstration. It is not certified as a medical diagnostic device and must not be used for autonomous clinical diagnosis.

---

## Key Highlights

- **Leakage-Free Partitioning:** Unlike standard random train/test splits that leak multiple images of the same lesion across partitions, this pipeline uses a two-stage `StratifiedGroupKFold` split grouped strictly by `lesion_id` (7,152 train / 1,432 val / 1,431 test).
- **Comprehensive Ablation Suite:** Systematically isolates each proposed module across 5 controlled experiments using identical seeds (`42`), image resolutions (`224x224`), and evaluation protocols.
- **Explainability by Design:** Integrates Grad-CAM on top convolutional and attention layers to provide visual saliency heatmaps highlighting discriminative pigment networks.
- **Multi-Modal Clinical Safeguards:** Fuses algorithmic ABCDE dermatological metrics (Asymmetry, Border, Color variegation, Diameter) with a 3-way Neural Gatekeeper (Lesion Skin vs Normal Healthy Skin vs Non-Skin) to prevent out-of-distribution errors and false negatives on high-risk lesions.
- **Interactive Clinical Studio:** A complete full-stack web application (`app/server.py`) offering real-time camera inference, model ensemble consensus, ABCDE scoring cards, and printable PDF diagnostic reports.

---

## Ablation Study & Benchmark Results

All models were evaluated on the exact same untouched lesion-grouped test partition (1,431 images):

| Model ID | Architecture / Configuration | Purpose / Mechanism | Test Accuracy | Macro-F1 | Macro Recall | Macro AUC (OvR) |
|---|---|---|:---:|:---:|:---:|:---:|
| **Model A** | Baseline EfficientNet-B0 + Cross-Entropy | Standard pretrained baseline reference | 83.44% | 64.07% | 61.57% | 0.9508 |
| **Model B** | Model A + Minority Targeted Augmentation | Isolates geometric & color jitter on minority classes | 77.92% | 53.04% | 47.07% | 0.9401 |
| **Model C** | Model B + Effective-Number Focal Loss ($\gamma=2.0$) | Isolates class-imbalance loss re-weighting | 77.71% | **57.42%** | **54.73%** | 0.9434 |
| **Model D** | Model B + CBAM Attention | Isolates channel and spatial attention | 77.36% | **55.53%** | 49.05% | 0.9324 |
| **Model E** | Full Proposed (CBAM + Focal Loss + Augmentation) | Joint activation of all proposed components | 74.00% | 49.09% | 44.02% | 0.9275 |
| **Ensemble** | **Validation-Calibrated Multi-Model Consensus** | **Optimal multi-architecture ensemble** | **84.35%** | **67.03%** | **64.20%** | **0.9650** |

### Research Takeaways
1. **Focal Loss Benefits (C vs B):** Imposing effective-number inverse class weights with Focal Loss produced a **+4.38% increase in Macro-F1** and a **+7.66% increase in Macro Recall/Sensitivity**, significantly improving sensitivity on rare classes like `mel`, `bcc`, and `akiec`.
2. **CBAM Attention Benefits (D vs B):** The dual-attention mechanism yielded a **+2.49% lift in Macro-F1** and increased Macro Precision to **75.84%** (+4.72% over B) by suppressing irrelevant peripheral skin regions.
3. **Hyperparameter Coupling Insight (E vs D/C):** Simultaneous activation of aggressive class-reweighting with spatial attention under fixed learning rates highlights the clinical AI challenge of gradient competition, providing valuable justification for staged learning rate warm-ups or ensemble aggregation.
4. **Ensemble Consensus:** Uniting the models achieves the framework's highest benchmark accuracy (**84.35%**) and peak discrimination ability (**0.9650 AUC**).

---

## System Architecture

```
skinlesionclassy/
├── app/                  # Interactive Clinical Web Studio
│   ├── server.py         # Multi-threaded HTTP backend with inference & Grad-CAM endpoints
│   ├── index.html        # Modern dashboard UI with live camera & report export
│   ├── app.js            # Frontend application logic & chart rendering
│   └── styles.css        # Responsive styling & themes
├── checkpoints/          # Saved best model weights (.pt) for all ablations & gatekeepers
├── configs/              # Reproducible YAML experiment configurations
│   ├── baseline_efficientnet_b0.yaml
│   ├── ablation_b_targeted_augmentation.yaml
│   ├── ablation_c_focal_loss.yaml
│   ├── ablation_d_cbam.yaml
│   └── proposed_efficientnet_b0_cbam_focal.yaml
├── data/
│   ├── raw/              # Official HAM10000 images and metadata (CC BY-NC 4.0)
│   ├── splits/           # Lesion-grouped train/val/test CSV splits and split_summary.json
│   └── skin_gate_3class/ # Dataset for lesion vs normal skin vs non-skin gatekeeper
├── docs/                 # Research protocols, ablation plans, and technical documentation
│   ├── dataset_protocol.md
│   ├── ablation_plan.md
│   └── cbam_and_focal_loss.md
├── figures/              # Generated Grad-CAM comparison galleries and figures
├── results/              # Per-experiment CSV metrics, confusion matrices, and ROC plots
├── scripts/              # Validation, calibration, and ensemble evaluation scripts
└── src/                  # Modular PyTorch framework modules
    ├── audit_ham10000.py # Metadata verification and sanity audit
    ├── cbam.py           # PyTorch CBAM (Channel + Spatial Attention) module
    ├── constants.py      # Class mappings and shared global constants
    ├── create_splits.py  # StratifiedGroupKFold lesion-based partitioner
    ├── dataset.py        # PyTorch Dataset with targeted augmentation policies
    ├── evaluate.py       # Metrics engine (Accuracy, F1, Recall, AUC, Confusion Matrix)
    ├── gradcam.py        # Native Grad-CAM hook and heatmap generator
    ├── losses.py         # Multi-class Focal Loss with effective-number weighting
    ├── model.py          # EfficientNet-B0 and CBAM network definitions
    ├── preprocessing.py  # Image transformations and deterministic evaluation pipelines
    ├── train.py          # Unified training engine with early stopping and checkpointing
    └── train_skin_gate_3class.py # 3-way neural gatekeeper training script
```

---

## Getting Started

### 1. Environment Setup

Clone the repository and install dependencies in a virtual environment:

```bash
git clone https://github.com/your-username/skinlesionclassy.git
cd skinlesionclassy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Setup & Leakage-Free Splitting

Download the official HAM10000 archive (`HAM10000_images_part_1.zip`, `HAM10000_images_part_2.zip`, `HAM10000_metadata.tab`) into `data/raw/`, then verify metadata and generate the splits:

```bash
python -m src.audit_ham10000
python -m src.create_splits
```

### 3. Training & Evaluation

To train any of the versioned ablation models:

```bash
# Example: Train Model D (EfficientNet-B0 + CBAM Attention)
python -m src.train --config configs/ablation_d_cbam.yaml

# Evaluate on untouched test split
python -m src.evaluate --config configs/ablation_d_cbam.yaml --checkpoint checkpoints/ablation_d_cbam_targeted_augmentation_cross_entropy/best_model.pt
```

To evaluate the calibrated multi-model ensemble:

```bash
python scripts/evaluate_ensemble.py
```

### 4. Running the Interactive Clinical Diagnostic Web Studio

Launch the full-featured web dashboard:

```bash
python -m app.server --port 8080
```

Open your browser to `http://localhost:8080` to access:
- **Live Camera Capture:** Capture real-time images directly from desktop webcam or mobile phone camera.
- **Benchmark Test Samples:** Inspect representative images from the untouched test set across all 7 diagnostic classes (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`).
- **Interactive Grad-CAM:** Toggle visual attention heatmaps indicating where the network focused.
- **Clinical ABCDE Analysis:** View real-time quantitative scores for Asymmetry, Border Irregularity, Color Variegation, and Diameter.
- **Neural Gatekeeper:** Automatic rejection of non-skin photos and recognition of normal healthy skin.
- **Export Diagnostic Report:** One-click generation of printable clinical summary reports.

---

## Diagnostic Classes Covered

| Code | Clinical Diagnosis | Type / Severity | Description |
|---|---|---|---|
| `akiec` | Actinic Keratosis / Bowen's Disease | Precancerous | Rough solar-induced patch; potential precursor to squamous carcinoma |
| `bcc` | Basal Cell Carcinoma | Malignant | Common, slow-growing epidermal cancer; locally destructive |
| `bkl` | Benign Keratosis (Seborrheic Keratoses) | Benign | Non-cancerous "pasted-on" warty or pigmented skin growth |
| `df` | Dermatofibroma | Benign | Firm cutaneous nodule, commonly triggered by minor trauma |
| `mel` | Melanoma | Malignant (Critical) | Highly aggressive malignant transformation of melanocytes |
| `nv` | Melanocytic Nevi (Common Moles) | Benign | Typical harmless cluster of pigment cells |
| `vasc` | Vascular Lesions (Angiomas, Hemangiomas) | Benign | Benign proliferation of blood vessels in dermis |

---

## Citation & Dataset Attribution

The dataset used in this study is the HAM10000 collection provided by Harvard Dataverse:
> Tschandl, P., Rosendahl, C., & Kittler, H. (2018). *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161. DOI: [10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161). Distributed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
