# Traffic Accident Detection from CCTV Footage via Generative Augmentation & FTCNN-ViT Hybrid Feature Fusion

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![Diffusers](https://img.shields.io/badge/🤗%20Diffusers-0.20%2B-yellow.svg)](https://huggingface.co/docs/diffusers)
[![Hardware](https://img.shields.io/badge/Hardware-Kaggle%20T4%20GPU%20(16GB)-green.svg)]()
[![Research Context](https://img.shields.io/badge/Benchmark-IEEE%20TITS%202025%20(Xi%20et%20al.)-purple.svg)]()

---

## 📌 Executive Summary

Real-time traffic accident detection from CCTV surveillance footage is a critical component of Intelligent Transportation Systems (ITS). However, real-world deployment faces two core machine learning challenges:
1. **Severe Data Scarcity & Class Imbalance**: CCTV camera feeds contain millions of normal traffic frames and relatively few accident instances.
2. **Feature Complementarity**: Accidents exhibit both fine-grained, localized spatial cues (e.g., bumper denting, glass shattering, tire marks) and broad contextual visual dependencies (e.g., unusual vehicle orientation across multiple lanes, traffic congestion build-up).

This repository contains the official implementation of a high-performance, generative-augmented traffic accident detection framework developed at **National Institute of Technology Delhi (NIT Delhi)**. The framework replicates and extends recent state-of-the-art benchmarks (Xi et al., *IEEE Transactions on Intelligent Transportation Systems*, 2025) by combining:
- **Advanced Generative Data Augmentation**: Comparing **StyleGAN2-ADA** against **Denoising Diffusion Probabilistic Models (DDPM)** to generate synthetic class-conditional accident samples without mode collapse.
- **Hybrid FTCNN-ViT Architecture**: A dual-branch network that simultaneously extracts local spatial features via a **Fine-Tuned Convolutional Neural Network (FTCNN)** and global dependencies via a **Vision Transformer (ViT)**, coupled with an attention-aware dense fusion head.
- **Explainable AI (XAI)**: Visual verification using **Grad-CAM** saliency maps, ensuring predictions align with true collision zones rather than background artifacts.

---

## 🏗️ Architecture & Pipeline Overview

```
                      ┌────────────────────────────────────────┐
                      │    Raw CCTV Traffic Video Frames       │
                      │       (Accident / No Accident)         │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    Color Preprocessing (Eq. 3–5)       │
                      │  Saturation, Brightness & Contrast     │
                      └───────────────────┬────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
      ┌───────────────────────────┐               ┌───────────────────────────┐
      │  Approach A: StyleGAN2-ADA│               │    Approach B: DDPM UNet  │
      │  Fine-Tuning / WGAN-GP    │               │ Class-Conditional Denoiser│
      └─────────────┬─────────────┘               └─────────────┬─────────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  Augmented Training Set (Real + Synth) │
                      └───────────────────┬────────────────────┘
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
      ┌───────────────────────────┐               ┌───────────────────────────┐
      │     Branch 1: FTCNN       │               │      Branch 2: ViT        │
      │ 4 Conv Blocks (32→64→128  │               │ 16×16 Patch Extraction +  │
      │ →256) + Dense(512)        │               │ Linear Proj + 8 Trafo Blks│
      │ [Captures Local Details]  │               │ [Captures Global Context] │
      └─────────────┬─────────────┘               └─────────────┬─────────────┘
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │        Feature Fusion & Classifier     │
                      │  Concat(1024) → Dense(512) → Dense(256)│
                      │      → Sigmoid Binary Prediction       │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │      Evaluation & Interpretability     │
                      │  ROC-AUC, PR Curves, Grad-CAM Maps     │
                      └────────────────────────────────────────┘
```

---

## 🔬 Key Methodologies

### 1. Data Preprocessing (Eq. 3–5)
Surveillance footage suffers from environmental variance (fog, low-light night conditions, glare). Inputs are normalized and standardized with targeted color transformations:
- **Saturation Scaling ($S_{factor} = 1.5$)**: Emphasizes vehicle paint boundaries and traffic lane demarcations.
- **Brightness Scaling ($B_{factor} = 1.2$)**: Compensates for dark road patches and shadows.
- **Contrast Adjustment ($C_{factor} = 1.3$)**: Sharpens metallic deformations and edge transitions.

### 2. Generative Data Augmentation: StyleGAN2-ADA vs. DDPM

| Dimension | StyleGAN2-ADA (GAN-based) | DDPM (Diffusion-based) |
|---|---|---|
| **Formulation** | Adversarial min-max game ($G$ vs. $D$) | Forward Gaussian noising + reverse UNet denoising |
| **Training Stability** | Prone to mode collapse and discriminator vanishing gradients | Exceptionally stable; optimizes fixed variational lower bound ($L_{simple}$) |
| **Sample Diversity** | Moderate; drops long-tail crash scenarios | High; full coverage of the empirical distribution |
| **Class Conditioning** | Requires conditioning projection layers | Direct class embedding injection into UNet blocks |
| **Compute Footprint** | Dual networks, delicate balance | Single UNet denoiser with linear noise scheduler ($T=1000$) |
| **Target Resolution** | $64 \times 64$ / $256 \times 256$ | $64 \times 64$ (interpolated to $224 \times 224$ for classifier) |

### 3. Dual-Branch FTCNN-ViT Hybrid Network

1. **FTCNN Branch (Local Feature Extractor)**:
   - 4 consecutive Convolutional blocks: `Conv2D(32)` $\to$ `Conv2D(64)` $\to$ `Conv2D(128)` $\to$ `Conv2D(256)`, each with $3 \times 3$ kernels, ReLU activation, and $2 \times 2$ MaxPooling.
   - Preserves spatial hierarchies and flattens into a `Dense(512)` with `Dropout(0.5)`.
2. **Vision Transformer (ViT) Branch (Global Context Extractor)**:
   - Divides images into $16 \times 16$ non-overlapping patches ($N = 196$ patches for $224 \times 224$).
   - Projects patches to dimension $D = 64$ with learnable 1D positional embeddings.
   - 8 stacked Transformer encoder blocks, each comprising LayerNorm, Multi-Head Self-Attention ($h = 4$ heads), and GELU Feed-Forward Networks.
   - Aggregated via `GlobalAveragePooling1D` into `Dense(512)` with `Dropout(0.1)`.
3. **Fusion Mechanism**:
   - Concatenates the $512$-d FTCNN vector and $512$-d ViT vector into a joint $1024$-d representation.
   - Fed into `Dense(512, gelu)` $\to$ `Dropout(0.5)` $\to$ `Dense(256, gelu)` $\to$ `Dense(1, sigmoid)`.
   - Trained with Adam optimizer ($\text{lr} = 3 \times 10^{-4}$) under Binary Cross-Entropy loss.

---

## 📊 Experimental Results & Comparison

Evaluated on the **Accident Detection from CCTV Footage** benchmark test set (100 hold-out frames: 53 Non-Accident, 47 Accident):

| Model Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|
| **Target Baseline (Xi et al., IEEE TITS 2025)** | $\ge 0.9500$ | $\ge 0.9800$ | $\ge 0.9200$ | $\ge 0.9500$ | — |
| **StyleGAN2-ADA + FTCNN-ViT Fusion** | **0.9200** | **0.9149** | **0.9149** | **0.9149** | **0.9631** |
| **DDPM + FTCNN-ViT Fusion** | **0.8600** | **0.9024** | **0.7872** | **0.8409** | **0.9482** |

### Per-Class Performance Breakdown

- **StyleGAN2-ADA + FTCNN-ViT**:
  - *No Accident*: Precision = 0.9245, Recall = 0.9245, F1 = 0.9245
  - *Accident*: Precision = 0.9149, Recall = 0.9149, F1 = 0.9149
- **DDPM + FTCNN-ViT**:
  - *No Accident*: Precision = 0.8305, Recall = 0.9245, F1 = 0.8750
  - *Accident*: Precision = 0.9024, Recall = 0.7872, F1 = 0.8409

---

## 📂 Repository Contents

```
.
├── defusion-02-fixed.ipynb                 # DDPM Diffusion Model training, sampling, and FTCNN-ViT fusion
├── pretrained-gan-ftcnn-vit-applied - Copy.ipynb  # StyleGAN2-ADA fine-tuning, WGAN-GP fallback & FTCNN-ViT fusion
└── README.md                               # Comprehensive project documentation
```

### Notebook Breakdown
1. **`defusion-02-fixed.ipynb`**:
   - Hugging Face `diffusers` `UNet2DModel` with 6 down/up blocks (including `AttnDownBlock2D` and `AttnUpBlock2D`).
   - Class-conditional reverse diffusion generation ($64 \times 64$).
   - Synthetic image pooling, dataset augmentation, and hybrid classifier training.
   - Comprehensive plotting: DDPM loss curve, generated samples, training history, ROC, PR curve, and Grad-CAM visualizations.
2. **`pretrained-gan-ftcnn-vit-applied - Copy.ipynb`**:
   - Script for setting up StyleGAN2-ADA and converting training folders to ZIP archives.
   - Checkpoint loader and fine-tuner, supplemented with a pure TensorFlow WGAN-GP fallback generator.
   - Full implementation and evaluation of the FTCNN-ViT fusion model (Tables III, IV, and V).

---

## 🚀 Setup & Execution Guide

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with $\ge 12\text{ GB}$ VRAM (tested on Kaggle NVIDIA Tesla T4 16GB)
- CUDA 11.8+ / 12.x

### Dependencies Installation
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow>=2.15.0
pip install diffusers transformers accelerate huggingface_hub
pip install opencv-python matplotlib seaborn pandas scikit-learn pillow
```

### Dataset Preparation
The project utilizes the public **Accident Detection from CCTV Footage** dataset:
```
data/
├── train/
│   ├── Accident/
│   └── No Accident/
├── val/
│   ├── Accident/
│   └── No Accident/
└── test/
    ├── Accident/
    └── No Accident/
```
In Kaggle environments, point `KAGGLE_DATA` to `/kaggle/input/datasets/ckay16/accident-detection-from-cctv-footage/data`.

### Running the Notebooks
- To run the **Diffusion-based Pipeline**:
  Open and execute [`defusion-02-fixed.ipynb`](file:///e:/nit%20delhi/research/final/defusion-02-fixed.ipynb).
- To run the **GAN-based Baseline & Replication**:
  Open and execute [`pretrained-gan-ftcnn-vit-applied - Copy.ipynb`](file:///e:/nit%20delhi/research/final/pretrained-gan-ftcnn-vit-applied%20-%20Copy.ipynb).

---

## 🔍 Model Interpretability & Grad-CAM

To ensure the model is not relying on spurious correlations (such as road markings or roadside trees):
- **Grad-CAM (Gradient-weighted Class Activation Mapping)** is computed on the final convolutional layer of the FTCNN branch (`ftcnn_conv4`).
- Gradients of the predicted accident class score with respect to feature maps are pooled and projected back to the input frame.
- High-intensity activations correspond directly to vehicle deformation zones, sudden lane deviations, and impact centers.

---

## 📚 References & Acknowledgments

- **Base Benchmark**:
  > Xi et al., *"Integrating Generative Adversarial Networks and Convolutional Neural Networks for Enhanced Traffic Accidents Detection and Analysis,"* *IEEE Transactions on Intelligent Transportation Systems (TITS)*, 2025.
- **Denoising Diffusion Models**:
  > Jonathan Ho, Ajay Jain, Pieter Abbeel, *"Denoising Diffusion Probabilistic Models,"* *NeurIPS*, 2020.
- **Vision Transformers**:
  > Alexey Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,"* *ICLR*, 2021.
- **Institutional Affiliation**:
  > Developed under research initiatives at **National Institute of Technology Delhi (NIT Delhi)**.
