# Synthetic Data Augmentation for Traffic Sign Recognition

This project related to a bachelor thesis research explores how synthetic data and advanced augmentation methods impact the real-world performance of traffic sign recognition models in intelligent transport systems. Using Conditional GANs and a structured augmentation pipeline, we simulate challenging real-world conditions like rain, snow, occlusions, and blur. Our experiments assess the impact of simple, complex, and hybrid augmentations on classification accuracy using models such as MobileNetV2, DenseNet, and ConvNeXt. We also evaluate how synthetic GAN-generated images contribute to robustness and generalization

---

## Overview

Autonomous vehicles must recognize traffic signs under challenging environmental conditions such as snow, rain, blur, and occlusion. This work:

- Builds a unified dataset from five real and synthetic sources.
- Implements a perception-based augmentation framework (Albumentations + SSIM/LPIPS).
- Trains Conditional GANs (DCGAN) to generate synthetic samples for underrepresented classes.
- Evaluates the effect of various augmentation strategies on MobileNetV2, DenseNet, and ConvNeXt.

---

## Project Structure

```
.
├── BP_KristianCervenka_BezOutputov.ipynb     # Main code notebook without outputs
├── BP_KristianCervenka.ipynb                 # Main code notebook with outputs
├── requirements.txt                          # Python dependencies
├── datasets/                                 # Folder for all datasets
│   └── dataset_description.txt               # Summary of used datasets
├── models/                                   # Trained model checkpoints (.pth)
├── results/                                   # Trained model checkpoints (.pth)
│   ├── final_evaluation_results_312_Classes.csv
│   ├── final_evaluation_results_43_Classes_NO_GAN.csv
│   └── final_evaluation_results_43_Classes_with_GAN.csv
├── generated/                                # GAN-generated images (per class)
├── utils/                                    # Helper scripts (augmentation, GANs, etc.)
├── license                                   # License file ( MIT )
├── docs/                                     # Final PDF, figures, plots
│   └── BP_Scientific_Part.pdf                # Scientific Part
└── README.md                                 # This file
```
---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/K1ko/data-generation-for-intelligent-driving-cv.git
cd traffic-sign-augmentation
```

### 2. Create Environment & Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
---

## Dataset Setup

This project uses the following datasets:

| Dataset         | Classes | Train Samples | Test Samples |
| --------------- | ------- | ------------- | ------------ |
| GTSRB           | 43      | 39,209        | 12,630       |
| BTSD            | 62      | 4,575         | 2,520        |
| ETSD            | 164     | 60,546        | 21,930       |
| Synthetic GTSRB | 211     | 84,400        | 21,100       |
| Synthetic AIL   | 196     | 12,649        | 3,314        |

Combined Datasets:

- **Full (312 classes)**: 151,605 train / 44,749 test
- **Filtered Subset (44 classes)**: 66,279 train / 21,659 test
- **+ GAN Augmentation**: 2,816 synthetic images (64 per class added)

Details in: `datasets/dataset_description.txt`

Download the datasets from the following source:

📎 [Download Datasets]([https://your-download-link.com](https://stubask-my.sharepoint.com/:f:/g/personal/xcervenkak_stuba_sk/EgEMqJhbrlNNn1vZ0pnLjiEBKfZcCu-DPK9uTWW1OBJ7Gg?e=Q77gfR))

---

## Running the Experiments

Open the notebook `BP_KristianCervenka_BezOutputov.ipynb` and run step-by-step:

1. **Requirements installation** Install all necessary requirements
2. **GAN training** using DCGAN (separate model per class)
3. **Data augmentation** using Albumentations:
   - Simple: contrast, blur, snow, posterize, color shift, etc.
   - Complex: rain, fog, perspective, motion blur, grid distortions
4. **Model training** (MobileNetV2, DenseNet, ConvNeXt)
5. **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix

---

## Model Architectures

- **MobileNetV2**: Lightweight for edge devices
- **DenseNet**: Efficient gradient propagation
- **ConvNeXt**: Modern transformer-CNN hybrid

Trained with:

- Adam optimizer, batch size = 64
- Epochs = 20
- Input size = 45x45 RGB
- CrossEntropyLoss
- Mixed precision training (torch.amp)

---
## GAN Training Overview
We use a class-conditional DCGAN to generate synthetic traffic sign images for underrepresented classes in the 44-class subset. For each class:

- A separate GAN is trained using real images of that class.

- The generator learns to produce 45×45 RGB images from noise + class embeddings.

- The discriminator distinguishes real vs. synthetic images, also conditioned on the class.

- After training, 64 synthetic images per class are generated and added to the training set (not used for validation or testing).
![image](https://github.com/user-attachments/assets/03765473-898d-4d93-889f-22d3c9c6ef65)

This enhances model robustness and improves generalization to unseen traffic sign variations.
---
## Key Results
Combined Dataset (312 Classes)
| Model       | Augmentation | Accuracy | Precision | Recall | F1 Score |
| ----------- | ------------ | -------- | --------- | ------ | -------- |
| MobileNetV2 | Simple       | 96.25%   | 0.96      | 0.96   | 0.96     |
| DenseNet    | Simple       | 97.60%   | 0.98      | 0.98   | 0.98     |
| ConvNeXt    | Simple       | 98.67%   | 0.99      | 0.99   | 0.99     |
| MobileNetV2 | Complex      | 96.41%   | 0.96      | 0.96   | 0.96     |
| DenseNet    | Complex      | 97.52%   | 0.98      | 0.98   | 0.98     |
| ConvNeXt    | Complex      | 98.48%   | 0.99      | 0.98   | 0.98     |
| ConvNeXt    | Hybrid       | 98.17%   | 0.98      | 0.98   | 0.98     |
| MobileNetV2 | Baseline     | 95.26%   | 0.95      | 0.95   | 0.95     |

Filtered Subset (44 Classes, No GAN)
| Model       | Augmentation | Accuracy | Precision | Recall | F1 Score |
| ----------- | ------------ | -------- | --------- | ------ | -------- |
| MobileNetV2 | Simple       | 97.28%   | 0.97      | 0.97   | 0.97     |
| DenseNet    | Simple       | 98.48%   | 0.98      | 0.98   | 0.98     |
| ConvNeXt    | Simple       | 98.97%   | 0.99      | 0.99   | 0.99     |
| DenseNet    | Complex      | 98.32%   | 0.98      | 0.98   | 0.98     |
| ConvNeXt    | Hybrid       | 98.85%   | 0.99      | 0.99   | 0.99     |
| MobileNetV2 | Baseline     | 96.35%   | 0.96      | 0.96   | 0.96     |

Filtered Subset (44 Classes, + GAN Augmentation)
| Model       | Augmentation  | Accuracy | Precision | Recall | F1 Score |
| ----------- | ------------- | -------- | --------- | ------ | -------- |
| MobileNetV2 | +GAN Baseline | 96.26%   | 0.96      | 0.96   | 0.96     |
| DenseNet    | +GAN Baseline | 97.93%   | 0.98      | 0.98   | 0.98     |
| ConvNeXt    | +GAN Baseline | 98.65%   | 0.99      | 0.99   | 0.99     |

## Citation

If you use this repository or its results in your work, please cite:

```bibtex
@bachelorthesis{cervenka2025,
  title     = {Synthetic Data for Computer Vision in Automotive},
  author    = {Kristián Červenka},
  school    = {Slovak University of Technology in Bratislava},
  year      = {2025}
}
```
---
## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is licensed under the [MIT License](LICENSE).

## 🙋 Contact

For questions or collaborations, contact: `xcervenkak@stuba.sk`

