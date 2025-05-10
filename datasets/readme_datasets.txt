Synthetic Data for Traffic Sign Recognition

This project utilizes a combined dataset of real and synthetic traffic sign images.
Each dataset contributes to training and evaluating models under diverse real-world conditions.

1. GTSRB (German Traffic Sign Recognition Benchmark)
      - Classes: 43
      - Train: 39,209 samples
      - Test: 12,630 samples
      - Source: [5] (Stallkamp et al.)

2. BTSD (Belgian Traffic Signs Dataset)
      - Classes: 62
      - Train: 4,575 samples
      - Test: 2,520 samples

3. ETSD (European Traffic Signs Dataset)
      - Classes: 164
      - Train: 60,546 samples
      - Test: 21,930 samples

4. Synthetic GTSRB
      - Classes: 211 (augmented)
      - Train: 84,400 samples
      - Test: 21,100 samples
      - Generated using synthetic transformation techniques

5. Synthetic AIL Dataset
      - Classes: 196
      - Train: 12,649 samples
      - Test: 3,314 samples
      - Captured from Slovak roads in various lighting and weather conditions
————————————————————————————————————————
Combined Datasets
————————————————————————————————————————
Final Combined Dataset (After Cleaning and Merging):
      - Classes: 312
      - Train: 151,605 images
      - Test: 44,749 images
      - Format: GTSRB-style (45x45 images with annotations)
      - Notes: Filtered for class balance, duplicate removal, histogram equalization

Filtered 44-Class Subset Dataset 
      - Classes: 44
      - Train: 66,279 images
      - Test: 21,659 images
      - Used for controlled training and classification experiments

GAN-Augmented 44-Class Dataset
      - Train images: +2,816 synthetic samples (64 per class)
      - Total: 69,095 training images, 
      - 21,659 test images
      - Notes: Synthetic samples used for training only (not in validation/test sets)

Notes
- All annotations follow the GTSRB format.
- Synthetic images were generated using DCGAN
- Augmentation pipelines are defined under baseline, simple, complex and hybrid scenario.

For more details, see the thesis: /docs/BP_Scientific_Part.pdf
