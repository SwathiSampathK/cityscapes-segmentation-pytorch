# Deep Learning for Cityscapes Segmentation
Semantic segmentation on the Cityscapes dataset using multiple architectures (FCN, SegNet, UNet, Pix2Pix) in PyTorch.

## Overview
This project benchmarks several deep learning models for semantic segmentation of urban street scenes from the Cityscapes dataset. Models are evaluated on IoU and Dice score across 30+ urban scene classes.

## Dataset

This project uses the **Cityscapes Dataset** for semantic segmentation.

- **Official Download Page:** [https://www.cityscapes-dataset.com/downloads/](https://www.cityscapes-dataset.com/downloads/)
- You must **register** (free for research/education) to access downloads.

**Recommended files:**
1. `leftImg8bit_trainvaltest.zip` → contains training/validation/test RGB images.
2. `gtFine_trainvaltest.zip` → contains ground truth segmentation labels.


## Models Implemented
- **FCN** (Fully Convolutional Network)
- **SegNet**
- **UNet**
- **Pix2Pix**

## Tech Stack
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-yellow)

## Results
- Pix2Pix outperformed others with:
  - **IoU:** +24% over FCN
  - **Dice Score:** +17% over FCN
- Results suggest GAN-based models can enhance segmentation accuracy in urban scenes.

