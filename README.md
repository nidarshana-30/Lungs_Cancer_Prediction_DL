# 🫁 Lung Cancer Prediction using Deep Learning

This project uses deep learning techniques to predict lung cancer from CT scan images. The goal is to aid radiologists and medical professionals in early and accurate detection of lung cancer using convolutional neural networks (CNN).

## 📌 Project Overview

- **Objective**: Classify CT scan images into cancerous or non-cancerous categories.
- **Model Used**: Pre-trained **VGG16** CNN architecture (fine-tuned).
- **Dataset**: LIDC-IDRI or custom-labeled CT scan image dataset.
- **Framework**: TensorFlow/Keras

## 📁 Dataset Description

- The dataset contains labeled CT scan images:
  - Class 0: Non-cancerous
  - Class 1: Cancerous

- Images were preprocessed to uniform size (e.g., 224x224) and normalized before feeding into the model.

📂 Dataset Sources:
- [LIDC-IDRI dataset on The Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)

## 🧠 Model Architecture

- Base model: `VGG16` (pre-trained on ImageNet)
- Top layers:
  - GlobalAveragePooling2D
  - Dense (ReLU + Dropout)
  - Output: Sigmoid (binary classification)

```python
model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
