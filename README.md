# Skin_Cancer_Detection_using_Dermoscopic_images

## Project Summary

This project builds a binary skin lesion classifier (Benign vs Malignant) using EfficientNetB0 with a custom hair-removal preprocessing pipeline.

Unlike standard transfer-learning implementations, this project integrates morphological black-hat filtering and OpenCV inpainting to remove hair artifacts before training, improving lesion clarity and robustness.

The model is trained and evaluated on the HAM10000 dataset along with its associated metadata file.

## Environment Setup

The project is implemented using Python and TensorFlow.

Before training the model, the environment verifies:

NumPy version

TensorFlow version

GPU availability

GPU acceleration is used to speed up deep learning training.

## Dataset

Dataset Used:

HAM10000 (Human Against Machine with 10,000 images)

Metadata file: HAM10000_metadata.csv

Data Includes:

Dermatoscopic images

Lesion metadata (diagnosis, age, sex, localization, etc.)

Binary mapping:

0 → Benign

1 → Malignant

Dataset distribution:

Benign → 9564 images  
Malignant → 2156 images

⚠️ Dataset is not included in this repository due to size constraints.

You must download it separately and place it inside the project directory.

## Data Preparation

Metadata is processed using Pandas.

Steps include:

Loading the metadata CSV file

Extracting diagnosis labels

Mapping diagnosis to binary classes

Creating image filenames from ISIC IDs

Linking metadata entries with image files

A stratified train–validation split is used to preserve class distribution.

## Technical Approach

### 1. Image Preprocessing (Key Differentiator)

Dermoscopic images often contain hair strands that obscure lesion features.

To address this issue, a custom hair removal pipeline is implemented using OpenCV.

Hair removal is performed using:

Morphological Black-Hat filtering

Binary thresholding

OpenCV inpainting (Telea algorithm)

This removes hair artifacts and improves lesion visibility.

### 2. Data Augmentation

To improve model generalization, image augmentation is applied using Keras ImageDataGenerator.

Augmentations include:

Rotation

Zoom

Width shift

Height shift

Horizontal flip

EfficientNet preprocessing is also applied to normalize images before training.

### 3. Handling Class Imbalance

The dataset contains significantly more benign samples than malignant samples.

To address this imbalance, class weights are calculated using:

compute_class_weight()

These weights are applied during model training so that malignant samples receive higher importance.

### 4. Model Architecture

Base Model: EfficientNetB0 (ImageNet pretrained)

Architecture:

Global Average Pooling

Dense layer (ReLU activation)

Dropout layer for regularization

Sigmoid output layer (Binary classification)

Training Strategy:

Freeze the EfficientNet base network initially

Train the classification head

Fine-tune upper layers later

Use a low learning rate for stable convergence.

## Model Training

Training is performed using TensorFlow/Keras.

The training pipeline uses:

ImageDataGenerator for batch loading

Training generator for augmented data

Validation generator for evaluation

Class weights for imbalance correction.

GPU acceleration is used to speed up training.

## Model Performance (Validation Set)

Metric            Benign     Malignant

Precision         ~0.94      ~0.39  
Recall            ~0.72      ~0.79  
F1 Score          ~0.82      ~0.52  

## Why Recall Matters

In medical diagnostics, missing a malignant lesion is more critical than a false positive.

This model prioritizes malignant recall (~79%) over raw accuracy.

Additional Metrics:

Confusion Matrix

ROC-AUC evaluation

Threshold tuning capability

## Inference Example

prediction = model.predict(processed_image)

if prediction > threshold:
    print("Malignant")
else:
    print("Benign")

Threshold can be tuned depending on sensitivity requirements.

## Key Skills Demonstrated

Transfer Learning (EfficientNetB0)

Medical Image Preprocessing

OpenCV Morphological Operations

Hair Artifact Removal in Dermoscopic Images

Data Augmentation Techniques

Handling Imbalanced Datasets

TensorFlow / Keras Deep Learning Pipeline

Classification Metrics and Model Evaluation

## Limitations

Image-level train-test split (not lesion-level grouping)

Moderate ROC-AUC

Class imbalance impacts malignant precision

No cross-validation implemented

## Future Improvements

Lesion-level split to avoid data leakage

Focal Loss / advanced class balancing

Cross-validation

Model comparison (EfficientNetB3, ResNet50)

Deployment via Streamlit or Flask
