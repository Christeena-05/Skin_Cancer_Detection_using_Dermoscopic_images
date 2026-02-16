# Skin_Cancer_Detection_using_Dermoscopic_images

## Project Summary

This project builds a binary skin lesion classifier (Benign vs Malignant) using EfficientNetB0 with a custom hair-removal preprocessing pipeline.

Unlike standard transfer-learning implementations, this project integrates morphological black-hat filtering and inpainting to remove hair artifacts before training, improving lesion clarity and robustness.

The model is trained and evaluated on the HAM10000 dataset along with its associated metadata file.

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

⚠️ Dataset is not included in this repository due to size constraints.
You must download it separately and place it inside the project directory.

## Technical Approach

 ### 1️. Preprocessing (Key Differentiator)

Hair removal is performed using:

Morphological Black-Hat Filtering

Binary thresholding

OpenCV Inpainting

This reduces noise caused by hair strands and enhances lesion visibility before model ingestion.

### 2️. Model Architecture

Base Model: EfficientNetB0 (ImageNet pretrained)

Global Average Pooling

Dense layer

Sigmoid output (Binary classification)

Training Strategy:

Freeze base model initially

Fine-tune top layers

Low learning rate for stable convergence

## Model Performance (Validation Set)
Metric	                   Benign	   Malignant
Precision	                 ~0.94	   ~0.39
Recall (Sensitivity)	     ~0.72	   ~0.79
F1 Score	                 ~0.82	   ~0.52

## Why Recall Matters

In medical diagnostics, missing a malignant lesion is more critical than a false positive.
This model prioritizes malignant recall (~79%) over raw accuracy.

Additional Metrics:

Confusion Matrix

ROC-AUC evaluation

Threshold tuning capability

### How to Run Locally (Anaconda Prompt)
Step 1: Clone the Repository

git clone <your-repo-link>
cd <project-folder>

Step 2: Activate Your Environment

conda activate <your-env-name>


(Or create one if needed)

conda create -n skin_env python=3.10
conda activate skin_env
pip install -r requirements.txt

Step 3: Place Dataset

Ensure your folder structure looks like:

project-folder/
│
 images/,
 HAM10000_metadata.csv,
 notebook.ipynb

Step 4: Run Jupyter Lab
cd <project-folder>
jupyter lab


Open the notebook and run all cells sequentially.

### How to Run on Google Colab
Step 1: Upload Dataset as ZIP File

Zip your dataset folder locally:

ham_dataset.zip


Upload to Colab.

Step 2: Unzip Inside Colab
!unzip /content/ham_dataset.zip


If stored in Google Drive:

from google.colab import drive
drive.mount('/content/drive')

!unzip /content/drive/MyDrive/ham_dataset.zip


Ensure image and metadata paths match the notebook configuration.

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

Imbalanced Classification Handling

ROC-AUC & Confusion Matrix Analysis

TensorFlow/Keras Implementation

Practical Model Evaluation Strategy

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
