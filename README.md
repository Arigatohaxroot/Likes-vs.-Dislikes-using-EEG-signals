# EEG Likes vs Dislikes Classification

This repository contains the code and datasets used for classifying consumer preferences (likes vs dislikes) using EEG signals. The project explores how brain activity data, recorded via EEG, can be leveraged to predict user preferences for different consumer products, particularly car images. We utilized various machine learning models to analyze the data and generate predictions.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Authors](#authors)

## Introduction
This project investigates the neural correlates of consumer behavior by analyzing EEG data collected while participants viewed various product images. The goal is to classify user responses as "likes" or "dislikes" based on the EEG signals using different machine learning algorithms.

### Neuromarketing
Neuromarketing blends neuroscience and marketing to better understand consumer behavior, using techniques like EEG, which provides high temporal resolution of brain activity. This project aims to enhance marketing strategies by analyzing brain responses.

## Dataset
The dataset consists of EEG recordings from 19 participants who were shown images of 10 different cars. The EEG data was collected using 14 electrodes with full-brain coverage and was preprocessed to remove noise and artifacts. The participants provided feedback on whether they "liked" or "disliked" each car image, which was recorded along with the EEG data.

### Dataset Description:
- **Source**: SEECS EEG lab, NUST Islamabad
- **Participants**: 19 individuals (aged 20-36)
- **Products**: 10 car images
- **EEG Channels**: 14 channels covering full brain activity
- **Response**: "Like" or "Dislike"
- **Total Recordings**: 190 EEG samples

### Sample Car Image:
<p align="center">
  <img src="https://i.imgur.com/GLhdo0f.jpeg" alt="Sample Car Image">
</p>

*(This is a sample image used in the dataset.)*

## Methodology

### Dataset Acquisition Process
EEG signals were recorded using BrainVision equipment, with modified electrode placement for full-brain coverage. The images were displayed for 4-5 seconds, and EEG data was collected during this period.

<p align="center">
  <img src="https://i.imgur.com/t8uDIfu.png" alt="EEG Acquisition Setup">
</p>

*(This image shows the EEG acquisition process.)*

### Data Preprocessing on MATLAB
- **Resampling**: From 2000Hz to 128Hz
- **High-Pass Filtering**: Removal of baseline drift (1-2 Hz for ICA)
- **Artifact Removal**: Using Artifact Subspace Reconstruction (ASR)
- **Independent Component Analysis (ICA)**: To eliminate eye-blink and muscle movement artifacts.

The clean EEG data was exported and used for model training and testing.

#### Visualizations:
<p align="center">
  <img src="https://i.imgur.com/IGVKJaM.png" alt="EEG Data Preprocessing in MATLAB">
</p>

*(This image shows EEG preprocessing in MATLAB.)*

#### Statistics of Recorded EEG Data Sample:
<p align="center">
  <img src="https://i.imgur.com/HJaozqq.png" alt="Statistics of Recorded EEG Data">
</p>

*(This image shows the statistics of the recorded EEG data.)*

#### Artifact Subspace Reconstruction (ASR):
<p align="center">
  <img src="https://i.imgur.com/pkZoHoM.png" alt="ASR Waveform Before and After">
</p>

*(Red waveform shows before performing ASR and blue waveform shows after performing ASR.)*

#### Independent Component Analysis (ICA):
- **Before Preprocessing**:
<p align="center">
  <img src="https://i.imgur.com/Wb0ld0U.png" alt="ICA Activity Before Preprocessing">
</p>

- **After Preprocessing**:
<p align="center">
  <img src="https://i.imgur.com/fmRNprn.png" alt="ICA Activity After Preprocessing">
</p>

*(These images show the ICA activity before and after preprocessing.)*

## Models
Several machine learning algorithms were tested:
- **Naive Bayes**
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Random Forest**
- **Neural Networks (ANN)**

### Feature Extraction
- Features were extracted using Discrete Wavelet Transform (DWT).
- Data was standardized and flattened for model training.

## Results

### Accuracy Comparison Between Different Classifiers:
<p align="center">
  <img src="https://i.imgur.com/rBy56u6.png" alt="Accuracy Comparison Between Classifiers">
</p>

### Cross-Validation Accuracy Comparison Between Classifiers:
<p align="center">
  <img src="https://i.imgur.com/dv9LYxo.png" alt="Cross-Validation Accuracy Comparison">
</p>

### Precision, Recall, and F1-Score Metrics Comparison:
<p align="center">
  <img src="https://i.imgur.com/a1UEqJf.png" alt="Precision, Recall, and F1-Score Metrics Comparison">
</p>

### Confusion Matrix of Different Classifiers:
<p align="center">
  <img src="https://i.imgur.com/JNfaM6P.png" alt="Confusion Matrix of Different Classifiers">
</p>

### t-SNE Plot:
<p align="center">
  <img src="https://i.imgur.com/4mJouJl.png" alt="t-SNE Visualization of ANN">
</p>

### Actual Labels vs. Predicted Labels:
<p align="center">
  <img src="https://i.imgur.com/P6t0bja.png" alt="Actual vs Predicted Labels">
</p>

*(These images show various metrics, including accuracy, cross-validation, confusion matrix, t-SNE visualization, and more.)*

## Usage
Clone this repository and run the following steps to reproduce the results:
1. Install the required dependencies listed below.
2. Load the dataset in the `/data` directory.
3. Run the preprocessing script to clean and prepare the EEG data.
4. Train the model using the provided scripts.
5. Evaluate the model's performance.

### Example:
```bash
git clone https://github.com/yourusername/eeg-likes-vs-dislikes.git
cd eeg-likes-vs-dislikes
python preprocess_data.py
python train_model.py
python evaluate_model.py
