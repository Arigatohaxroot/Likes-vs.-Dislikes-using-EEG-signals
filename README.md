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
In this project, we combined the fields of neuroscience and machine learning to decode consumer preferences using EEG data. By analyzing brain activity through EEG, we can predict whether a person "likes" or "dislikes" a product based on their neural responses.

### Neuromarketing
Neuromarketing blends neuroscience and marketing to better understand consumer behavior. EEG, a key tool in neuromarketing, allows researchers to track brain responses with high temporal accuracy, making it ideal for real-time analysis of preferences.

## Dataset
We collected EEG recordings from 19 participants who were shown images of 10 different cars. Their responses were categorized as "like" or "dislike" based on their brain activity.

### Dataset Description:
- **Source**: SEECS EEG lab, NUST Islamabad
- **Participants**: 19 individuals (aged 20-36)
- **Products**: 10 car images
- **EEG Channels**: 14 channels covering full brain activity
- **Response**: "Like" or "Dislike"
- **Total Recordings**: 190 EEG samples

### Sample Car Image:
<p align="center">
  <img src="https://i.imgur.com/GLhdo0f.jpeg" alt="Sample Car Image" width="400">
</p>

*(This is a sample image used in the dataset.)*

## Methodology

### Dataset Acquisition Process
EEG signals were recorded using BrainVision equipment, with modified electrode placement to cover the full brain. Participants viewed car images for 4-5 seconds, and their brain activity was recorded.

<p align="center">
  <img src="https://i.imgur.com/t8uDIfu.png" alt="EEG Acquisition Setup" width="400">
</p>

*(This image shows the EEG acquisition process.)*

### Data Preprocessing on MATLAB
After acquiring the data, several preprocessing steps were performed using MATLAB. These steps include resampling, high-pass filtering, artifact removal, and independent component analysis (ICA) to remove noise and artifacts from the EEG signals.

#### Visualizations:
<p align="center">
  <img src="https://i.imgur.com/IGVKJaM.png" alt="EEG Data Preprocessing in MATLAB" width="400">
</p>

*(This image shows EEG preprocessing in MATLAB.)*

#### Statistics of Recorded EEG Data Sample:
<p align="center">
  <img src="https://i.imgur.com/HJaozqq.png" alt="Statistics of Recorded EEG Data" width="400">
</p>

*(This image displays the statistics of the recorded EEG data.)*

#### Artifact Subspace Reconstruction (ASR):
ASR was applied to reduce artifacts in the EEG signals caused by noise such as muscle movements or electrical interference. The following image shows the difference between the signal before and after applying ASR.

<p align="center">
  <img src="https://i.imgur.com/pkZoHoM.png" alt="ASR Waveform Before and After" width="400">
</p>

*(Red waveform: before ASR, Blue waveform: after ASR.)*

#### Independent Component Analysis (ICA):
ICA was used to isolate and remove noise, such as eye-blink or muscle-related artifacts. The two images below show the EEG signal before and after ICA.

- **Before Preprocessing**:
<p align="center">
  <img src="https://i.imgur.com/Wb0ld0U.png" alt="ICA Activity Before Preprocessing" width="400">
</p>

- **After Preprocessing**:
<p align="center">
  <img src="https://i.imgur.com/fmRNprn.png" alt="ICA Activity After Preprocessing" width="400">
</p>

*(These images show the EEG signal before and after applying ICA.)*

## Models
Several machine learning algorithms were applied to the EEG data to classify the user responses as either "like" or "dislike". The following models were used:
- **Naive Bayes**
- **Logistic Regression**
- **Support Vector Machines (SVM)**
- **Random Forest**
- **Neural Networks (ANN)**

### Feature Extraction
We extracted features from the EEG data using the Discrete Wavelet Transform (DWT) and standardized the data for model training.

## Results

We evaluated the performance of different machine learning models based on accuracy, precision, recall, F1-score, and confusion matrix.

### Accuracy Comparison Between Different Classifiers:
The bar graph below compares the accuracy of various classifiers tested on the EEG data.

<p align="center">
  <img src="https://i.imgur.com/rBy56u6.png" alt="Accuracy Comparison Between Classifiers" width="400">
</p>

### Cross-Validation Accuracy Comparison Between Classifiers:
The line graph below shows cross-validation accuracies of different classifiers, highlighting their generalizability.

<p align="center">
  <img src="https://i.imgur.com/dv9LYxo.png" alt="Cross-Validation Accuracy Comparison" width="400">
</p>

### Precision, Recall, and F1-Score Metrics Comparison:
We compared the precision, recall, and F1-score of various classifiers to assess their performance.

<p align="center">
  <img src="https://i.imgur.com/a1UEqJf.png" alt="Precision, Recall, and F1-Score Metrics Comparison" width="400">
</p>

### Confusion Matrix of Different Classifiers:
The confusion matrix below provides a visual representation of the classification accuracy of the models for "like" vs. "dislike" predictions.

<p align="center">
  <img src="https://i.imgur.com/JNfaM6P.png" alt="Confusion Matrix of Different Classifiers" width="400">
</p>

### t-SNE Plot:
The t-SNE plot provides a visualization of the EEG data distribution for the "like" and "dislike" classes, as learned by the Artificial Neural Network (ANN) model.

<p align="center">
  <img src="https://i.imgur.com/4mJouJl.png" alt="t-SNE Visualization of ANN" width="400">
</p>

### Actual Labels vs. Predicted Labels:
This figure compares the actual labels from the dataset versus the predicted labels generated by the classifiers for a few samples.

<p align="center">
  <img src="https://i.imgur.com/P6t0bja.png" alt="Actual vs Predicted Labels" width="400">
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
