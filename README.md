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
### Data Acquisition
EEG signals were recorded using BrainVision equipment, with modified electrode placement for full-brain coverage. The images were displayed for 4-5 seconds, and EEG data was collected during this period.

### Data Preprocessing
- **Resampling**: From 2000Hz to 128Hz
- **High-Pass Filtering**: Removal of baseline drift (1-2 Hz for ICA)
- **Artifact Removal**: Using Artifact Subspace Reconstruction (ASR)
- **Independent Component Analysis (ICA)**: To eliminate eye-blink and muscle movement artifacts.

The clean EEG data was exported and used for model training and testing.

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
- **Support Vector Machines (SVM)** showed the best precision and recall performance.
- **Artificial Neural Networks (ANN)** demonstrated balanced performance in terms of precision, recall, and F1-score.
- **Random Forest** and **Logistic Regression** exhibited moderate performance.

### Accuracy Comparison:
<p align="center">
  <img src="link_to_accuracy_graph_image" alt="Accuracy Comparison Graph">
</p>

### t-SNE Plot:
<p align="center">
  <img src="link_to_tsne_plot_image" alt="t-SNE Plot">
</p>

*(Add links to your result images once you provide them.)*

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
