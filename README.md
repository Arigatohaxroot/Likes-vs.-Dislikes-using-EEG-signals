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
This project combines neuroscience and machine learning to investigate how EEG signals can be used to predict consumer preferences. EEG data was recorded from participants viewing images of different cars, and their brain activity was analyzed to classify the images into "like" or "dislike." Several machine learning models were applied to the data to identify the best-performing approach for predicting preferences.

### Neuromarketing
Neuromarketing applies brain-scanning technology, such as EEG, to understand consumer behavior. This project uses EEG’s high temporal resolution to track brain activity while participants make real-time decisions on whether they like or dislike a product.

## Dataset
The dataset consists of EEG recordings from 19 participants who were shown images of 10 different cars. Their responses were recorded based on whether they liked or disliked the product. This EEG data was used to train machine learning models to classify consumer preferences.

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

*This image is an example of the car images shown to participants. After viewing each image, participants' brain activity was recorded, and they were asked whether they liked or disliked the car.*

## Methodology

### Dataset Acquisition Process
The EEG data was recorded using BrainVision equipment, with electrodes placed on the participant's scalp to capture brain activity. Each participant was shown images of cars for 4-5 seconds, and EEG data was collected during this period. After each image, participants provided feedback on whether they liked or disliked the car.

<p align="center">
  <img src="https://i.imgur.com/t8uDIfu.png" alt="EEG Acquisition Setup" width="400">
</p>

*This image shows the setup used for EEG data acquisition. Participants wore EEG caps, and brain signals were recorded as they viewed car images.*

### Data Preprocessing on MATLAB
The raw EEG data was preprocessed using MATLAB to remove noise and artifacts. Preprocessing steps included resampling, filtering, and artifact removal using methods like ASR (Artifact Subspace Reconstruction) and ICA (Independent Component Analysis). These steps ensured the data was clean and ready for machine learning analysis.

#### Visualizations of EEG Preprocessing:
<p align="center">
  <img src="https://i.imgur.com/IGVKJaM.png" alt="EEG Data Preprocessing in MATLAB" width="400">
</p>

*This image shows how the raw EEG data was processed using MATLAB. Preprocessing was crucial to eliminate noise from brain activity signals.*

#### Statistics of Recorded EEG Data Sample:
<p align="center">
  <img src="https://i.imgur.com/HJaozqq.png" alt="Statistics of Recorded EEG Data" width="400">
</p>

*The above image shows statistical analysis of the recorded EEG data, providing insights into data distribution and signal properties.*

#### Artifact Subspace Reconstruction (ASR):
<p align="center">
  <img src="https://i.imgur.com/pkZoHoM.png" alt="ASR Waveform Before and After" width="400">
</p>

*This image compares the EEG data before and after applying ASR. The red waveform represents the noisy signal, while the blue waveform shows the cleaned signal.*

#### Independent Component Analysis (ICA):
ICA was used to separate brain signals from artifacts such as eye blinks or muscle movements. The following images show the EEG signal before and after applying ICA.

- **Before ICA**:
<p align="center">
  <img src="https://i.imgur.com/Wb0ld0U.png" alt="ICA Activity Before Preprocessing" width="400">
</p>

- **After ICA**:
<p align="center">
  <img src="https://i.imgur.com/fmRNprn.png" alt="ICA Activity After Preprocessing" width="400">
</p>

*The two images above show how ICA was used to clean the EEG data, making it more suitable for analysis by removing artifacts.*

## Models
We applied a range of machine learning models to the processed EEG data to classify the brain responses as either "like" or "dislike." Each model was evaluated based on its ability to correctly predict the participants' preferences.

The following models were used:
- **Naive Bayes**: Assumes feature independence and provides fast classification.
- **Logistic Regression**: Suitable for binary classification problems like this.
- **Support Vector Machines (SVM)**: Effective in high-dimensional spaces.
- **Random Forest**: An ensemble learning method that uses multiple decision trees.
- **Artificial Neural Networks (ANN)**: Designed to capture complex, non-linear patterns in data.

### Feature Extraction
Features were extracted from the EEG data using the Discrete Wavelet Transform (DWT) method, which helped to break down the signals into different frequency components. These features were then standardized before being fed into the machine learning models.

## Results

We compared the performance of the models using various evaluation metrics, including accuracy, cross-validation, precision, recall, F1-score, and confusion matrices.

### Accuracy Comparison Between Different Classifiers:
<p align="center">
  <img src="https://i.imgur.com/rBy56u6.png" alt="Accuracy Comparison Between Classifiers" width="400">
</p>

*This bar graph compares the test and cross-validation accuracies of different models. The ANN model shows the best overall accuracy, followed closely by SVM.*

### Cross-Validation Accuracy Comparison Between Classifiers:
<p align="center">
  <img src="https://i.imgur.com/dv9LYxo.png" alt="Cross-Validation Accuracy Comparison" width="400">
</p>

*This line graph highlights the generalizability of each model based on cross-validation accuracy. Again, ANN and SVM perform the best, indicating stable performance.*

### Precision, Recall, and F1-Score Metrics Comparison:
<p align="center">
  <img src="https://i.imgur.com/a1UEqJf.png" alt="Precision, Recall, and F1-Score Metrics Comparison" width="400">
</p>

*The above chart compares the precision, recall, and F1-scores of the models. ANN and SVM show balanced performance, meaning they are good at minimizing false positives and false negatives.*

### Confusion Matrix of Different Classifiers:
<p align="center">
  <img src="https://i.imgur.com/JNfaM6P.png" alt="Confusion Matrix of Different Classifiers" width="400">
</p>

*The confusion matrix provides a detailed view of the model’s performance in distinguishing between "like" and "dislike." The ANN model performed well, correctly identifying most of the cases.*

### t-SNE Plot:
<p align="center">
  <img src="https://i.imgur.com/4mJouJl.png" alt="t-SNE Visualization of ANN" width="400">
</p>

*This t-SNE plot visualizes the separation between "like" and "dislike" categories in the data, as modeled by the Artificial Neural Network (ANN). Clear clustering indicates the model effectively differentiates between the two classes.*

### Actual Labels vs. Predicted Labels:
<p align="center">
  <img src="https://i.imgur.com/P6t0bja.png" alt="Actual vs Predicted Labels" width="400">
</p>

*This graph compares the actual labels from the dataset to the predicted labels from the classifiers. A good alignment indicates accurate predictions.*

### Summary of Results:
Based on the results, the **Artificial Neural Network (ANN)** model performed the best overall, with the highest accuracy, strong generalizability, and well-balanced precision and recall scores. The **Support Vector Machine (SVM)** also showed competitive performance, making it a reliable alternative.


