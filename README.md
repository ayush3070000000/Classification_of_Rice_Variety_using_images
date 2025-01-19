# Rice Classification Project

This project focuses on automating the classification of different rice varieties using image-based machine learning techniques. By leveraging machine vision and Convolutional Neural Networks (CNNs), it aims to provide an efficient, non-destructive solution for classifying rice varieties based on grain attributes like shape, texture, and color.

---

## Table of Contents

- [Abstract](#abstract)
- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Dataset Details](#dataset-details)
- [Data Preprocessing](#data-preprocessing)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [Contributions](#contributions)
- [References](#references)

---

## Abstract

Rice, a vital global grain, varies in appearance, taste, and nutrition across its varieties. Traditional manual methods for classifying rice are labor-intensive, costly, and inconsistent. This project addresses these challenges by implementing machine learning techniques to classify rice varieties based on grain images. A Convolutional Neural Network (CNN) achieved the highest accuracy of 96.12%, outperforming traditional machine learning models.

---

## Project Overview

### Objectives

1. Classify five distinct rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.
2. Extract meaningful features from images, including color, texture, and shape.
3. Evaluate the performance of various machine learning algorithms.

### Key Features

- Large dataset with 75,000 images (15,000 per variety).
- Advanced preprocessing techniques, including outlier removal and feature normalization.
- Comprehensive comparison of traditional models and CNNs.

---

## Setup and Installation

### Prerequisites

Ensure the following are installed:

- Python 3.8+
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `torch`, `torchvision`, `scikit-learn`

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/ayush-mourya/Rice-classification-using-Machine-Learning-and-CNN.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Rice-classification-using-Machine-Learning-and-CNN
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Details

The dataset consists of:

- 75,000 images of rice grains.
- Five varieties: Arborio, Basmati, Ipsala, Jasmine, Karacadag.
- Visual features include:
  - Morphological attributes: Area, Perimeter, Axis Lengths, Roundness, Compactness.
  - Color features from RGB, HSV, Lab\*, YCbCr, and XYZ color spaces.

---

## Data Preprocessing

Key preprocessing steps:

1. **Uniform Dataset**:
   - Balanced dataset with 15,000 images per variety.
2. **Image Resizing**:
   - Resized to 32x32 pixels for consistency.
3. **Data Augmentation**:
   - Horizontal flipping for variability.
4. **Pixel Normalization**:
   - Normalized pixel values to [0, 1].
5. **Grayscale and Binary Conversion**:
   - Simplifies images to structural features.
6. **Outlier Removal**:
   - Removed extreme values in area, perimeter, and axis lengths, improving class separability.

---

## Methodology

### Feature Extraction

- Morphological Features: Area, Perimeter, Major/Minor Axis Lengths.
- Color Features: 90 features from RGB, HSV, Lab\*, YCbCr, XYZ color spaces.
- Additional Features: Color Histograms, HOG, LBP, and Edge Features.

### Machine Learning Models

1. **Traditional Models**:
   - Logistic Regression
   - Decision Trees
   - Random Forest
   - K-Nearest Neighbor (KNN)
   - Support Vector Machines (SVM)
   - Multi-Layer Perceptron (MLP)
2. **Convolutional Neural Network (CNN)**:
   - Achieved the highest test accuracy (96.12%).

### CNN Implementation Details

The CNN architecture was implemented in PyTorch with the following steps:
1. **Dataset Preparation**:
   - Images resized to 32x32 pixels using `torchvision.transforms.Resize`.
   - Converted to grayscale with `Grayscale(num_output_channels=1)`.
   - Normalized to the range [-0.5, 0.5].
2. **Data Loading**:
   - Used `ImageFolder` to load images, splitting them into training and testing datasets.
   - Applied random shuffling for training batches to ensure robust learning.
3. **Training Setup**:
   - Defined a multi-layer CNN with ReLU activations and max pooling layers.
   - Used Cross-Entropy Loss for classification and Adam optimizer for model training.
4. **Evaluation Metrics**:
   - Model accuracy and loss monitored across epochs.
   - Visualized predictions for validation using Matplotlib.

---

## Results

### Initial Results

- Traditional models like Logistic Regression and Naive Bayes struggled due to high-dimensional data and feature overlap.
- Ensemble models suffered from overfitting, achieving perfect training accuracy but poor test performance.

### Improvements After Preprocessing

- Outlier removal and feature refinement improved test accuracies to 88-92%.
- CNN achieved the best accuracy (96.12%).

---

## Conclusion

This project highlights the effectiveness of advanced preprocessing and CNNs for image-based classification tasks. By addressing challenges like high-dimensionality and feature overlap, the model achieved a significant improvement in accuracy. The findings underscore the importance of robust data preparation and feature extraction in machine learning.

### Additional Notes from Deep Learning Notebook

The CNN model implementation emphasized:
- Efficient data transformation and normalization to enhance learning.
- Real-time visualization of data distribution and model predictions.
- Insights into training progress, including accuracy and loss trends over epochs.

These aspects strengthened the project’s ability to achieve high accuracy while maintaining generalization.

---

## References

- [Study on Machine Learning Techniques for Rice Classification](https://dergipark.org.tr/tr/download/article-file/3060874)
- [A Comprehensive Review of Rice Image Processing Techniques](https://dergipark.org.tr/en/download/article-file/1513632)
