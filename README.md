# Kernel Principal Component Analysis (Kernel PCA) Analysis

This directory contains an implementation of **Kernel Principal Component Analysis (Kernel PCA)**, a non-linear dimensionality reduction technique that uses kernel methods to project data into a higher-dimensional space where it becomes linearly separable.

## Dataset Overview

The model uses the `Wine.csv` dataset, which contains:

- **Features**: 13 chemical constituents (Alcohol, Malic acid, Ash, etc.) used to determine the origin of wines.
- **Target**: Wine Class (three different cultivators).
- **Goal**: Reduce the high-dimensional feature space to two principal components while preserving non-linear relationships, and then classify the wine types.

## Implementation Steps

The implementation follows these key steps:

1.  **Data Preprocessing**:
    - Split the dataset into Training (80%) and Test (20%) sets.
    - Applied **Feature Scaling** (Standardization) to all 13 features.
2.  **Applying Kernel PCA**:
    - Used `sklearn.decomposition.KernelPCA` with the **RBF (Gaussian) kernel**.
    - Reduced the feature space to **2 principal components** (PC1 and PC2).
3.  **Model Training**:
    - Trained a **Logistic Regression** classifier on the reduced 2-dimensional feature space.
4.  **Evaluation**:
    - Evaluated the model using a **Confusion Matrix** and **Accuracy Score**.
5.  **Visualization**:
    - Plotted decision boundaries for both training and test sets in the 2D principal component space.

## Results

The Kernel PCA combined with Logistic Regression yielded exceptional results:

- **Accuracy**: 100% on the test set (1.0 accuracy score).
- **Confusion Matrix**: Perfect classification for all three wine classes on the test set.
- The visualization demonstrates that the RBF kernel effectively captured the non-linear structure of the wine data, projecting it into a 2D space where the classes are perfectly separable by linear boundaries.
