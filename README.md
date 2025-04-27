# Credit Card Fraud Detection using Autoencoders in PyTorch

This project demonstrates the use of Autoencoders, a type of unsupervised neural network, to detect fraudulent credit card transactions. Built with PyTorch, this model learns to reconstruct normal transactions and flags anomalies (potential fraud) based on reconstruction error.

# Project Overview

Credit card fraud detection is a highly imbalanced classification problem — fraudulent transactions are rare compared to legitimate ones. Traditional classification models struggle with this imbalance.

Autoencoders offer an alternative by learning to reconstruct normal transactions. Since they are not trained on fraudulent data, they tend to perform poorly at reconstructing them, which makes high reconstruction error a good indicator of potential fraud.

In anomaly detection with autoencoders, choosing the right threshold for reconstruction error is crucial. This threshold determines whether a transaction is classified as normal or fraudulent based on how well the model can reconstruct it. If the reconstriction error is greater than threshold, we flag the transaction as fraud, and normal otherwise.

The right threshold depends on business priorities — better to catch more fraud (higher recall) or avoid false alarms (higher precision)?
In this project, we explore different approches to set the threshold:
1. The threshold that maximizes F1 Score.
2. The percentile-based threshold(e.g., 95th percentile of the reconstruction errors associated with the normal transcations).
3. The threshold that maximizes the difference between True Positive Rate (TPR) and False Positive Rate (FPR).


# Anomaly detection auto-encoder model Architecture

- The autoencoder:
    - Input Layer: Takes the full set of features from the dataset (typically 30 for the Kaggle credit card dataset).

    - Encoder: Compresses the input into a lower-dimensional latent space.
        - Reduces the dimensionality from input_dim → 16 → 8 using ReLU activations.

    - Decoder: Attempts to reconstruct the original input from the latent space
        - Reconstructs the original input with the reverse architecture 8 → 16 → input_dim.

- Loss Function: Mean Squared Error (MSE) is used to measure reconstruction quality.
- Optimizer: Adam optimizer.



# Dataset

We use the Kaggle Credit Card Fraud Detection dataset with 284807 transaction among which about 0.172% are fraudulent.
