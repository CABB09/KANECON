# Using Kolmogorov Arnold Network for Stock Market Values


# Neural Network Project - Financial Market Time Series Analysis

## Project Overview

This project presents a comparative analysis of three neural network models for predicting financial markets: **Artificial Neural Networks (ANN)**, **Convolutional Neural Networks (CNN)**, and **Kolmogorov-Arnold Networks (KAN)**. Using historical data from the financial markets of Google, Amazon, and Apple, these models are trained and tested to predict opening, closing, low, and high price values.

The project demonstrates that the **KAN model**, with its spline-based structure and learnable activation functions, outperforms the more traditional ANN and CNN models in terms of prediction accuracy, as measured by metrics like Mean Squared Error (MSE) and Mean Absolute Error (MAE).

---

## Table of Contents

1. [Introduction](#introduction)
2. [Theoretical Framework](#theoretical-framework)
   - [Backpropagation in Feedforward Neural Networks](#backpropagation-in-feedforward-neural-networks)
   - [Convolutional Neural Networks](#convolutional-neural-networks)
   - [Kolmogorov-Arnold Networks (KAN)](#kolmogorov-arnold-networks-kan)
3. [Dataset](#dataset)
4. [Proposed Models](#proposed-models)
   - [ANN Model](#ann-model)
   - [CNN Model](#cnn-model)
   - [KAN Model](#kan-model)
5. [Data Preparation](#data-preparation)
   - [Data for KAN and ANN](#data-for-kan-and-ann)
   - [Data for CNN](#data-for-cnn)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Introduction

Predicting stock market behavior using artificial intelligence (AI) has been an ongoing area of research since the late 20th century. This project compares three AI-based methods for financial market prediction: **Artificial Neural Networks (ANN)**, **Convolutional Neural Networks (CNN)**, and the more recent **Kolmogorov-Arnold Networks (KAN)**. The aim is to assess which model provides the best predictive accuracy for stock prices using historical data from popular markets such as Google (GOOG), Amazon (AMZN), and Apple (AAPL).

---

## Theoretical Framework

### Backpropagation in Feedforward Neural Networks

Feedforward neural networks (ANNs) map high-dimensional input spaces into output spaces using hierarchical layers of abstraction. These networks rely on **backpropagation** for updating the weights and biases through gradient descent. The ANN model in this project consists of:

- Multiple layers of neurons connected in a feedforward manner.
- Sigmoid, ReLU, and linear activation functions for various layers.
- A final output layer corresponding to the stock market predictions.

### Convolutional Neural Networks

**Convolutional Neural Networks (CNNs)** are highly effective at exploiting local spatial structures within data. In this project, CNNs are applied to synthetic images generated from stock price time series data, treating financial data as two-dimensional histograms. The CNN architecture includes convolutional layers followed by max-pooling and fully connected layers, optimized for financial market prediction.

### Kolmogorov-Arnold Networks (KAN)

The **Kolmogorov-Arnold Networks (KAN)** model is inspired by the Kolmogorov-Arnold representation theorem. Unlike traditional ANNs, which use fixed activation functions, KAN employs learnable splines for activation functions at the edges of the network, making it highly effective for non-linear function approximation. The KAN model in this project is shown to outperform both ANN and CNN in most cases.

---

## Dataset

The dataset used in this project was obtained from **Yahoo Finance**, containing historical data of stock prices for Google (GOOG), Amazon (AMZN), and Apple (AAPL). The features used include:

- **Open** price
- **Close** price
- **Low** price
- **High** price

Data was collected from January 1, 2014, to April 1, 2024, with daily intervals.

---

## Proposed Models

### ANN Model

The **Artificial Neural Network (ANN)** in this project has the following structure:

- **Input Layer**: 8 neurons (corresponding to the data from the previous two days and four stock price features).
- **Hidden Layers**: Two layers, each with 2 neurons.
- **Output Layer**: 4 neurons corresponding to the predicted stock prices (Open, Close, Low, High).
- Activation functions include sigmoid, ReLU, and linear functions, with a learning rate of 0.01. The loss function used is **Mean Squared Error (MSE)**.

### CNN Model

The **Convolutional Neural Network (CNN)** uses the following architecture:

- Two convolutional layers with ReLU activation and max-pooling.
- A fully connected layer with 160 neurons, followed by an output layer.
- Data is processed as synthetic 2D histograms from stock price data.

The optimizer used is **Adam** with a learning rate of 0.01 and 180 epochs.

### KAN Model

The **Kolmogorov-Arnold Network (KAN)** was implemented using the KAN Python library. The model has:

- **Input Layer**: 8 neurons (same as ANN).
- **Hidden Layers**: Three layers with 16, 12, and 8 neurons respectively.
- **Output Layer**: 4 neurons.
- The optimizer used is **LBFGS** with 30 steps. Special hyperparameters include spline degree (quadratic and cubic) and grid interval settings.

---

## Data Preparation

### Data for KAN and ANN

For KAN and ANN models, the data was preprocessed as follows:

- **Missing Values**: Forward-filled to handle gaps in the dataset.
- **Detrending**: The `detrend` function from Python was used to remove linear trends from the time series, ensuring the models focus on short-term fluctuations.
- **Scaling**: The data was normalized to a range of [0, 1].
- The data was split into 80% for training and 20% for testing.

### Data for CNN

The CNN model required generating synthetic 2D histograms from the time series data. The histograms were created from 50 sets of consecutive stock price points, producing 20x20 grayscale matrices. The CNN reads these histograms and predicts the stock prices for the next day.

---

## Results

The table below summarizes the results of each model for predicting stock prices across the three stocks (Amazon, Google, and Apple). Performance metrics include **R²**, **MSE**, and **MAE** for the test set, with the average across all stock price features:

### Results for Amazon (AMZN)

| Metric    | ANN      | CNN      | KAN (k=3) | KAN (k=2) |
|-----------|----------|----------|-----------|-----------|
| R²        | 0.4201   | 0.3511   | 0.7176    | 0.8604    |
| MSE       | 0.0166   | 0.0129   | 0.0081    | 0.0040    |
| MAE       | 0.0912   | 0.0920   | 0.0657    | 0.0482    |

### Results for Google (GOOG)

| Metric    | ANN      | CNN      | KAN (k=3) | KAN (k=2) |
|-----------|----------|----------|-----------|-----------|
| R²        | 0.8541   | 0.6532   | 0.9477    | 0.9489    |
| MSE       | 0.0053   | 0.0064   | 0.0019    | 0.0019    |
| MAE       | 0.0631   | 0.0652   | 0.0325    | 0.0322    |

### Results for Apple (AAPL)

| Metric    | ANN      | CNN      | KAN (k=3) | KAN (k=2) |
|-----------|----------|----------|-----------|-----------|
| R²        | 0.8761   | 0.2808   | 0.9287    | 0.9169    |
| MSE       | 0.0026   | 0.0076   | 0.0015    | 0.0018    |
| MAE       | 0.0397   | 0.0734   | 0.0303    | 0.0321    |

---

## Conclusion

The **Kolmogorov-Arnold Networks (KAN)** significantly outperformed traditional **ANN** and **CNN** models in most cases, especially when using quadratic splines. While ANN models showed signs of overfitting, the KAN model demonstrated robustness even after detrending the data. Future work could involve optimizing the ANN model to compete more closely with KAN or further exploring KAN's potential in financial market predictions.

---

## References

- Coqueret, G. (2021). *Machine Learning in Finance: From Theory to Practice*. Quantitative Finance.
- Liu, Z., et al. (2024). *Kolmogorov-Arnold Networks*. arXiv: 2404.19756.
- Mokhtari, S., Yen, K. K., & Liu, J. (2021). Effectiveness of AI in Stock Market Prediction. *International Journal of Computer Applications*.
- Mukherjee, S., et al. (2021). *Stock Market Prediction Using Deep Learning Algorithms*. CAAI Transactions on Intelligence Technology.
- Yahoo Finance. Historical Data Accessed: 2024-06-04.

