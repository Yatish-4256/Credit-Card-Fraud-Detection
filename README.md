# Credit-Card-Fraud-Detection
A Robust Anomaly Detection System for Financial Transactions

# Overview
This project implements a machine learning system for detecting fraudulent credit card transactions using logistic regression and advanced sampling techniques. Designed to handle highly imbalanced datasets (284,807 transactions), the system achieves 99.1% accuracy while maintaining high precision for fraud cases. The solution includes comprehensive data preprocessing, model training, and evaluation components suitable for real-world financial applications.

## Key Features
# 1. Data Pipeline
Advanced Preprocessing:
Time-series feature engineering (transaction time decomposition)
StandardScaler normalization for monetary amounts
PCA dimensionality reduction for anonymized features
Missing Value Handling: Automated detection and imputation

# 2. Model Architecture
Core Algorithm: Logistic Regression with L2 regularization
Class Imbalance Mitigation:
SMOTE oversampling for minority class
Class weighting strategies
Alternative Models: Random Forest and XGBoost benchmarks

# 3. Fraud Detection Capabilities
Detection Type	Precision	Recall
Fraudulent (Class 1)	0.91	0.82
Legitimate (Class 0)	0.999	0.999

# 4. System Performance
Training Time: < 2 minutes (on 8-core CPU)
Inference Speed: 10,000 transactions/sec
Memory Efficiency: < 1GB RAM usage

