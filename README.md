# Fraud Detection System

A complete machine learning–based project that detects fraudulent transactions using Python. This project includes data preprocessing, model training & evaluation, exploratory data analysis(EDA), and deployment logic to identify fraud patterns and predict fraud with a tuned classifier.

---

## Objective

Financial fraud poses a significant risk to digital payment systems and financial institutions. The objective of this project is to **build an efficient machine learning model** that can accurately identify fraudulent transactions based on historical transaction data.

This project focuses on:
- Identifying patterns associated with fraudulent behavior  
- Handling class imbalance in fraud datasets  
- Training and evaluating classification models  
- Improving fraud detection using calibrated predictions  

---

## Dataset Summary

The dataset contains transaction-level financial records used to identify fraudulent activities. Each row represents a single transaction, with features describing transaction type, amount, and account balances before and after the transaction.

### Key Features

- **type** – Type of transaction (e.g., transfer, cash-out, payment)  
- **amount** – Transaction amount  
- **oldbalanceOrg** – Sender’s account balance before the transaction  
- **newbalanceOrig** – Sender’s account balance after the transaction  
- **oldbalanceDest** – Receiver’s account balance before the transaction  
- **newbalanceDest** – Receiver’s account balance after the transaction  
- **target** – Fraud indicator  
  - `1` → Fraudulent transaction  
  - `0` → Legitimate transaction

### Key characteristics:
- Highly imbalanced dataset  
- Numerical transaction-related features  
- Binary classification target

---

## Exploratory Data Analysis
