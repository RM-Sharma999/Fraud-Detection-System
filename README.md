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

Exploratory Data Analysis was conducted to understand transaction behavior, fraud patterns, and class imbalance in the dataset.

### Class Imbalance

- Fraudulent transactions account for **~0.13%** of the total data.
- Non-fraud transactions dominate the dataset (**~99.87%**).

This confirms a **highly imbalanced classification problem**, requiring careful model evaluation using precision, recall, and threshold tuning rather than accuracy alone.

---

### Fraud Percentage by Transaction Type

- **TRANSFER** transactions have the **highest fraud percentage**
- **CASH_OUT** transactions also show a significant fraud rate
- All other transaction types have nearly **0% fraud**

This insight helps the model focus on **high-risk transaction categories**.

---

### Transaction Amount vs Fraud Status

- Fraudulent transactions generally involve **higher transaction amounts**
- Fraud cases show a **higher median and wider spread** compared to non-fraud cases
- Legitimate transactions tend to cluster around lower amounts

This highlights **transaction amount** as an important feature for fraud detection.

---

### Fraud Cases by Transaction Type

- Fraud cases occur primarily in **TRANSFER** and **CASH_OUT** transactions.
- **CASH_OUT** transactions show a higher number of fraud cases compared to **TRANSFER** transactions.

This highlights the concentration of fraudulent activity in specific transaction types.

---

### Zero Sender Balance in Fraudulent Transactions

A significant number of fraudulent transactions occur when the **sender’s account balance before the transaction is zero**, particularly in:
- **TRANSFER**
- **CASH_OUT**

This suggests abnormal balance behavior and inconsistencies commonly associated with fraudulent activity.

---
