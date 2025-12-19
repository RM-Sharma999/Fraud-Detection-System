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

<img width="969" height="502" alt="image" src="https://github.com/user-attachments/assets/781229c1-3f20-4509-8ef5-d703ffddf4a3" />

---

### Fraud Percentage by Transaction Type

- **TRANSFER** transactions have the **highest fraud percentage**
- **CASH_OUT** transactions also show a significant fraud rate
- All other transaction types have nearly **0% fraud**

This insight helps the model focus on **high-risk transaction categories**.

<img width="856" height="456" alt="image" src="https://github.com/user-attachments/assets/92c05e14-56d7-402a-b391-78a8f3e0fc2c" />

---

### Transaction Amount vs Fraud Status

- Fraudulent transactions generally involve **higher transaction amounts**
- Fraud cases show a **higher median and wider spread** compared to non-fraud cases
- Legitimate transactions tend to cluster around lower amounts

This highlights **transaction amount** as an important feature for fraud detection.

<img width="734" height="479" alt="image" src="https://github.com/user-attachments/assets/b690a7bb-88df-4eef-8c99-79b34a8d3e44" />

---

### Fraud Cases by Transaction Type

- Fraud cases occur primarily in **TRANSFER** and **CASH_OUT** transactions.
- **CASH_OUT** transactions show a higher number of fraud cases compared to **TRANSFER** transactions.

This highlights the concentration of fraudulent activity in specific transaction types.

<img width="700" height="456" alt="image" src="https://github.com/user-attachments/assets/16483adc-96af-4ec9-ae05-2e5116346d84" />

---

### Zero Sender Balance in Fraudulent Transactions

A significant number of fraudulent transactions occur when the **sender’s account balance before the transaction is zero**, particularly in:
- **TRANSFER**
- **CASH_OUT**

This suggests abnormal balance behavior and inconsistencies commonly associated with fraudulent activity.

<img width="715" height="456" alt="image" src="https://github.com/user-attachments/assets/0d6f61e9-86f1-4394-95eb-477ffb64fe33" />

---
