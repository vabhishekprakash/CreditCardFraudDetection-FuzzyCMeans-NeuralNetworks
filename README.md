# 💳 Credit Card Fraud Detection using Hidden Markov Model

 # Overview
 This project implements a multi-layered system designed to detect fraudulent credit card transactions by combining Fuzzy C-Means (FCM) Clustering and Neural Networks. The system aims to identify suspicious spending behaviors while significantly reducing the rate of false alarms.

# 🧠 How It Works
The detection process is executed in three distinct phases:

Authentication: Initial verification of card details and user identity.

Behavioral Analysis: Uses Fuzzy C-Means Clustering to find normal usage patterns based on past activity. A "suspicion score" is calculated based on how much a transaction deviates from these patterns.

Learning Phase: Suspicious transactions are passed to a Neural Network to determine if they are truly fraudulent or just an occasional deviation by a genuine user.

# 🛠️ Tech StackLanguage: 

Python Framework: Django (for web integration) 

Database: MySQL 

Design: HTML, CSS, JavaScript 

Environment: MATLAB-2014 (used for core simulation and results) 

## 📊 Dataset
The dataset used for this project is too large for GitHub. You can download it here: 

https://drive.google.com/file/d/18N1rzgXZ-CweeKU89BdI19ObF5BcFW1e/view?usp=sharing

Once downloaded, place the file in the `/dataset` folder.

# 📸 System Preview

Phase 1: Secure Authentication
The system begins with a secure login to verify the cardholder's identity.

Phase 2: Transaction Monitoring
Every transaction is passed through the Fuzzy C-Means logic to check for behavioral deviations.

Phase 3: Fraud Detection Results
When a transaction exceeds the "Suspicion Score," the Neural Network flags it.

## 📊 Results & Performance
The system was evaluated based on its ability to distinguish between genuine transactions and fraudulent ones.

### Accuracy and Metrics
* **Overall Accuracy:** 93.90%
* **FCM Clustering:** Successfully categorized users into spending profiles.
* **Neural Network:** Minimized false alarms by learning from suspicious deviations.

