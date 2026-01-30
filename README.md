# Credit Card Fraud Detection (Machine Learning)

This project demonstrates an end-to-end **Credit Card Fraud Detection System** using **Python, Machine Learning, and Streamlit**.

## 🚀 Features
- Synthetic dataset with anonymous card transactions
- Fraud detection using Logistic Regression
- Feature scaling and categorical encoding
- Real-time fraud prediction via Streamlit dashboard
- Explainable ML: shows top contributing features
- Beginner-friendly & deployable


## 📁 Project Structure
Credit Card Fraud Detection/
│
├── app.py
├── train_model.py
├── fraud_model.pkl
├── scaler.pkl
├── model_columns.pkl
├── requirements.txt
├── README.md
└── data/
└── transactions.csv


## 📊 Dataset
Synthetic dataset with the following columns:
- card_id
- amount
- hour
- txn_type
- merchant_category
- txns_last_24h
- avg_amount_7d
- is_fraud


## ▶️ How to Run Locally
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py


 🌐 Deployment

This app can be deployed using Streamlit Cloud directly from GitHub.
🧠 Model
Algorithm: Logistic Regression
Binary classification (Fraud / Not Fraud)
Explainable using feature contribution scores

🔮 Future Improvements
Card-wise transaction history analysis
Advanced ML models (Random Forest, XGBoost)
Visualization dashboards
Real-time transaction simulation


