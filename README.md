# 🤖 FactoryGuard AI – Predictive Maintenance System

## 🚀 Overview

FactoryGuard AI is a Machine Learning-based Predictive Maintenance System designed to forecast machine failures using sensor data such as temperature, vibration, and pressure.

The system helps industries prevent unexpected breakdowns by predicting failures in advance, reducing downtime and maintenance costs.

---

## 🧠 Features

* 📊 Time-series feature engineering (rolling mean, lag features)
* ⚙️ Machine Learning model (XGBoost / RandomForest)
* 🔍 Real-time failure prediction
* 🌐 Flask API for deployment
* 🎨 Streamlit UI for interactive dashboard
* 📈 Prediction history tracking

---

## 🏗️ Tech Stack

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost
* **Backend:** Flask
* **Frontend:** Streamlit
* **Model Saving:** Joblib

---

## 📁 Project Structure

```
predictive-iot/
│
├── data.csv
├── train.py
├── app.py
├── ui.py
├── model.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/YOUR_USERNAME/predictive-iot.git
cd predictive-iot
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Step 1: Train the model

```
python train.py
```

### Step 2: Start Flask API

```
python app.py
```

### Step 3: Run Streamlit UI

```
streamlit run ui.py
```

---

## 🌐 API Endpoint

### POST `/predict`

#### Request JSON:

```
{
  "temperature": 80,
  "vibration": 45,
  "pressure": 130,
  "temp_mean_3": 75,
  "vib_std_3": 5,
  "temp_lag1": 72,
  "temp_lag2": 70
}
```

#### Response:

```
{
  "failure_prediction": 1,
  "failure_probability": 0.87
}
```

---

## 📊 Model Details

* Handles class imbalance using weighting techniques
* Evaluated using **Precision-Recall AUC**
* Optimized for real-time prediction

---

## 🎯 Use Case

* Industrial IoT monitoring
* Predictive maintenance in manufacturing
* Smart factory automation

---

## ⚠️ Limitations

* Uses sample dataset (not real industrial data)
* Feature inputs are manually provided in UI
* No database integration (session-based history)

---

## 🚀 Future Enhancements

* 📊 Real-time sensor integration (IoT devices)
* 🧠 Advanced models (LightGBM, Deep Learning)
* 🔐 Authentication system
* ☁️ Cloud deployment (AWS / Render)
* 📈 Live dashboards & analytics

---

## 👨‍💻 Author

Developed as part of a Production AI / MLOps Project.

---

## ⭐ Acknowledgment

This project is inspired by real-world industrial predictive maintenance systems used in smart manufacturing.

---
