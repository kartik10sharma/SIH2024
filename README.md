# ⚡ Power Demand Prediction System

A web-based application to forecast **daily, weekly, and monthly power demand** using LSTM models. Built with **FastAPI (backend)** and **ReactJS (frontend)**, this project aims to solve real-world energy forecasting problems with intuitive visualization and ML-driven predictions.

---

## 📌 Problem Statement

**SIH Problem Statement ID: 1624**
"Develop a system that predicts future electricity demand using historical load data, helping authorities manage grid load and energy production efficiently."

---

## 🚀 Features

* 🔁 **LSTM-based Time Series Forecasting**
* 📈 **Predict for Daily, Weekly, and Monthly ranges**
* 🎯 **MAPE (Mean Absolute Percentage Error) calculation**
* 📊 **Chart Visualization (React Chart.js)**
* 🧪 Integrated with a working dataset for Delhi region
* 🌐 Cross-origin support using CORS Middleware

---

## 🛠️ Tech Stack

| Component    | Tech                                       |
| ------------ | ------------------------------------------ |
| **Frontend** | ReactJS, Chart.js, Axios                   |
| **Backend**  | FastAPI, TensorFlow (Keras), Pandas, NumPy |
| **Model**    | LSTM (Long Short-Term Memory)              |
| **Language** | Python & JavaScript                        |

---

## 📁 Project Structure

```

├── node_modules/                 # React dependencies
├── public/                       # Static assets for React
├── src/
│   ├── App.jsx                   # React main component
│   ├── App.css                   # Styling
│   ├── assets/                   # Static images, logos
│   ├── main.jsx                  # React entry point
│   └── __pycache__/              # Python bytecode
├── delhi_load_with_peaks.csv    # Historical power load data
├── delhi_load_with_peaks_modi.csv
├── arima.joblib                 # (if used in advanced versions)
├── *.keras                      # LSTM model files
├── processed.csv                # Preprocessed dataset (optional)
├── pro.csv                      # Raw or transformed file (optional)
├── modelmaking.py               # Script to generate/train LSTM model
├── main.py                      # FastAPI backend entry point
├── package.json                 # React dependencies
├── vite.config.js               # ReactJS Vite config
├── .gitignore
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

```

---

## 📦 Installation & Setup

### 🔧 Backend (FastAPI)

1. **Create a virtual environment** (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the FastAPI server**:

```bash
uvicorn main:app --reload
```

It will start on: `http://127.0.0.1:8000`

---

### 🌐 Frontend (React)

1. **Navigate to `frontend/`**:

```bash
cd frontend
```

2. **Install dependencies**:

```bash
npm install
```

3. **Run the React App**:

```bash
npm run dev
```

Frontend will run at: `http://localhost:5173`

---

## 📡 API Endpoints

| Method | Endpoint         | Description              |
| ------ | ---------------- | ------------------------ |
| GET    | `/predict/day`   | Predict load for 1 day   |
| GET    | `/predict/week`  | Predict load for 7 days  |
| GET    | `/predict/month` | Predict load for 30 days |

**Query Parameter**: `?date=YYYY-MM-DD`
**Example**: `http://127.0.0.1:8000/predict/day?date=2024-09-20`

---


## ⚠️ Important Notes

* Make sure `main.py` is running **before** the frontend triggers predictions.
* `localhost:5173` (frontend) should match the backend on `localhost:8000`
* Modify CORS origins as needed in `main.py`

---

## 📊 Model Architecture

* **Model Type**: LSTM
* **Framework**: TensorFlow Keras
* **Input Features**: Previous 3 years of same day/month load
* **Target**: Load value for input date
* **Loss Function**: Mean Squared Error

---

## 📘 Dataset Description

* File: `delhi_load_with_peaks - delhi_load_with_peaks_modi.csv`
* Columns:

  * `Datetime`
  * `Load`
  * Additional engineered features: `Year`, `Month`, `Day`

---

## 🤝 Contributors

* Kartik Sharma

---

## 🏁 Future Improvements

*  Add actual load values for MAPE comparison
*  Improve weekly/monthly model accuracy
*  Dockerize for easy deployment
*  Add user authentication for different stakeholders

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


