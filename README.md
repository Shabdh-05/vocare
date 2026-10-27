# 🎙️ Vocare – AI-Based Voice Health Screening System

Vocare is an AI-powered system designed for **early detection of vocal health irregularities** using acoustic signal analysis and machine learning.
It allows users to upload or record voice samples and receive **instant screening results, risk levels, and clinical insights**.

---

## 🚀 Key Features

* 🎤 Upload or record voice samples (WAV)
* 🤖 AI-based vocal health screening using ML model (AVHFM)
* 📊 Real-time acoustic feature extraction
* ⚠️ Risk classification (Healthy / Mild / Moderate / High Risk)
* 📈 Detailed metrics (Pitch, Jitter, Shimmer, HNR, etc.)
* 📜 Recommendations & insights
* 🧾 Downloadable PDF report
* 🕘 Analysis history tracking

---

## 🧠 Machine Learning Model (Core of the Project)

### 🔬 Model Name: **AVHFM (Acoustic Voice Health Feature Model)**

The AVHFM model is a **tree-based ensemble learning model** designed to detect vocal irregularities from acoustic signals.

---

## ⚙️ How the Model Works

### 1. 🎧 Raw Audio Input

* Voice is recorded at ~16kHz sampling rate
* Example: 5 seconds → ~80,000 audio samples

---

### 2. 🔄 Feature Extraction

Raw audio is converted into meaningful acoustic features:

| Feature    | Description                  |
| ---------- | ---------------------------- |
| Pitch (F0) | Frequency of voice           |
| Jitter     | Frequency variation          |
| Shimmer    | Amplitude variation          |
| HNR        | Harmonic-to-noise ratio      |
| MFCC       | Spectral voice features      |
| AVHI       | Composite voice health index |

👉 These reduce thousands of samples into **~20–50 structured features**

---

### 3. 🌳 Ensemble Learning (Multiple Trees)

Instead of using a single model:

* 100+ decision trees are trained
* Each tree learns **different patterns** using:

  * Different subsets of data
  * Different subsets of features

---

### 4. 🗳️ Voting Mechanism

Each tree gives a prediction:

```
Tree 1 → Irregular  
Tree 2 → Healthy  
Tree 3 → Irregular  
```

Final output is based on **majority voting**:

```
Final Prediction → Irregular
Confidence → % of trees voting
```

---

### 5. 📊 Output Interpretation

The model returns:

* ✅ Status → Healthy / Warning / Risk
* 📈 Confidence Score (0–100)
* ⚠️ Risk Level
* 📊 Metrics
* 💡 Insights & Recommendations

---

## 📊 Performance Comparison of All Models

| Model | Acc. | Prec. | Rec. | F1 |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.78 | 0.76 | 0.74 | 0.74 |
| Support Vector Machine | 0.85 | 0.84 | 0.82 | 0.83 |
| Random Forest | 0.88 | 0.87 | 0.86 | 0.87 |
| Gradient Boosting | 0.89 | 0.88 | 0.89 | 0.89 |
| **AVHFM (Proposed)** | **0.93** | **0.92** | **0.91** | **0.92** |

👉 AVHFM performs best because it:

* Captures **non-linear patterns**
* Reduces overfitting using ensemble learning
* Handles complex voice variations

---

## 🧪 Backend (Flask API)

* Built using **Flask**
* Handles:

  * File upload
  * Feature extraction
  * Model prediction
* Returns JSON response with:

  * Score
  * Diagnosis
  * Metrics
  * Recommendations

### API Endpoint

```
POST /predict
```

---

## 💻 Frontend (Next.js)

* Built using **Next.js + Tailwind CSS**
* Features:

  * Voice upload & recording UI
  * Real-time analysis progress
  * Interactive dashboards
  * History tracking
  * PDF report generation

---

## 📁 Project Structure

```
ml_data/
│
├── frontend/           # Next.js UI
├── vocare_backend/     # Flask ML API
├── README.md
```

---

## ⚠️ Important Note

* Model files (`.pkl`) are not included due to size constraints
* Please download them separately and place inside:

```
vocare_backend/
```

---

## 🧠 Why This Project is Important

* Enables **early detection of vocal disorders**
* Helps in **preventive healthcare**
* Can be extended to:

  * Telemedicine
  * Clinical screening tools
  * Mobile health applications

---

## 🚀 Future Improvements

* 🎤 Real-time voice streaming analysis
* 📱 Mobile app integration
* ☁️ Cloud deployment (AWS / GCP)
* 🧠 Deep learning models (CNN / RNN)
* 👥 Multi-user profiles & authentication
* 📊 Advanced analytics dashboard

---

## 👨‍💻 Author

Developed as part of an AI/ML project for vocal health screening.

---

## 📜 Disclaimer

This system is intended for **early screening purposes only** and is not a substitute for professional medical diagnosis.

---
