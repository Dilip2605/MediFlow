# 🏥 MediFlow AI

<div align="center">

## An AI-Powered Healthcare Platform for Smarter Hospital Management

MediFlow AI is a healthcare-focused project that combines Artificial Intelligence, Machine Learning, NLP, and Hospital Management into one intelligent platform.

The goal of this project is to help hospitals and clinics improve diagnosis support, patient management, reporting, and overall healthcare workflow efficiency using AI technologies.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge\&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-red?style=for-the-badge\&logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-orange?style=for-the-badge\&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-orange?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-blue?style=for-the-badge\&logo=sqlite)

### Links

[Live Demo](http://10.211.157.208:8501) • [GitHub](https://github.com/Dilip2605/MediFlow) • [LinkedIn](https://linkedin.com/in/dilipkumar-p-20020526cse)

</div>

---

# 📌 Overview

Access to quality healthcare and early diagnosis is still a major challenge in many areas, especially in rural and resource-limited environments.

MediFlow AI was built to explore how Artificial Intelligence can support healthcare professionals by combining:

* Medical image analysis
* Disease prediction models
* Natural language symptom analysis
* Patient management workflows
* Hospital analytics and reporting

The system combines Deep Learning, NLP, and Machine Learning models with a modular hospital management platform.

---

# 🚀 Key Highlights

* AI-assisted disease prediction system
* DenseNet121 CNN for chest X-ray analysis
* NLP-based symptom understanding engine
* XGBoost-based diabetes and heart disease prediction
* Role-based hospital management system
* PDF medical report generation
* Inventory and billing management
* Appointment scheduling system
* Patient history and analytics dashboard
* Modular service-oriented architecture

---

# ⚙️ Core Features

## AI Modules

| Module                   | Technology                   | Purpose                          |
| ------------------------ | ---------------------------- | -------------------------------- |
| Chest X-ray Analysis     | DenseNet121 CNN              | Pneumonia and TB detection       |
| Diabetes Prediction      | XGBoost                      | Diabetes risk prediction         |
| Heart Disease Prediction | XGBoost                      | Cardiac risk assessment          |
| Symptom Analysis         | TF-IDF + Logistic Regression | Disease prediction from symptoms |

---

## Hospital Management Features

* Patient Registration and Records
* Appointment Booking
* Digital Prescriptions
* Lab Test Management
* Medical Billing System
* Inventory Tracking
* PDF Report Generation
* Doctor Notes and Patient History
* Analytics Dashboard
* Role-Based Authentication

---

# 🏗️ System Architecture

```text
User Interface (Streamlit)
        ↓
Application Layer (Python Services)
        ↓
AI/ML Models + Database Layer
        ↓
Reports, Predictions, Analytics
```

---

# 🛠️ Technology Stack

| Category         | Technologies                      |
| ---------------- | --------------------------------- |
| Frontend         | Streamlit, Custom CSS             |
| Backend          | Python                            |
| Machine Learning | XGBoost, scikit-learn             |
| Deep Learning    | TensorFlow, Keras                 |
| NLP              | TF-IDF, Logistic Regression, NLTK |
| Database         | SQLite                            |
| Visualization    | Matplotlib                        |
| Reporting        | FPDF                              |

---

# 📁 Project Structure

```text
MediFlow/
│
├── 📄 app.py                    ← Main Streamlit application (16 pages)
│
├── 📁 database/
│   ├── db.py                    ← Cached SQLite connection
│   └── init_db.py               ← Creates all 11 tables + sample data
│
├── 📁 services/
│   ├── diabetes_service.py      ← XGBoost diabetes prediction logic
│   ├── heart_service.py         ← XGBoost heart disease prediction logic
│   ├── nlp_service.py           ← TF-IDF NLP symptom analysis logic
│   └── xray_service.py          ← DenseNet121 CNN X-ray analysis logic
│
├── 📁 utils/
│   ├── helpers.py               ← Shared utility functions
│   └── pdf_generator.py         ← Medical PDF report generation
│
├── 📁 models/                   ← Trained ML model files (not in GitHub)
│   ├── diabetes_model.pkl
│   ├── diabetes_scaler.pkl
│   ├── heart_model.pkl
│   ├── heart_scaler.pkl
│   ├── heart_features.pkl
│   ├── nlp_pipeline.pkl
│   ├── nlp_label_encoder.pkl
│   └── cnn_xray_model.keras
│
├── 📁 scripts/                  ← Model training scripts (run once)
│   ├── diabetes_model.py
│   ├── heart_model.py
│   ├── nlp_model.py
│   └── cnn_model.py
│
├── 📁 data/                     ← Datasets (not in GitHub)
│   ├── ml/diabetes.csv
│   ├── ml/heart.csv
│   ├── nlp/Symptom2Disease.csv
│   └── xray/pneumonia/
│
├── 📄 requirements.txt
├── 📄 .gitignore
└── 📄 README.md
```

---

# 🤖 Machine Learning Models

## Diabetes Prediction

* Algorithm: XGBoost
* Dataset: Pima Indians Diabetes Dataset
* Cross Validation Accuracy: ~74%
* Feature Engineering applied
* Risk classification support

## Heart Disease Prediction

* Algorithm: XGBoost
* Dataset: UCI Heart Disease Dataset
* Cross Validation Accuracy: ~84%
* Engineered clinical features included

## Symptom NLP Engine

* TF-IDF Vectorization
* Logistic Regression Classifier
* Multi-disease symptom prediction
* Natural language symptom understanding

## X-ray Analysis

* DenseNet121 Transfer Learning
* TensorFlow/Keras implementation
* Chest X-ray classification
* Pneumonia detection workflow

---

# 🧠 Engineering Concepts Used

* Modular Service Architecture
* Feature Engineering
* Transfer Learning
* Database Normalization
* Role-Based Access Control
* Machine Learning Pipelines
* PDF Report Automation
* Analytics Dashboard Design
* Data Processing Pipelines

---

# ⚡ Installation Guide

## 1. Clone Repository

```bash
git clone https://github.com/Dilip2605/MediFlow.git
cd MediFlow
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Initialize Database

```bash
python database/init_db.py
```

## 4. Run Application

```bash
streamlit run app.py
```

---

# Default Login

```text
Username: admin
Password: admin123
```

---

# 📸 Screenshots

> Add dashboard and prediction screenshots here.

| Dashboard  | Prediction Module | Analytics  |
| ---------- | ----------------- | ---------- |
| Screenshot | Screenshot        | Screenshot |

---

# 🌐 Deployment

## Streamlit Cloud

```bash
git push origin main
```

Deploy directly using Streamlit Community Cloud.

---

# 🔮 Future Enhancements

* Voice-based symptom input
* Multi-language support
* Telemedicine integration
* Mobile application
* REST API integration
* Docker deployment
* PostgreSQL migration
* Multi-hospital support
* Drug interaction analysis

---

# ❤️ Why This Project Matters

This project was created with the idea of making healthcare systems smarter, more accessible, and more efficient using AI. Instead of building only prediction models, the focus was also placed on real hospital workflows such as appointments, patient history, billing, and reporting.

MediFlow AI combines technical AI concepts with practical healthcare management to create a more realistic end-to-end healthcare platform.

---

# 👨‍💻 Developer

## Dilipkumar P

Machine Learning Engineer | AI Developer

* BE Computer Science Engineering
* ML Intern at VCodeZ, Chennai
* Certified AI Programmer — NCVET & TN Skill Corporation
* Focus Areas: NLP, Computer Vision, Healthcare AI

### Connect

* GitHub: [https://github.com/Dilip2605](https://github.com/Dilip2605)
* LinkedIn: [https://linkedin.com/in/dilipkumar-p-20020526cse](https://linkedin.com/in/dilipkumar-p-20020526cse)
* Email: [dilipkumar26052002@gmail.com](mailto:dilipkumar26052002@gmail.com)

---

# License

This project is licensed under the MIT License.

---

<div align="center">

### If you found this project useful, consider giving it a star.

Built with a focus on AI-driven healthcare innovation.

</div>
