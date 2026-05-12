# 🏥 MediFlow AI — Hospital Management System

AI-powered hospital management system combining CNN, NLP, and Machine Learning for intelligent disease prediction.

## 🎯 Problem Solved
80% of rural Indian hospitals lack specialist doctors. MediFlow gives every hospital the power of AI-assisted diagnosis.

## 🚀 Features
- 🫁 **X-ray CNN Analysis** — DenseNet121, 90%+ accuracy
- 🩸 **Diabetes Prediction** — XGBoost, 74% CV accuracy
- ❤️ **Heart Disease** — XGBoost, 84% CV accuracy, 87% recall
- 📝 **Symptom NLP** — TF-IDF+LR, 24 diseases, 90%+ accuracy
- 👥 **Patient Management** — Full CRUD with SQLite
- 📊 **Analytics Dashboard** — Charts and insights
- 💊 **Inventory Management** — Low stock alerts
- 📄 **PDF Reports** — Downloadable medical reports
- 🔐 **Login System** — Role-based access (admin/doctor/nurse)

## 📊 Model Performance
| Model | Algorithm | CV Accuracy | CV Recall |
|-------|-----------|-------------|-----------|
| Diabetes | XGBoost | ~74% | ~74% |
| Heart Disease | XGBoost | ~84% | ~87% |
| Symptom NLP | TF-IDF+LR | ~90% | ~90% |
| X-ray CNN | DenseNet121 | ~90% | ~90% |

## 🛠️ Tech Stack
Python | XGBoost | TensorFlow | Streamlit | SQLite | scikit-learn | fpdf

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/Dilip2605/MediFlow.git
cd MediFlow

# Install dependencies
pip install -r requirements.txt

# Initialize database
python database/init_db.py

# Train ML models (run once)
python scripts/diabetes_model.py
python scripts/heart_model.py
python scripts/nlp_model.py
# For CNN: use Google Colab (free GPU)

# Run application
streamlit run app.py
```

## 🔑 Default Login
- Username: `admin`
- Password: `admin123`
- ⚠️ Change password after first login!

## 📁 Project Structure
```
MediFlow/
├── app.py              ← Main Streamlit UI
├── services/           ← ML prediction logic
├── database/           ← DB connection and setup
├── utils/              ← Helper functions and PDF
├── models/             ← Trained model files
├── scripts/            ← Training scripts
└── data/               ← Datasets
```

## 👤 Developer
**Dilipkumar P** | BE CSE, Anna University (8.1 CGPA)  
ML Intern — VCodeZ, Chennai | Govt Certified AI Programmer (NCVET)  
📧 dilipkumar26052002@gmail.com | 🔗 github.com/Dilip2605
