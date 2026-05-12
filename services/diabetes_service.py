import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ═══════════════════════════════════════════════════════════════
# DIABETES PREDICTION SERVICE
# Purpose: All diabetes prediction logic in ONE place
# Algorithm: XGBoost with Feature Engineering
# Dataset: Pima Indians Diabetes (768 patients)
# CV Accuracy: ~74% | CV Recall: ~74%
# ═══════════════════════════════════════════════════════════════

# Feature column names — must match training order exactly!
DIABETES_FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
    'BMI_Category', 'Age_Group', 'Glucose_BMI', 'High_Risk'
]


@st.cache_resource
def load_diabetes_models():
    """
    Load diabetes model and scaler from disk.
    Cached — loads once, reused for all predictions.
    
    Returns:
        (model, scaler, True) if successful
        (None, None, False) if files not found
    """
    try:
        model = joblib.load("models/diabetes_model.pkl")
        scaler = joblib.load("models/diabetes_scaler.pkl")
        return model, scaler, True
    except FileNotFoundError:
        return None, None, False
    except Exception as e:
        print(f"Error loading diabetes model: {e}")
        return None, None, False


def engineer_diabetes_features(
    pregnancies: int,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    dpf: float,
    age: int
) -> pd.DataFrame:
    """
    Create feature-engineered DataFrame for diabetes prediction.
    
    Original features (8):
        Pregnancies, Glucose, BloodPressure, SkinThickness,
        Insulin, BMI, DiabetesPedigreeFunction, Age
    
    Engineered features (4 new):
        BMI_Category: Medical obesity classification (0-3)
        Age_Group: Age-based risk grouping (0-3)
        Glucose_BMI: Combined risk factor (Glucose × BMI / 1000)
        High_Risk: Both Glucose > 140 AND BMI > 30 flag (0/1)
    
    Total: 12 features
    """
    from utils.helpers import (
        calculate_bmi_category,
        calculate_age_group,
        calculate_high_risk_flag,
        calculate_glucose_bmi
    )

    # Calculate engineered features
    bmi_category = calculate_bmi_category(bmi)
    age_group = calculate_age_group(age)
    glucose_bmi = calculate_glucose_bmi(glucose, bmi)
    high_risk = calculate_high_risk_flag(glucose, bmi)

    # Create DataFrame with exact feature order used in training
    features = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age,
        bmi_category, age_group, glucose_bmi, high_risk
    ]], columns=DIABETES_FEATURES)

    return features


def predict_diabetes(
    pregnancies: int,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    dpf: float,
    age: int
) -> dict:
    """
    Main diabetes prediction function.
    
    Steps:
    1. Load model and scaler
    2. Engineer features
    3. Scale features
    4. Predict
    5. Return structured result
    
    Returns dict with:
        prediction: 0 (healthy) or 1 (diabetic)
        probability: float 0.0 to 1.0
        probability_healthy: float 0.0 to 1.0
        result_text: "DIABETIC" or "HEALTHY"
        risk_level: "LOW RISK" / "MODERATE RISK" / "HIGH RISK"
        model_loaded: True/False
        error: None or error message
    """
    from utils.helpers import get_risk_level

    # Default error response
    error_response = {
        "prediction": None,
        "probability": 0.0,
        "probability_healthy": 0.0,
        "result_text": "ERROR",
        "risk_level": "UNKNOWN",
        "model_loaded": False,
        "error": None
    }

    # Load models
    model, scaler, loaded = load_diabetes_models()
    if not loaded:
        error_response["error"] = "Diabetes model not loaded. Run: python scripts/diabetes_model.py"
        return error_response

    try:
        # Step 1: Engineer features
        features_df = engineer_diabetes_features(
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        )

        # Step 2: Scale features
        features_scaled = scaler.transform(features_df)

        # Step 3: Predict
        prediction = int(model.predict(features_scaled)[0])
        probabilities = model.predict_proba(features_scaled)[0]
        prob_diabetic = float(probabilities[1])
        prob_healthy = float(probabilities[0])

        return {
            "prediction": prediction,
            "probability": prob_diabetic,
            "probability_healthy": prob_healthy,
            "result_text": "DIABETIC" if prediction == 1 else "HEALTHY",
            "risk_level": get_risk_level(prob_diabetic),
            "model_loaded": True,
            "error": None,
            "features_used": {
                "Glucose": glucose,
                "BMI": bmi,
                "Age": age,
                "Pregnancies": pregnancies,
                "High Risk Flag": "Yes" if (glucose > 140 and bmi > 30) else "No"
            }
        }

    except Exception as e:
        error_response["error"] = f"Prediction error: {str(e)}"
        return error_response


def get_diabetes_recommendations(prediction: int, glucose: float, bmi: float) -> list:
    """
    Generate medical recommendations based on prediction result.
    
    Returns list of recommendation strings for display and PDF.
    """
    if prediction == 1:
        return [
            "Consult an endocrinologist or diabetologist immediately.",
            "Request HbA1c blood test for confirmed diagnosis.",
            "Monitor fasting blood glucose daily.",
            f"{'Urgent weight management required — BMI is in obese range.' if bmi >= 30 else 'Maintain current BMI and monitor weight monthly.'}",
            f"{'Urgent glucose control needed — levels critically high.' if glucose > 180 else 'Follow diabetic diet — avoid high sugar foods.'}",
            "Regular exercise: 30 minutes walking daily.",
            "Follow-up appointment in 2 weeks.",
        ]
    else:
        return [
            "No diabetes detected — maintain healthy lifestyle.",
            "Annual blood glucose screening recommended.",
            "Maintain balanced diet with low sugar intake.",
            f"{'Work towards reducing BMI to normal range (18.5-24.9).' if bmi >= 25 else 'Maintain current healthy BMI.'}",
            "Regular exercise: minimum 150 minutes per week.",
            "Next check-up in 12 months.",
        ]
