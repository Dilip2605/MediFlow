import pandas as pd
import numpy as np
import joblib
import streamlit as st

# ═══════════════════════════════════════════════════════════════
# HEART DISEASE PREDICTION SERVICE
# Purpose: All heart disease prediction logic
# Algorithm: XGBoost with Clinical Feature Engineering
# Dataset: UCI Heart Disease (302 unique patients after dedup)
# CV Accuracy: ~84% | CV Recall: ~87%
# ═══════════════════════════════════════════════════════════════


@st.cache_resource
def load_heart_models():
    """Load heart disease model, scaler, and feature names"""
    try:
        model = joblib.load("models/heart_model.pkl")
        scaler = joblib.load("models/heart_scaler.pkl")
        features = joblib.load("models/heart_features.pkl")
        return model, scaler, features, True
    except FileNotFoundError:
        return None, None, None, False
    except Exception as e:
        print(f"Error loading heart model: {e}")
        return None, None, None, False


def engineer_heart_features(
    age: int, sex: int, cp: int, trestbps: float,
    chol: float, fbs: int, restecg: int, thalach: float,
    exang: int, oldpeak: float, slope: int, ca: int, thal: int
) -> pd.DataFrame:
    """
    Create feature-engineered DataFrame for heart disease prediction.
    
    Original features (13):
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    
    Engineered features (6 new):
        Age_Risk: Age-based cardiac risk (0-3)
        BP_Category: WHO blood pressure classification (0-3)
        Chol_Risk: Cholesterol risk level (0-2)
        HR_Efficiency: Achieved HR / Predicted max HR
        Critical_Risk: Combined high-risk flag (0/1)
        Age_Sex_Risk: Men over 55 high risk flag (0/1)
    
    Total: 19 features
    """
    from utils.helpers import (
        calculate_age_risk,
        calculate_bp_category,
        calculate_chol_risk,
        calculate_hr_efficiency,
        calculate_critical_risk,
        calculate_age_sex_risk
    )

    age_risk = calculate_age_risk(age)
    bp_category = calculate_bp_category(trestbps)
    chol_risk = calculate_chol_risk(chol)
    hr_efficiency = calculate_hr_efficiency(thalach, age)
    critical_risk = calculate_critical_risk(cp, exang, oldpeak)
    age_sex_risk = calculate_age_sex_risk(age, sex)

    _, _, features_list, loaded = load_heart_models()
    if not loaded:
        return None

    features = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal,
        age_risk, bp_category, chol_risk,
        hr_efficiency, critical_risk, age_sex_risk
    ]], columns=features_list)

    return features


def predict_heart_disease(
    age: int, sex: int, cp: int, trestbps: float,
    chol: float, fbs: int, restecg: int, thalach: float,
    exang: int, oldpeak: float, slope: int, ca: int, thal: int
) -> dict:
    """
    Main heart disease prediction function.
    
    Returns structured prediction result dict.
    """
    from utils.helpers import get_risk_level

    error_response = {
        "prediction": None, "probability": 0.0,
        "result_text": "ERROR", "risk_level": "UNKNOWN",
        "model_loaded": False, "error": None
    }

    model, scaler, features_list, loaded = load_heart_models()
    if not loaded:
        error_response["error"] = "Heart model not loaded. Run: python scripts/heart_model.py"
        return error_response

    try:
        features_df = engineer_heart_features(
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        )

        if features_df is None:
            error_response["error"] = "Feature engineering failed"
            return error_response

        # Scale features
        features_scaled = scaler.transform(features_df)
        features_scaled_df = pd.DataFrame(features_scaled, columns=features_list)

        # Predict
        prediction = int(model.predict(features_scaled_df)[0])
        probabilities = model.predict_proba(features_scaled_df)[0]
        prob_disease = float(probabilities[1])

        # Calculate engineered values for display
        from utils.helpers import (
            calculate_age_risk, calculate_bp_category,
            calculate_critical_risk, calculate_age_sex_risk
        )

        return {
            "prediction": prediction,
            "probability": prob_disease,
            "probability_healthy": float(probabilities[0]),
            "result_text": "HEART DISEASE" if prediction == 1 else "HEALTHY",
            "risk_level": get_risk_level(prob_disease),
            "model_loaded": True,
            "error": None,
            "clinical_flags": {
                "Chest Pain Type": f"{'⚠️ Typical Angina' if cp == 0 else '✅ Not typical angina'}",
                "Exercise Angina": f"{'⚠️ Yes' if exang == 1 else '✅ No'}",
                "Critical Risk": f"{'⚠️ All 3 risk factors present' if calculate_critical_risk(cp, exang, oldpeak) else '✅ Not all risk factors'}",
                "Age+Sex Risk": f"{'⚠️ Elevated (Male > 55)' if calculate_age_sex_risk(age, sex) else '✅ Normal'}",
            },
            "features_used": {
                "Age": age,
                "Chest Pain": cp,
                "Cholesterol (mg/dl)": chol,
                "Max Heart Rate": thalach,
                "ST Depression": oldpeak,
                "Major Vessels": ca,
            }
        }

    except Exception as e:
        error_response["error"] = f"Prediction error: {str(e)}"
        return error_response


def get_heart_recommendations(prediction: int, cp: int, chol: float) -> list:
    """Generate heart disease medical recommendations"""
    if prediction == 1:
        return [
            "Urgent cardiology consultation required — do not delay.",
            "Request ECG (electrocardiogram) immediately.",
            "Echocardiogram recommended to assess heart function.",
            "Avoid strenuous physical activity until evaluated.",
            f"{'Critical: Cholesterol very high — statin therapy discussion needed.' if chol > 240 else 'Monitor cholesterol levels monthly.'}",
            f"{'Typical angina detected — coronary artery disease possible.' if cp == 0 else 'Continue monitoring chest symptoms.'}",
            "Blood pressure monitoring: check twice daily.",
            "Follow-up within 48-72 hours.",
        ]
    else:
        return [
            "No heart disease detected — maintain heart-healthy lifestyle.",
            "Annual cardiac check-up recommended.",
            f"{'Work to reduce cholesterol below 200 mg/dl through diet.' if chol > 200 else 'Maintain healthy cholesterol levels.'}",
            "Regular aerobic exercise: 150 minutes per week.",
            "Heart-healthy diet: reduce saturated fats and sodium.",
            "Quit smoking if applicable — major cardiac risk factor.",
            "Next check-up in 12 months.",
        ]
