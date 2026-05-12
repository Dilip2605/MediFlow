import joblib
import streamlit as st

# ═══════════════════════════════════════════════════════════════
# NLP SYMPTOM ANALYSIS SERVICE
# Purpose: Predict disease from symptom text description
# Algorithm: TF-IDF Vectorizer + Logistic Regression Pipeline
# Dataset: Symptom2Disease (1200 records, 24 diseases)
# CV Accuracy: 90%+
# ═══════════════════════════════════════════════════════════════


@st.cache_resource
def load_nlp_models():
    """Load NLP pipeline and label encoder"""
    try:
        pipeline = joblib.load("models/nlp_pipeline.pkl")
        encoder = joblib.load("models/nlp_label_encoder.pkl")
        return pipeline, encoder, True
    except FileNotFoundError:
        return None, None, False
    except Exception as e:
        print(f"Error loading NLP model: {e}")
        return None, None, False


def analyze_symptoms(symptom_text: str, top_n: int = 5) -> dict:
    """
    Analyze symptom text and predict possible diseases.
    
    How it works:
    1. Clean the text (lowercase, remove special chars)
    2. TF-IDF converts words to numbers
    3. Logistic Regression classifies disease
    4. Returns top N most likely diseases with probabilities
    
    Parameters:
        symptom_text: Natural language symptom description
        top_n: Number of top diseases to return (default 5)
    
    Returns dict with:
        primary_disease: Most likely disease name
        primary_confidence: Probability of primary disease (0-1)
        top_diseases: List of (disease_name, probability) tuples
        all_diseases: Full list of all disease classes
        model_loaded: True/False
        error: None or error message
    
    Example:
        result = analyze_symptoms("I have chest pain and fever")
        → primary_disease: "Pneumonia"
        → primary_confidence: 0.82
    """
    from utils.helpers import clean_text

    error_response = {
        "primary_disease": "ERROR",
        "primary_confidence": 0.0,
        "top_diseases": [],
        "all_diseases": [],
        "model_loaded": False,
        "error": None
    }

    pipeline, encoder, loaded = load_nlp_models()
    if not loaded:
        error_response["error"] = "NLP model not loaded. Run: python scripts/nlp_model.py"
        return error_response

    if not symptom_text or len(symptom_text.strip()) < 5:
        error_response["error"] = "Please enter at least 5 characters describing symptoms"
        return error_response

    try:
        # Clean the text
        cleaned = clean_text(symptom_text)

        # Get prediction
        pred_encoded = pipeline.predict([cleaned])[0]
        pred_disease = encoder.inverse_transform([pred_encoded])[0]

        # Get all probabilities
        all_probs = pipeline.predict_proba([cleaned])[0]

        # Get top N diseases
        top_n_indices = all_probs.argsort()[-top_n:][::-1]
        top_diseases = [
            (encoder.classes_[i], float(all_probs[i]))
            for i in top_n_indices
        ]

        return {
            "primary_disease": pred_disease,
            "primary_confidence": float(all_probs[pred_encoded]),
            "top_diseases": top_diseases,
            "all_diseases": list(encoder.classes_),
            "cleaned_text": cleaned,
            "model_loaded": True,
            "error": None
        }

    except Exception as e:
        error_response["error"] = f"Analysis error: {str(e)}"
        return error_response


def get_disease_info(disease_name: str) -> dict:
    """
    Return general information about a disease.
    Used to provide context after NLP prediction.
    
    Returns dict with description, common_symptoms, urgency
    """
    disease_info = {
        "Pneumonia": {
            "description": "Lung infection causing air sacs to fill with fluid",
            "common_symptoms": "Chest pain, cough, fever, difficulty breathing",
            "urgency": "HIGH — Seek immediate medical attention",
            "specialist": "Pulmonologist or General Physician"
        },
        "Tuberculosis": {
            "description": "Bacterial infection primarily affecting the lungs",
            "common_symptoms": "Night sweats, weight loss, cough with blood, fever",
            "urgency": "HIGH — Requires immediate testing and isolation",
            "specialist": "Pulmonologist or Infectious Disease Specialist"
        },
        "Diabetes": {
            "description": "Metabolic disorder affecting blood sugar regulation",
            "common_symptoms": "Excessive thirst, frequent urination, fatigue, blurred vision",
            "urgency": "MEDIUM — Consult doctor within 1 week",
            "specialist": "Endocrinologist or General Physician"
        },
        "Heart Attack": {
            "description": "Blockage of blood supply to heart muscle",
            "common_symptoms": "Severe chest pain, arm pain, shortness of breath, sweating",
            "urgency": "CRITICAL — Call emergency services immediately",
            "specialist": "Emergency Medicine / Cardiologist"
        },
        "Migraine": {
            "description": "Severe recurring headache often with nausea",
            "common_symptoms": "Throbbing headache, nausea, light sensitivity, aura",
            "urgency": "LOW — Schedule doctor appointment",
            "specialist": "Neurologist"
        },
    }

    return disease_info.get(disease_name, {
        "description": f"Medical condition: {disease_name}",
        "common_symptoms": "Symptoms vary — consult physician for details",
        "urgency": "MEDIUM — Consult a qualified physician",
        "specialist": "General Physician"
    })


def get_supported_diseases() -> list:
    """Return list of all diseases the NLP model can detect"""
    _, encoder, loaded = load_nlp_models()
    if loaded:
        return list(encoder.classes_)
    return []
