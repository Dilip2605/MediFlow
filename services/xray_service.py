import numpy as np
import streamlit as st

# ═══════════════════════════════════════════════════════════════
# X-RAY ANALYSIS SERVICE
# Purpose: Detect disease in chest X-ray images using CNN
# Algorithm: DenseNet121 with Transfer Learning
# Training: Google Colab with GPU (90%+ accuracy expected)
# Classes: NORMAL, PNEUMONIA (expandable to TB, COVID)
# ═══════════════════════════════════════════════════════════════


@st.cache_resource
def load_cnn_model():
    """
    Load CNN model and class labels.
    
    Why separate function:
    - CNN model is large (~80MB)
    - Cache ensures it loads only once
    - Avoids memory issues from repeated loading
    """
    try:
        import tensorflow as tf
        import joblib

        model = tf.keras.models.load_model("models/cnn_xray_model.keras")
        labels = joblib.load("models/cnn_class_labels.pkl")
        return model, labels, True

    except FileNotFoundError:
        return None, None, False
    except ImportError:
        return None, None, False
    except Exception as e:
        print(f"Error loading CNN model: {e}")
        return None, None, False


def preprocess_xray_image(image_file) -> np.ndarray:
    """
    Preprocess X-ray image for CNN input.
    
    Steps:
    1. Open image with PIL
    2. Convert to RGB (X-rays might be grayscale)
    3. Resize to 224×224 (DenseNet121 input size)
    4. Normalize pixels to 0-1 range (divide by 255)
    5. Add batch dimension (1, 224, 224, 3)
    
    Parameters:
        image_file: Uploaded file from st.file_uploader
    
    Returns:
        numpy array of shape (1, 224, 224, 3)
    """
    from PIL import Image

    # Open and convert to RGB
    image = Image.open(image_file).convert('RGB')

    # Resize to DenseNet121 input size
    image_resized = image.resize((224, 224))

    # Convert to numpy array and normalize
    img_array = np.array(image_resized, dtype=np.float32) / 255.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def analyze_xray(image_file) -> dict:
    """
    Analyze chest X-ray image for disease detection.
    
    How it works:
    1. Load DenseNet121 model
    2. Preprocess image to 224×224×3
    3. CNN predicts disease probabilities
    4. Return structured result
    
    Parameters:
        image_file: File from st.file_uploader
    
    Returns dict with:
        predicted_class: Disease name (NORMAL/PNEUMONIA)
        confidence: Prediction confidence (0-1)
        all_predictions: {class_name: probability} dict
        model_loaded: True/False
        error: None or error message
    """
    error_response = {
        "predicted_class": "ERROR",
        "confidence": 0.0,
        "all_predictions": {},
        "model_loaded": False,
        "error": None
    }

    model, labels, loaded = load_cnn_model()
    if not loaded:
        error_response["error"] = (
            "CNN model not loaded. "
            "Train on Google Colab: python scripts/cnn_model.py"
        )
        return error_response

    try:
        # Preprocess image
        img_array = preprocess_xray_image(image_file)

        # Run prediction
        predictions = model.predict(img_array, verbose=0)
        pred_class_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][pred_class_idx])
        predicted_class = labels[pred_class_idx]

        # Build all predictions dict
        all_predictions = {
            labels[i]: float(predictions[0][i])
            for i in range(len(labels))
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_predictions": all_predictions,
            "is_normal": predicted_class == "NORMAL",
            "model_loaded": True,
            "error": None
        }

    except Exception as e:
        error_response["error"] = f"Analysis error: {str(e)}"
        return error_response


def get_xray_recommendations(predicted_class: str, confidence: float) -> list:
    """Generate recommendations based on X-ray analysis"""
    if predicted_class == "PNEUMONIA":
        return [
            f"Pneumonia detected with {int(confidence*100)}% confidence — immediate attention needed.",
            "Consult pulmonologist or emergency physician immediately.",
            "Chest CT scan may be required for confirmation.",
            "Sputum culture test recommended to identify bacteria/virus.",
            "Antibiotic therapy likely needed — do not self-medicate.",
            "Monitor oxygen saturation continuously.",
        ]
    elif predicted_class == "TB":
        return [
            "Tuberculosis patterns detected — immediate isolation and testing needed.",
            "Sputum AAFB test and GeneXpert test required for confirmation.",
            "Contact tracing of household members necessary.",
            "Chest HRCT scan for detailed assessment.",
            "Specialist referral: Pulmonologist / TB specialist.",
        ]
    else:
        return [
            "X-ray appears normal — no obvious disease detected.",
            "Continue regular health monitoring.",
            "Annual chest X-ray recommended if in high-risk environment.",
            "Consult physician if respiratory symptoms persist.",
        ]


def get_cnn_model_status() -> dict:
    """Return CNN model loading status and info"""
    model, labels, loaded = load_cnn_model()
    if loaded:
        return {
            "loaded": True,
            "classes": labels,
            "n_classes": len(labels),
            "architecture": "DenseNet121 Transfer Learning",
            "input_size": "224×224×3"
        }
    return {
        "loaded": False,
        "classes": [],
        "n_classes": 0,
        "architecture": "Not loaded",
        "input_size": "N/A"
    }
