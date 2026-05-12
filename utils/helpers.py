import hashlib
import re
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# HELPERS
# Purpose: Small utility functions used across entire project
# No ML logic here — only helper functions
# ═══════════════════════════════════════════════════════════════


# ─── PASSWORD HASHING ────────────────────────────────────────
def hash_password(password: str) -> str:
    """
    Convert plain text password to SHA256 hash.
    
    Why: Never store plain text passwords in database!
    If database is hacked, hashed passwords cannot be reversed.
    
    Example:
        hash_password("admin123") 
        → "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9"
    """
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Check if entered password matches stored hash.
    
    Example:
        verify_password("admin123", stored_hash) → True
        verify_password("wrongpass", stored_hash) → False
    """
    return hash_password(plain_password) == stored_hash


# ─── TEXT CLEANING ────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean symptom text before NLP processing.
    
    Steps:
    1. Convert to lowercase: 'Chest Pain' → 'chest pain'
    2. Remove special chars: 'chest-pain!!' → 'chest pain'
    3. Remove extra spaces: 'chest  pain' → 'chest pain'
    4. Strip edges: '  chest pain  ' → 'chest pain'
    
    Example:
        clean_text("I have CHEST-PAIN and Fever!!!")
        → "i have chest pain and fever"
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─── RISK ASSESSMENT ─────────────────────────────────────────
def get_risk_color(probability: float) -> str:
    """
    Return color code based on disease probability.
    
    Used for: confidence bars, result cards, risk indicators
    
    < 0.30 = Green  = Low risk
    0.30-0.60 = Yellow = Moderate risk  
    > 0.60 = Red = High risk
    """
    if probability < 0.30:
        return "#34d399"   # Green
    elif probability < 0.60:
        return "#fbbf24"   # Yellow/Orange
    else:
        return "#f87171"   # Red


def get_risk_level(probability: float) -> str:
    """
    Return risk level text based on probability.
    
    Example:
        get_risk_level(0.85) → "HIGH RISK"
        get_risk_level(0.45) → "MODERATE RISK"
        get_risk_level(0.15) → "LOW RISK"
    """
    if probability < 0.30:
        return "LOW RISK"
    elif probability < 0.60:
        return "MODERATE RISK"
    else:
        return "HIGH RISK"


def get_risk_emoji(probability: float) -> str:
    """Return emoji based on risk level"""
    if probability < 0.30:
        return "✅"
    elif probability < 0.60:
        return "⚠️"
    else:
        return "🚨"


# ─── DATE HELPERS ─────────────────────────────────────────────
def get_today() -> str:
    """Return today's date as string YYYY-MM-DD"""
    return datetime.now().strftime("%Y-%m-%d")


def get_now() -> str:
    """Return current datetime as string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_date_display(date_str: str) -> str:
    """
    Convert database date to readable format.
    
    Example:
        format_date_display("2024-03-15") → "15 Mar 2024"
    """
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%d %b %Y")
    except Exception:
        return date_str


# ─── VALIDATION ───────────────────────────────────────────────
def validate_phone(phone: str) -> bool:
    """
    Validate Indian phone number format.
    Accepts: 10 digit numbers, with or without +91
    
    Example:
        validate_phone("9876543210") → True
        validate_phone("123") → False
    """
    # Remove spaces and +91 prefix
    phone = phone.replace(" ", "").replace("+91", "").replace("-", "")
    return len(phone) == 10 and phone.isdigit()


def validate_age(age: int) -> bool:
    """Validate age is realistic (1-120)"""
    return 1 <= age <= 120


def validate_glucose(glucose: float) -> bool:
    """Validate glucose is in realistic range"""
    return 0 < glucose <= 600


def validate_bmi(bmi: float) -> bool:
    """Validate BMI is in realistic range"""
    return 5.0 <= bmi <= 100.0


# ─── FEATURE ENGINEERING ─────────────────────────────────────
def calculate_bmi_category(bmi: float) -> int:
    """
    Convert BMI to medical risk category.
    
    WHO BMI Classifications:
    0 = Underweight (BMI < 18.5)
    1 = Normal weight (18.5 - 24.9)
    2 = Overweight (25.0 - 29.9)
    3 = Obese (≥ 30.0)
    """
    if bmi < 18.5:
        return 0
    elif bmi < 25.0:
        return 1
    elif bmi < 30.0:
        return 2
    else:
        return 3


def get_bmi_label(bmi: float) -> str:
    """Return human-readable BMI category"""
    category = calculate_bmi_category(bmi)
    labels = {0: "Underweight", 1: "Normal", 2: "Overweight", 3: "Obese"}
    return labels[category]


def calculate_age_group(age: int) -> int:
    """
    Convert age to risk group.
    
    0 = Young (< 30)
    1 = Middle age (30 - 44)
    2 = Senior (45 - 59)
    3 = Elder (≥ 60)
    """
    if age < 30:
        return 0
    elif age < 45:
        return 1
    elif age < 60:
        return 2
    else:
        return 3


def calculate_high_risk_flag(glucose: float, bmi: float) -> int:
    """
    Combined high risk flag — both glucose AND BMI elevated.
    
    High risk = Glucose > 140 AND BMI > 30
    Returns 1 if high risk, 0 otherwise
    """
    return 1 if (glucose > 140 and bmi > 30) else 0


def calculate_glucose_bmi(glucose: float, bmi: float) -> float:
    """
    Combined risk factor: Glucose × BMI interaction.
    Captures combined effect of both risk factors.
    """
    return (glucose * bmi) / 1000.0


# ─── HEART DISEASE FEATURES ──────────────────────────────────
def calculate_age_risk(age: int) -> int:
    """
    Age-based heart disease risk.
    0 = Low risk (< 40)
    1 = Medium risk (40-54)
    2 = High risk (55-64)
    3 = Very high risk (≥ 65)
    """
    if age < 40:
        return 0
    elif age < 55:
        return 1
    elif age < 65:
        return 2
    else:
        return 3


def calculate_bp_category(trestbps: float) -> int:
    """
    WHO Blood Pressure Classification.
    0 = Normal (< 120)
    1 = Elevated (120-129)
    2 = High Stage 1 (130-139)
    3 = High Stage 2 (≥ 140)
    """
    if trestbps < 120:
        return 0
    elif trestbps < 130:
        return 1
    elif trestbps < 140:
        return 2
    else:
        return 3


def calculate_chol_risk(chol: float) -> int:
    """
    Cholesterol risk level.
    0 = Desirable (< 200)
    1 = Borderline (200-239)
    2 = High (≥ 240)
    """
    if chol < 200:
        return 0
    elif chol < 240:
        return 1
    else:
        return 2


def calculate_hr_efficiency(thalach: float, age: int) -> float:
    """
    Heart rate efficiency ratio.
    Formula: achieved_max_hr / predicted_max_hr
    Predicted max HR = 220 - age
    """
    predicted_max = 220 - age
    if predicted_max <= 0:
        return 0.0
    return thalach / predicted_max


def calculate_critical_risk(cp: int, exang: int, oldpeak: float) -> int:
    """
    Critical combined risk flag for heart disease.
    Returns 1 if all three high-risk indicators are present.
    """
    return 1 if (cp == 0 and exang == 1 and oldpeak > 2.0) else 0


def calculate_age_sex_risk(age: int, sex: int) -> int:
    """
    Age and sex combined risk.
    Men over 55 have significantly higher heart disease risk.
    """
    return 1 if (age > 55 and sex == 1) else 0
