import pandas as pd
import numpy as np
from sklearn.model_selection import (train_test_split,
    cross_val_score, StratifiedKFold)
from sklearn.metrics import (accuracy_score,
    classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import re
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════
# MEDIFLOW — NLP SYMPTOM ANALYSIS MODEL
# Purpose: Predict disease from symptom text
# Dataset: Symptom2Disease — 1200 records, 24 diseases
# Algorithm: TF-IDF + Logistic Regression Pipeline
# Why NLP: Doctors type symptoms as text
#          NLP converts text to disease prediction
# ═══════════════════════════════════════════════════

print("=" * 60)
print("   MEDIFLOW — NLP SYMPTOM ANALYSIS MODEL")
print("=" * 60)

# ─────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────────────
print("\n📂 STEP 1: Loading Data...")

df = pd.read_csv("data/nlp/Symptom2Disease.csv")

# Remove unnamed index column if exists
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

print(f"✅ Dataset loaded: {df.shape}")
print(f"\n📋 Columns: {list(df.columns)}")
print(f"\n📊 First 3 rows:\n{df.head(3)}")
print(f"\n📈 Unique diseases: {df['label'].nunique()}")
print(f"\n🏥 Disease list:\n{sorted(df['label'].unique())}")

# ─────────────────────────────────────────────────────
# STEP 2 — CHECK DATA QUALITY
# ─────────────────────────────────────────────────────
print("\n🔍 STEP 2: Data Quality Check...")

print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Remove duplicates
df = df.drop_duplicates()
df = df.reset_index(drop=True)
print(f"✅ After cleaning: {df.shape}")

# Disease distribution
print(f"\n📊 Samples per disease:")
print(df['label'].value_counts())

# ─────────────────────────────────────────────────────
# STEP 3 — TEXT PREPROCESSING
# Why: Raw text has noise — clean it for better results
# Steps: lowercase → remove special chars →
#        remove extra spaces → remove numbers
# ─────────────────────────────────────────────────────
print("\n🔧 STEP 3: Text Preprocessing...")

def clean_text(text):
    """
    Clean raw symptom text for NLP processing

    Steps:
    1. Lowercase: 'Chest Pain' → 'chest pain'
    2. Remove special chars: 'chest-pain!' → 'chest pain'
    3. Remove extra spaces: 'chest  pain' → 'chest pain'
    4. Strip whitespace from edges
    """
    # Convert to lowercase
    text = text.lower()

    # Remove special characters — keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

print("✅ Text cleaning complete!")
print("\nExample before cleaning:")
print(f"  Raw:     {df['text'].iloc[0][:80]}...")
print(f"  Cleaned: {df['cleaned_text'].iloc[0][:80]}...")

# ─────────────────────────────────────────────────────
# STEP 4 — ENCODE LABELS
# Why: Model needs numbers not text labels
# LabelEncoder converts disease names to numbers
# Pneumonia → 0, Diabetes → 1, etc.
# ─────────────────────────────────────────────────────
print("\n🔢 STEP 4: Encoding Disease Labels...")

label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

print(f"✅ {len(label_encoder.classes_)} diseases encoded")
print("\nEncoding mapping:")
for i, disease in enumerate(label_encoder.classes_):
    print(f"  {i:2d} → {disease}")

# ─────────────────────────────────────────────────────
# STEP 5 — PREPARE FEATURES AND TARGET
# ─────────────────────────────────────────────────────
print("\n📊 STEP 5: Preparing Data...")

X = df['cleaned_text']
y = df['label_encoded']

print(f"✅ Features (X): {len(X)} symptom texts")
print(f"✅ Target (y): {len(label_encoder.classes_)} disease classes")

# ─────────────────────────────────────────────────────
# STEP 6 — SPLIT DATA
# Why: Evaluate on unseen text descriptions
# stratify ensures all diseases in both train/test
# ─────────────────────────────────────────────────────
print("\n✂️ STEP 6: Splitting Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)

print(f"✅ Training: {len(X_train)} samples")
print(f"✅ Testing: {len(X_test)} samples")

# ─────────────────────────────────────────────────────
# STEP 7 — BUILD NLP PIPELINE
# Why: Pipeline chains TF-IDF + Model together
#      Ensures same preprocessing for train and test
#
# TF-IDF Explained:
# TF = how often word appears in THIS document
# IDF = how rare word is ACROSS all documents
# TF-IDF = TF × IDF = important AND unique words score high
#
# Example:
# 'chest pain' appears in many pneumonia descriptions
# → High TF for pneumonia documents
# 'chest pain' rarely in diabetes descriptions
# → High IDF for chest pain
# TF-IDF = high score → strong pneumonia indicator!
# ─────────────────────────────────────────────────────
print("\n🔧 STEP 7: Building NLP Pipeline...")

# TF-IDF Configuration
# ngram_range=(1,2) captures both single words AND pairs
# Example: 'chest' AND 'chest pain' both captured
# max_features=5000 keeps top 5000 most important terms
# min_df=2 ignores terms appearing in only 1 document

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2,
    sublinear_tf=True)  # Log scaling of TF

print("✅ TF-IDF Vectorizer configured:")
print("   ngram_range=(1,2) — single words AND word pairs")
print("   max_features=5000 — top 5000 important terms")
print("   min_df=2 — ignore very rare terms")
print("   sublinear_tf=True — log scaling")

# ─────────────────────────────────────────────────────
# STEP 8 — COMPARE NLP ALGORITHMS
# Why: Different algorithms work differently for text
# Naive Bayes — great for text classification
# Logistic Regression — good with TF-IDF features
# Random Forest — slower but handles complex patterns
# ─────────────────────────────────────────────────────
print("\n🏆 STEP 8: Comparing NLP Algorithms...")
print("-" * 60)

skf = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42)

pipelines = {
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            sublinear_tf=True)),
        ('model', MultinomialNB(alpha=0.1))
    ]),
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            min_df=2,
            sublinear_tf=True)),
        ('model', LogisticRegression(
            max_iter=1000,
            C=5.0,
            random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            min_df=2)),
        ('model', RandomForestClassifier(
            n_estimators=100,
            random_state=42))
    ])
}

results = {}
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, pred)
    cv = cross_val_score(
        pipeline, X, y,
        cv=skf, scoring='accuracy')
    results[name] = {
        'accuracy': acc,
        'cv_mean': cv.mean(),
        'cv_std': cv.std(),
        'pipeline': pipeline
    }
    print(f"{name:25} → "
          f"Test: {acc*100:.2f}% | "
          f"CV: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

best_name = max(results,
    key=lambda x: results[x]['cv_mean'])
print(f"\n🏆 Best Algorithm: {best_name}")

# ─────────────────────────────────────────────────────
# STEP 9 — TRAIN FINAL MODEL
# Why: Use best pipeline for production
# ─────────────────────────────────────────────────────
print("\n🤖 STEP 9: Training Final Model...")

# Use Logistic Regression — most consistent for medical NLP
final_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True)),
    ('model', LogisticRegression(
        max_iter=1000,
        C=5.0,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42))
])

final_pipeline.fit(X_train, y_train)
print("✅ Final NLP pipeline trained!")

# ─────────────────────────────────────────────────────
# STEP 10 — COMPREHENSIVE EVALUATION
# ─────────────────────────────────────────────────────
print("\n📊 STEP 10: Evaluation...")
print("=" * 60)

y_pred = final_pipeline.predict(X_test)
y_prob = final_pipeline.predict_proba(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
print("\n📋 Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_))

# ─────────────────────────────────────────────────────
# STEP 11 — CROSS VALIDATION
# ─────────────────────────────────────────────────────
print("\n🔄 STEP 11: Cross Validation...")

cv_scores = cross_val_score(
    final_pipeline, X, y,
    cv=skf, scoring='accuracy')

print(f"✅ CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print(f"✅ Individual folds: {[f'{s*100:.1f}%' for s in cv_scores]}")

# ─────────────────────────────────────────────────────
# STEP 12 — TEST SAMPLE PREDICTIONS
# Why: Verify model understands medical text correctly
# ─────────────────────────────────────────────────────
print("\n🧪 STEP 12: Sample Predictions...")

test_symptoms = [
    {
        'text': 'I have severe chest pain and difficulty breathing for 3 days',
        'expected': 'Heart Disease / Pneumonia'
    },
    {
        'text': 'experiencing excessive thirst frequent urination and fatigue',
        'expected': 'Diabetes'
    },
    {
        'text': 'skin rash itching all over body with redness',
        'expected': 'Psoriasis / Fungal Infection'
    },
    {
        'text': 'high fever cough night sweats and weight loss',
        'expected': 'Tuberculosis'
    },
    {
        'text': 'severe headache nausea vomiting and light sensitivity',
        'expected': 'Migraine'
    }
]

print("\nSymptom Text → Disease Prediction:")
for symptom in test_symptoms:
    cleaned = clean_text(symptom['text'])
    pred_encoded = final_pipeline.predict([cleaned])[0]
    pred_disease = label_encoder.inverse_transform(
        [pred_encoded])[0]
    probs = final_pipeline.predict_proba([cleaned])[0]
    top3_idx = probs.argsort()[-3:][::-1]
    top3 = [(label_encoder.classes_[i],
             f"{probs[i]*100:.1f}%")
            for i in top3_idx]

    print(f"\n  Symptoms: {symptom['text'][:60]}...")
    print(f"  Expected: {symptom['expected']}")
    print(f"  Top Predictions:")
    for disease, prob in top3:
        print(f"    → {disease}: {prob}")

# ─────────────────────────────────────────────────────
# STEP 13 — SAVE ALL ARTIFACTS
# ─────────────────────────────────────────────────────
print("\n💾 STEP 13: Saving Artifacts...")

# Save complete pipeline (TF-IDF + Model together)
joblib.dump(final_pipeline, "models/nlp_pipeline.pkl")
print("✅ nlp_pipeline.pkl (TF-IDF + Model)")

# Save label encoder
joblib.dump(label_encoder, "models/nlp_label_encoder.pkl")
print("✅ nlp_label_encoder.pkl")

# Save disease list
joblib.dump(list(label_encoder.classes_),
    "models/nlp_diseases.pkl")
print("✅ nlp_diseases.pkl")

# Save clean_text function reference
joblib.dump(clean_text, "models/nlp_clean_text.pkl")
print("✅ nlp_clean_text.pkl")

# Save metadata
metadata = {
    'model_type': 'TF-IDF + Logistic Regression Pipeline',
    'n_diseases': len(label_encoder.classes_),
    'diseases': list(label_encoder.classes_),
    'accuracy': float(accuracy),
    'cv_accuracy': float(cv_scores.mean()),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'tfidf_features': 5000,
    'ngram_range': '(1,2)'
}
joblib.dump(metadata, "models/nlp_metadata.pkl")
print("✅ nlp_metadata.pkl")

print("\n" + "=" * 60)
print("   NLP MODEL — SUMMARY")
print("=" * 60)
print(f"  Dataset: {len(df)} symptom descriptions")
print(f"  Diseases: {len(label_encoder.classes_)}")
print(f"  Algorithm: TF-IDF + Logistic Regression")
print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}%")
print(f"  Files Saved: 5")
print("=" * 60)
print("✅ NLP MODEL COMPLETE!")