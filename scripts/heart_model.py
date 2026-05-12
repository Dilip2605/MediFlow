import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (train_test_split,
    cross_val_score, StratifiedKFold)
from sklearn.metrics import (accuracy_score,
    classification_report, confusion_matrix,
    roc_auc_score)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════
# MEDIFLOW — HEART DISEASE PREDICTION MODEL
# Purpose: Predict heart disease from clinical data
# Dataset: UCI Heart Disease — 1025 patients
# Algorithm: XGBoost with ensemble comparison
# ═══════════════════════════════════════════════════

print("=" * 60)
print("   MEDIFLOW — HEART DISEASE PREDICTION MODEL")
print("=" * 60)

# ─────────────────────────────────────────────────────
# STEP 1 — LOAD DATA
# Why: Must load and understand data first
# ─────────────────────────────────────────────────────
print("\n📂 STEP 1: Loading Data...")

df = pd.read_csv("data/ml/heart.csv")
print(f"✅ Original shape: {df.shape}")

# ─────────────────────────────────────────────────────
# STEP 2 — REMOVE DUPLICATES
# Why: Duplicate rows cause data leakage!
# Model sees test data during training → fake high accuracy
# This is why we got 99% — dataset had duplicates!
# ─────────────────────────────────────────────────────
print("\n🔍 STEP 2: Removing Duplicates...")

duplicates = df.duplicated().sum()
print(f"⚠️ Duplicate rows found: {duplicates}")
df = df.drop_duplicates()
df = df.reset_index(drop=True)
print(f"✅ After removing duplicates: {df.shape}")

# ─────────────────────────────────────────────────────
# STEP 3 — EXPLORE DATA
# Why: Understand what we have before processing
# ─────────────────────────────────────────────────────
print("\n📊 STEP 3: Exploring Data...")

print(f"\n📋 Columns: {list(df.columns)}")
print(f"\n📊 First 3 rows:\n{df.head(3)}")
print(f"\n📈 Target distribution:\n{df['target'].value_counts()}")
print(f"\n🔍 Missing values:\n{df.isnull().sum()}")
print(f"\n📉 Statistics:\n{df.describe()}")

# ─────────────────────────────────────────────────────
# STEP 4 — UNDERSTAND FEATURES
# Why: Medical domain knowledge helps feature engineering
# ─────────────────────────────────────────────────────
print("\n📚 STEP 4: Feature Understanding...")

feature_info = {
    'age': 'Patient age in years',
    'sex': '1=Male, 0=Female',
    'cp': 'Chest pain: 0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120: 1=True, 0=False',
    'restecg': 'Resting ECG: 0=normal, 1=ST-T abnormality, 2=LV hypertrophy',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina: 1=Yes, 0=No',
    'oldpeak': 'ST depression by exercise vs rest',
    'slope': 'Slope of peak exercise ST segment',
    'ca': 'Major vessels colored by fluoroscopy (0-3)',
    'thal': 'Thalassemia: 0=normal, 1=fixed defect, 2=reversible defect',
    'target': '1=Heart Disease, 0=No Heart Disease'
}

for feature, description in feature_info.items():
    print(f"  {feature:12} → {description}")

# ─────────────────────────────────────────────────────
# STEP 5 — DATA QUALITY CHECK
# Why: Bad data = bad model
# ─────────────────────────────────────────────────────
print("\n🔧 STEP 5: Data Quality Check...")

print(f"\nAge range: {df['age'].min()} - {df['age'].max()}")
print(f"Cholesterol zeros: {(df['chol'] == 0).sum()}")
print(f"Blood pressure zeros: {(df['trestbps'] == 0).sum()}")

# Fix zero values in medical columns
medical_columns = ['trestbps', 'chol']
for col in medical_columns:
    zeros = (df[col] == 0).sum()
    if zeros > 0:
        df[col] = df[col].replace(0, df[col].median())
        print(f"✅ Fixed {zeros} zeros in {col}")
    else:
        print(f"✅ {col}: No zero values found")

# ─────────────────────────────────────────────────────
# STEP 6 — FEATURE ENGINEERING
# Why: New features capture medical relationships
# ─────────────────────────────────────────────────────
print("\n⚙️ STEP 6: Feature Engineering...")

# Age risk category — medical standard
df['Age_Risk'] = pd.cut(
    df['age'],
    bins=[0, 40, 55, 65, 100],
    labels=[0, 1, 2, 3],
    include_lowest=True)
df['Age_Risk'] = df['Age_Risk'].fillna(3).astype(int)

# Blood pressure category — WHO standard
df['BP_Category'] = pd.cut(
    df['trestbps'],
    bins=[0, 120, 130, 140, 300],
    labels=[0, 1, 2, 3],
    include_lowest=True)
df['BP_Category'] = df['BP_Category'].fillna(3).astype(int)

# Cholesterol risk — medical standard
df['Chol_Risk'] = pd.cut(
    df['chol'],
    bins=[0, 200, 239, 1000],
    labels=[0, 1, 2],
    include_lowest=True)
df['Chol_Risk'] = df['Chol_Risk'].fillna(2).astype(int)

# Heart rate efficiency
# Formula: achieved rate / predicted max rate
# Predicted max = 220 - age
df['HR_Efficiency'] = df['thalach'] / (220 - df['age'])

# Critical risk flag
# All three high risk indicators together
df['Critical_Risk'] = (
    (df['cp'] == 0) &       # Typical angina
    (df['exang'] == 1) &    # Exercise angina
    (df['oldpeak'] > 2)     # High ST depression
).astype(int)

# Age × Sex interaction
# Men over 55 have higher heart disease risk
df['Age_Sex_Risk'] = (
    (df['age'] > 55) &
    (df['sex'] == 1)
).astype(int)

print("✅ Age_Risk: 4 medical risk categories")
print("✅ BP_Category: WHO blood pressure categories")
print("✅ Chol_Risk: Cholesterol risk levels")
print("✅ HR_Efficiency: Heart rate efficiency ratio")
print("✅ Critical_Risk: Combined high risk flag")
print("✅ Age_Sex_Risk: Age-gender interaction")
print(f"✅ New dataset shape: {df.shape}")

# ─────────────────────────────────────────────────────
# STEP 7 — PREPARE FEATURES AND TARGET
# Why: Separate what we predict from what we use
# ─────────────────────────────────────────────────────
print("\n📊 STEP 7: Preparing Features and Target...")

X = df.drop('target', axis=1)
y = df['target']

# Calculate class ratio for imbalance handling
no_disease = (y == 0).sum()
has_disease = (y == 1).sum()
ratio = no_disease / has_disease

print(f"✅ Features (X): {X.shape[1]} features")
print(f"✅ Target (y): Heart Disease (0=No, 1=Yes)")
print(f"✅ No Heart Disease: {no_disease} patients")
print(f"✅ Heart Disease: {has_disease} patients")
print(f"✅ Class ratio: {ratio:.2f}")

# ─────────────────────────────────────────────────────
# STEP 8 — SPLIT DATA WITH STRATIFICATION
# Why: Ensures balanced train/test split
# stratify=y = same disease ratio in both sets
# ─────────────────────────────────────────────────────
print("\n✂️ STEP 8: Splitting Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)

print(f"✅ Training: {len(X_train)} patients")
print(f"✅ Testing: {len(X_test)} patients")
print(f"✅ Train disease ratio: {y_train.mean():.2f}")
print(f"✅ Test disease ratio: {y_test.mean():.2f}")

# ─────────────────────────────────────────────────────
# STEP 9 — SCALE FEATURES
# Why: Different scales confuse model
# age(29-77) vs fbs(0-1) — huge difference!
# StandardScaler makes all features mean=0, std=1
# ─────────────────────────────────────────────────────
print("\n⚖️ STEP 9: Scaling Features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to keep feature names
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns)
X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns)

print("✅ StandardScaler applied")
print("   All features: mean=0, std=1")

# ─────────────────────────────────────────────────────
# STEP 10 — COMPARE ALGORITHMS
# Why: Find best algorithm for this specific dataset
# Never assume one algorithm is always best!
# ─────────────────────────────────────────────────────
print("\n🏆 STEP 10: Comparing Algorithms...")
print("-" * 60)

algorithms = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=ratio)
}

results = {}
skf = StratifiedKFold(n_splits=5,
    shuffle=True, random_state=42)

for name, algo in algorithms.items():
    algo.fit(X_train_scaled, y_train)
    pred = algo.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    cv = cross_val_score(
        algo, X_train_scaled, y_train,
        cv=skf, scoring='accuracy')
    results[name] = {
        'accuracy': acc,
        'cv_mean': cv.mean(),
        'cv_std': cv.std(),
        'model': algo
    }
    print(f"{name:25} → "
          f"Test: {acc*100:.2f}% | "
          f"CV: {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")

best_name = max(results,
    key=lambda x: results[x]['cv_mean'])
print(f"\n🏆 Best Algorithm: {best_name}")

# ─────────────────────────────────────────────────────
# STEP 11 — TRAIN FINAL MODEL
# Why: Use XGBoost — most consistent for medical data
# max_depth=4 prevents overfitting
# subsample=0.8 adds randomness — reduces overfitting
# ─────────────────────────────────────────────────────
print("\n🤖 STEP 11: Training Final Model...")

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    scale_pos_weight=ratio
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False)

print("✅ Final XGBoost model trained!")

# ─────────────────────────────────────────────────────
# STEP 12 — COMPREHENSIVE EVALUATION
# Why: Multiple metrics give complete picture
# Accuracy alone is not enough for medical AI!
# ─────────────────────────────────────────────────────
print("\n📊 STEP 12: Comprehensive Evaluation...")
print("=" * 60)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")
print(f"📈 AUC-ROC: {auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
    target_names=['No Disease', 'Heart Disease']))

cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(f"  True Negative:  {cm[0][0]:3d} ✅ (No Disease → No Disease)")
print(f"  False Positive: {cm[0][1]:3d} ⚠️  (No Disease → Disease)")
print(f"  False Negative: {cm[1][0]:3d} ❌ (Disease → No Disease) DANGEROUS!")
print(f"  True Positive:  {cm[1][1]:3d} ✅ (Disease → Disease)")
print(f"\n  Missed patients: {cm[1][0]}")
print(f"  Caught patients: {cm[1][1]}")
print(f"  Catch Rate: {cm[1][1]/(cm[1][0]+cm[1][1])*100:.1f}%")

# ─────────────────────────────────────────────────────
# STEP 13 — CROSS VALIDATION
# Why: More reliable than single test score
# ─────────────────────────────────────────────────────
print("\n🔄 STEP 13: 5-Fold Cross Validation...")

cv_acc = cross_val_score(
    model, X, y, cv=skf, scoring='accuracy')
cv_rec = cross_val_score(
    model, X, y, cv=skf, scoring='recall')
cv_pre = cross_val_score(
    model, X, y, cv=skf, scoring='precision')
cv_f1 = cross_val_score(
    model, X, y, cv=skf, scoring='f1')
cv_auc = cross_val_score(
    model, X, y, cv=skf, scoring='roc_auc')

print(f"✅ CV Accuracy:  {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")
print(f"✅ CV Recall:    {cv_rec.mean()*100:.2f}% ± {cv_rec.std()*100:.2f}%")
print(f"✅ CV Precision: {cv_pre.mean()*100:.2f}% ± {cv_pre.std()*100:.2f}%")
print(f"✅ CV F1 Score:  {cv_f1.mean()*100:.2f}% ± {cv_f1.std()*100:.2f}%")
print(f"✅ CV AUC-ROC:   {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# ─────────────────────────────────────────────────────
# STEP 14 — FEATURE IMPORTANCE
# Why: Understand which clinical features predict disease
# Validates model makes medical sense
# ─────────────────────────────────────────────────────
print("\n📊 STEP 14: Feature Importance...")

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Rankings:")
for _, row in importance.iterrows():
    bar = "█" * int(row['Importance'] * 100)
    desc = feature_info.get(row['Feature'], 'Engineered')
    print(f"  {row['Feature']:15} {bar:20} {row['Importance']:.4f}")

print(f"\n✅ Most predictive: {importance.iloc[0]['Feature']}")
print(f"✅ Least predictive: {importance.iloc[-1]['Feature']}")

# ─────────────────────────────────────────────────────
# STEP 15 — SAMPLE PREDICTIONS
# Why: Verify model makes medically correct predictions
# If predictions are wrong here — something is wrong!
# ─────────────────────────────────────────────────────
print("\n🧪 STEP 15: Sample Predictions...")

# Use ACTUAL patients from test set for verification
print("\nUsing real patients from test set:")
for i in range(min(5, len(X_test))):
    patient = X_test_scaled.iloc[i:i+1]
    actual = y_test.iloc[i]
    pred = model.predict(patient)[0]
    prob = model.predict_proba(patient)[0][1]
    status = "✅" if actual == pred else "❌"
    print(f"  Patient {i+1}: "
          f"Actual={'Disease' if actual==1 else 'No Disease'} | "
          f"Predicted={'Disease' if pred==1 else 'No Disease'} | "
          f"Confidence={max(prob,1-prob)*100:.1f}% {status}")

# ─────────────────────────────────────────────────────
# STEP 16 — SAVE ALL ARTIFACTS
# Why: Production needs all files saved
# model.pkl = trained model
# scaler.pkl = feature scaler
# features.pkl = feature names (for consistent prediction)
# metadata.pkl = model info and performance
# ─────────────────────────────────────────────────────
print("\n💾 STEP 16: Saving Artifacts...")

joblib.dump(model, "models/heart_model.pkl")
print("✅ heart_model.pkl")

joblib.dump(scaler, "models/heart_scaler.pkl")
print("✅ heart_scaler.pkl")

joblib.dump(list(X.columns), "models/heart_features.pkl")
print("✅ heart_features.pkl")

metadata = {
    'model_type': 'XGBoost',
    'version': '2.0',
    'features': list(X.columns),
    'n_features': len(X.columns),
    'target': 'Heart Disease',
    'accuracy': float(accuracy),
    'auc_roc': float(auc),
    'cv_accuracy': float(cv_acc.mean()),
    'cv_recall': float(cv_rec.mean()),
    'cv_f1': float(cv_f1.mean()),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'duplicates_removed': int(duplicates)
}
joblib.dump(metadata, "models/heart_metadata.pkl")
print("✅ heart_metadata.pkl")

# ─────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("   HEART DISEASE MODEL — SUMMARY")
print("=" * 60)
print(f"  Dataset: {len(df)} patients after deduplication")
print(f"  Features: {X.shape[1]} (13 original + 6 engineered)")
print(f"  Algorithm: XGBoost")
print(f"  Test Accuracy: {accuracy*100:.2f}%")
print(f"  AUC-ROC: {auc:.4f}")
print(f"  CV Accuracy: {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")
print(f"  CV Recall: {cv_rec.mean()*100:.2f}%")
print(f"  Top Feature: {importance.iloc[0]['Feature']}")
print(f"  Duplicates Removed: {duplicates}")
print(f"  Files Saved: 4")
print("=" * 60)
print("✅ HEART DISEASE MODEL COMPLETE!")