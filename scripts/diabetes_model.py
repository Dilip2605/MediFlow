import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# STEP 1 — Load and Explore Data
# ─────────────────────────────────────────────
print("=" * 50)
print("MEDIFLOW — DIABETES PREDICTION MODEL")
print("=" * 50)

df = pd.read_csv("data/ml/diabetes.csv")
print("\n📊 Dataset shape:", df.shape)
print("\n📋 First 3 rows:\n", df.head(3))
print("\n📈 Disease distribution:\n", df['Outcome'].value_counts())
print("\n🔍 Missing values:\n", df.isnull().sum())
print("\n📉 Statistics:\n", df.describe())

# ─────────────────────────────────────────────
# STEP 2 — Fix Zero Values (Hidden Missing Data)
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("FIXING HIDDEN MISSING VALUES (ZEROS)")
print("=" * 50)

# These columns CANNOT be 0 in real life
# 0 means data was missing — entered as 0
zero_columns = ['Glucose', 'BloodPressure',
                'SkinThickness', 'Insulin', 'BMI']

for col in zero_columns:
    zero_count = (df[col] == 0).sum()
    median_val = df[col].median()
    df[col] = df[col].replace(0, median_val)
    print(f"✅ {col}: {zero_count} zeros replaced with median ({median_val:.1f})")

# ─────────────────────────────────────────────
# STEP 3 — Feature Engineering
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("FEATURE ENGINEERING — CREATING NEW FEATURES")
print("=" * 50)

# Create new meaningful features
# BMI Category — medical standard
df['BMI_Category'] = pd.cut(df['BMI'],
    bins=[0, 18.5, 25, 30, 100],
    labels=[0, 1, 2, 3])  # Under/Normal/Over/Obese
df['BMI_Category'] = df['BMI_Category'].astype(int)

# Age Group
df['Age_Group'] = pd.cut(df['Age'],
    bins=[0, 30, 45, 60, 100],
    labels=[0, 1, 2, 3])  # Young/Middle/Senior/Elder
df['Age_Group'] = df['Age_Group'].astype(int)

# Glucose x BMI — combined risk factor
df['Glucose_BMI'] = df['Glucose'] * df['BMI'] / 1000

# High risk flag — both glucose and BMI high
df['High_Risk'] = ((df['Glucose'] > 140) &
                   (df['BMI'] > 30)).astype(int)

print("✅ BMI_Category created (0=Under, 1=Normal, 2=Over, 3=Obese)")
print("✅ Age_Group created (0=Young, 1=Middle, 2=Senior, 3=Elder)")
print("✅ Glucose_BMI interaction feature created")
print("✅ High_Risk flag created")
print("\n📊 New dataset shape:", df.shape)

# ─────────────────────────────────────────────
# STEP 4 — Calculate Class Ratio
# ─────────────────────────────────────────────
healthy = df['Outcome'].value_counts()[0]
diabetic = df['Outcome'].value_counts()[1]
ratio = healthy / diabetic
print(f"\n⚖️ Class ratio (healthy/diabetic): {ratio:.2f}")
print(f"This means: model will pay {ratio:.2f}x more attention to diabetic patients")

# ─────────────────────────────────────────────
# STEP 5 — Prepare Features and Target
# ─────────────────────────────────────────────
X = df.drop('Outcome', axis=1)
y = df['Outcome']
print(f"\n✅ Features (X): {list(X.columns)}")
print(f"✅ Target (y): Outcome (0=Healthy, 1=Diabetic)")

# ─────────────────────────────────────────────
# STEP 6 — Split Data with Stratify
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("SPLITTING DATA")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y)  # Ensures balanced split!

print(f"✅ Training samples: {len(X_train)}")
print(f"✅ Testing samples: {len(X_test)}")
print(f"✅ Train diabetic ratio: {y_train.mean():.2f}")
print(f"✅ Test diabetic ratio: {y_test.mean():.2f}")

# ─────────────────────────────────────────────
# STEP 7 — Scale Features
# ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n✅ Features scaled with StandardScaler")

# ─────────────────────────────────────────────
# STEP 8 — Train Final Model
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("TRAINING XGBOOST MODEL")
print("=" * 50)

model = XGBClassifier(
    n_estimators=100,       # 100 trees
    learning_rate=0.1,      # Learning speed
    max_depth=4,            # Tree depth — prevents overfitting
    subsample=0.8,          # Use 80% data per tree
    colsample_bytree=0.8,   # Use 80% features per tree
    random_state=42,        # Reproducibility
    scale_pos_weight=ratio  # Handle class imbalance
)

model.fit(X_train_scaled, y_train)
print("✅ Model training complete!")

# ─────────────────────────────────────────────
# STEP 9 — Evaluate Model
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Detailed Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Healthy', 'Diabetic']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(f"True Negative  (Healthy→Healthy):   {cm[0][0]}")
print(f"False Positive (Healthy→Diabetic):  {cm[0][1]}")
print(f"False Negative (Diabetic→Healthy):  {cm[1][0]} ← Missed diabetics!")
print(f"True Positive  (Diabetic→Diabetic): {cm[1][1]}")

# ─────────────────────────────────────────────
# STEP 10 — Cross Validation
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("CROSS VALIDATION — 5 FOLD")
print("=" * 50)

cv_accuracy = cross_val_score(
    model, X, y, cv=5, scoring='accuracy')
cv_recall = cross_val_score(
    model, X, y, cv=5, scoring='recall')
cv_f1 = cross_val_score(
    model, X, y, cv=5, scoring='f1')

print(f"✅ CV Accuracy: {cv_accuracy.mean():.3f} ± {cv_accuracy.std():.3f}")
print(f"✅ CV Recall:   {cv_recall.mean():.3f} ± {cv_recall.std():.3f}")
print(f"✅ CV F1 Score: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

# ─────────────────────────────────────────────
# STEP 11 — Feature Importance
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance.to_string(index=False))
print(f"\n✅ Most important feature: {importance.iloc[0]['Feature']}")
print(f"✅ Least important feature: {importance.iloc[-1]['Feature']}")

# ─────────────────────────────────────────────
# STEP 12 — Test with Sample Patient
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("SAMPLE PREDICTION TEST")
print("=" * 50)

sample_patient = pd.DataFrame({
    'Pregnancies': [2],
    'Glucose': [148],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [0],
    'BMI': [33.6],
    'DiabetesPedigreeFunction': [0.627],
    'Age': [50],
    'BMI_Category': [2],
    'Age_Group': [2],
    'Glucose_BMI': [148 * 33.6 / 1000],
    'High_Risk': [1]
})

sample_scaled = scaler.transform(sample_patient)
prediction = model.predict(sample_scaled)[0]
probability = model.predict_proba(sample_scaled)[0][1]

print(f"Sample Patient: Glucose=148, BMI=33.6, Age=50")
print(f"Prediction: {'DIABETIC ⚠️' if prediction == 1 else 'HEALTHY ✅'}")
print(f"Probability: {probability * 100:.1f}%")

# ─────────────────────────────────────────────
# STEP 13 — Save Everything
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("SAVING MODEL")
print("=" * 50)

joblib.dump(model, "models/diabetes_model.pkl")
joblib.dump(scaler, "models/diabetes_scaler.pkl")
joblib.dump(list(X.columns), "models/diabetes_features.pkl")
print("✅ diabetes_model.pkl saved!")
print("✅ diabetes_scaler.pkl saved!")
print("✅ diabetes_features.pkl saved!")

print("\n" + "=" * 50)
print("DIABETES MODEL COMPLETE! ✅")
print("=" * 50)