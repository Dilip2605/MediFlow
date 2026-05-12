import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report,
    confusion_matrix, accuracy_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense,
    Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint)
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator)

# ═══════════════════════════════════════════════════
# MEDIFLOW — CNN X-RAY ANALYSIS MODEL
# Purpose: Detect Pneumonia and TB from chest X-rays
# Dataset: Chest X-Ray images (train/test/val folders)
# Algorithm: DenseNet121 with Transfer Learning
#
# WHY DENSENET121?
# - Same architecture used in HealthLens project
# - Specifically designed for medical imaging
# - Dense connections: each layer connects to ALL
#   previous layers → better feature reuse
# - Pre-trained on 1.2M ImageNet images
#   → Already knows edges, textures, shapes
# - Fine-tune only top layers → fast training
# ═══════════════════════════════════════════════════

print("=" * 60)
print("   MEDIFLOW — CNN X-RAY ANALYSIS MODEL")
print("=" * 60)

# ─────────────────────────────────────────────────────
# STEP 1 — CONFIGURATION
# Why: Centralize all settings for easy modification
# ─────────────────────────────────────────────────────
print("\n⚙️ STEP 1: Configuration...")

CONFIG = {
    # Image settings
    'img_size': (224, 224),    # DenseNet121 input size
    'img_channels': 3,          # RGB channels
    'batch_size': 32,           # Images per batch

    # Training settings
    'epochs': 20,               # Max training epochs
    'learning_rate': 0.0001,    # Initial learning rate

    # Data paths
    'train_path': 'data/x_ray/pneumonia/train',
    'test_path': 'data/x_ray/pneumonia/test',

    # Model settings
    'dropout_rate': 0.5,        # Dropout to prevent overfitting
    'dense_units': 256,         # Dense layer size

    # Classes
    'classes': ['NORMAL', 'PNEUMONIA'],
    'n_classes': 2
}

for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ─────────────────────────────────────────────────────
# STEP 2 — CHECK DATA
# Why: Verify images exist before building model
# ─────────────────────────────────────────────────────
print("\n📂 STEP 2: Checking Data...")

train_path = CONFIG['train_path']
test_path = CONFIG['test_path']

for class_name in CONFIG['classes']:
    train_count = len(os.listdir(
        os.path.join(train_path, class_name)))
    test_count = len(os.listdir(
        os.path.join(test_path, class_name)))
    print(f"✅ {class_name}: "
          f"{train_count} train, "
          f"{test_count} test images")

total_train = sum([
    len(os.listdir(os.path.join(train_path, c)))
    for c in CONFIG['classes']])
total_test = sum([
    len(os.listdir(os.path.join(test_path, c)))
    for c in CONFIG['classes']])

print(f"\n✅ Total training images: {total_train}")
print(f"✅ Total test images: {total_test}")

# ─────────────────────────────────────────────────────
# STEP 3 — DATA AUGMENTATION
# Why: Increases effective dataset size without new images
# Augmentation techniques:
# - Rotation: X-rays can be slightly tilted
# - Zoom: Different distances from patient
# - Flip: Left/right flip (horizontal only for X-rays)
# - Brightness: Different X-ray machine settings
# - Shear: Slight perspective changes
#
# IMPORTANT: Only augment TRAINING data!
# Test data must be original — augmenting test = cheating!
# ─────────────────────────────────────────────────────
print("\n🔧 STEP 3: Data Augmentation...")

# Training data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel 0-255 to 0-1
    rotation_range=15,        # Rotate up to 15 degrees
    zoom_range=0.1,           # Zoom in/out 10%
    width_shift_range=0.1,    # Shift horizontally 10%
    height_shift_range=0.1,   # Shift vertically 10%
    horizontal_flip=True,     # Flip left/right
    brightness_range=[0.8, 1.2],  # Brightness variation
    fill_mode='nearest'       # Fill new pixels with nearest
)

# Test data — NO augmentation, only normalization
test_datagen = ImageDataGenerator(
    rescale=1./255)           # Only normalize pixels

print("✅ Training augmentation configured:")
print("   - Rotation: ±15°")
print("   - Zoom: ±10%")
print("   - Horizontal flip: Yes")
print("   - Brightness: 0.8-1.2x")
print("✅ Test normalization only")

# ─────────────────────────────────────────────────────
# STEP 4 — DATA GENERATORS
# Why: Load images in batches — saves memory!
# Loading 5000 images at once = 2GB+ memory
# Batch loading = only 32 images at a time
# ─────────────────────────────────────────────────────
print("\n📊 STEP 4: Creating Data Generators...")

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=True,
    seed=42)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size'],
    class_mode='categorical',
    shuffle=False)  # No shuffle for evaluation

print(f"✅ Training generator: {train_generator.samples} images")
print(f"✅ Test generator: {test_generator.samples} images")
print(f"✅ Classes found: {train_generator.class_indices}")

# ─────────────────────────────────────────────────────
# STEP 5 — BUILD MODEL WITH TRANSFER LEARNING
# Why Transfer Learning:
# Training CNN from scratch needs millions of images
# DenseNet121 already trained on 1.2M ImageNet images
# Already knows: edges, textures, shapes, patterns
# We just teach it: these patterns = Pneumonia/Normal
#
# Architecture:
# DenseNet121 Base (frozen) → Global Avg Pool
# → Dense(256) → BatchNorm → Dropout(0.5)
# → Dense(2) → Softmax → [Normal, Pneumonia]
# ─────────────────────────────────────────────────────
print("\n🏗️ STEP 5: Building CNN Model...")

# Load pre-trained DenseNet121
# include_top=False: remove original classification head
# weights='imagenet': use pre-trained weights
base_model = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=(*CONFIG['img_size'],
                 CONFIG['img_channels']))

# FREEZE base model weights
# Why: We don't want to overwrite pre-trained knowledge
# Only train our custom classification head
base_model.trainable = False

print(f"✅ DenseNet121 loaded")
print(f"   Base model layers: {len(base_model.layers)}")
print(f"   Base model trainable: {base_model.trainable}")

# Build custom classification head
x = base_model.output

# Global Average Pooling
# Converts 7×7×1024 feature maps to 1024 values
# Reduces parameters dramatically!
x = GlobalAveragePooling2D()(x)

# Dense layer for learning disease patterns
x = Dense(CONFIG['dense_units'],
    activation='relu')(x)

# Batch Normalization
# Normalizes activations → faster training, more stable
x = BatchNormalization()(x)

# Dropout
# Randomly drops 50% neurons during training
# Prevents overfitting!
x = Dropout(CONFIG['dropout_rate'])(x)

# Output layer
# n_classes outputs, softmax for probability
outputs = Dense(CONFIG['n_classes'],
    activation='softmax')(x)

# Create final model
model = Model(
    inputs=base_model.input,
    outputs=outputs)

print(f"✅ Custom head added")
print(f"   Total layers: {len(model.layers)}")
print(f"   Trainable params: {model.count_params():,}")

# ─────────────────────────────────────────────────────
# STEP 6 — COMPILE MODEL
# Why: Define how model learns
# Optimizer: Adam — adapts learning rate automatically
# Loss: categorical_crossentropy — standard for classification
# Metrics: accuracy — easy to understand
# ─────────────────────────────────────────────────────
print("\n⚙️ STEP 6: Compiling Model...")

model.compile(
    optimizer=Adam(learning_rate=CONFIG['learning_rate']),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

print(f"✅ Optimizer: Adam (lr={CONFIG['learning_rate']})")
print(f"✅ Loss: categorical_crossentropy")
print(f"✅ Metrics: accuracy")

# ─────────────────────────────────────────────────────
# STEP 7 — TRAINING CALLBACKS
# Why: Smart training that stops at right time
#
# EarlyStopping:
# Monitors validation loss
# Stops when no improvement for 5 epochs
# Prevents overfitting and wasted time!
#
# ReduceLROnPlateau:
# Reduces learning rate when stuck
# Like slowing down to find exact minimum
#
# ModelCheckpoint:
# Saves BEST model during training
# Not just the last model!
# ─────────────────────────────────────────────────────
print("\n🔧 STEP 7: Setting Up Callbacks...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1),
    ModelCheckpoint(
        filepath='models/cnn_best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1)
]

print("✅ EarlyStopping: stops if no improvement for 5 epochs")
print("✅ ReduceLROnPlateau: reduces LR if stuck for 3 epochs")
print("✅ ModelCheckpoint: saves best model automatically")

# ─────────────────────────────────────────────────────
# STEP 8 — PHASE 1 TRAINING — FROZEN BASE
# Why: Train only classification head first
# Base model frozen → only 4 custom layers train
# Fast training — few parameters to update
# ─────────────────────────────────────────────────────
print("\n🤖 STEP 8: Phase 1 Training (Frozen Base)...")
print("Training only classification head...")

history_phase1 = model.fit(
    train_generator,
    epochs=CONFIG['epochs'],
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1)

print("✅ Phase 1 training complete!")

# ─────────────────────────────────────────────────────
# STEP 9 — PHASE 2 TRAINING — FINE TUNING
# Why: Unfreeze last layers of DenseNet for fine-tuning
# Last layers learned general features
# We teach them medical-specific features
# Use VERY small learning rate — don't destroy pre-trained weights!
# ─────────────────────────────────────────────────────
print("\n🔧 STEP 9: Phase 2 — Fine Tuning...")

# Unfreeze last 20 layers of DenseNet
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile with smaller learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

print(f"✅ Unfroze last 20 DenseNet layers")
print(f"✅ Recompiled with learning_rate=1e-5")

history_phase2 = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    callbacks=callbacks,
    verbose=1)

print("✅ Phase 2 fine-tuning complete!")

# ─────────────────────────────────────────────────────
# STEP 10 — EVALUATE MODEL
# ─────────────────────────────────────────────────────
print("\n📊 STEP 10: Evaluation...")
print("=" * 60)

# Evaluate on test set
test_loss, test_acc = model.evaluate(
    test_generator, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc * 100:.2f}%")
print(f"📉 Test Loss: {test_loss:.4f}")

# Detailed predictions
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Class labels
class_labels = list(
    test_generator.class_indices.keys())

print(f"\n📋 Classification Report:")
print(classification_report(
    y_true, y_pred,
    target_names=class_labels))

cm = confusion_matrix(y_true, y_pred)
print(f"\n📊 Confusion Matrix:")
for i, label in enumerate(class_labels):
    for j, pred_label in enumerate(class_labels):
        status = "✅" if i == j else "❌"
        print(f"  Actual {label:10} → "
              f"Predicted {pred_label:10}: "
              f"{cm[i][j]:4d} {status}")

# ─────────────────────────────────────────────────────
# STEP 11 — SAVE MODEL
# ─────────────────────────────────────────────────────
print("\n💾 STEP 11: Saving Model...")

# Save full model
model.save("models/cnn_xray_model.keras")
print("✅ cnn_xray_model.keras saved!")

# Save class labels
joblib.dump(class_labels,
    "models/cnn_class_labels.pkl")
print("✅ cnn_class_labels.pkl saved!")

# Save configuration
joblib.dump(CONFIG, "models/cnn_config.pkl")
print("✅ cnn_config.pkl saved!")

# Save metadata
metadata = {
    'model_type': 'DenseNet121 Transfer Learning',
    'architecture': 'DenseNet121 + Custom Head',
    'input_size': CONFIG['img_size'],
    'classes': class_labels,
    'n_classes': CONFIG['n_classes'],
    'test_accuracy': float(test_acc),
    'test_loss': float(test_loss),
    'training_images': total_train,
    'test_images': total_test
}
joblib.dump(metadata, "models/cnn_metadata.pkl")
print("✅ cnn_metadata.pkl saved!")

print("\n" + "=" * 60)
print("   CNN X-RAY MODEL — SUMMARY")
print("=" * 60)
print(f"  Architecture: DenseNet121 + Custom Head")
print(f"  Training images: {total_train}")
print(f"  Test images: {total_test}")
print(f"  Classes: {class_labels}")
print(f"  Test Accuracy: {test_acc*100:.2f}%")
print(f"  Files Saved: 4")
print("=" * 60)
print("✅ CNN X-RAY MODEL COMPLETE!")