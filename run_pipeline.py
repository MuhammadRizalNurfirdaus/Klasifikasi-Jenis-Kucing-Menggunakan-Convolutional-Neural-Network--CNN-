#!/usr/bin/env python3
"""
Complete evaluation and export pipeline for CNN cat classification model.
Runs while notebook training is in progress.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import shutil

# TensorFlow/Keras imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflowjs as tfjs

print("=" * 70)
print("RUNNING OFFLINE EVALUATION & EXPORT PIPELINE")
print("=" * 70)
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {keras.__version__}")

# Paths
MODEL_PATH = 'best_model.h5'
DATASET_ROOT = '/root/.cache/kagglehub/datasets/mibrahimhanif/jenis-kucing/versions/1/train val'
SPLIT_ROOT = 'dataset_split'
SAVED_MODEL_PATH = 'submission/saved_model'
TFLITE_PATH = 'submission/tflite'
TFJS_PATH = 'submission/tfjs_model'

CLASS_NAMES = ['Belang Tiga', 'Hitam', 'Kampung']

# ============================================================================
# LOAD MODEL & GENERATORS
# ============================================================================
print("\n[1/7] Loading model and data generators...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    sys.exit(1)

# Create generators
test_augmentation = ImageDataGenerator(rescale=1./255)
test_generator = test_augmentation.flow_from_directory(
    os.path.join(SPLIT_ROOT, 'test'),
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
print(f"✓ Test generator created with {test_generator.samples} samples")

# ============================================================================
# EVALUATION
# ============================================================================
print("\n[2/7] Evaluating model on test set...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
print(f"✓ Test Loss: {test_loss:.4f}")
print(f"✓ Test Accuracy: {test_accuracy:.4f}")

# ============================================================================
# PREDICTIONS & CONFUSION MATRIX
# ============================================================================
print("\n[3/7] Computing predictions and confusion matrix...")
y_pred_probs = model.predict(test_generator, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

cm = confusion_matrix(y_true, y_pred)
print("✓ Confusion Matrix:")
print(cm)

# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix - Test Set')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix_test.png', dpi=100, bbox_inches='tight')
plt.close()
print("✓ Confusion matrix plot saved to confusion_matrix_test.png")

# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================
print("\n[4/7] Generating classification report...")
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
print(report)
with open('classification_report.txt', 'w') as f:
    f.write(report)
print("✓ Classification report saved to classification_report.txt")

# ============================================================================
# EXPORT: SavedModel
# ============================================================================
print("\n[5/7] Exporting SavedModel (Keras 3 format)...")
if os.path.exists(SAVED_MODEL_PATH):
    shutil.rmtree(SAVED_MODEL_PATH)
try:
    model.export(SAVED_MODEL_PATH)
    print(f"✓ SavedModel exported to {SAVED_MODEL_PATH}")
except Exception as e:
    print(f"✗ SavedModel export failed: {e}")

# ============================================================================
# EXPORT: TFLite
# ============================================================================
print("\n[6/7] Exporting TFLite (quantized)...")
if not os.path.exists(TFLITE_PATH):
    os.makedirs(TFLITE_PATH)

try:
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_file = os.path.join(TFLITE_PATH, 'model.tflite')
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    print(f"✓ TFLite model saved to {tflite_file}")
    print(f"  Size: {os.path.getsize(tflite_file) / 1024 / 1024:.2f} MB")
    
    # Save label file
    label_file = os.path.join(TFLITE_PATH, 'label.txt')
    with open(label_file, 'w') as f:
        for name in CLASS_NAMES:
            f.write(name + '\n')
    print(f"✓ Label file saved to {label_file}")
except Exception as e:
    print(f"✗ TFLite export failed: {e}")

# ============================================================================
# EXPORT: TensorFlow.js
# ============================================================================
print("\n[7/7] Exporting TensorFlow.js...")
if os.path.exists(TFJS_PATH):
    shutil.rmtree(TFJS_PATH)
try:
    tfjs.converters.save_keras_model(model, TFJS_PATH)
    print(f"✓ TFJS model exported to {TFJS_PATH}")
    print(f"  Files: {os.listdir(TFJS_PATH)}")
except Exception as e:
    print(f"✗ TFJS export failed: {e}")

# ============================================================================
# VERIFY SUBMISSION STRUCTURE
# ============================================================================
print("\n" + "=" * 70)
print("SUBMISSION FOLDER STRUCTURE")
print("=" * 70)
submission_struct = {}
for root, dirs, files in os.walk('submission'):
    level = root.replace('submission', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    sub_indent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        size = os.path.getsize(os.path.join(root, file)) / 1024 / 1024
        print(f"{sub_indent}{file} ({size:.2f} MB)")
    if len(files) > 5:
        print(f"{sub_indent}... and {len(files) - 5} more files")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE!")
print("=" * 70)
print("\nNOTE: Notebook training is still running in the background.")
print("Once it completes, all notebook cells will execute automatically.")
print("Then run: git add . && git commit -m 'Training complete with exports'")
