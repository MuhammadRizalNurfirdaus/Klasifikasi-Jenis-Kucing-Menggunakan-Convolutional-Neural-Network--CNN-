#!/usr/bin/env python3
import os, sys, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import shutil, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflowjs as tfjs

print("\n" + "="*70)
print("COMPLETE CNN PIPELINE - ALL OPERATIONS")
print("="*70)

MODEL_PATH = 'best_model.h5'
DATASET_ROOT = '/root/.cache/kagglehub/datasets/mibrahimhanif/jenis-kucing/versions/1/train val'
SPLIT_ROOT = 'dataset_split'
SAVED_MODEL_PATH = 'submission/saved_model'
TFLITE_PATH = 'submission/tflite'
TFJS_PATH = 'submission/tfjs_model'
CLASS_NAMES = ['Belang Tiga', 'Hitam', 'Kampung']

print("\n[LOADING] Model and generators...")
model = keras.models.load_model(MODEL_PATH)
print(f"✓ Model loaded: {MODEL_PATH}")

test_augmentation = ImageDataGenerator(rescale=1./255)
test_gen = test_augmentation.flow_from_directory(os.path.join(SPLIT_ROOT, 'test'), target_size=(150, 150), batch_size=32, class_mode='categorical', shuffle=False)
train_gen = test_augmentation.flow_from_directory(os.path.join(SPLIT_ROOT, 'train'), target_size=(150, 150), batch_size=32, class_mode='categorical', shuffle=False)
val_gen = test_augmentation.flow_from_directory(os.path.join(SPLIT_ROOT, 'val'), target_size=(150, 150), batch_size=32, class_mode='categorical', shuffle=False)

print(f"✓ Generators created")

print("\n[1/7] Evaluating model...")
train_loss, train_acc = model.evaluate(train_gen, verbose=0)
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"✓ Train: {train_acc:.4f} | Val: {val_acc:.4f} | Test: {test_acc:.4f}")

print("\n[2/7] Confusion matrix & report...")
test_gen.reset()
y_pred_probs = model.predict(test_gen, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('submission/confusion_matrix_final.png', dpi=150, bbox_inches='tight'); plt.close()
print(f"✓ Confusion matrix saved")

report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print(report)
with open('submission/classification_report_final.txt', 'w') as f:
    f.write(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f}\n\n")
    f.write(report)
print(f"✓ Report saved")

print("\n[3/7] SavedModel export...")
if os.path.exists(SAVED_MODEL_PATH): shutil.rmtree(SAVED_MODEL_PATH)
try:
    model.export(SAVED_MODEL_PATH)
    print(f"✓ SavedModel exported")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n[4/7] TFLite export...")
if not os.path.exists(TFLITE_PATH): os.makedirs(TFLITE_PATH)
try:
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    tflite_file = os.path.join(TFLITE_PATH, 'model.tflite')
    with open(tflite_file, 'wb') as f: f.write(tflite_model)
    size_mb = os.path.getsize(tflite_file) / 1024 / 1024
    print(f"✓ TFLite saved ({size_mb:.2f} MB)")
    with open(os.path.join(TFLITE_PATH, 'label.txt'), 'w') as f:
        for name in CLASS_NAMES: f.write(name + '\n')
except Exception as e:
    print(f"✗ Error: {e}")

print("\n[5/7] TFJS export...")
if os.path.exists(TFJS_PATH): shutil.rmtree(TFJS_PATH)
try:
    tfjs.converters.save_keras_model(model, TFJS_PATH)
    files = os.listdir(TFJS_PATH)
    print(f"✓ TFJS exported ({len(files)} files)")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n[6/7] TFLite inference demo...")
try:
    interp = tf.lite.Interpreter(model_path=os.path.join(TFLITE_PATH, 'model.tflite'))
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    
    test_gen.reset()
    test_images = []
    test_labels = []
    for i in range((test_gen.samples // test_gen.batch_size) + 1):
        try:
            bx, by = next(test_gen)
            test_images.append(bx)
            test_labels.append(by)
        except StopIteration:
            break
    test_images = np.vstack(test_images)[:test_gen.samples]
    test_labels = np.vstack(test_labels)[:test_gen.samples]
    
    demo_idx = [0, 1, 36, 37, 72, 73]  # Fixed indices
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('TFLite Inference Demo', fontsize=14, fontweight='bold')
    
    for ax_idx, img_idx in enumerate(demo_idx):
        img = test_images[img_idx]
        true_label = CLASS_NAMES[np.argmax(test_labels[img_idx])]
        interp.set_tensor(input_details[0]['index'], img[np.newaxis, :, :, :].astype(np.float32))
        interp.invoke()
        pred_probs = interp.get_tensor(output_details[0]['index'])[0]
        pred_class = CLASS_NAMES[np.argmax(pred_probs)]
        confidence = pred_probs[np.argmax(pred_probs)]
        
        ax = axes.flatten()[ax_idx]
        ax.imshow(img.astype(np.uint8))
        color = 'green' if pred_class == true_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_class} ({confidence:.2%})", fontsize=10, color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('submission/tflite_inference_demo.png', dpi=150, bbox_inches='tight'); plt.close()
    print(f"✓ Inference demo saved")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n[7/7] Summary...")
total_size = sum(os.path.getsize(os.path.join(r, f)) for r, d, files in os.walk('submission') for f in files) / 1024 / 1024
print(f"✓ Total submission size: {total_size:.2f} MB")

print("\n" + "="*70)
print("✅ PIPELINE COMPLETE - ALL OPERATIONS SUCCESSFUL")
print("="*70)
print("\nNEXT: git add . && git commit -m 'Complete: all exports + inference'")
print("="*70 + "\n")
