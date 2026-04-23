import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report

# --- Configurations ---
DATA_DIRS = [
    r"E:\UCL\CASA0018-ML\Dataset_Part1",
    r"E:\UCL\CASA0018-ML\Dataset_Part2",
    r"E:\UCL\CASA0018-ML\Dataset_Part3",
    r"E:\UCL\CASA0018-ML\Dataset_Part4"
]
LOG_FILE_PATH = "exposure_training_log.csv"
EXPORT_MODEL_PATH = "exposure_expert_v4.tflite"

IMG_SHAPE = (224, 224)
BATCH_SIZE = 32
STAGE1_EPOCHS = 70
STAGE2_EPOCHS = 50
LR_HEAD = 1e-4
LR_FINETUNE = 1e-6


# --- Dataset Parsing and Splitting ---
def parse_dataset_metadata(paths):
    """
    Scans directories to assign exposure labels (0: Under, 1: Normal, 2: Over)
    and flags 0 EV images for specific augmentation.
    """
    img_paths, labels, zero_ev_flags = [], [], []

    for base_path in paths:
        print(f"Scanning directory: {base_path}")
        folders = [f for f in glob.glob(os.path.join(base_path, "*")) if os.path.isdir(f)]

        for folder in folders:
            all_imgs = [img for img in os.listdir(folder) if img.endswith(".jpg")]
            count = len(all_imgs)

            # Identify the 0 EV image based on the number of exposures in the set
            base_ev_file = ""
            if count == 9:
                base_ev_file = "5.jpg"
            elif count == 7:
                base_ev_file = "4.jpg"
            elif count == 5:
                base_ev_file = "3.jpg"

            if not base_ev_file: continue

            base_idx = int(base_ev_file.split('.')[0])

            for img_name in all_imgs:
                full_path = os.path.join(folder, img_name)
                idx = int(img_name.split('.')[0])

                # Assign labels relative to the 0EV image
                if idx < base_idx - 1:
                    label = 0
                elif idx > base_idx + 1:
                    label = 2
                else:
                    label = 1

                img_paths.append(full_path)
                labels.append(label)
                zero_ev_flags.append(1 if img_name == base_ev_file else 0)

    return np.array(img_paths), np.array(labels), np.array(zero_ev_flags)


X_paths, y_labels, flags = parse_dataset_metadata(DATA_DIRS)

# Dataset split: 80% Train, 10% Validation, 10% Test
X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
    X_paths, y_labels, flags, test_size=0.2, random_state=42, stratify=y_labels)
X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
    X_temp, y_temp, f_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Handle class imbalance
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights_array))


# --- Data Loader ---
def preprocess_image(file_path, label, is_zero_ev):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SHAPE)
    img = tf.cast(img, tf.float32) / 255.0

    def apply_exposure_aug(image):
        image = tf.image.random_brightness(image, max_delta=0.08)
        image = tf.image.random_contrast(image, 0.92, 1.08)
        return image

    # Conditionally apply photometric augmentation only to 0 EV samples
    img = tf.cond(tf.equal(is_zero_ev, 1), lambda: apply_exposure_aug(img), lambda: img)

    # Spatial augmentation for all samples
    img = tf.image.random_flip_left_right(img)
    return img, label


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, f_train)) \
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
    .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val, f_val)) \
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test, f_test)) \
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Learning Rate Schedule ---
steps_per_epoch = len(X_train) // BATCH_SIZE
total_steps_stage1 = STAGE1_EPOCHS * steps_per_epoch

lr_schedule_stage1 = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR_HEAD,
    decay_steps=total_steps_stage1,
    alpha=0.1
)


# --- Neural Network ---
def build_classifier():
    backbone = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    backbone.trainable = False

    model = models.Sequential([
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(3, activation='softmax')
    ])
    return model


model = build_classifier()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule_stage1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# --- Two-Stage Training ---
csv_logger = callbacks.CSVLogger(LOG_FILE_PATH, append=True)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

print("\n--- Stage 1: Training Classification Head ---")
history_stage1 = model.fit(train_ds, validation_data=val_ds, epochs=STAGE1_EPOCHS,
                           callbacks=[csv_logger, early_stopping], class_weight=class_weights_dict)

print("\n--- Stage 2: Fine-Tuning Backbone (BN layers frozen) ---")
backbone_layer = model.layers[0]
backbone_layer.trainable = True

# Freeze BN layers to prevent moving statistics from updating during fine-tuning
for layer in backbone_layer.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
    elif layer in backbone_layer.layers[:-20]:
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_FINETUNE),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history_stage2 = model.fit(train_ds, validation_data=val_ds,
                           epochs=STAGE1_EPOCHS + STAGE2_EPOCHS,
                           initial_epoch=history_stage1.epoch[-1],
                           callbacks=[csv_logger, early_stopping],
                           class_weight=class_weights_dict)


# --- Evaluation and Visualization ---
def plot_training_history(h1, h2):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.axvline(x=len(h1.history['accuracy']) - 1, color='r', linestyle='--', label='Fine-tuning start')
    plt.title('Model Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.axvline(x=len(h1.history['loss']) - 1, color='r', linestyle='--', label='Fine-tuning start')
    plt.title('Model Loss')
    plt.legend()
    plt.show()


plot_training_history(history_stage1, history_stage2)

all_acc = history_stage1.history['accuracy'] + history_stage2.history['accuracy']
all_val_acc = history_stage1.history['val_accuracy'] + history_stage2.history['val_accuracy']

print("\n" + "=" * 40)
print("Training Summary:")
print(f"Peak Training Accuracy:   {max(all_acc):.4f}")
print(f"Peak Validation Accuracy: {max(all_val_acc):.4f}")
print("=" * 40)


# --- Confusion Matrix ---
def evaluate_and_plot_cm(model, test_ds, class_names):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Test Set')
    plt.show()

    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))


evaluate_and_plot_cm(model, test_ds, ['Under', 'Normal', 'Over'])

# --- TFLite Export ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(EXPORT_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"\nModel successfully exported to: {EXPORT_MODEL_PATH}")