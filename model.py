import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# CONFIG
B_DATASET_PATHS = [
    r"E:\UCL\CASA0018-ML\Dataset_Part1",
    r"E:\UCL\CASA0018-ML\Dataset_Part2",
    r"E:\UCL\CASA0018-ML\Dataset_Part3",
    r"E:\UCL\CASA0018-ML\Dataset_Part4"
]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4


def get_data_with_aug_flags(paths):
    """
    Parse folders to map labels (Under/Normal/Over) and flag 0 EV images for augmentation.
    Label mapping: 0=Under, 1=Normal (0EV +/- 1), 2=Over.
    """
    image_paths, labels, is_base_ev_flags = [], [], []

    for base_path in paths:
        print(f"Scanning: {base_path} ...")
        folders = [f for f in glob.glob(os.path.join(base_path, "*")) if os.path.isdir(f)]

        for folder in folders:
            all_imgs = [img for img in os.listdir(folder) if img.endswith(".jpg")]
            count = len(all_imgs)

            # Map total image count to its 0 EV filename
            base_ev_map = {9: "5.jpg", 7: "4.jpg", 5: "3.jpg"}
            base_ev_file = base_ev_map.get(count, "")

            if not base_ev_file:
                continue

            base_idx = int(base_ev_file.split('.')[0])

            for img_name in all_imgs:
                idx = int(img_name.split('.')[0])

                # Group adjacent exposures into 'Normal'
                if idx < base_idx - 1:
                    label = 0
                elif idx > base_idx + 1:
                    label = 2
                else:
                    label = 1

                image_paths.append(os.path.join(folder, img_name))
                labels.append(label)
                is_base_ev_flags.append(1 if img_name == base_ev_file else 0)

    return np.array(image_paths), np.array(labels), np.array(is_base_ev_flags)


# --- Data Prep ---
X_paths, y_labels, flags = get_data_with_aug_flags(B_DATASET_PATHS)

# 80/10/10 split
X_train, X_temp, y_train, y_temp, f_train, f_temp = train_test_split(
    X_paths, y_labels, flags, test_size=0.2, random_state=42, stratify=y_labels)

X_val, X_test, y_val, y_test, f_val, f_test = train_test_split(
    X_temp, y_temp, f_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Samples -> Total: {len(X_paths)} | Train: {len(X_train)} | Val: {len(X_val)}")


def preprocess_image(file_path, label, is_base_ev):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    def augment_exposure(image):
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        return image

    # Only apply brightness/contrast shift to 0 EV images to preserve exposure labels
    img = tf.cond(tf.equal(is_base_ev, 1), lambda: augment_exposure(img), lambda: img)

    # Generic augs for all
    img = tf.image.random_flip_left_right(img)
    return img, label


# Build TF datasets
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train, f_train)) \
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
    .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val, f_val)) \
    .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- Model setup ---
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cb_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

# --- Train ---
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb_list)


# --- Eval & Export ---
def plot_results(hist):
    acc, val_acc = hist.history['accuracy'], hist.history['val_accuracy']
    loss, val_loss = hist.history['loss'], hist.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show()


plot_results(history)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('exposure_smart_v2.tflite', 'wb') as f:
    f.write(tflite_model)