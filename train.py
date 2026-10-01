
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

# ---------------- CONFIGURATION ----------------

DATA_DIR = Path("flower_images")
MODEL_PATH = "flower_cnn_model.keras"
CLASS_NAMES_PATH = "class_names.json"
THRESHOLD_PATH = "rejection_threshold.json"

IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 10

tf.keras.utils.set_random_seed(SEED)

# ---------------- CHECK DATASET ----------------

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR.resolve()}"
    )

class_names = sorted(
    [p.name for p in DATA_DIR.iterdir() if p.is_dir()]
)

if len(class_names) < 2:
    raise ValueError("At least two class folders are required.")

print("Classes:", class_names)

# Collect images and their labels.
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
paths = []
labels = []

for label, class_name in enumerate(class_names):
    folder = DATA_DIR / class_name
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]

    print(f"{class_name}: {len(files)} images")

    if len(files) < 10:
        raise ValueError(
            f"Not enough images in {class_name}. "
            "Please add at least 10 images."
        )

    paths.extend([str(p) for p in files])
    labels.extend([label] * len(files))

paths = np.array(paths)
labels = np.array(labels, dtype=np.int32)

# Make sure the Other class exists.
if "Other" not in class_names:
    raise ValueError(
        "A folder named 'Other' is required."
    )

# Stratified 80/10/10 split.
train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )
)

val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels
    )
)

print("\nDataset split:")
print("Training:", len(train_paths))
print("Validation:", len(val_paths))
print("Testing:", len(test_paths))

# Save class order for the Streamlit app.
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

# ---------------- DATA PIPELINE ----------------

AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label

def make_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

train_ds = make_dataset(
    train_paths, train_labels, training=True
)
val_ds = make_dataset(val_paths, val_labels)
test_ds = make_dataset(test_paths, test_labels)

# ---------------- DATA AUGMENTATION ----------------

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.12),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomTranslation(0.10, 0.10),
    tf.keras.layers.RandomContrast(0.15),
], name="augmentation")

# ---------------- BUILD MODEL ----------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))

x = augmentation(inputs)

# MobileNetV2 expects pixels in the [-1, 1] range.
x = tf.keras.layers.Rescaling(
    scale=1.0 / 127.5,
    offset=-1
)(x)

x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.35)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.25)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------- CLASS WEIGHTS ----------------

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = {
    int(label): float(weight)
    for label, weight in zip(np.unique(train_labels), weights)
}

print("\nClass weights:", class_weights)

# ---------------- TRAINING CALLBACKS ----------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

# ---------------- STAGE 1: TRAIN CLASSIFIER ----------------

print("\nStage 1: Training classification head...")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ---------------- STAGE 2: FINE-TUNING ----------------

print("\nStage 2: Fine-tuning MobileNetV2...")

base_model.trainable = True

# Fine-tune only the last 30 layers.
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Keep Batch Normalization layers frozen for stability.
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# ---------------- LOAD BEST MODEL ----------------

print("\nLoading best saved model...")
model = tf.keras.models.load_model(MODEL_PATH)

# ---------------- FINAL TEST ----------------

print("\nEvaluating on unseen test images...")
test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

predictions = model.predict(test_ds, verbose=1)
predicted_labels = np.argmax(predictions, axis=1)

print("\nClassification report:")
print(classification_report(
    test_labels,
    predicted_labels,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
))

# ---------------- CONFUSION MATRIX ----------------

cm = confusion_matrix(
    test_labels,
    predicted_labels,
    labels=np.arange(len(class_names))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 8))
display.plot(
    ax=ax,
    xticks_rotation=45,
    cmap="Blues",
    values_format="d"
)
plt.title("Flower Classifier - Test Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()

# ---------------- REJECTION THRESHOLD ----------------
# Compare the maximum flower probability with the
# Other probability on the validation set.
#
# If a flower is uncertain, the app can classify it
# as Other using this threshold.

other_index = class_names.index("Other")
flower_indices = [
    i for i, name in enumerate(class_names)
    if name != "Other"
]

val_predictions = model.predict(val_ds, verbose=1)

flower_scores = np.max(
    val_predictions[:, flower_indices],
    axis=1
)

# The validation set contains known flowers and known Other
# images. Select a threshold that maximizes balanced accuracy
# for distinguishing these two groups.
val_is_flower = (val_labels != other_index).astype(int)

best_threshold = 0.50
best_score = -1.0

for threshold in np.arange(0.20, 0.96, 0.01):
    predicted_is_flower = (
        flower_scores >= threshold
    ).astype(int)

    true_positive = np.sum(
        (predicted_is_flower == 1) & (val_is_flower == 1)
    )
    false_negative = np.sum(
        (predicted_is_flower == 0) & (val_is_flower == 1)
    )
    true_negative = np.sum(
        (predicted_is_flower == 0) & (val_is_flower == 0)
    )
    false_positive = np.sum(
        (predicted_is_flower == 1) & (val_is_flower == 0)
    )

    sensitivity = true_positive / max(
        true_positive + false_negative, 1
    )
    specificity = true_negative / max(
        true_negative + false_positive, 1
    )

    balanced_accuracy = (sensitivity + specificity) / 2

    if balanced_accuracy > best_score:
        best_score = balanced_accuracy
        best_threshold = float(threshold)

with open(THRESHOLD_PATH, "w") as f:
    json.dump(
        {
            "flower_threshold": best_threshold,
            "validation_balanced_accuracy": best_score
        },
        f,
        indent=4
    )

print(f"\nFlower rejection threshold: {best_threshold:.2f}")
print(f"Validation balanced accuracy: {best_score * 100:.2f}%")

print("\nTraining completed!")
print(f"Model saved to: {MODEL_PATH}")
print(f"Class names saved to: {CLASS_NAMES_PATH}")
print(f"Threshold saved to: {THRESHOLD_PATH}")
print("Confusion matrix saved to: confusion_matrix.png")