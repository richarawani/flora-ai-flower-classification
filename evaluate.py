import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. SETTINGS
# ==========================================

dataset_path = "flower_images"

IMAGE_SIZE = (180, 180)
BATCH_SIZE = 32
SEED = 42


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model(
    "flower_cnn_model.keras"
)

print("Model loaded successfully!")


# ==========================================
# 3. LOAD THE SAME 20% TEMPORARY DATA
# ==========================================

temp_dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# 4. GET CLASS NAMES
# ==========================================

class_names = temp_dataset.class_names

print("\nFlower Classes:")
print(class_names)


# ==========================================
# 5. GET ONLY THE 500 TEST IMAGES
# ==========================================

temp_dataset = temp_dataset.unbatch()

# Skip the first 500 validation images
# Keep the last 500 test images
test_dataset = temp_dataset.skip(500)

test_dataset = test_dataset.batch(BATCH_SIZE)


print("\nEvaluating on 500 unseen test images...")


# ==========================================
# 6. GET PREDICTIONS
# ==========================================

true_labels = []
predicted_labels = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    true_labels.extend(
        labels.numpy()
    )

    predicted_labels.extend(
        predicted_classes
    )


true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)


# ==========================================
# 7. TEST ACCURACY
# ==========================================

accuracy = np.mean(
    true_labels == predicted_labels
)


print("\n================================")
print("TEST ACCURACY")
print("================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 9. DISPLAY CONFUSION MATRIX
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    xticks_rotation=45
)

plt.title(
    "Flower Classification - Test Set"
)

plt.tight_layout()

plt.show()


# ==========================================
# 10. CLASSIFICATION REPORT
# ==========================================

print("\n================================")
print("CLASSIFICATION REPORT")
print("================================")

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names
)

print(report)