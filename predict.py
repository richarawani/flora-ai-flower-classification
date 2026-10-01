import tensorflow as tf
import numpy as np


# ==========================================
# 1. LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(
    "flower_cnn_model.keras"
)


# ==========================================
# 2. FLOWER CLASS NAMES
# ==========================================

class_names = [
    "Lilly",
    "Lotus",
    "Orchid",
    "Sunflower",
    "Tulip"
]


# ==========================================
# 3. LOAD IMAGE
# ==========================================

image_path = "sunflower.jpg"

image = tf.keras.utils.load_img(
    image_path,
    target_size=(180, 180)
)


# ==========================================
# 4. CONVERT IMAGE
# ==========================================

image_array = tf.keras.utils.img_to_array(
    image
)

image_array = tf.expand_dims(
    image_array,
    axis=0
)


# ==========================================
# 5. PREDICT
# ==========================================

predictions = model.predict(
    image_array
)


# ==========================================
# 6. FIND PREDICTED CLASS
# ==========================================

predicted_index = np.argmax(
    predictions[0]
)

predicted_flower = class_names[
    predicted_index
]


# ==========================================
# 7. GET CONFIDENCE
# ==========================================

confidence = (
    predictions[0][predicted_index] * 100
)


# ==========================================
# 8. DISPLAY RESULT
# ==========================================

print("--------------------------------")
print("Flower Classification Result")
print("--------------------------------")

print(
    "Predicted Flower:",
    predicted_flower
)

print(
    "Confidence:",
    round(confidence, 2),
    "%"
)

print("--------------------------------")