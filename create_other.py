import tensorflow as tf
from PIL import Image
import os


# ==========================================
# SETTINGS
# ==========================================

OTHER_FOLDER = "flower_images/Other"

NUMBER_OF_IMAGES = 1000


# ==========================================
# CREATE OTHER FOLDER
# ==========================================

os.makedirs(
    OTHER_FOLDER,
    exist_ok=True
)


# ==========================================
# DOWNLOAD CIFAR-10
# ==========================================

print("Downloading CIFAR-10 dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)


# Combine train and test images
images = list(x_train) + list(x_test)


# ==========================================
# SAVE NON-FLOWER IMAGES
# ==========================================

print("Creating Other class...")

count = 0

for image in images:

    if count >= NUMBER_OF_IMAGES:
        break

    # Convert NumPy array to PIL image
    pil_image = Image.fromarray(image)

    # Resize to reasonable size
    pil_image = pil_image.resize(
        (180, 180)
    )

    # Save image
    image_path = os.path.join(
        OTHER_FOLDER,
        f"other_{count}.jpg"
    )

    pil_image.save(image_path)

    count += 1


print("\n================================")
print("OTHER CLASS CREATED")
print("================================")

print(
    f"Images saved: {count}"
)

print(
    f"Location: {OTHER_FOLDER}"
)

print("================================")