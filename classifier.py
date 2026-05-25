from PIL import Image
import tensorflow as tf
import numpy as np

cnn_model = tf.keras.models.load_model(
    "cnn_model.h5"
)

CLASS_NAMES = [
    "pothole",
    "garbage",
    "pipe_leak"
]

def classify_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = cnn_model.predict(
        img_array
    )

    class_index = np.argmax(
        prediction
    )

    return CLASS_NAMES[class_index]