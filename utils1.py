import os
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from PIL import Image
import tensorflow as tf
import numpy as np

from math import radians, sin, cos, sqrt, atan2

from google import genai

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

tf.get_logger().setLevel("ERROR")


@lru_cache(maxsize=1)
def get_cnn_model():

    model_path = "efficientnet_model2.keras"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing model file: {model_path}"
        )

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


CLASS_NAMES = [
    "garbage",
    "overflow",
    "pothole"
]


def classify_image(image_path):

    cnn_model = get_cnn_model()

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = cnn_model.predict(
        img_array,
        verbose=0
    )

    class_index = np.argmax(prediction)

    return CLASS_NAMES[class_index]


def extract_intent(title, description):

    prompt = f"""
    Classify the civic complaint.

    Return ONLY one word.

    Allowed outputs:

    pothole
    garbage
    overflow

    Title:
    {title}

    Description:
    {description}
    """

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        output = response.text.lower().strip()

        if "pothole" in output:
            return "pothole"

        if "garbage" in output:
            return "garbage"

        if "overflow" in output:
            return "overflow"

        return "unknown"

    except Exception as e:

        print(f"Gemini error: {e}")

        return "unknown"

def detect_department(category):

    mapping = {
        "pothole": "PWD",
        "garbage": "Sanitation",
        "overflow": "Sanitation"
    }

    return mapping.get(
        category,
        "Unknown"
    )


def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    R = 6371000

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1))
        * cos(radians(lat2))
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c



def should_cluster(
    category1,
    category2,
    lat1,
    lon1,
    lat2,
    lon2
):

    if category1 != category2:
        return False

    distance = haversine_distance(
        lat1,
        lon1,
        lat2,
        lon2
    )

    print(
        f"Distance = {distance:.2f} meters"
    )

    return distance <= 25