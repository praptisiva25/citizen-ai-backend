import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from PIL import Image

import tensorflow as tf
import numpy as np

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity

from math import radians
from math import sin
from math import cos
from math import sqrt
from math import atan2

import ollama

# -----------------------------------
# CNN MODEL
# -----------------------------------

tf.get_logger().setLevel(
    "ERROR"
)

cnn_model = tf.keras.models.load_model(
    "efficientnet_model2.keras",
    compile=False
)

CLASS_NAMES = [

    "garbage",

    "overflow",

    "pothole"
]

# -----------------------------------
# IMAGE CLASSIFICATION
# -----------------------------------

def classify_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    img_array = np.array(
        image
    ) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = cnn_model.predict(
        img_array,
        verbose=0
    )

    class_index = np.argmax(
        prediction
    )

    return CLASS_NAMES[
        class_index
    ]

# -----------------------------------
# SLM
# -----------------------------------

def extract_intent(
    title,
    description
):

    prompt = f"""
    Classify the civic complaint.

    Return ONLY one word.

    Possible classes:

    pothole
    garbage
    overflow

    Title:
    {title}

    Description:
    {description}
    """

    response = ollama.chat(

        model="qwen2.5:1.5b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    output = response[
        "message"
    ]["content"].lower().strip()

    if "pothole" in output:
        return "pothole"

    if "garbage" in output:
        return "garbage"

    if "overflow" in output:
        return "overflow"

    return "unknown"

# -----------------------------------
# EMBEDDINGS
# -----------------------------------

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

departments = {

    "PWD": [
        "pothole",
        "road damage",
        "broken road"
    ],

    "Sanitation": [
        "garbage",
        "overflow",
        "waste"
    ]
}

def detect_department(text):

    text_embedding = embedder.encode(
        text
    )

    best_department = None

    best_score = -1

    for dept, keywords in departments.items():

        keyword_text = " ".join(
            keywords
        )

        keyword_embedding = embedder.encode(
            keyword_text
        )

        similarity = cosine_similarity(

            [text_embedding],

            [keyword_embedding]

        )[0][0]

        if similarity > best_score:

            best_score = similarity

            best_department = dept

    return best_department

# -----------------------------------
# DISTANCE
# -----------------------------------

def haversine_distance(

    lat1,
    lon1,

    lat2,
    lon2
):

    R = 6371000

    dlat = radians(
        lat2 - lat1
    )

    dlon = radians(
        lon2 - lon1
    )

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

# -----------------------------------
# CLUSTER MATCH
# -----------------------------------

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

    return distance <= 5