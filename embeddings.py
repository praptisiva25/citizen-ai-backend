from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
        "trash",
        "waste"
    ],

    "Water Department": [
        "pipe leak",
        "water leakage"
    ]
}

def detect_department(text):

    text_embedding = embedder.encode(
        text
    )

    best_department = None

    best_score = -1

    for dept, keywords in departments.items():

        keyword_sentence = " ".join(
            keywords
        )

        keyword_embedding = embedder.encode(
            keyword_sentence
        )

        similarity = cosine_similarity(
            [text_embedding],
            [keyword_embedding]
        )[0][0]

        if similarity > best_score:

            best_score = similarity

            best_department = dept

    return best_department