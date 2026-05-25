from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import Form

from fastapi.middleware.cors import CORSMiddleware

import os
import uuid
import shutil

from sqlalchemy.orm import Session

from database import SessionLocal
from database import engine
from database import Base

from models import Complaint
from models import Cluster

from classifier import classify_image

from llm import extract_intent

from embeddings import detect_department

from clustering import should_cluster

# -----------------------------------
# CREATE TABLES
# -----------------------------------

Base.metadata.create_all(
    bind=engine
)

# -----------------------------------
# FASTAPI
# -----------------------------------

app = FastAPI()

# -----------------------------------
# CORS
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# UPLOADS
# -----------------------------------

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# -----------------------------------
# SUBMIT COMPLAINT
# -----------------------------------

@app.post("/submit")
async def submit_complaint(

    title: str = Form(...),

    description: str = Form(...),

    latitude: float = Form(...),

    longitude: float = Form(...),

    image: UploadFile = File(...)

):

    db = SessionLocal()

    # -----------------------------------
    # SAVE IMAGE
    # -----------------------------------

    extension = image.filename.split(".")[-1]

    filename = f"{uuid.uuid4()}.{extension}"

    image_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    with open(
        image_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            image.file,
            buffer
        )

    # -----------------------------------
    # IMAGE CLASSIFICATION
    # -----------------------------------

    image_category = classify_image(
        image_path
    )

    # -----------------------------------
    # TITLE WEIGHTAGE
    # -----------------------------------

    combined_text = (
        title * 3
        + " "
        + description
    )

    # -----------------------------------
    # LLM INTENT
    # -----------------------------------

    text_category = extract_intent(
        title,
        description
    )

    # -----------------------------------
    # VALIDATION
    # -----------------------------------

    if image_category != text_category:

        return {
            "status": "rejected",
            "reason": "Image and text mismatch",
            "image_prediction": image_category,
            "text_prediction": text_category
        }

    # -----------------------------------
    # DEPARTMENT
    # -----------------------------------

    department = detect_department(
        combined_text
    )

    # -----------------------------------
    # FIND EXISTING CLUSTER
    # -----------------------------------

    all_clusters = db.query(
        Cluster
    ).all()

    assigned_cluster = None

    for cluster in all_clusters:

        first_complaint = db.query(
            Complaint
        ).filter(
            Complaint.cluster_id == cluster.id
        ).first()

        if not first_complaint:
            continue

        match = should_cluster(

            image_category,

            first_complaint.category,

            latitude,
            longitude,

            first_complaint.latitude,
            first_complaint.longitude
        )

        if match:

            assigned_cluster = cluster

            break

    # -----------------------------------
    # CREATE NEW CLUSTER
    # -----------------------------------

    if assigned_cluster is None:

        assigned_cluster = Cluster(

            category=image_category,

            department=department
        )

        db.add(
            assigned_cluster
        )

        db.commit()

        db.refresh(
            assigned_cluster
        )

    # -----------------------------------
    # SAVE COMPLAINT
    # -----------------------------------

    complaint = Complaint(

        title=title,

        description=description,

        intent=text_category,

        category=image_category,

        department=department,

        latitude=latitude,

        longitude=longitude,

        image_path=image_path,

        cluster_id=assigned_cluster.id
    )

    db.add(
        complaint
    )

    db.commit()

    return {

        "status": "success",

        "cluster_id": assigned_cluster.id,

        "department": department
    }

# -----------------------------------
# GET CLUSTERS
# -----------------------------------

@app.get("/clusters")
def get_clusters():

    db = SessionLocal()

    clusters = db.query(
        Cluster
    ).all()

    output = []

    for cluster in clusters:

        complaints = db.query(
            Complaint
        ).filter(
            Complaint.cluster_id == cluster.id
        ).all()

        complaint_data = []

        for c in complaints:

            complaint_data.append({

                "id": c.id,

                "title": c.title,

                "description": c.description,

                "intent": c.intent,

                "category": c.category,

                "department": c.department,

                "latitude": c.latitude,

                "longitude": c.longitude,

                "image_path": c.image_path
            })

        output.append({

            "cluster_id": cluster.id,

            "category": cluster.category,

            "department": cluster.department,

            "status": cluster.status,

            "complaints": complaint_data
        })

    return output

# -----------------------------------
# DELETE CLUSTER
# -----------------------------------

@app.delete("/cluster/{cluster_id}")
def delete_cluster(
    cluster_id: int
):

    db = SessionLocal()

    cluster = db.query(
        Cluster
    ).filter(
        Cluster.id == cluster_id
    ).first()

    db.delete(cluster)

    db.commit()

    return {
        "message": "Cluster deleted"
    }

# -----------------------------------
# DELETE COMPLAINT
# -----------------------------------

@app.delete("/complaint/{complaint_id}")
def delete_complaint(
    complaint_id: int
):

    db = SessionLocal()

    complaint = db.query(
        Complaint
    ).filter(
        Complaint.id == complaint_id
    ).first()

    cluster_id = complaint.cluster_id

    db.delete(
        complaint
    )

    db.commit()

    # DELETE EMPTY CLUSTER

    remaining = db.query(
        Complaint
    ).filter(
        Complaint.cluster_id == cluster_id
    ).count()

    if remaining == 0:

        cluster = db.query(
            Cluster
        ).filter(
            Cluster.id == cluster_id
        ).first()

        db.delete(
            cluster
        )

        db.commit()

    return {
        "message": "Complaint deleted"
    }