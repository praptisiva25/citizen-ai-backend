from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import Form

from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

import os
import uuid
import shutil

import uvicorn

load_dotenv(
    ".env.local"
)

from database import SessionLocal
from database import engine
from database import Base

from models import Complaint
from models import Cluster

from utils1 import classify_image
from utils1 import extract_intent
from utils1 import detect_department
from utils1 import haversine_distance


Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


@app.post("/submit")
async def submit_complaint(

    title: str = Form(...),

    description: str = Form(...),

    latitude: float = Form(...),

    longitude: float = Form(...),

    image: UploadFile = File(...)

):

    db = SessionLocal()

    try:

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

        image_category = classify_image(
            image_path
        )

        text_category = extract_intent(
            title,
            description
        )

        if image_category != text_category:

            return {
                "status": "rejected",
                "reason": "Image and text mismatch",
                "image_prediction": image_category,
                "text_prediction": text_category
            }

        department = detect_department(
            image_category
        )

        all_clusters = db.query(
            Cluster
        ).all()

        assigned_cluster = None

        nearest_distance = float("inf")

        for cluster in all_clusters:

            complaints = db.query(
                Complaint
            ).filter(
                Complaint.cluster_id == cluster.id
            ).all()

            if not complaints:
                continue

            if complaints[0].category != image_category:
                continue

            avg_lat = sum(
                c.latitude
                for c in complaints
            ) / len(complaints)

            avg_lon = sum(
                c.longitude
                for c in complaints
            ) / len(complaints)

            distance = haversine_distance(

                latitude,
                longitude,

                avg_lat,
                avg_lon
            )

            print(
                f"Cluster {cluster.id} centroid distance = {distance:.2f} m"
            )

            if distance <= 25 and distance < nearest_distance:

                nearest_distance = distance

                assigned_cluster = cluster

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

    finally:

        db.close()


@app.get("/clusters")
def get_clusters():

    db = SessionLocal()

    try:

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

    finally:

        db.close()


if __name__ == "__main__":

    uvicorn.run(

        app,

        host="0.0.0.0",

        port=8000
    )