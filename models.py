from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Float
from sqlalchemy import ForeignKey

from sqlalchemy.orm import relationship

from database import Base

# -----------------------------------
# CLUSTER TABLE
# -----------------------------------

class Cluster(Base):

    __tablename__ = "clusters"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    category = Column(String)

    department = Column(String)

    status = Column(
        String,
        default="pending"
    )

    complaints = relationship(
        "Complaint",
        back_populates="cluster",
        cascade="all, delete"
    )

# -----------------------------------
# COMPLAINT TABLE
# -----------------------------------

class Complaint(Base):

    __tablename__ = "complaints"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    title = Column(String)

    description = Column(String)

    intent = Column(String)

    category = Column(String)

    department = Column(String)

    latitude = Column(Float)

    longitude = Column(Float)

    image_path = Column(String)

    cluster_id = Column(
        Integer,
        ForeignKey(
            "clusters.id",
            ondelete="CASCADE"
        )
    )

    cluster = relationship(
        "Cluster",
        back_populates="complaints"
    )