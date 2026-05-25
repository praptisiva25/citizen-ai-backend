from math import radians
from math import sin
from math import cos
from math import sqrt
from math import atan2

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

    # CATEGORY CHECK

    if category1 != category2:
        return False

    # DISTANCE CHECK

    distance = haversine_distance(
        lat1,
        lon1,
        lat2,
        lon2
    )

    if distance <= 5:
        return True

    return False