import math

def is_inside_office(user_lat, user_lng, office):
    """
    Returns True if user is inside office GPS radius
    """
    R = 6371000  # Earth radius (meters)

    lat1 = math.radians(float(user_lat))
    lon1 = math.radians(float(user_lng))
    lat2 = math.radians(float(office.latitude))
    lon2 = math.radians(float(office.longitude))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    distance = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return distance <= office.radius_meters