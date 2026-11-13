import math

def distance_to_office_m(user_lat, user_lng, office):
    R = 6371000  # Earth radius in meters

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
    a = min(1.0, max(0.0, a))
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def is_inside_office(user_lat, user_lng, office):
    """Return whether a GPS coordinate is within an office's configured radius."""
    distance = distance_to_office_m(user_lat, user_lng, office)

    return distance <= office.radius_meters
