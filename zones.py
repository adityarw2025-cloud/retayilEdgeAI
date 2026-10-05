# zones.py
"""
Zone definitions and geometric containment calculations.
Defines Store Zone, Shelf Zone, and Checkout Zone for retail analytics.
"""

# =====================================================================
# Configurable Rectangular Zones (Store Area, Shelf Area, Checkout Area)
# Format: "coords": (x1, y1, x2, y2)
# Supports normalized ratios (0.0 to 1.0) or absolute pixel coordinates
# =====================================================================
STORE_AREA_CONFIG = {
    "coords": (0.05, 0.15, 0.48, 0.52),   # Top-left browse / aisle area
    "color": (255, 191, 0),               # Cyan / Sky Blue (BGR)
}
SHELF_AREA_CONFIG = {
    "coords": (0.05, 0.55, 0.48, 0.90),   # Bottom-left retail shelf area
    "color": (0, 220, 100),               # Emerald Green (BGR)
}
CHECKOUT_AREA_CONFIG = {
    "coords": (0.52, 0.15, 0.95, 0.90),   # Right area (Checkout queue)
    "color": (0, 165, 255),               # Bright Orange (BGR)
}

# Configurable zones representing: Store Area, Shelf Area, Checkout Area
# Includes both Area names and legacy Zone aliases
ZONES = {
    "Store Zone": STORE_AREA_CONFIG,
    "Shelf Zone": SHELF_AREA_CONFIG,
    "Checkout Zone": CHECKOUT_AREA_CONFIG,
    "Store Area": STORE_AREA_CONFIG,
    "Shelf Area": SHELF_AREA_CONFIG,
    "Checkout Area": CHECKOUT_AREA_CONFIG,
}

# Display / Primary Zones for UI overlays
PRIMARY_ZONES = ["Store Area", "Shelf Area", "Checkout Area"]


# Primary Zone Identifiers
QUEUE_ZONE_NAME = "Checkout Area"
DWELL_ZONE_NAME = "Store Area"
SHELF_ZONE_NAME = "Shelf Area"

# Backward-compatibility aliases for legacy scripts and test runners
LEGACY_ZONE_ALIASES = {
    "Store Zone": "Store Area",
    "Shelf Zone": "Shelf Area",
    "Checkout Zone": "Checkout Area",
    "Store Area": "Store Area",
    "Shelf Area": "Shelf Area",
    "Checkout Area": "Checkout Area",
}




def get_zone_pixels(coords, width, height):
    """
    Resolves normalized ratios (<= 1.0) or absolute pixel coordinates to integers.
    Returns: (x1, y1, x2, y2) in absolute pixel coordinates.
    """
    x1, y1, x2, y2 = coords
    if x2 <= 1.0 and y2 <= 1.0:
        return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
    return int(x1), int(y1), int(x2), int(y2)


def resolve_all_zones(zones_dict, width, height):
    """
    Resolves pixel coordinates for all zones in the dictionary.
    Returns: dict mapping zone_name -> (x1, y1, x2, y2)
    """
    resolved = {}
    for name, info in zones_dict.items():
        resolved[name] = get_zone_pixels(info["coords"], width, height)
    return resolved


def is_point_in_zone(point, zone_rect):
    """
    Checks if point (px, py) is inside zone_rect (x1, y1, x2, y2).
    """
    px, py = point
    x1, y1, x2, y2 = zone_rect
    return (x1 <= px <= x2) and (y1 <= py <= y2)


def get_zone_for_point(point, resolved_zones):
    """
    Determines which zone contains point (px, py).
    Returns zone_name string if inside a zone, else None.
    """
    for zone_name, rect in resolved_zones.items():
        if is_point_in_zone(point, rect):
            return zone_name
    return None

