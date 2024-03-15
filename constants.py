coco_categories = {
    "chair": 0,
    "couch": 1,
    "potted plant": 2,
    "bed": 3,
    "toilet": 4,
    "tv": 5,
    "dining-table": 6,
    "oven": 7,
    "sink": 8,
    "refrigerator": 9,
    "book": 10,
    "clock": 11,
    "vase": 12,
    "cup": 13,
    "bottle": 14
}

category_to_id = [
        "chair",
        "bed",
        "plant",
        "toilet",
        "tv_monitor",
        "sofa"
] # 6 classes

category_to_id_gibson = [
        "chair",
        "couch",
        "potted plant",
        "bed",
        "toilet",
        "tv"
]

mp3d_category_id = {
    'void': 1,
    'chair': 2,
    'sofa': 3,
    'plant': 4,
    'bed': 5,
    'toilet': 6,
    'tv_monitor': 7,
    'table': 8,
    'refrigerator': 9,
    'sink': 10,
    'stairs': 16,
    'fireplace': 12
}

# mp_categories_mapping = [4, 11, 15, 12, 19, 23, 6, 7, 15, 38, 40, 28, 29, 8, 17]

mp_categories_mapping = [4, 11, 15, 12, 19, 23, 26, 24, 28, 38, 21, 16, 14, 6, 16]

hm3d_category = [
        "chair",
        "sofa",
        "plant",
        "bed",
        "toilet",
        "tv_monitor",
        "bathtub",
        "shower",
        "fireplace",
        "appliances",
        "towel",
        "sink",
        "chest_of_drawers",
        "table",
        "stairs"
] # DO NOT USE ANYMORE

object_category = [
    # --- goal categories (DO NOT CHANGE)
    "chair",
    "bed",
    "plant",
    "toilet",
    "tv",
    "couch",
    # --- add as many categories in between as you wish, which are used for LLM reasoning
    # "table",
    "desk",
    "refrigerator",
    "sink",
    "bathtub",
    "shower",
    "towel",
    "painting",
    "trashcan",
    # --- stairs must be put here (DO NOT CHANGE)
    "stairs",
    # --- void category (DO NOT CHANGE)
    "void"
]

coco_categories_mapping = {
    56: 0,  # chair
    57: 1,  # couch
    58: 2,  # potted plant
    59: 3,  # bed
    61: 4,  # toilet
    62: 5,  # tv
    60: 6,  # dining-table
    69: 7,  # oven
    71: 8,  # sink
    72: 9,  # refrigerator
    73: 10,  # book
    74: 11,  # clock
    75: 12,  # vase
    41: 13,  # cup
    39: 14,  # bottle
}

def generate_gradient(color1, color2, steps):
    gradient_colors = []
    for i in range(1, steps + 1):
        r = color1[0] * (1 - i/steps) + color2[0] * (i/steps)
        g = color1[1] * (1 - i/steps) + color2[1] * (i/steps)
        b = color1[2] * (1 - i/steps) + color2[2] * (i/steps)
        gradient_colors.append(r)
        gradient_colors.append(g)
        gradient_colors.append(b)
    return gradient_colors

color_palette = [
    1.0, 1.0, 1.0,
    0.6, 0.6, 0.6,
    0.95, 0.95, 0.95,
    0.96, 0.36, 0.26,
    0.12156862745098039, 0.47058823529411764, 0.7058823529411765,
    0.9400000000000001, 0.7818, 0.66,
    0.9400000000000001, 0.8868, 0.66,
    0.8882000000000001, 0.9400000000000001, 0.66,
    0.7832000000000001, 0.9400000000000001, 0.66,
    0.6782000000000001, 0.9400000000000001, 0.66,
    0.66, 0.9400000000000001, 0.7468000000000001,
    0.66, 0.9400000000000001, 0.8518000000000001,
    0.66, 0.9232, 0.9400000000000001,
    0.66, 0.8182, 0.9400000000000001,
    0.66, 0.7132, 0.9400000000000001,
    0.7117999999999999, 0.66, 0.9400000000000001,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999,
    # NEW COLORS
    0.66, 0.9400000000000001, 0.9531999999999998,
    0.7600000000000001, 0.66, 0.9400000000000001,
    0.9400000000000001, 0.66, 0.9531999999999998,
    0.9400000000000001, 0.66, 0.748199999999999,
    0.66, 0.9400000000000001, 0.8531999999999998,
    0.9400000000000001, 0.66, 0.9581999999999998,
    0.8882000000000001, 0.9400000000000001, 0.9531999999999998,
    0.7832000000000001, 0.9400000000000001, 0.9581999999999998,
    0.6782000000000001, 0.9400000000000001, 0.9531999999999998,
    0.66, 0.9400000000000001, 0.7618000000000001,
    0.66, 0.9400000000000001, 0.9661999999999998,
    0.8168, 0.66, 0.9400000000000001,
    0.9218, 0.66, 0.9661999999999998,
    0.9400000000000001, 0.66, 0.7668000000000001,
    0.66, 0.9661999999999998, 0.9400000000000001,
    0.7832000000000001, 0.9661999999999998, 0.66,
    0.9400000000000001, 0.8531999999999998, 0.66,
    0.66, 0.9661999999999998, 0.9681999999999998,
    0.8882000000000001, 0.66, 0.9661999999999998,
    0.66, 0.7657999999999998, 0.9661999999999998,
]

