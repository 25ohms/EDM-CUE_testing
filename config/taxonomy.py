"""Shared taxonomy and visualization defaults for EDM-CUE charts."""

GENRE_FONT_STACK = [
    "Arial Unicode MS",
    "DejaVu Sans",
    "Helvetica",
    "Arial",
]

GENRE_DELIMITER_PATTERN = r"[\\/]+"

GENRE_SCORING = {
    # Base score applied when any keyword matches a genre token.
    "keyword_match": 10,
    # Additional bonus when the BPM is inside the taxonomy entry's range.
    "bpm_match_bonus": 5,
    # Lower priority wins when scores tie. This default is used if an entry omits priority.
    "default_priority": 1000,
}

GENRE_TAXONOMY = [
    {
        "label": "Drum & Bass / Jungle",
        "keywords": (
            "drum and bass",
            "drum & bass",
            "drum n bass",
            "dnb",
            "jungle",
            "neurofunk",
        ),
        "bpm_range": (160, None),
        "priority": 10,
    },
    {
        "label": "Dubstep / Bass",
        "keywords": (
            "dubstep",
            "riddim",
            "brostep",
            "future bass",
            "bass music",
        ),
        "bpm_range": (150, 160),
        "priority": 20,
    },
    {
        "label": "Trap / Hip-Hop",
        "keywords": (
            "trap",
            "hip hop",
            "hip-hop",
        ),
        "bpm_range": (130, 160),
        "priority": 30,
    },
    {
        "label": "House / Deep House",
        "keywords": (
            "deep house",
            "tech house",
            "progressive house",
            "future house",
            "house",
        ),
        "bpm_range": (112, 125),
        "priority": 40,
    },
    {
        "label": "Electro / Dance",
        "keywords": (
            "electro",
            "dance",
            "electronic",
            "edm",
            "club",
        ),
        "bpm_range": (120, 135),
        "priority": 50,
    },
    {
        "label": "Techno",
        "keywords": (
            "techno",
            "minimal",
            "acid techno",
        ),
        "bpm_range": (125, 140),
        "priority": 60,
    },
    {
        "label": "Trance / Progressive",
        "keywords": (
            "trance",
            "psytrance",
            "uplifting",
            "progressive",
        ),
        "bpm_range": (138, 150),
        "priority": 70,
    },
    {
        "label": "Hardstyle / Hardcore",
        "keywords": (
            "hardstyle",
            "hard dance",
            "hardcore",
            "gabber",
        ),
        "bpm_range": (145, None),
        "priority": 80,
    },
    {
        "label": "Chill / Downtempo",
        "keywords": (
            "ambient",
            "downtempo",
            "chill",
            "chillout",
            "lofi",
        ),
        "bpm_range": (0, 112),
        "priority": 90,
    },
    {
        "label": "Pop / Mainstream",
        "keywords": (
            "pop",
            "synthpop",
            "indie pop",
        ),
        "bpm_range": (90, 130),
        "priority": 100,
    },
]

BPM_BUCKETS = [
    ("Chill / Downtempo", None, 112),
    ("House / Deep House", 112, 125),
    ("Electro / Dance", 125, 133),
    ("Techno", 133, 138),
    ("Trance / Progressive", 138, 150),
    ("Dubstep / Bass", 150, 160),
    ("Drum & Bass / Jungle", 160, None),
]
