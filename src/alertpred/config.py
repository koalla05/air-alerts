from pathlib import Path
from typing import Dict, List

# Base Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "official_data_uk.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Prediction horizons
TASK_A_HORIZONS_MIN = [15, 30, 60]      # Probabilities that active alert ends within X min
TASK_B_HORIZONS_MIN = [60, 180, 360]   # Probabilities that next alert starts within 1h, 3h, 6h

# Target quantiles for uncertainty intervals
QUANTILES = [0.10, 0.50, 0.90]

# Standard sampling grid for elapsed time during alerts (minutes)
ELAPSED_SAMPLE_STEPS = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]

# Mapping Ukrainian Oblast names to clean standardized English & canonical IDs
REGION_MAPPING: Dict[str, Dict[str, str]] = {
    "м. Київ": {"en": "Kyiv City", "id": "kyiv_city"},
    "Київська область": {"en": "Kyiv Oblast", "id": "kyiv"},
    "Харківська область": {"en": "Kharkiv Oblast", "id": "kharkiv"},
    "Дніпропетровська область": {"en": "Dnipropetrovsk Oblast", "id": "dnipropetrovsk"},
    "Донецька область": {"en": "Donetsk Oblast", "id": "donetsk"},
    "Запорізька область": {"en": "Zaporizhzhia Oblast", "id": "zaporizhzhia"},
    "Миколаївська область": {"en": "Mykolaiv Oblast", "id": "mykolaiv"},
    "Одеська область": {"en": "Odesa Oblast", "id": "odesa"},
    "Полтавська область": {"en": "Poltava Oblast", "id": "poltava"},
    "Сумська область": {"en": "Sumy Oblast", "id": "sumy"},
    "Чернігівська область": {"en": "Chernihiv Oblast", "id": "chernihiv"},
    "Черкаська область": {"en": "Cherkasy Oblast", "id": "cherkasy"},
    "Житомирська область": {"en": "Zhytomyr Oblast", "id": "zhytomyr"},
    "Вінницька область": {"en": "Vinnytsia Oblast", "id": "vinnytsia"},
    "Кіровоградська область": {"en": "Kirovohrad Oblast", "id": "kirovohrad"},
    "Хмельницька область": {"en": "Khmelnytskyi Oblast", "id": "khmelnytskyi"},
    "Рівненська область": {"en": "Rivne Oblast", "id": "rivne"},
    "Волинська область": {"en": "Volyn Oblast", "id": "volyn"},
    "Львівська область": {"en": "Lviv Oblast", "id": "lviv"},
    "Тернопільська область": {"en": "Ternopil Oblast", "id": "ternopil"},
    "Івано-Франківська область": {"en": "Ivano-Frankivsk Oblast", "id": "ivano_frankivsk"},
    "Чернівецька область": {"en": "Chernivtsi Oblast", "id": "chernivtsi"},
    "Закарпатська область": {"en": "Zakarpattia Oblast", "id": "zakarpattia"},
    "Херсонська область": {"en": "Kherson Oblast", "id": "kherson"},
    "Луганська область": {"en": "Luhansk Oblast", "id": "luhansk"},
    "Автономна Республіка Крим": {"en": "Autonomous Republic of Crimea", "id": "crimea"},
    "м. Севастополь": {"en": "Sevastopol", "id": "sevastopol"},
}

# Adjacency graph for Ukrainian Oblasts (Neighboring regions)
OBLAST_NEIGHBORS: Dict[str, List[str]] = {
    "kyiv_city": ["kyiv"],
    "kyiv": ["kyiv_city", "chernihiv", "poltava", "cherkasy", "vinnytsia", "zhytomyr"],
    "chernihiv": ["kyiv", "poltava", "sumy"],
    "sumy": ["chernihiv", "poltava", "kharkiv"],
    "kharkiv": ["sumy", "poltava", "dnipropetrovsk", "donetsk", "luhansk"],
    "luhansk": ["kharkiv", "donetsk"],
    "donetsk": ["luhansk", "kharkiv", "dnipropetrovsk", "zaporizhzhia"],
    "zaporizhzhia": ["donetsk", "dnipropetrovsk", "kherson"],
    "dnipropetrovsk": ["poltava", "kharkiv", "donetsk", "zaporizhzhia", "kherson", "mykolaiv", "kirovohrad"],
    "poltava": ["chernihiv", "sumy", "kharkiv", "dnipropetrovsk", "kirovohrad", "cherkasy", "kyiv"],
    "cherkasy": ["kyiv", "poltava", "kirovohrad", "vinnytsia"],
    "kirovohrad": ["cherkasy", "poltava", "dnipropetrovsk", "mykolaiv", "odesa", "vinnytsia"],
    "mykolaiv": ["kirovohrad", "dnipropetrovsk", "kherson", "odesa"],
    "kherson": ["mykolaiv", "dnipropetrovsk", "zaporizhzhia", "crimea"],
    "odesa": ["vinnytsia", "kirovohrad", "mykolaiv"],
    "vinnytsia": ["zhytomyr", "kyiv", "cherkasy", "kirovohrad", "odesa", "chernivtsi", "khmelnytskyi"],
    "zhytomyr": ["kyiv", "vinnytsia", "khmelnytskyi", "rivne"],
    "khmelnytskyi": ["rivne", "zhytomyr", "vinnytsia", "chernivtsi", "ternopil"],
    "rivne": ["volyn", "zhytomyr", "khmelnytskyi", "ternopil"],
    "volyn": ["rivne", "lviv"],
    "ternopil": ["rivne", "khmelnytskyi", "chernivtsi", "ivano_frankivsk", "lviv"],
    "chernivtsi": ["ivano_frankivsk", "ternopil", "khmelnytskyi", "vinnytsia"],
    "ivano_frankivsk": ["lviv", "ternopil", "chernivtsi", "zakarpattia"],
    "lviv": ["volyn", "rivne", "ternopil", "ivano_frankivsk", "zakarpattia"],
    "zakarpattia": ["lviv", "ivano_frankivsk"],
    "crimea": ["kherson", "sevastopol"],
    "sevastopol": ["crimea"],
}

