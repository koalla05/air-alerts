"""alerts.in.ua region identifiers.

These map our internal `region_id` (see config.REGION_MAPPING) to the
identifiers used by the alerts.in.ua API, which are independent of both
our IDs and the GeoJSON ISO codes in `live.ISO_MAPPING`.

Source: https://devs.alerts.in.ua/ (oblast UID table and the fixed
27-character oblast ordering used by the `active_air_raid_alerts_by_oblast`
endpoint). The by-oblast order is NOT the same as the UID order.
"""
from typing import Dict, List

# region_id -> alerts.in.ua region uid, used for
# /v1/regions/{uid}/alerts/{period}.json and /v1/iot/active_air_raid_alerts/{uid}.json
OBLAST_UID: Dict[str, int] = {
    "khmelnytskyi": 3,
    "vinnytsia": 4,
    "rivne": 5,
    "volyn": 8,
    "dnipropetrovsk": 9,
    "zhytomyr": 10,
    "zakarpattia": 11,
    "zaporizhzhia": 12,
    "ivano_frankivsk": 13,
    "kyiv": 14,
    "kirovohrad": 15,
    "luhansk": 16,
    "mykolaiv": 17,
    "odesa": 18,
    "poltava": 19,
    "sumy": 20,
    "ternopil": 21,
    "kharkiv": 22,
    "kherson": 23,
    "cherkasy": 24,
    "chernihiv": 25,
    "chernivtsi": 26,
    "lviv": 27,
    "donetsk": 28,
    "crimea": 29,
    "sevastopol": 30,
    "kyiv_city": 31,
}

UID_TO_REGION: Dict[int, str] = {uid: reg for reg, uid in OBLAST_UID.items()}

# Fixed position order (1-indexed in the docs, 0-indexed here) of the 27
# characters returned by /v1/iot/active_air_raid_alerts_by_oblast.json.
# This is NOT sorted by uid.
OBLAST_STATUS_ORDER: List[str] = [
    "crimea",
    "volyn",
    "vinnytsia",
    "dnipropetrovsk",
    "donetsk",
    "zhytomyr",
    "zakarpattia",
    "zaporizhzhia",
    "ivano_frankivsk",
    "kyiv_city",
    "kyiv",
    "kirovohrad",
    "luhansk",
    "lviv",
    "mykolaiv",
    "odesa",
    "poltava",
    "rivne",
    "sevastopol",
    "sumy",
    "ternopil",
    "kharkiv",
    "kherson",
    "khmelnytskyi",
    "cherkasy",
    "chernivtsi",
    "chernihiv",
]
