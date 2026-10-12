"""Raion-level geography: loads `data/geo/ukraine_raions.geojson` (136 post-2020
raions, CC BY 4.0, see `data/geo/SOURCES.md`) and builds a stable registry that
assigns each raion to its parent oblast via a point-in-polygon spatial join
against the (already correctly-wound) oblast geojson - no third-party oblast
attribution data is used for this, just our own geometry.

Kyiv city and Sevastopol have no raion-level subdivision in the source data
(matching reality: alerts.in.ua issues their alerts at the whole-city level),
so they are included in the registry as single "pseudo-raion" entries equal to
their own oblast-level polygon.
"""
import json
from pathlib import Path
from typing import Any, Dict, List

from matplotlib.path import Path as MplPath

from ..config import DATA_DIR, ISO_MAPPING, REGION_MAPPING
from .oblasts import load_ukraine_geojson

RAION_GEOJSON_PATH = DATA_DIR / "geo" / "ukraine_raions.geojson"

# alerts.in.ua's current `location_title` -> this geojson's (pre-2023/2024
# decommunization rename) `rayon` name. Found by diffing raion names observed
# in live API data against this file; see data/geo/SOURCES.md.
RAION_NAME_ALIASES: Dict[str, str] = {
    "Звягельський район": "Новоград-Волинський район",     # Zhytomyr oblast, renamed 2023
    "Володимирський район": "Володимир-Волинський район",   # Volyn oblast
    "Берестинський район": "Красноградський район",         # Kharkiv oblast, renamed Sep 2024
    "Самарівський район": "Новомосковський район",          # Dnipropetrovsk oblast, renamed Sep 2024
    "Шептицький район": "Червоноградський район",           # Lviv oblast, renamed Sep 2024
}

_OLD_TO_CURRENT_NAME: Dict[str, str] = {old: current for current, old in RAION_NAME_ALIASES.items()}

_ISO_TO_REGION = {iso: region_id for region_id, iso in ISO_MAPPING.items()}
_REGION_ID_TO_EN = {meta["id"]: meta["en"] for meta in REGION_MAPPING.values()}

# Ukrainian national transliteration system (simplified, word-initial rules
# applied per hyphen-separated segment). Best-effort only - not hand-curated
# like config.REGION_MAPPING's oblast English names.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e", "є": "ie",
    "ж": "zh", "з": "z", "и": "y", "і": "i", "ї": "i", "й": "i", "к": "k", "л": "l",
    "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "shch", "ю": "iu",
    "я": "ia", "ь": "", "'": "", "’": "",
}
_TRANSLIT_INITIAL = {**_TRANSLIT, "є": "ye", "ї": "yi", "й": "y", "ю": "yu", "я": "ya"}


def transliterate(text: str) -> str:
    words = text.split(" ")
    out_words = []
    for word in words:
        parts = word.split("-")
        out_parts = []
        for part in parts:
            chars = []
            for i, ch in enumerate(part):
                table = _TRANSLIT_INITIAL if i == 0 else _TRANSLIT
                sub = table.get(ch.lower(), ch.lower())
                if ch.isupper() and sub:
                    sub = sub[0].upper() + sub[1:]
                chars.append(sub)
            out_parts.append("".join(chars))
        out_words.append("-".join(out_parts))
    return " ".join(out_words)


def raion_name_en(name_uk: str) -> str:
    base = name_uk.replace(" район", "").strip()
    return f"{transliterate(base)} Raion"


def _polygon_centroid(ring: List[List[float]]) -> tuple:
    """Area-weighted centroid of a simple (non-self-intersecting) ring."""
    n = len(ring)
    area = cx = cy = 0.0
    for i in range(n):
        x0, y0 = ring[i][0], ring[i][1]
        x1, y1 = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    area *= 0.5
    if abs(area) < 1e-12:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (sum(xs) / n, sum(ys) / n)
    return (cx / (6 * area), cy / (6 * area))


def _largest_ring(geometry: Dict[str, Any]) -> List[List[float]]:
    """Exterior ring of the largest-area part, for Polygon or MultiPolygon."""
    rings = (
        [geometry["coordinates"][0]]
        if geometry["type"] == "Polygon"
        else [part[0] for part in geometry["coordinates"]]
    )

    def ring_area(r):
        n = len(r)
        return abs(sum(r[i][0] * r[(i + 1) % n][1] - r[(i + 1) % n][0] * r[i][1] for i in range(n)))

    return max(rings, key=ring_area)


def load_raion_geojson() -> Dict[str, Any]:
    with open(RAION_GEOJSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_oblast_paths() -> List[tuple]:
    """One (matplotlib Path, region_id) per ring/part of every oblast polygon."""
    oblast_geojson = load_ukraine_geojson()  # rewound to the correct winding order
    paths = []
    for feat in oblast_geojson["features"]:
        iso = feat["properties"].get("shapeISO")
        region_id = _ISO_TO_REGION.get(iso)
        if region_id is None:
            continue
        geom = feat["geometry"]
        rings = (
            [geom["coordinates"][0]]
            if geom["type"] == "Polygon"
            else [part[0] for part in geom["coordinates"]]
        )
        for ring in rings:
            paths.append((MplPath(ring), region_id))
    return paths


def _assign_oblast(centroid: tuple, oblast_paths: List[tuple]) -> Any:
    for path, region_id in oblast_paths:
        if path.contains_point(centroid):
            return region_id
    return None


def _build_registry() -> List[Dict[str, Any]]:
    raion_geojson = load_raion_geojson()
    oblast_geojson = load_ukraine_geojson()
    oblast_paths = _build_oblast_paths()

    slug_counts: Dict[str, int] = {}
    registry = []

    for feat in raion_geojson["features"]:
        # Display the current (post-decommunization-rename) name where it
        # differs from this geojson's name, so the UI matches what
        # alerts.in.ua and the public currently call the raion.
        raw_name_uk = feat["properties"]["rayon"]
        name_uk = _OLD_TO_CURRENT_NAME.get(raw_name_uk, raw_name_uk)
        ring = _largest_ring(feat["geometry"])
        centroid = _polygon_centroid(ring)
        oblast_id = _assign_oblast(centroid, oblast_paths)
        if oblast_id is None:
            raise ValueError(f"Could not assign raion '{name_uk}' to any oblast (centroid {centroid})")

        base_slug = transliterate(name_uk.replace(" район", "")).lower().replace(" ", "_").replace("-", "_")
        slug = f"{oblast_id}__{base_slug}"
        slug_counts[slug] = slug_counts.get(slug, 0) + 1
        if slug_counts[slug] > 1:
            slug = f"{slug}_{slug_counts[slug]}"

        registry.append({
            "raion_id": slug,
            "name_uk": name_uk,
            "name_en": raion_name_en(name_uk),
            "oblast_id": oblast_id,
            "oblast_en": _REGION_ID_TO_EN.get(oblast_id, oblast_id),
            "is_pseudo_oblast": False,
            "geometry": feat["geometry"],
        })

    # Kyiv city and Sevastopol have no raion split in the source data; keep
    # them as single whole-region "pseudo-raion" entries using their own
    # oblast polygon, so the raion map still covers the whole country.
    for feat in oblast_geojson["features"]:
        iso = feat["properties"].get("shapeISO")
        region_id = _ISO_TO_REGION.get(iso)
        if region_id not in ("kyiv_city", "sevastopol"):
            continue
        registry.append({
            "raion_id": region_id,
            "name_uk": next(k for k, v in REGION_MAPPING.items() if v["id"] == region_id),
            "name_en": _REGION_ID_TO_EN.get(region_id, region_id),
            "oblast_id": region_id,
            "oblast_en": _REGION_ID_TO_EN.get(region_id, region_id),
            "is_pseudo_oblast": True,
            "geometry": feat["geometry"],
        })

    return registry


RAION_REGISTRY: List[Dict[str, Any]] = _build_registry()
RAION_BY_ID: Dict[str, Dict[str, Any]] = {r["raion_id"]: r for r in RAION_REGISTRY}

# alerts.in.ua `location_title` -> our raion_id. registry name_uk is already
# the current name, so this covers current names directly; also map the old
# (pre-rename) names from RAION_NAME_ALIASES for robustness.
API_NAME_TO_RAION_ID: Dict[str, str] = {}
for _r in RAION_REGISTRY:
    API_NAME_TO_RAION_ID[_r["name_uk"]] = _r["raion_id"]
for _current_name, _old_name in RAION_NAME_ALIASES.items():
    _target = API_NAME_TO_RAION_ID.get(_current_name)
    if _target:
        API_NAME_TO_RAION_ID[_old_name] = _target

RAIONS_BY_OBLAST: Dict[str, List[str]] = {}
for _r in RAION_REGISTRY:
    RAIONS_BY_OBLAST.setdefault(_r["oblast_id"], []).append(_r["raion_id"])
