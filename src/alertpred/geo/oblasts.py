"""Oblast-level geography: loads `data/geo/ukraine_oblasts.geojson` (27 oblasts,
geoBoundaries, see `data/geo/SOURCES.md`)."""
import json
from typing import Any, Dict, List

from ..config import DATA_DIR

OBLAST_GEOJSON_PATH = DATA_DIR / "geo" / "ukraine_oblasts.geojson"


def _rewind_ring(ring: List[List[float]]) -> List[List[float]]:
    return list(reversed(ring))


def _rewind_geometry(geometry: Dict[str, Any]) -> None:
    # The source file's rings are wound the opposite way from what Plotly's
    # geo renderer (d3-geo) expects, which makes it fill the area *outside*
    # each oblast instead of the oblast itself. Reversing every ring fixes
    # the fill without needing a different geojson source.
    geom_type = geometry.get("type")
    if geom_type == "Polygon":
        geometry["coordinates"] = [_rewind_ring(r) for r in geometry["coordinates"]]
    elif geom_type == "MultiPolygon":
        geometry["coordinates"] = [
            [_rewind_ring(r) for r in polygon] for polygon in geometry["coordinates"]
        ]


def load_ukraine_geojson() -> Dict[str, Any]:
    with open(OBLAST_GEOJSON_PATH, "r", encoding="utf-8") as f:
        geojson = json.load(f)
    for feature in geojson.get("features", []):
        _rewind_geometry(feature["geometry"])
    return geojson
