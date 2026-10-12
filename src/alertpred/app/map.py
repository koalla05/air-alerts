from typing import Dict, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from ..config import REGION_MAPPING
from ..ingestion.live import ISO_MAPPING
from ..geo.oblasts import load_ukraine_geojson
from ..geo.raions import RAION_REGISTRY


def create_ukraine_map(
    live_status: Dict[str, Any],
    selected_region_id: Optional[str] = None,
) -> go.Figure:
    geojson = load_ukraine_geojson()
    # Ensure each feature has a unique 'id' matching its ISO code for Plotly mapping
    for feature in geojson.get("features", []):
        iso_code = feature.get("properties", {}).get("shapeISO")
        if iso_code:
            feature["id"] = iso_code

    records = []
    for reg_id, iso in ISO_MAPPING.items():
        st = live_status["regions"].get(reg_id, {"is_active": False, "elapsed_min": 0, "elapsed_quiet_min": 0})
        info = next((v for v in REGION_MAPPING.values() if v["id"] == reg_id), {"en": reg_id})

        is_active = st["is_active"]
        elapsed = st["elapsed_min"] if is_active else st["elapsed_quiet_min"]

        if is_active:
            status_text = f"🚨 ALERT ({int(elapsed)}m ago)"
            color_group = "Active Alert"
        else:
            status_text = f"✅ Quiet ({int(elapsed / 60)}h ago)"
            color_group = "No Alert"

        records.append({
            "region_id": reg_id,
            "shapeISO": iso,
            "region_name": info["en"],
            "status_text": status_text,
            "color_group": color_group,
            "is_selected": (reg_id == selected_region_id),
        })

    df = pd.DataFrame(records)

    fig = px.choropleth(
        df,
        geojson=geojson,
        locations="shapeISO",
        featureidkey="id",
        color="color_group",
        color_discrete_map={
            "Active Alert": "#ef4444",   # Vivid Alert Red
            "No Alert": "#1e293b",       # Deep Slate Blue/Dark Gray
        },
        hover_name="region_name",
        hover_data={
            "shapeISO": False,
            "color_group": False,
            "status_text": True,
        },
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="#0e1117",
    )

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=640,
        margin={"r": 0, "t": 10, "l": 0, "b": 10},
        dragmode=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#e2e8f0", size=13),
            title=None,
        ),
    )

    fig.update_traces(
        marker_line_width=1.2,
        marker_line_color="#475569",
    )

    return fig


def create_raion_map(
    live_status: Dict[str, Any],
    selected_raion_id: Optional[str] = None,
) -> go.Figure:
    """
    Choropleth at raion granularity (true alert locations - see data/geo/SOURCES.md
    and the EDA notebook on why oblast-level alerts are now rare), with the 27
    oblast boundaries drawn as a thicker, unfilled outline layer on top so the
    map still reads as "Ukraine by oblast" at a glance.
    """
    raion_geojson = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": r["raion_id"], "properties": {}, "geometry": r["geometry"]}
            for r in RAION_REGISTRY
        ],
    }

    records = []
    for r in RAION_REGISTRY:
        raion_id = r["raion_id"]
        st = live_status.get("raions", {}).get(raion_id, {"is_active": False, "elapsed_min": 0, "elapsed_quiet_min": 0})
        is_active = st["is_active"]
        elapsed = st["elapsed_min"] if is_active else st["elapsed_quiet_min"]

        if is_active:
            status_text = f"🚨 ALERT ({int(elapsed)}m ago)"
            color_group = "Active Alert"
        else:
            status_text = f"✅ Quiet ({int(elapsed / 60)}h ago)"
            color_group = "No Alert"

        records.append({
            "raion_id": raion_id,
            "raion_name": r["name_en"],
            "oblast_name": r["oblast_en"],
            "status_text": status_text,
            "color_group": color_group,
            "is_selected": (raion_id == selected_raion_id),
        })

    df = pd.DataFrame(records)

    fig = px.choropleth(
        df,
        geojson=raion_geojson,
        locations="raion_id",
        featureidkey="id",
        color="color_group",
        color_discrete_map={
            "Active Alert": "#ef4444",
            "No Alert": "#1e293b",
        },
        hover_name="raion_name",
        hover_data={
            "raion_id": False,
            "color_group": False,
            "oblast_name": True,
            "status_text": True,
        },
    )
    fig.update_traces(marker_line_width=0.4, marker_line_color="#334155")

    # Oblast outlines on top: unfilled, just a thicker border, for visual grouping.
    oblast_geojson = load_ukraine_geojson()
    for feature in oblast_geojson["features"]:
        feature["id"] = feature["properties"].get("shapeISO")
    oblast_ids = [f["id"] for f in oblast_geojson["features"]]

    fig.add_trace(
        go.Choropleth(
            geojson=oblast_geojson,
            locations=oblast_ids,
            featureidkey="id",
            z=[0] * len(oblast_ids),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
            showscale=False,
            marker_line_width=1.4,
            marker_line_color="#94a3b8",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="#0e1117",
    )

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        height=640,
        margin={"r": 0, "t": 10, "l": 0, "b": 10},
        dragmode=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#e2e8f0", size=13),
            title=None,
        ),
    )

    return fig
