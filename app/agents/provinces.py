"""Experimental province-level summaries on the JMA Himawari r2w image grid.

JMA Users' Guide: northwest (90 E, 30 N), southeast (165 E, 15 S),
0.05 degrees per sample. Large r2w images are 1501 x 901 inclusive.
This is approximate screen-space attribution to 38 Indonesian provinces,
NOT rainfall observation or future rainfall forecasting.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .comparison import jma_magenta_mask

ASSETS = Path(__file__).resolve().parents[1] / "assets"
BOUNDARIES = ASSETS / "indonesia_38_provinces.geojson"
LON_W, LAT_N, RES_DEGREES = 90.0, 30.0, 0.05
GRID_SIZE = (1501, 901)
MIN_INDICATED_PIXELS = 3


def longitude_latitude_to_pixel(lon: float, lat: float) -> tuple[float, float]:
    if not (80 <= lon <= 170 and -20 <= lat <= 35):
        raise ValueError("Coordinate out of supported Southeast Asia region")
    return ((lon - LON_W) / RES_DEGREES,
            (LAT_N - lat) / RES_DEGREES)


@lru_cache(maxsize=1)
def _features() -> tuple:
    dataset = json.loads(BOUNDARIES.read_text(encoding="utf-8"))
    features = dataset.get("features")
    if not isinstance(features, list) or len(features) != 38:
        raise ValueError("Province geometry must contain exactly 38 features")
    names = [f.get("properties", {}).get("PROVINSI") for f in features]
    if any(not name for name in names) or len(set(names)) != 38:
        raise ValueError("Province boundary source has missing/duplicate names")
    return tuple(features)


def _pixel_ring(ring: list) -> list:
    if len(ring) < 4:
        return []
    return [(round((float(point[0]) - LON_W) / RES_DEGREES, 2),
             round((LAT_N - float(point[1])) / RES_DEGREES, 2))
            for point in ring]


def _polygons(feature: dict):
    geom = feature.get("geometry") or {}
    geom_type = geom.get("type")
    coords = geom.get("coordinates") or []
    if geom_type == "Polygon":
        return [coords]
    if geom_type == "MultiPolygon":
        return coords
    raise ValueError("Province geometry must be Polygon/MultiPolygon")


def province_mask(feature: dict, size: tuple[int, int] = GRID_SIZE) -> np.ndarray:
    if tuple(size) != GRID_SIZE:
        raise ValueError(f"JMA r2w expected {GRID_SIZE}; got {size}. "
                         "Province assignment is disabled to avoid incorrect coordinates.")
    image = Image.new("L", size, 0)
    draw = ImageDraw.Draw(image)
    for polygon in _polygons(feature):
        if not polygon:
            continue
        outline = _pixel_ring(polygon[0])
        if len(outline) >= 4:
            draw.polygon(outline, fill=255)
        for hole in polygon[1:]:
            ring = _pixel_ring(hole)
            if len(ring) >= 4:
                draw.polygon(ring, fill=0)
    return np.asarray(image) > 0


def _classify(fraction: float, positive_pixels: int) -> str:
    """Descriptive scene-coverage bins, NOT probabilistic weather-risk levels."""
    if positive_pixels < MIN_INDICATED_PIXELS or fraction < 0.5:
        return "Belum terindikasi"
    if fraction >= 10:
        return "Sebaran sangat luas"
    if fraction >= 3:
        return "Sebaran luas"
    return "Sebaran lokal"


def analyze_provinces(predicted_probabilities: np.ndarray, hrp: Image.Image,
                      *, threshold: float = 0.5) -> list:
    if hrp.size != GRID_SIZE or predicted_probabilities.shape != (901, 1501):
        raise ValueError("Province analysis only supports native JMA r2w 1501x901 imagery")
    predicted = predicted_probabilities >= threshold
    reference = jma_magenta_mask(hrp)
    rows = []
    for f in _features():
        land = province_mask(f)
        pixels = int(np.count_nonzero(land))
        rain_count = int(np.count_nonzero(predicted & land))
        hrp_count = int(np.count_nonzero(reference & land))
        overlapping = int(np.count_nonzero(predicted & reference & land))
        fraction = round(100.0 * rain_count / max(1, pixels), 3)
        jma_pct = round(100.0 * hrp_count / max(1, pixels), 3)
        valid = (predicted & land)
        yy, xx = np.nonzero(valid)
        if len(xx) >= MIN_INDICATED_PIXELS:
            location = [float(np.median(xx)), float(np.median(yy))]
        else:
            yy, xx = np.nonzero(land)
            location = ([float(np.median(xx)), float(np.median(yy))]
                        if len(xx) else None)
        rows.append({
            "provinsi": f["properties"]["PROVINSI"],
            "indikasi": _classify(fraction, rain_count),
            "luas_indikasi_rainnet_pct": fraction,
            "area_hrp_jma_pct": jma_pct,
            "piksel_provinsi": pixels,
            "piksel_rainnet": rain_count,
            "piksel_hrp_jma": hrp_count,
            "piksel_tumpang_tindih": overlapping,
            "koordinat_label": location,
        })
    return sorted(rows, key=lambda r: (-r["luas_indikasi_rainnet_pct"],
                                       -r["piksel_rainnet"], r["provinsi"]))


def _display_font(size: int, bold: bool = False):
    for name in ("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
                 "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_province_labels(overlay: Image.Image, summary: list,
                         *, max_labels: int = 12) -> Image.Image:
    if overlay.size != GRID_SIZE:
        raise ValueError("Province labels require native r2w image dimensions")
    result = overlay.convert("RGBA")
    layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    pencil = ImageDraw.Draw(layer)
    font = _display_font(17, True)
    small = _display_font(14)
    occupied = []
    labels = 0
    for row in summary:
        if row["indikasi"] == "Belum terindikasi" or labels >= max_labels:
            continue
        position = row["koordinat_label"]
        if not position:
            continue
        px, py = map(int, position)
        text = f'{row["provinsi"]} · {row["luas_indikasi_rainnet_pct"]:.1f}%'
        try:
            x0, y0, x1, y1 = pencil.textbbox((0, 0), text, font=font)
            width = x1 - x0
            height = y1 - y0
        except AttributeError:
            width, height = (len(text) * 10, 22)
        width += 18
        height += 13
        placed = None
        for dx, dy in ((12, -32), (-width-12, -32), (12, 10),
                       (-width-12, 10), (-width//2, -50), (-width//2, 18),
                       (20, -65), (20, 50)):
            x = min(max(1, px + dx), GRID_SIZE[0] - width - 1)
            y = min(max(1, py + dy), GRID_SIZE[1] - height - 1)
            rect = (x, y, x + width, y + height)
            if not any(not (rect[2] + 5 < old[0] or rect[0] - 5 > old[2]
                            or rect[3] + 5 < old[1] or rect[1] - 5 > old[3])
                       for old in occupied):
                placed = rect
                break
        if placed is None:
            continue
        occupied.append(placed)
        pencil.rounded_rectangle(placed, radius=7, fill=(9, 19, 31, 225),
                                outline=(58, 152, 255, 255), width=2)
        pencil.text((placed[0] + 9, placed[1] + 5), text,
                    font=font, fill=(237, 247, 255, 255))
        pencil.line((px, py, min(max(px, placed[0]), placed[2]),
                     min(max(py, placed[1]), placed[3])),
                    fill=(58, 152, 255, 230), width=2)
        pencil.ellipse((px-4, py-4, px+4, py+4),
                       fill=(3, 78, 250, 255), outline=(0, 0, 0, 255), width=1)
        labels += 1
    return Image.alpha_composite(result, layer).convert("RGB")
