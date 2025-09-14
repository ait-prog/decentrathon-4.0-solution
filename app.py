#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Динамический сервер: 10 карт Folium генерируются на лету по путям /maps/... и /output/...
Ничего не сохраняем на диск. Совместим с твоими эндпоинтами /, /list, /health, /maps/<file>, /output/<file>.

Запуск:
    python app.py
Открыть:
    http://127.0.0.1:8081/output/points_centers(1).html
    http://127.0.0.1:8081/maps/folium_heatmap.html
    ...

Данные:
- Берём CSV из ./data/ (первый .csv) либо укажи:
    $env:CSV_NAME = "my.csv"    # => .\data\my.csv
    # или полный путь:
    # $env:CLEAN_CSV = "C:\path\big.csv"
- Колонки lat/lng обязательны (автопереименование latitude/longitude/x/y/lon поддерживается).
- Колонка spd опциональна (для «пробок», «безопасности», «аномалий»).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from flask import Flask, jsonify, Response, request
from flask_cors import CORS

# ----------------------- Конфиг -----------------------

ROOT_DIR = Path(".").resolve()
DATA_DIR = ROOT_DIR / "data"

# env-переменные (по желанию)
CLEAN_CSV_ENV = os.environ.get("CLEAN_CSV", "").strip()
CSV_NAME_ENV  = os.environ.get("CSV_NAME", "").strip()

DEFAULT_CENTER = (51.1801, 71.4460)  # центр Астаны
DEFAULT_ZOOM   = 12
ASTANA_BBOX    = {"lat_min": 50.8, "lat_max": 51.4, "lng_min": 71.1, "lng_max": 71.8}

# ----------------------- Flask ------------------------

app = Flask(__name__)
CORS(app)

_data_cache: Dict[str, pd.DataFrame] = {}

# ---------------------- Утилиты -----------------------

def _auto_pick_csv() -> Optional[Path]:
    """Приоритет: CLEAN_CSV -> CSV_NAME в ./data -> первый *.csv в ./data."""
    if CLEAN_CSV_ENV:
        p = Path(CLEAN_CSV_ENV).expanduser()
        return p if p.exists() else None
    if CSV_NAME_ENV:
        p = (DATA_DIR / CSV_NAME_ENV).resolve()
        return p if p.exists() else None
    for p in DATA_DIR.glob("*.csv"):
        return p.resolve()
    return None

def _ensure_lat_lng(df: pd.DataFrame) -> pd.DataFrame:
    """Переименовываем известные варианты колонок в lat/lng."""
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(cands: List[str]) -> Optional[str]:
        for k in cands:
            if k in cols:
                return cols[k]
        return None
    lat = pick(["lat","latitude","y"])
    lng = pick(["lng","lon","longitude","x"])
    if lat and lat != "lat":
        df = df.rename(columns={lat: "lat"})
    if lng and lng != "lng":
        df = df.rename(columns={lng: "lng"})
    return df

def load_df() -> pd.DataFrame:
    """Читаем CSV -> приводим колонки -> фильтруем по bbox -> кэшируем."""
    if 'df' in _data_cache:
        return _data_cache['df']

    csv_path = _auto_pick_csv()
    if not csv_path or not csv_path.exists():
        df = pd.DataFrame(columns=["lat","lng","spd"])
        _data_cache['df'] = df
        return df

    df = pd.read_csv(csv_path)
    df = _ensure_lat_lng(df)
    if "spd" not in df.columns:
        df["spd"] = np.nan

    if {"lat","lng"}.issubset(df.columns):
        df = df[
            df["lat"].between(ASTANA_BBOX["lat_min"], ASTANA_BBOX["lat_max"]) &
            df["lng"].between(ASTANA_BBOX["lng_min"], ASTANA_BBOX["lng_max"])
        ].copy()
    else:
        df = pd.DataFrame(columns=["lat","lng","spd"])

    _data_cache['df'] = df
    return df

def _sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Безопасная подвыборка для больших датасетов."""
    if len(df) <= n:
        return df
    return df.sample(n, random_state=seed)

def _parse_params() -> Tuple[Tuple[float,float], int, int]:
    """
    Параметры в URL:
        ?center=51.18,71.44&zoom=12&sample=30000
    """
    zoom = int(request.args.get("zoom", DEFAULT_ZOOM))
    sample = int(request.args.get("sample", 50000))
    center_param = request.args.get("center", "")
    if center_param:
        try:
            lat_s, lng_s = center_param.split(",")
            center = (float(lat_s), float(lng_s))
        except Exception:
            center = DEFAULT_CENTER
    else:
        center = DEFAULT_CENTER
    return center, zoom, sample

def _make_map(center: Tuple[float,float], zoom: int, tiles: str = "cartodbpositron") -> folium.Map:
    return folium.Map(location=list(center), zoom_start=zoom, tiles=tiles, control_scale=True)

def _render(m: folium.Map) -> Response:
    html = m.get_root().render()
    return Response(html, mimetype="text/html; charset=utf-8")

# ------------------- Генераторы карт (10) -------------------

# 1) Heatmap базовая
def map_heatmap(df: pd.DataFrame, center, zoom, sample) -> Response:
    m = _make_map(center, zoom)
    if df.empty:
        lat0, lng0 = center
        pts = np.column_stack([np.random.normal(lat0, 0.03, 2000),
                               np.random.normal(lng0, 0.05, 2000)])
        HeatMap(pts.tolist(), radius=15, blur=20, min_opacity=0.3).add_to(m)
        return _render(m)
    sdf = _sample_df(df[["lat","lng"]].dropna(), sample)
    HeatMap(sdf.values.tolist(), radius=15, blur=20, min_opacity=0.3).add_to(m)
    return _render(m)

# 2) Heatmap "реалистичная" (другие тайлы)
def map_heatmap_realistic(df: pd.DataFrame, center, zoom, sample) -> Response:
    m = _make_map(center, zoom, tiles="OpenStreetMap")
    if df.empty:
        HeatMap([[center[0], center[1]]], radius=25).add_to(m)
        return _render(m)
    sdf = _sample_df(df[["lat","lng"]].dropna(), sample)
    HeatMap(sdf.values.tolist(), radius=16, blur=22, min_opacity=0.35).add_to(m)
    return _render(m)

# 3) Hex density (эмуляция hex через другой радиус/blur)
def map_hex_density(df: pd.DataFrame, center, zoom, sample) -> Response:
    m = _make_map(center, zoom)
    if df.empty:
        HeatMap([[center[0], center[1]]], radius=25).add_to(m)
        return _render(m)
    sdf = _sample_df(df[["lat","lng"]].dropna(), sample)
    HeatMap(sdf.values.tolist(), radius=22, blur=28, min_opacity=0.35).add_to(m)
    return _render(m)

# 4) Пробки — вес ниже при высокой скорости (ниже speed => выше вес)
def map_congestion(df: pd.DataFrame, center, zoom, sample) -> Response:
    m = _make_map(center, zoom)
    if df.empty:
        HeatMap([[center[0], center[1]]], radius=25).add_to(m)
        return _render(m)
    use_spd = "spd" in df.columns
    sdf = _sample_df(df[(["lat","lng"] + (["spd"] if use_spd else []))].dropna(subset=["lat","lng"]), sample)
    if use_spd and sdf["spd"].notna().any():
        spd = sdf["spd"].fillna(sdf["spd"].median())
        wt  = 1.0 / (1.0 + spd.clip(lower=0))
        data = [[row.lat, row.lng, float(w)] for (_, row), w in zip(sdf.iterrows(), wt)]
    else:
        data = sdf[["lat","lng"]].values.tolist()
    HeatMap(data, radius=18, blur=25, min_opacity=0.35).add_to(m)
    return _render(m)

# 5) Пробки (вариация)
def map_congestion_heatmap(df: pd.DataFrame, center, zoom, sample) -> Response:
    m = _make_map(center, zoom)
    if df.empty:
        HeatMap([[center[0], center[1]]], radius=25).add_to(m)
        return _render(m)
    sdf = _sample_df(df[["lat","lng","spd"]].dropna(subset=["lat","lng"]), sample)
    if "spd" in sdf.columns and sdf["spd"].notna().any():
        spd = sdf["spd"].fillna(sdf["spd"].median())
        wt  = 1.0 / (0.5 + spd.clip(lower=0))  # мягче, шире
        data = [[row.lat, row.lng, float(w)] for (_, row), w in zip(sdf.iterrows(), wt)]
    else:
        data = sdf[["lat","lng"]].values.tolist()
    HeatMap(data, radius=20, blur=28, min_opacity=0.35).add_to(m)
    return _render(m)

# 6) Центры кластеров
def map_points_centers(df: pd.DataFrame, center, zoom, sample, k: int = 50) -> Response:
    m = _make_map(center, zoom)
    if df.empty or not {"lat","lng"}.issubset(df.columns):
        folium.Marker(center, tooltip="No data — demo").add_to(m)
        return _render(m)

    sdf = _sample_df(df[["lat","lng"]].dropna(), sample)
    try:
        from sklearn.cluster import KMeans
        kk = min(k, max(1, len(sdf)//600))
        km = KMeans(n_clusters=kk, n_init="auto", random_state=42)
        km.fit(sdf[["lat","lng"]])
        centers = km.cluster_centers_
    except Exception:
        step = max(1, len(sdf)//k)
        centers = sdf[["lat","lng"]].iloc[::step].head(k).values

    mc = MarkerCluster().add_to(m)
    for lat, lng in _sample_df(sdf, min(3000, len(sdf)))[["lat","lng"]].values:
        folium.CircleMarker([lat, lng], radius=2, opacity=0.6, fill=True, fill_opacity=0.6).add_to(mc)

    for lat, lng in centers:
        folium.CircleMarker([lat, lng], radius=8, color="#000", weight=2, fill=True, fill_opacity=0.7,
                            tooltip=f"Center: {lat:.5f}, {lng:.5f}").add_to(m)
    return _render(m)

# 7) Цветные кластеры
def map_points_clusters_colored(df: pd.DataFrame, center, zoom, sample, k: int = 8) -> Response:
    m = _make_map(center, zoom)
    if df.empty or not {"lat","lng"}.issubset(df.columns):
        folium.Marker(center, tooltip="No data — demo").add_to(m)
        return _render(m)

    sdf = _sample_df(df[["lat","lng"]].dropna(), min(sample, 20000))
    try:
        from sklearn.cluster import KMeans
        kk = min(k, max(2, len(sdf)//3000))
        km = KMeans(n_clusters=kk, n_init="auto", random_state=42)
        labels = km.fit_predict(sdf[["lat","lng"]])
        sdf = sdf.assign(lbl=labels)
        colors = [
            "#e6194B","#3cb44b","#ffe119","#4363d8","#f58231",
            "#911eb4","#46f0f0","#f032e6","#bcf60c","#fabebe",
            "#008080","#e6beff","#9A6324","#fffac8","#800000",
            "#aaffc3","#808000","#ffd8b1","#000075","#808080"
        ]
        for lbl, g in sdf.groupby("lbl"):
            col = colors[lbl % len(colors)]
            for lat, lng in g[["lat","lng"]].values:
                folium.CircleMarker([lat, lng], radius=2, color=col, fill=True, fill_opacity=0.7).add_to(m)
    except Exception:
        mc = MarkerCluster().add_to(m)
        for lat, lng in sdf[["lat","lng"]].values:
            folium.CircleMarker([lat, lng], radius=2, opacity=0.6, fill=True, fill_opacity=0.6).add_to(mc)

    return _render(m)

# 8) Grid heatmap + немного scatter
def map_grid_heatmap_scatter(df: pd.DataFrame, center, zoom, sample) -> Response:
    m = _make_map(center, zoom)
    if df.empty:
        HeatMap([[center[0], center[1]]], radius=25).add_to(m)
        return _render(m)
    sdf = _sample_df(df[["lat","lng"]].dropna(), sample)
    HeatMap(sdf.values.tolist(), radius=16, blur=22, min_opacity=0.3).add_to(m)
    for lat, lng in _sample_df(sdf, min(2000, len(sdf))).values:
        folium.CircleMarker([lat, lng], radius=1, opacity=0.3, fill=True, fill_opacity=0.3).add_to(m)
    return _render(m)

# 9) Safety heatmap (вес по высокой скорости)
def map_safety_heatmap_scatter(df: pd.DataFrame, center, zoom, sample) -> Response:
    m = _make_map(center, zoom)
    if df.empty:
        HeatMap([[center[0], center[1]]], radius=25).add_to(m)
        return _render(m)
    sdf = _sample_df(df[["lat","lng","spd"]].dropna(subset=["lat","lng"]), sample)
    if "spd" in sdf.columns and sdf["spd"].notna().any():
        spd = sdf["spd"].fillna(sdf["spd"].median())
        spd_n = (spd - spd.min()) / max(spd.max() - spd.min(), 1e-9)
        data = [[row.lat, row.lng, float(w)] for (_, row), w in zip(sdf.iterrows(), spd_n.clip(0.1, 1.0))]
    else:
        data = sdf[["lat","lng"]].values.tolist()
    HeatMap(data, radius=16, blur=22, min_opacity=0.35).add_to(m)
    return _render(m)

# 10) Аномалии (>99-й перцентиль скорости)
def map_anomalies_heatmap(df: pd.DataFrame, center, zoom, sample) -> Response:
    m = _make_map(center, zoom)
    if df.empty:
        HeatMap([[center[0], center[1]]], radius=25).add_to(m)
        return _render(m)
    if "spd" in df.columns and df["spd"].notna().any():
        sdf = _sample_df(df[["lat","lng","spd"]].dropna(subset=["lat","lng"]), sample)
        if sdf["spd"].notna().any():
            thr = float(sdf["spd"].quantile(0.99))
            ano = sdf[sdf["spd"] > thr]
            if len(ano) > 0:
                max_spd = max(ano["spd"].max(), 1e-9)
                data = [[r.lat, r.lng, float(r.spd / max_spd)] for _, r in ano.iterrows()]
            else:
                data = sdf[["lat","lng"]].values.tolist()
        else:
            data = sdf[["lat","lng"]].values.tolist()
    else:
        data = _sample_df(df[["lat","lng"]].dropna(), sample).values.tolist()
    HeatMap(data, radius=20, blur=26, min_opacity=0.35).add_to(m)
    return _render(m)

# ----------------------- Роуты -----------------------

@app.route("/")
def index():
    df = load_df()
    maps = {
        "heatmaps": [
            {"name": "Карта загруженности (основная)",     "url": "/maps/folium_heatmap.html"},
            {"name": "Карта загруженности (реалистичная)", "url": "/maps/folium_heatmap_realistic.html"},
            {"name": "HEX плотность",                      "url": "/maps/folium_hex_density.html"}
        ],
        "congestion": [
            {"name": "Карта пробок (основная)",       "url": "/maps/folium_congestion.html"},
            {"name": "Карта пробок (тепловая)",       "url": "/maps/folium_congestion_heatmap.html"}
        ],
        "analysis": [
            {"name": "Кластеры точек (цвет)", "url": "/output/points_clusters_colored.html"},
            {"name": "Центры кластеров",      "url": "/output/points_centers(1).html"},
            {"name": "Сетка плотности",       "url": "/output/grid_heatmap_scatter.html"},
            {"name": "Карта безопасности",    "url": "/output/safety_heatmap_scatter.html"},
            {"name": "Карта аномалий",        "url": "/output/anomalies_heatmap.html"}
        ]
    }
    return jsonify({
        "message": "Онлайн-генерация 10 карт Folium (без файлов).",
        "version": "2.0.0",
        "maps": maps,
        "total_maps": sum(len(v) for v in maps.values()),
        "data_rows": int(len(df)),
        "hint": "К любой карте можно добавить ?center=51.18,71.44&zoom=12&sample=30000"
    })

@app.route("/health")
def health():
    df = load_df()
    return jsonify({
        "status": "healthy",
        "data_rows": int(len(df)),
        "bbox": ASTANA_BBOX,
        "virtual_maps": 10
    })

@app.route("/list")
def list_virtual():
    return jsonify({
        "maps": [
            {"type":"maps",   "name":"folium_heatmap",               "url":"/maps/folium_heatmap.html"},
            {"type":"maps",   "name":"folium_heatmap_realistic",     "url":"/maps/folium_heatmap_realistic.html"},
            {"type":"maps",   "name":"folium_hex_density",           "url":"/maps/folium_hex_density.html"},
            {"type":"maps",   "name":"folium_congestion",            "url":"/maps/folium_congestion.html"},
            {"type":"maps",   "name":"folium_congestion_heatmap",    "url":"/maps/folium_congestion_heatmap.html"},
            {"type":"output", "name":"points_centers(1)",            "url":"/output/points_centers(1).html"},
            {"type":"output", "name":"points_clusters_colored",      "url":"/output/points_clusters_colored.html"},
            {"type":"output", "name":"grid_heatmap_scatter",         "url":"/output/grid_heatmap_scatter.html"},
            {"type":"output", "name":"safety_heatmap_scatter",       "url":"/output/safety_heatmap_scatter.html"},
            {"type":"output", "name":"anomalies_heatmap",            "url":"/output/anomalies_heatmap.html"}
        ],
        "params": "?center=51.18,71.44&zoom=12&sample=30000"
    })

@app.route("/maps/<path:filename>")
def maps_virtual(filename: str):
    center, zoom, sample = _parse_params()
    df = load_df()
    name = filename.lower()

    if name.startswith("folium_heatmap_realistic"):
        return map_heatmap_realistic(df, center, zoom, sample)
    if name.startswith("folium_hex_density"):
        return map_hex_density(df, center, zoom, sample)
    if name.startswith("folium_congestion_heatmap"):
        return map_congestion_heatmap(df, center, zoom, sample)
    if name.startswith("folium_congestion"):
        return map_congestion(df, center, zoom, sample)
    if name.startswith("folium_heatmap"):
        return map_heatmap(df, center, zoom, sample)

    # неизвестное имя — дефолт на обычный heatmap
    return map_heatmap(df, center, zoom, sample)

@app.route("/output/<path:filename>")
def output_virtual(filename: str):
    center, zoom, sample = _parse_params()
    df = load_df()
    name = filename.lower()

    if name.startswith("points_centers"):
        return map_points_centers(df, center, zoom, sample)
    if name.startswith("points_clusters_colored"):
        return map_points_clusters_colored(df, center, zoom, sample)
    if name.startswith("grid_heatmap_scatter"):
        return map_grid_heatmap_scatter(df, center, zoom, sample)
    if name.startswith("safety_heatmap_scatter"):
        return map_safety_heatmap_scatter(df, center, zoom, sample)
    if name.startswith("anomalies_heatmap"):
        return map_anomalies_heatmap(df, center, zoom, sample)

    # дефолт
    return map_heatmap(df, center, zoom, sample)

# ----------------------- Старт сервера -----------------------

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8081"))

    print("🗺️ Динамический сервер карт (10 штук, без файлов на диске)")
    print("=" * 60)
    print(f"🌐 Запуск на http://{host}:{port}")
    print("\n📋 Эндпоинты:")
    print("  GET /            — обзор + ссылки")
    print("  GET /list        — список 10 виртуальных карт")
    print("  GET /health      — статус и размер датасета")
    print("  GET /maps/<file> — виртуальные folium_* карты")
    print("  GET /output/<f>  — виртуальные аналитические карты")
    print("\n🔗 Примеры:")
    print("  http://127.0.0.1:8081/maps/folium_heatmap.html")
    print("  http://127.0.0.1:8081/output/points_centers(1).html")
    print("  http://127.0.0.1:8081/output/anomalies_heatmap.html?center=51.17,71.45&zoom=13&sample=40000")
    app.run(host=host, port=port, debug=False)