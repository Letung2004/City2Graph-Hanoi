# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH TRỰC QUAN HÓA DỮ LIỆU & ĐỒ THỊ KHÔNG GIAN HÀ NỘI (ĐẦY ĐỦ 9 CELLS)
Kế thừa & Chuyển đổi 1:1 từ: 05_visualization.py (Sato et al., CEUS 2026)
Đối tượng khảo sát: 985 ô lục giác Uber H3 Resolution 9 tại Thủ đô Hà Nội
Hệ quy chiếu phẳng: WGS 84 / UTM Zone 48N (EPSG:32648)
Định dạng hỗ trợ: VS Code Interactive Window (# %%) hoặc python dòng lệnh
"""

from __future__ import annotations

# %% [Cell 1]: Thiết lập môi trường thực thi và tự động định vị thư mục gốc
# --- Ghi chú giải thích từ Notebook gốc ---
# # 05. Visualization (Hanoi Case Study)
#
# This notebook visualizes the processed data and graph structures,
# including POI distributions, urban land use, and metapaths on dark layouts.
# -----------------------------------------

import os
import sys
import warnings
from pathlib import Path

# Cấu hình OpenMP an toàn cho môi trường Windows chống xung đột runtime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

# Cơ chế Dynamic Root Discovery tự động định vị gốc repository City2Graph-Hanoi
CWD = Path.cwd().resolve()
REPO_ROOT = CWD
while not (REPO_ROOT / "data").exists() and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent

SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("=" * 90)
print(f"✓ [Cell 1] Đã định vị thư mục gốc dự án: {REPO_ROOT}")
print(f"✓ [Cell 1] Đã kết nối module mã nguồn lõi: {SRC_DIR}")
print("=" * 90)


# %% [Cell 2]: Khai báo thư viện, thiết lập phong cách đồ họa và cấu hình đường dẫn
# --- Ghi chú giải thích từ Notebook gốc ---
# Import dependencies, plotting styles, scalebar and directory structure
# -----------------------------------------

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.lines import Line2D
from shapely import wkt
from shapely.geometry import Point, LineString
import contextily as ctx
import torch

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    HAS_SCALEBAR = True
except ImportError:
    HAS_SCALEBAR = False

try:
    from cmcrameri import cm as cmc
    HAS_CMC = True
except ImportError:
    HAS_CMC = False

try:
    import city2graph as c2g
except ImportError:
    c2g = None

# Thiết lập phong cách đồ họa chuẩn
plt.style.use("default")

TARGET_CRS = "EPSG:32648"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "hanoi"
PROCESSED_DIR = DATA_DIR / "processed" / "hanoi"
OUTPUT_DIR = DATA_DIR / "outputs" / "hanoi"

FEATURES_DIR = PROCESSED_DIR / "features"
GRAPHS_DIR = PROCESSED_DIR / "graphs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print(f"✓ [Cell 2] Hệ tọa độ chuẩn: {TARGET_CRS}")
print(f"✓ [Cell 2] Thư mục nạp đặc trưng: {FEATURES_DIR}")
print(f"✓ [Cell 2] Thư mục lưu ấn phẩm: {FIGURES_DIR}")


# %% [Cell 3]: Danh mục cố định 16 nhóm đặc trưng tiện ích tầng trệt (Table A.1)
# --- Ghi chú giải thích từ Notebook gốc ---
# OA node feature columns (fixed list for this case study, Sato et al., CEUS 2026)
# -----------------------------------------

OA_FEATURE_COLS = [
    "arts_culture",
    "automotive_facility",
    "consumer_service",
    "corporate_service",
    "education",
    "entertainment",
    "food_beverage",
    "healthcare",
    "hotel_lodging",
    "industrial_service",
    "park",
    "public_service",
    "religion",
    "retail",
    "sports_fitness",
    "transportation_facility"
]

print(f"✓ [Cell 3] Danh mục {len(OA_FEATURE_COLS)} nhóm POIs chuẩn Bảng A.1 đã sẵn sàng.")


# %% [Cell 4]: Trực quan hóa ma trận 16 bản đồ POIs tầng trệt (Hình 10b trong bài báo)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 1. POI Visualization
# Visualize POI density/categories across OAs.
# -----------------------------------------

print("\n-> [Cell 4] Đang nạp đặc trưng và kết xuất ma trận 16 bản đồ POIs tầng trệt...")

oa_features_path = FEATURES_DIR / "oa_with_features.gpkg"
if not oa_features_path.exists():
    oa_features_path = FEATURES_DIR / "hanoi_normalized_features.parquet"

if oa_features_path.suffix == ".parquet":
    oa_with_features = gpd.read_parquet(oa_features_path)
else:
    oa_with_features = gpd.read_file(oa_features_path)

if oa_with_features.crs is None or str(oa_with_features.crs) != TARGET_CRS:
    oa_with_features = oa_with_features.to_crs(TARGET_CRS)

# Lọc các cột số hợp lệ trong danh mục 16 POIs
categories = [
    c for c in oa_with_features.columns
    if c in OA_FEATURE_COLS and pd.api.types.is_numeric_dtype(oa_with_features[c])
]

n_cols = 4
n_rows = (len(categories) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4.5 * n_rows), dpi=200)
axes = axes.flatten()

# Chuyển đổi sang Web Mercator (EPSG:3857) để khớp lớp bản đồ nền Esri
oa_3857 = oa_with_features.to_crs("EPSG:3857")

for i, col in enumerate(categories):
    ax = axes[i]
    col_vals = oa_3857[col]
    n_unique = col_vals.nunique()

    if n_unique >= 5:
        oa_3857.plot(
            column=col,
            ax=ax,
            scheme="NaturalBreaks",
            k=5,
            legend=True,
            cmap="YlOrRd",
            alpha=0.75,
            linewidth=0.15,
            edgecolor="white",
            legend_kwds={"loc": "lower right", "fmt": "{:.0f}", "fontsize": "x-small"}
        )
    elif n_unique > 1:
        oa_3857.plot(
            column=col,
            ax=ax,
            scheme="NaturalBreaks",
            k=min(n_unique, 3),
            legend=True,
            cmap="YlOrRd",
            alpha=0.75,
            linewidth=0.15,
            edgecolor="white",
            legend_kwds={"loc": "lower right", "fmt": "{:.0f}", "fontsize": "x-small"}
        )
    else:
        oa_3857.plot(column=col, ax=ax, color="#fffae6", alpha=0.75, linewidth=0.15, edgecolor="white")

    # Bản đồ nền mở Esri Gray Canvas
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas, attribution=False)
    except Exception:
        pass

    ax.set_title(f"POI: {col}", fontsize=12, fontweight="bold", pad=4)
    ax.set_axis_off()

# Ẩn các ô subplot thừa nếu có
for j in range(len(categories), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
poi_img_path = FIGURES_DIR / "poi.jpg"
plt.savefig(poi_img_path, bbox_inches="tight", dpi=200)
print(f"✓ [Cell 4] Đã lưu ma trận bản đồ POIs tại: {poi_img_path.resolve()}")
plt.show()
plt.close(fig)


# %% [Cell 5]: Tính toán tổng số lượng và mật độ tiện ích tầng trệt (POIs) trên từng ô H3
# --- Ghi chú phương pháp luận ---
# Đồng bộ hóa 100% với quyết định thiết kế chỉ sử dụng 16 nhóm POIs tầng trệt (Overture Maps)
# cho ca nghiên cứu Hà Nội, loại bỏ hoàn toàn việc nội suy giả định lớp sử dụng đất (Land Use).
# -----------------------------------------

print("\n-> [Cell 5] Đang tổng hợp số lượng và mật độ tiện ích tầng trệt (16 nhóm POIs) trên 985 ô H3...")

# Tự động nhận diện GeoDataFrame từ Cell 4 hoặc nạp trực tiếp từ oa_with_features.gpkg
if "oa_gdf" not in globals():
    if "oa_with_features" in globals():
        oa_gdf = oa_with_features.copy()
    elif "gdf" in globals():
        oa_gdf = gdf.copy()
    elif "hanoi_oa" in globals():
        oa_gdf = hanoi_oa.copy()
    else:
        _oa_path = (
            FEATURES_DIR / "oa_with_features.gpkg"
            if "FEATURES_DIR" in globals()
            else REPO_ROOT / "data" / "processed" / "hanoi" / "features" / "oa_with_features.gpkg"
        )
        oa_gdf = gpd.read_file(_oa_path)

# Đảm bảo hệ quy chiếu phẳng mét EPSG:32648
if oa_gdf.crs is None or str(oa_gdf.crs) != "EPSG:32648":
    oa_gdf = oa_gdf.to_crs("EPSG:32648")

# Đảm bảo danh sách 16 cột POIs có sẵn
if "OA_FEATURE_COLS" not in globals():
    OA_FEATURE_COLS = [
        "arts_culture", "automotive_facility", "consumer_service", "corporate_service",
        "education", "entertainment", "food_beverage", "healthcare",
        "hotel_lodging", "industrial_service", "park", "public_service",
        "religion", "retail", "sports_fitness", "transportation_facility"
    ]

# Tính tổng số lượng POIs trên toàn bộ 16 danh mục chuẩn Bảng A.1
oa_gdf["total_pois"] = oa_gdf[OA_FEATURE_COLS].sum(axis=1).astype(float)

# Tính diện tích từng ô lục giác H3 theo km2 (Hệ quy chiếu phẳng EPSG:32648 - đơn vị mét)
oa_gdf["area_km2"] = oa_gdf.geometry.area / 1e6
oa_gdf["poi_density_km2"] = oa_gdf["total_pois"] / oa_gdf["area_km2"].replace(0, np.nan)

total_poi_sum = int(oa_gdf["total_pois"].sum())
mean_poi = oa_gdf["total_pois"].mean()
max_poi = int(oa_gdf["total_pois"].max())
median_poi = oa_gdf["total_pois"].median()

print(f"✓ [Cell 5] Thống kê phân bổ tổng tiện ích tầng trệt trên {len(oa_gdf)} ô H3:")
print(f"  • Tổng số điểm POIs toàn vùng : {total_poi_sum:,} điểm")
print(f"  • Trung bình / Trung vị mỗi ô : {mean_poi:.1f} / {median_poi:.1f} POIs/ô H3")
print(f"  • Cực đại tại ô trung tâm CBD : {max_poi:,} POIs/ô H3")


# %% [Cell 6]: Kết xuất bản đồ phân bố tổng mật độ tiện ích đô thị (Thay thế Figure 10a)
# --- Ghi chú giải thích ---
# Vẽ bản đồ phân cấp định lượng (Quantiles/FisherJenks) tổng số lượng POIs trên từng ô H3,
# tích hợp nền bản đồ mở CARTO Positron, Kim chỉ Bắc (North Arrow) và Thước tỷ lệ 2 km.
# -----------------------------------------

print("\n-> [Cell 6] Đang kết xuất bản đồ tổng mật độ tiện ích tầng trệt (Thay thế Hình 10a)...")

try:
    import contextily as ctx
except ImportError:
    ctx = None

fig, ax = plt.subplots(figsize=(12, 12), dpi=300)

# Sử dụng phân khoảng FisherJenks (k=5) để làm nổi bật cấu trúc lõi - vùng ven của Hà Nội
try:
    oa_gdf.plot(
        column="total_pois",
        ax=ax,
        cmap="YlOrRd",
        scheme="FisherJenks",
        k=5,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.25,
        legend=True,
        legend_kwds={
            "title": "Total POIs / H3 Zone",
            "loc": "upper right",
            "frameon": True,
            "facecolor": "white",
            "framealpha": 0.9,
            "fontsize": 11,
            "title_fontsize": 12,
            "fmt": "{:.0f}",
        },
    )
except Exception:
    # Dự phòng nếu thiếu thư viện mapclassify
    oa_gdf.plot(
        column="total_pois",
        ax=ax,
        cmap="YlOrRd",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.25,
        legend=True,
        legend_kwds={
            "label": "Total POIs per H3 Zone",
            "shrink": 0.6,
        },
    )

# Thêm lớp bản đồ nền (Basemap)
if ctx is not None:
    try:
        ctx.add_basemap(
            ax,
            crs=oa_gdf.crs.to_string(),
            source=ctx.providers.Esri.WorldGrayCanvas,
            attribution=False,
            zoom=13,
        )
    except Exception:
        pass

# Vẽ Kim chỉ Bắc (North Arrow) ở góc trên bên trái
ax.annotate(
    "N",
    xy=(0.06, 0.93),
    xytext=(0.06, 0.85),
    xycoords="axes fraction",
    textcoords="axes fraction",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    arrowprops=dict(facecolor="black", width=4, headwidth=12, headlength=12),
    zorder=20,
)

# Vẽ Thước tỷ lệ 2 km (Scale Bar) ở góc dưới bên phải
bounds = oa_gdf.total_bounds  # [minx, miny, maxx, maxy]
bar_len_m = 2000.0  # 2 km = 2000 m trong EPSG:32648
x_end = bounds[2] - (bounds[2] - bounds[0]) * 0.04
x_start = x_end - bar_len_m
y_pos = bounds[1] + (bounds[3] - bounds[1]) * 0.03

ax.plot([x_start, x_end], [y_pos, y_pos], color="black", linewidth=6, solid_capstyle="butt", zorder=20)
ax.text(
    (x_start + x_end) / 2,
    y_pos - (bounds[3] - bounds[1]) * 0.022,
    "2 km",
    ha="center",
    va="top",
    fontsize=12,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.8),
    zorder=20,
)

ax.set_title(
    "Hanoi Ground-Floor Urban Amenity Density (Total POIs per H3 Zone)",
    fontsize=15,
    fontweight="bold",
    pad=14,
)
ax.set_axis_off()
plt.tight_layout()

# Xác định thư mục xuất ảnh an toàn
_fig_dir = (
    FIGURES_DIR
    if "FIGURES_DIR" in globals()
    else REPO_ROOT / "data" / "outputs" / "hanoi" / "figures"
)
_fig_dir.mkdir(parents=True, exist_ok=True)

# Lưu đồng thời ra file chuẩn và file tương thích ngược với script kiểm toán
poi_density_fig_path = _fig_dir / "hanoi_total_poi_density_fig10a.jpg"
compat_fig_path = _fig_dir / ".jpg"

plt.savefig(poi_density_fig_path, dpi=300, bbox_inches="tight")
plt.savefig(compat_fig_path, dpi=300, bbox_inches="tight")

print(f"✓ [Cell 6] Đã lưu bản đồ tổng mật độ tiện ích tại: {poi_density_fig_path}")
print(f"✓ [Cell 6] Đã cập nhật đè tệp tương thích kiểm toán tại: {compat_fig_path}")
plt.show()
plt.close(fig)


# %% [Cell 7]: Nạp đồ thị dị thể và trực quan hóa Metapaths nền tối (Hình 1A & Hình 1B)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3. Metapath Visualization (Figures 1A & 1B in paper)
# - Figure 1A: OA polygons + centroids
# - Figure 1B: Contiguity edges with travel-time colorbar
# -----------------------------------------

print("\n-> [Cell 7] Đang nạp đồ thị dị thể và kết xuất Hình 1A & Hình 1B nền tối...")

graph_path = GRAPHS_DIR / "hetero_multi.pt"
if not graph_path.exists():
    raise FileNotFoundError(f"Không tìm thấy đồ thị hetero_multi.pt tại: {graph_path}")

hetero_data_obj = torch.load(graph_path, map_location="cpu", weights_only=False)

# Chuyển đổi PyG sang GeoDataFrames nếu có city2graph, hoặc bóc tách trực tiếp
if c2g is not None and hasattr(c2g, "pyg_to_gdf"):
    try:
        hetero_nodes_multi, hetero_edges_multi = c2g.pyg_to_gdf(hetero_data_obj)
    except Exception:
        hetero_nodes_multi = None
        hetero_edges_multi = None
else:
    hetero_nodes_multi = None
    hetero_edges_multi = None

# Chuẩn bị GeoDataFrame đỉnh 'oa' từ oa_with_features
oa_polygons = oa_with_features.copy()
minx, miny, maxx, maxy = oa_polygons.total_bounds

node_gdf = oa_polygons.copy()
if "pop_weighted_centroid" in node_gdf.columns:
    sample_pt = node_gdf["pop_weighted_centroid"].dropna().iloc[0]
    if isinstance(sample_pt, str):
        node_gdf["geometry"] = node_gdf["pop_weighted_centroid"].map(wkt.loads)
    else:
        node_gdf["geometry"] = node_gdf["pop_weighted_centroid"]
else:
    node_gdf["geometry"] = node_gdf.geometry.centroid

oa_centroids_dict = {str(k): (p.x, p.y) for k, p in zip(node_gdf.index, node_gdf["geometry"])}

# Trích xuất 3 loại quan hệ cốt lõi
edge_types = [
    ("oa", "contig", "oa"),
    ("oa", "15_min_walk", "oa"),
    ("oa", "15_min_multi", "oa"),
]

# Tự động dựng GeoDataFrame cạnh từ tensor PyG nếu cần
edge_gdfs = {}
for rel in edge_types:
    if hetero_edges_multi is not None and rel in hetero_edges_multi:
        edge_gdfs[rel] = hetero_edges_multi[rel]
    elif rel in hetero_data_obj.edge_types:
        e_idx = hetero_data_obj[rel].edge_index.numpy()
        e_attr = hetero_data_obj[rel].edge_attr.numpy().flatten() if hasattr(hetero_data_obj[rel], "edge_attr") and hetero_data_obj[rel].edge_attr is not None else np.full(e_idx.shape[1], 450.0)
        u_keys = [str(node_gdf.index[u]) for u in e_idx[0]]
        v_keys = [str(node_gdf.index[v]) for v in e_idx[1]]
        lines = [LineString([oa_centroids_dict[u], oa_centroids_dict[v]]) for u, v in zip(u_keys, v_keys)]
        df_rel = gpd.GeoDataFrame({"travel_time_sec": e_attr, "geometry": lines}, crs=TARGET_CRS)
        edge_gdfs[rel] = df_rel

edge_lws = [1.2, 0.45, 0.35]
vmin, vmax = 0, 900
norm_time = colors.Normalize(vmin=vmin, vmax=vmax)

# Chuẩn dải màu davos đảo ngược hoặc viridis
if HAS_CMC:
    base_cmap = cmc.davos.reversed()
    def truncate_cmap(c, lo=0.0, hi=0.75, n=256):
        return LinearSegmentedColormap.from_list("trunc", c(np.linspace(lo, hi, n)))
    cmap_meta = truncate_cmap(base_cmap)
else:
    cmap_meta = plt.get_cmap("viridis")

SPLIT_FIG_SIZE = (12, 12)
SPLIT_SAVE_KW = {"facecolor": "black", "dpi": 300, "bbox_inches": "tight", "pad_inches": 0}
BACKGROUND_COLOR = "black"
CBAR_LABEL = r"$d_{\mathrm{euc}}(i,j) \,/\, v_{\mathrm{walk}}$"

# --- Figure 1A: OA polygons + weighted centroids ---
fig, ax = plt.subplots(1, 1, figsize=SPLIT_FIG_SIZE, facecolor="black", constrained_layout=True)
ax.set_facecolor("black")

oa_polygons.plot(facecolor="none", edgecolor="white", linewidth=0.5, ax=ax, alpha=0.6)
node_gdf.plot(ax=ax, color="white", markersize=6, alpha=1.0)

# Colorbar khớp khung hình
sm_black = cm.ScalarMappable(cmap=ListedColormap([BACKGROUND_COLOR]), norm=norm_time)
sm_black.set_array([])
cbar_1a = fig.colorbar(sm_black, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
cbar_1a.set_label(CBAR_LABEL, color="black", fontsize=18)
cbar_1a.ax.yaxis.set_tick_params(color="black", labelcolor="black")
cbar_1a.ax.set_facecolor(BACKGROUND_COLOR)
cbar_1a.outline.set_edgecolor(BACKGROUND_COLOR)

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_axis_off()

fig1a_path = FIGURES_DIR / "hanoi_contig_oa.jpg"
plt.savefig(fig1a_path, **SPLIT_SAVE_KW)
print(f"✓ [Cell 7] Đã lưu Hình 1A (Đa giác & Centroids) tại: {fig1a_path.resolve()}")
plt.show()
plt.close(fig)

# --- Figure 1B: Contiguity edges ---
fig, ax = plt.subplots(1, 1, figsize=SPLIT_FIG_SIZE, facecolor="black", constrained_layout=True)
ax.set_facecolor("black")

df_contig = edge_gdfs[edge_types[0]]
edge_colors_contig = [cmap_meta(norm_time(val)) for val in df_contig["travel_time_sec"]]

# Vẽ cạnh tiếp giáp màu phát quang
df_contig.plot(ax=ax, color=edge_colors_contig, linewidth=edge_lws[0], alpha=0.9)
node_gdf.plot(ax=ax, color="white", markersize=6, alpha=1.0)

sm_contig = cm.ScalarMappable(cmap=cmap_meta, norm=norm_time)
sm_contig.set_array([])
cbar_1b = fig.colorbar(sm_contig, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
cbar_1b.set_label(CBAR_LABEL, color="white", fontsize=18)
cbar_1b.ax.yaxis.set_tick_params(color="white", labelcolor="white")

ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_axis_off()

fig1b_path = FIGURES_DIR / "hanoi_contig_edges.jpg"
plt.savefig(fig1b_path, **SPLIT_SAVE_KW)
print(f"✓ [Cell 7] Đã lưu Hình 1B (Cạnh Contiguity) tại: {fig1b_path.resolve()}")
plt.show()
plt.close(fig)


# %% [Cell 8]: Trực quan hóa các siêu quan hệ Metapaths tách rời (Hình 2 trong bài báo)
# --- Ghi chú giải thích từ Notebook gốc ---
# --- Figure 2: Metapaths (split into one figure per relation, CEUS 2026) ---
# - hanoi_metapaths_walk.jpg (15-min walk edges)
# - hanoi_metapaths_multi.jpg (15-min multimodal transit edges)
# -----------------------------------------

print("\n-> [Cell 8] Đang kết xuất các siêu đường đi Metapaths tách rời (Hình 2)...")

metapath_configs = [
    (1, "hanoi_metapaths_walk.jpg", "15-Minute Walk Accessibility Metapath"),
    (2, "hanoi_metapaths_multi.jpg", "15-Minute Multimodal Transit Metapath"),
]

for idx, out_name, title_desc in metapath_configs:
    fig, ax = plt.subplots(1, 1, figsize=SPLIT_FIG_SIZE, facecolor="black", constrained_layout=True)
    ax.set_facecolor("black")

    rel_key = edge_types[idx]
    lw_val = edge_lws[idx]
    df_meta = edge_gdfs[rel_key]
    edge_colors_meta = [cmap_meta(norm_time(val)) for val in df_meta["travel_time_sec"]]

    # Lấy mẫu thông minh nếu số cạnh quá lớn (> 20.000) để vẽ nhanh và sắc nét
    if len(df_meta) > 20000:
        sample_step = len(df_meta) // 20000
        df_meta_plot = df_meta.iloc[::sample_step]
        colors_plot = edge_colors_meta[::sample_step]
    else:
        df_meta_plot = df_meta
        colors_plot = edge_colors_meta

    df_meta_plot.plot(ax=ax, color=colors_plot, linewidth=lw_val, alpha=0.85)
    node_gdf.plot(ax=ax, color="white", markersize=3.5, alpha=0.95)

    sm_meta = cm.ScalarMappable(cmap=cmap_meta, norm=norm_time)
    sm_meta.set_array([])
    cbar_meta = fig.colorbar(sm_meta, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
    cbar_meta.set_label("Travel Time (seconds)", color="white", fontsize=18)
    cbar_meta.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()

    out_fig_meta_path = FIGURES_DIR / out_name
    plt.savefig(out_fig_meta_path, **SPLIT_SAVE_KW)
    print(f"✓ [Cell 8] Đã lưu {title_desc} tại: {out_fig_meta_path.resolve()}")
    plt.show()
    plt.close(fig)


# %% [Cell 9]: Bảng thống kê mô tả thời gian di chuyển trên từng quan hệ cạnh (Phụ lục Bảng A.2)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 4. Edge Relation Descriptive Statistics (Appendix Table A.2 in Sato et al., CEUS 2026)
# Create a comprehensive table of descriptive statistics for each edge relation.
# Reports undirected pairs: count, min, mean, max, std of travel_time_sec.
# -----------------------------------------

print("\n-> [Cell 9] Đang tổng hợp Bảng thống kê mô tả Phụ lục A.2 (Table A.2 Benchmark)...")

# Danh sách 7 quan hệ mạng lưới chuẩn mực đối chuẩn bài báo quốc tế
desired_edge_types = [
    ("oa", "is_nearby", "street_connector"),
    ("street_connector", "is_connected_to", "street_connector"),
    ("bus_station", "is_nearby", "street_connector"),
    ("bus_station", "is_next_to", "bus_station"),
    ("oa", "contig", "oa"),
    ("oa", "15_min_walk", "oa"),
    ("oa", "15_min_multi", "oa"),
]

# Nạp thêm dữ liệu đường bộ và trạm buýt nếu cần bổ sung cho đủ 7 quan hệ
extra_edge_dfs = {}

st_edges_f = FEATURES_DIR / "street_edges.gpkg"
if st_edges_f.exists():
    df_st_e = gpd.read_file(st_edges_f)
    if "travel_time_sec" in df_st_e.columns:
        extra_edge_dfs[("street_connector", "is_connected_to", "street_connector")] = df_st_e

tr_edges_f = FEATURES_DIR / "travel_summary_edges.gpkg"
if tr_edges_f.exists():
    df_tr_e = gpd.read_file(tr_edges_f)
    if "travel_time_sec" in df_tr_e.columns:
        extra_edge_dfs[("bus_station", "is_next_to", "bus_station")] = df_tr_e

relation_rows = []

for edge_type in desired_edge_types:
    target_df = None
    if edge_type in edge_gdfs:
        target_df = edge_gdfs[edge_type]
    elif edge_type in extra_edge_dfs:
        target_df = extra_edge_dfs[edge_type]
    elif hetero_edges_multi is not None and edge_type in hetero_edges_multi:
        target_df = hetero_edges_multi[edge_type]

    if target_df is None:
        continue

    # Lọc cột số thời gian di chuyển
    time_col = "travel_time_sec" if "travel_time_sec" in target_df.columns else "weight"
    if time_col not in target_df.columns:
        continue

    s_vals = pd.to_numeric(target_df[time_col], errors="coerce").dropna()
    if s_vals.empty:
        continue

    # Tính toán chính xác theo chuẩn Table A.2
    c_count = int(len(s_vals))
    c_min = float(s_vals.min())
    c_mean = float(s_vals.mean())
    c_max = float(s_vals.max())
    c_std = float(s_vals.std())

    relation_rows.append({
        "edge_type": str(edge_type),
        "count": c_count,
        "min": round(c_min, 2),
        "mean": round(c_mean, 2),
        "max": round(c_max, 2),
        "std": round(c_std, 2),
    })

# Tạo DataFrame Bảng A.2
edge_relation_stats = pd.DataFrame(relation_rows)

print("\n" + "=" * 90)
print("PHỤ LỤC BẢNG A.2: THỐNG KÊ MÔ TẢ CÁC QUAN HỆ CẠNH KHÔNG GIAN (HÀ NỘI CASE STUDY)")
print("=" * 90)
print(edge_relation_stats.to_string(index=False))
print("=" * 90)

# Xuất tệp CSV
out_stats_csv_path = TABLES_DIR / "edge_relation_descriptive_stats.csv"
edge_relation_stats.to_csv(out_stats_csv_path, index=False)
print(f"✓ [Cell 9] Đã xuất Bảng A.2 thành công tại: {out_stats_csv_path.resolve()}")

print("\n" + "=" * 90)
print("✓ HOÀN THÀNH TOÀN BỘ 9 CELLS CỦA FILE 05_VISUALIZATION CHO HÀ NỘI!")
print("==========================================================================================")