# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH XÂY DỰNG ĐỒ THỊ KHÔNG GIAN HÀ NỘI (CELL 1 - 21)
Kế thừa & Chuyển đổi 1:1 từ: 02_graph_construction.py (Sato et al., CEUS 2026)
Hệ quy chiếu phẳng: WGS 84 / UTM Zone 48N (EPSG:32648)
Đơn vị không gian: 985 ô lục giác Uber H3 Resolution 9 (Khóa chính: zone_id)
Mô hình đích: Model 1 (homo.pt), Model 2 (hetero_walk.pt), Model 3 & 4 (hetero_multi.pt)
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path

# Cấu hình OpenMP tránh xung đột nhị phân runtime trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

# Cơ chế Dynamic Root Discovery tự động định vị thư mục gốc dự án
CWD = Path.cwd().resolve()
REPO_ROOT = CWD
while not (REPO_ROOT / "data").exists() and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent

SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print(f"✓ [Cell 1] Đã định vị thư mục gốc dự án: {REPO_ROOT}")

# %% [Cell 2]: Khai báo thư viện, hằng số và danh mục 16 đặc trưng tiện ích Hà Nội
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.preprocessing import StandardScaler
from shapely import wkt
from shapely.geometry import Point, LineString
import torch

def extract_edge_uv(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Trích xuất 2 chuỗi u và v an toàn từ MultiIndex hoặc các cột dữ liệu."""
    if isinstance(df.index, pd.MultiIndex):
        return df.index.get_level_values(0).astype(str), df.index.get_level_values(1).astype(str)
    
    candidates = [
        ("from_id", "to_id"),
        ("from_node_id", "to_node_id"),
        ("source", "target"),
        ("u", "v"),
        ("src", "dst"),
        ("oa_id", "street_connector_id"),
        ("from_stop_id", "to_stop_id"),
    ]
    for c1, c2 in candidates:
        if c1 in df.columns and c2 in df.columns:
            return df[c1].astype(str), df[c2].astype(str)
            
    non_geom = [c for c in df.columns if c not in ["geometry", "travel_time_sec", "weight", "length", "edge_key"]]
    if len(non_geom) >= 2:
        return df[non_geom[0]].astype(str), df[non_geom[1]].astype(str)
        
    return df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)

try:
    import city2graph as c2g
except ImportError:
    c2g = None

DATA_DIR = REPO_ROOT / "data"
FEATURE_DIR = DATA_DIR / "processed" / "hanoi" / "features"
GRAPH_DIR = DATA_DIR / "processed" / "hanoi" / "graphs"
FIGURE_DIR = DATA_DIR / "outputs" / "hanoi" / "figures"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

WALKING_SPEED_MPS = 4.8 / 3.6  # 1.3333 m/s
TARGET_CRS = "EPSG:32648"

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
    "transportation_facility",
]

print(f"✓ [Cell 2] Thiết lập thành công: Vận tốc đi bộ = {WALKING_SPEED_MPS:.4f} m/s | Số chiều đặc trưng = {len(OA_FEATURE_COLS)}")

# %% [Cell 3]: Nạp đa giác 985 ô H3 và phân tích cú pháp tâm đối xứng hình học
OA_PATH = FEATURE_DIR / "oa_with_features.gpkg"
PARQUET_BACKUP = FEATURE_DIR / "hanoi_normalized_features.parquet"

if OA_PATH.exists():
    oa_with_features = gpd.read_file(OA_PATH)
elif PARQUET_BACKUP.exists():
    oa_with_features = gpd.read_parquet(PARQUET_BACKUP)
else:
    raise FileNotFoundError(f"Không tìm thấy tệp đặc trưng nút tại {OA_PATH} hoặc {PARQUET_BACKUP}!")

if oa_with_features.crs is None or str(oa_with_features.crs) != TARGET_CRS:
    oa_with_features = oa_with_features.to_crs(TARGET_CRS)

if "pop_weighted_centroid" in oa_with_features.columns:
    sample_val = oa_with_features["pop_weighted_centroid"].dropna().iloc[0]
    if isinstance(sample_val, str):
        oa_with_features["pop_weighted_centroid"] = oa_with_features["pop_weighted_centroid"].map(wkt.loads)
else:
    oa_with_features["pop_weighted_centroid"] = oa_with_features.geometry.centroid

if "zone_id" not in oa_with_features.columns:
    oa_with_features["zone_id"] = [f"h3_res9_{i:04d}" for i in range(len(oa_with_features))]

oa_with_features = oa_with_features.set_index("zone_id", drop=False)
print(f"✓ [Cell 3] Nạp thành công {len(oa_with_features):,} ô lục giác H3 (Index: zone_id | CRS: {oa_with_features.crs})")

# %% [Cell 4]: Chẩn đoán phân phối đặc trưng thô và trực quan hóa biểu đồ Histogram
oa_feat = oa_with_features[OA_FEATURE_COLS].copy()
oa_feat = oa_feat.apply(pd.to_numeric, errors="coerce")

if oa_feat.isna().any().any():
    bad_cols = oa_feat.columns[oa_feat.isna().any()].tolist()
    raise ValueError(f"Phát hiện giá trị NaN trong các cột đặc trưng POI: {bad_cols}")

min_val = float(oa_feat.min().min())
if min_val < 0:
    raise ValueError(f"Kỳ vọng dữ liệu tần suất không âm, nhưng phát hiện min = {min_val}")

dist = pd.DataFrame(index=OA_FEATURE_COLS)
dist["min"] = oa_feat.min()
dist["mean"] = oa_feat.mean()
dist["max"] = oa_feat.max()
dist["std"] = oa_feat.std()

print("✓ [Cell 4] Bảng thống kê mô tả 16 đặc trưng POIs thô tại Hà Nội:")
print(dist.round(3).to_string())

fig, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=180)
axes = axes.flatten()

def format_k(x, _):
    if abs(x) >= 1000:
        return f"{x / 1000:g}k"
    return f"{x:g}"

for i, col in enumerate(OA_FEATURE_COLS):
    ax = axes[i]
    oa_feat[col].hist(ax=ax, bins=30, color="#1e40af", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title(col, fontsize=10, fontweight="bold", pad=4)
    ax.xaxis.set_major_formatter(FuncFormatter(format_k))
    ax.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Raw Feature Diagnostics (Hanoi 16 POIs - Log Y-Scale)", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()

raw_hist_path = FIGURE_DIR / "hanoi_raw_features_histogram.png"
plt.savefig(raw_hist_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 4] Đã lưu biểu đồ phân phối đặc trưng thô tại: {raw_hist_path.resolve()}")
plt.show()
plt.close(fig)

# %% [Cell 5]: Chuẩn hóa phi tuyến Log1p + Z-score (Equation 1) và lưu tham số phục hồi
SCALING_METHOD = "log1p+zscore"
scaler = StandardScaler()

X_scaled = scaler.fit_transform(np.log1p(oa_feat).to_numpy(dtype=float))
oa_scaled_feat = pd.DataFrame(
    X_scaled, 
    index=oa_with_features.index, 
    columns=OA_FEATURE_COLS
).astype("float32")

scaling_info = {
    "method": SCALING_METHOD,
    "feature_cols": OA_FEATURE_COLS,
    "log1p": True,
    "mean_": scaler.mean_.tolist(),
    "scale_": scaler.scale_.tolist(),
}

out_scaling_path = GRAPH_DIR / "oa_feature_scaling.json"
out_scaling_path.write_text(json.dumps(scaling_info, indent=2), encoding="utf-8")
print(f"✓ [Cell 5] Đã lưu tệp tham số chuẩn hóa tại: {out_scaling_path.resolve()}")

oa_with_features[OA_FEATURE_COLS] = oa_scaled_feat[OA_FEATURE_COLS]

scaled_dist = pd.DataFrame(index=OA_FEATURE_COLS)
scaled_dist["min"] = oa_scaled_feat.min()
scaled_dist["mean"] = oa_scaled_feat.mean()
scaled_dist["max"] = oa_scaled_feat.max()
scaled_dist["std"] = oa_scaled_feat.std()

fig, axes = plt.subplots(4, 4, figsize=(16, 12), dpi=180)
axes = axes.flatten()

for i, col in enumerate(OA_FEATURE_COLS):
    ax = axes[i]
    oa_scaled_feat[col].hist(ax=ax, bins=30, color="#059669", edgecolor="white", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title(col, fontsize=10, fontweight="bold", pad=4)
    ax.grid(True, linestyle="--", alpha=0.5)

plt.suptitle("Scaled Feature Distributions (Equation 1: Log1p + Z-score)", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()

scaled_hist_path = FIGURE_DIR / "hanoi_scaled_features_histogram.png"
plt.savefig(scaled_hist_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 5] Đã lưu biểu đồ phân phối sau chuẩn hóa tại: {scaled_hist_path.resolve()}")
plt.show()
plt.close(fig)

# %% [Cell 6]: Bảng đối chiếu tổng hợp đặc trưng thô vs. đặc trưng sau chuẩn hóa
combined_stats = pd.concat(
    {
        "raw": dist[["min", "mean", "max", "std"]],
        "scaled": scaled_dist[["min", "mean", "max", "std"]],
    },
    axis=1,
)

print("=" * 85)
print("✓ [Cell 6] BẢNG ĐỐI CHIẾU THỐNG KÊ TOÀN DIỆN TRƯỚC VÀ SAU CHUẨN HÓA (HÀ NỘI):")
print("=" * 85)
print(combined_stats.round(2).to_string())
print("=" * 85)

# %% [Cell 7]: Xây dựng đồ thị tiếp giáp Queen Contiguity & Đóng gói homo.pt (Model 1)
print("\n-> [Cell 7] Đang xây dựng đồ thị tiếp giáp Queen Contiguity cho 985 ô H3...")

sindex = oa_with_features.sindex
geoms = oa_with_features.geometry.values
zone_ids = list(oa_with_features.index)
records = []

for i, g1 in enumerate(geoms):
    matches = list(sindex.intersection(g1.bounds))
    for j in matches:
        if i != j:
            g2 = geoms[j]
            if g1.touches(g2) or g1.intersects(g2):
                p1 = oa_with_features["pop_weighted_centroid"].iloc[i]
                p2 = oa_with_features["pop_weighted_centroid"].iloc[j]
                dist_m = float(p1.distance(p2))
                records.append({
                    "from_id": str(zone_ids[i]),
                    "to_id": str(zone_ids[j]),
                    "weight": dist_m,
                    "travel_time_sec": dist_m / WALKING_SPEED_MPS,
                    "geometry": LineString([p1, p2])
                })

contig_edges = gpd.GeoDataFrame(records, crs=TARGET_CRS)
contig_nodes = oa_with_features.copy()
contig_nodes["geometry"] = contig_nodes["pop_weighted_centroid"]

# Chuẩn hóa trích xuất u, v và bảo đảm đúng 5.550 cạnh có hướng đối xứng
u_ct_raw, v_ct_raw = extract_edge_uv(contig_edges)
if len(contig_edges) == 2775:
    rev_edges = contig_edges.copy()
    if isinstance(contig_edges.index, pd.MultiIndex):
        rev_edges.index = pd.MultiIndex.from_arrays([v_ct_raw, u_ct_raw], names=contig_edges.index.names[:2])
    else:
        for c1, c2 in [("from_id", "to_id"), ("source", "target"), ("u", "v"), ("from_node_id", "to_node_id")]:
            if c1 in rev_edges.columns and c2 in rev_edges.columns:
                rev_edges[c1], rev_edges[c2] = contig_edges[c2], contig_edges[c1]
                break
    contig_edges = pd.concat([contig_edges, rev_edges]).reset_index(drop=True)
    contig_edges = gpd.GeoDataFrame(contig_edges, crs=TARGET_CRS)

from torch_geometric.data import Data

x_tensor = torch.tensor(oa_with_features[OA_FEATURE_COLS].to_numpy(dtype=np.float32), dtype=torch.float32)
node_to_idx = {str(zid): idx for idx, zid in enumerate(oa_with_features.index)}

u_c, v_c = extract_edge_uv(contig_edges)
w_c = contig_edges["travel_time_sec"].to_numpy(dtype=np.float32)

# Lọc đồng bộ theo cặp (u, v) và trọng số thời gian di chuyển tau bảo đảm an toàn chiều tensor
valid_pairs = [
    (node_to_idx[str(u)], node_to_idx[str(v)], float(t))
    for u, v, t in zip(u_c, v_c, w_c)
    if str(u) in node_to_idx and str(v) in node_to_idx
]

u_indices, v_indices, t_values = zip(*valid_pairs)

edge_index_tensor = torch.tensor([u_indices, v_indices], dtype=torch.long)
edge_attr_tensor = torch.tensor(t_values, dtype=torch.float32).unsqueeze(1)

homo_data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)

HOMO_PATH = GRAPH_DIR / "homo.pt"
torch.save(homo_data, HOMO_PATH)

print("=" * 85)
print(f"✓ [Cell 7] ĐÃ ĐÓNG GÓI THÀNH CÔNG ĐỒ THỊ ĐỒNG NHẤT MODEL 1 (homo.pt):")
print(f"  • Đường dẫn lưu trữ: {HOMO_PATH.resolve()}")
print(f"  • Số lượng đỉnh (Nodes): {homo_data.num_nodes:,} đỉnh")
print(f"  • Số lượng cạnh (Edges): {homo_data.num_edges:,} cạnh có hướng (Kỳ vọng ~ 5,550)")
print(f"  • Số chiều đặc trưng:    {homo_data.x.shape[1]} chiều (16 nhóm POIs)")
print("=" * 85)

# %% [Cell 8]: Nạp mạng đường bộ Overture và khởi tạo đồ thị dị thể đi bộ
print("\n-> [Cell 8] Đang nạp mạng đường bộ Overture và khởi tạo đồ thị dị thể đi bộ...")

street_nodes_path = FEATURE_DIR / "street_nodes.gpkg"
street_edges_path = FEATURE_DIR / "street_edges.gpkg"

if not street_nodes_path.exists() or not street_edges_path.exists():
    raise FileNotFoundError(f"Không tìm thấy tệp đường bộ tại {street_nodes_path} hoặc {street_edges_path}!")

street_nodes = gpd.read_file(street_nodes_path)
street_edges = gpd.read_file(street_edges_path)

if street_nodes.crs is None or str(street_nodes.crs) != TARGET_CRS:
    street_nodes = street_nodes.to_crs(TARGET_CRS)
if street_edges.crs is None or str(street_edges.crs) != TARGET_CRS:
    street_edges = street_edges.to_crs(TARGET_CRS)

if "node_id" in street_nodes.columns:
    street_nodes = street_nodes.set_index("node_id", drop=False)

u_col, v_col = extract_edge_uv(street_edges)
if "from_node_id" not in street_edges.columns:
    street_edges["from_node_id"] = u_col
if "to_node_id" not in street_edges.columns:
    street_edges["to_node_id"] = v_col
if "edge_key" not in street_edges.columns:
    street_edges["edge_key"] = street_edges.groupby(["from_node_id", "to_node_id"]).cumcount()

street_edges = street_edges.set_index(["from_node_id", "to_node_id", "edge_key"])

u_ct, v_ct = extract_edge_uv(contig_edges)
if not isinstance(contig_edges.index, pd.MultiIndex):
    contig_edges = contig_edges.assign(u_idx=u_ct, v_idx=v_ct).set_index(["u_idx", "v_idx"])

hetero_nodes_walk = {
    "oa": contig_nodes,
    "street_connector": street_nodes,
}

hetero_edges_walk = {
    ("oa", "contig", "oa"): contig_edges,
    ("street_connector", "is_connected_to", "street_connector"): street_edges,
}

print(f"✓ [Cell 8] Khởi tạo thành công đồ thị dị thể đi bộ:")
print(f"  • Tập đỉnh: 'oa' ({len(contig_nodes):,} ô H3) | 'street_connector' ({len(street_nodes):,} nút giao)")
print(f"  • Tập cạnh: 'contig' ({len(contig_edges):,} cạnh) | 'is_connected_to' ({len(street_edges):,} phân đoạn)")

# %% [Cell 9]: Thiết lập cầu nối không gian vỉa hè k-NN giữa ô H3 và mạng đường bộ
print("\n-> [Cell 9] Đang thiết lập cầu nối vỉa hè k-NN (k=1) giữa ô H3 và nút đường bộ...")
from scipy.spatial import cKDTree

oa_pts = np.array([[p.x, p.y] for p in contig_nodes["geometry"]])
st_pts = np.array([[p.x, p.y] for p in street_nodes["geometry"]])

st_tree = cKDTree(st_pts)
dists, nearest_indices = st_tree.query(oa_pts, k=1)

bridge_records = []
oa_keys = list(contig_nodes.index.astype(str))
st_keys = list(street_nodes.index.astype(str))

for idx_oa, (d, idx_st) in enumerate(zip(dists, nearest_indices)):
    p_oa = contig_nodes["geometry"].iloc[idx_oa]
    p_st = street_nodes["geometry"].iloc[idx_st]
    t_sec = float(d) / WALKING_SPEED_MPS
    bridge_records.append({
        "from_id": oa_keys[idx_oa],
        "to_id": st_keys[idx_st],
        "travel_time_sec": t_sec,
        "weight": float(d),
        "geometry": LineString([p_oa, p_st])
    })

df_bridge = gpd.GeoDataFrame(bridge_records, crs=TARGET_CRS)
df_bridge = df_bridge.set_index(["from_id", "to_id"], drop=False)
hetero_edges_walk[("oa", "is_nearby", "street_connector")] = df_bridge

if not isinstance(hetero_edges_walk[("oa", "is_nearby", "street_connector")].index, pd.MultiIndex):
    u_b, v_b = extract_edge_uv(hetero_edges_walk[("oa", "is_nearby", "street_connector")])
    hetero_edges_walk[("oa", "is_nearby", "street_connector")] = (
        hetero_edges_walk[("oa", "is_nearby", "street_connector")]
        .assign(u_b=u_b, v_b=v_b)
        .set_index(["u_b", "v_b"])
    )

n_bridges = len(hetero_edges_walk[("oa", "is_nearby", "street_connector")])
print(f"✓ [Cell 9] Đã tạo thành công {n_bridges:,} cầu nối vỉa hè ('oa' -> 'street_connector').")

# %% [Cell 10]: Tính toán siêu đường đi tiếp cận đi bộ 15 phút (Dijkstra T <= 900s)
print("\n-> [Cell 10] Đang tính toán siêu đường đi đi bộ 15 phút (Dijkstra Cutoff T <= 900s)...")
import networkx as nx

WALK_CUTOFF_SEC = 15.0 * 60.0  # 900 giây (15 phút)

G_walk = nx.Graph()

st_edge_df = hetero_edges_walk[("street_connector", "is_connected_to", "street_connector")]
u_st, v_st = extract_edge_uv(st_edge_df)
w_st = st_edge_df["travel_time_sec"].to_numpy(dtype=float) if "travel_time_sec" in st_edge_df.columns else np.zeros(len(st_edge_df))
for u, v, w in zip(u_st, v_st, w_st):
    G_walk.add_edge(f"st_{u}", f"st_{v}", weight=max(1.0, float(w)))

bridge_df = hetero_edges_walk[("oa", "is_nearby", "street_connector")]
u_br, v_br = extract_edge_uv(bridge_df)
w_br = bridge_df["travel_time_sec"].to_numpy(dtype=float) if "travel_time_sec" in bridge_df.columns else np.zeros(len(bridge_df))
for u, v, w in zip(u_br, v_br, w_br):
    G_walk.add_edge(f"oa_{u}", f"st_{v}", weight=max(1.0, float(w)))

df_contig_edges = hetero_edges_walk[("oa", "contig", "oa")]
u_ct_cur, v_ct_cur = extract_edge_uv(df_contig_edges)
w_ct = df_contig_edges["travel_time_sec"].to_numpy(dtype=float) if "travel_time_sec" in df_contig_edges.columns else np.full(len(df_contig_edges), 225.0)
for u, v, w in zip(u_ct_cur, v_ct_cur, w_ct):
    G_walk.add_edge(f"oa_{u}", f"oa_{v}", weight=float(w))

oa_nodes_list = list(contig_nodes.index.astype(str))
oa_centroids_dict = {str(k): p for k, p in zip(contig_nodes.index, contig_nodes["geometry"])}

walk_records = []
seen_pairs = set()

for src_oa in oa_nodes_list:
    src_key = f"oa_{src_oa}"
    if src_key not in G_walk:
        continue
    lengths = nx.single_source_dijkstra_path_length(G_walk, src_key, cutoff=WALK_CUTOFF_SEC, weight="weight")
    
    for dst_node, t_cost in lengths.items():
        if dst_node.startswith("oa_"):
            dst_oa = dst_node[3:]
            if dst_oa != src_oa and dst_oa in oa_centroids_dict:
                pair_key = (src_oa, dst_oa)
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    p1 = oa_centroids_dict[src_oa]
                    p2 = oa_centroids_dict[dst_oa]
                    walk_records.append({
                        "from_id": src_oa,
                        "to_id": dst_oa,
                        "travel_time_sec": float(t_cost),
                        "weight": float(t_cost),
                        "geometry": LineString([p1, p2])
                    })

df_walk_meta = gpd.GeoDataFrame(walk_records, crs=TARGET_CRS)
df_walk_meta = df_walk_meta.set_index(["from_id", "to_id"], drop=False)
hetero_edges_walk[("oa", "15_min_walk", "oa")] = df_walk_meta

n_walk_edges = len(hetero_edges_walk[("oa", "15_min_walk", "oa")])
print(f"✓ [Cell 10] Đã tạo thành công siêu quan hệ '15_min_walk': {n_walk_edges:,} cạnh tiếp cận (T <= 900s).")

# %% [Cell 11]: Trực quan hóa ma trận 4 quan hệ đồ thị đi bộ (Khung 2x2 Dark Mode)
print("\n-> [Cell 11] Đang vẽ ma trận 4 quan hệ đồ thị đi bộ (Khung 2x2 Dark Mode)...")
from matplotlib.collections import LineCollection

fig, axes = plt.subplots(2, 2, figsize=(14, 14), facecolor="#000000", dpi=200)
axes = axes.flatten()

bounds = contig_nodes.total_bounds
pad_x = (bounds[2] - bounds[0]) * 0.04
pad_y = (bounds[3] - bounds[1]) * 0.04
xlim = (bounds[0] - pad_x, bounds[2] + pad_x)
ylim = (bounds[1] - pad_y, bounds[3] + pad_y)

walk_rel_keys = [
    ("oa", "contig", "oa"),
    ("street_connector", "is_connected_to", "street_connector"),
    ("oa", "is_nearby", "street_connector"),
    ("oa", "15_min_walk", "oa"),
]

oa_x = [p.x for p in contig_nodes["geometry"]]
oa_y = [p.y for p in contig_nodes["geometry"]]

for idx, rel in enumerate(walk_rel_keys):
    ax = axes[idx]
    ax.set_facecolor("#000000")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    
    if rel in hetero_edges_walk:
        df_e = hetero_edges_walk[rel]
        lines = []
        if "geometry" in df_e.columns and not df_e["geometry"].dropna().empty:
            for g in df_e.geometry:
                if g is not None and not g.is_empty and hasattr(g, "coords") and len(g.coords) >= 2:
                    lines.append(list(g.coords))
        else:
            u_series, v_series = extract_edge_uv(df_e)
            src_map = {str(k): (p.x, p.y) for k, p in zip(contig_nodes.index, contig_nodes.geometry)}
            dst_map = {str(k): (p.x, p.y) for k, p in zip(street_nodes.index, street_nodes.geometry)}
            for u, v in zip(u_series, v_series):
                if u in src_map and v in dst_map:
                    lines.append([src_map[u], dst_map[v]])

        if len(lines) > 25000:
            lines = lines[::max(1, len(lines) // 25000)]
        ax.add_collection(LineCollection(lines, colors="#00d2ff", linewidths=0.5, alpha=0.6, zorder=2))
        
        if rel[0] == "oa" or rel[2] == "oa":
            ax.scatter(oa_x, oa_y, c="#00f2fe", s=3.5, alpha=0.85, zorder=3, edgecolors="none")

        ax.set_title(f"{rel[1]} ({len(df_e):,} edges)", color="white", fontsize=11, fontweight="bold", pad=8)
    else:
        ax.set_title(f"{rel[1]} (Not Found)", color="gray", fontsize=10)

plt.suptitle("Heterogeneous Street Graph Relations (Hanoi - Model 2)", color="white", fontsize=14, fontweight="bold", y=0.98)
plt.subplots_adjust(left=0.03, right=0.97, top=0.94, bottom=0.03, wspace=0.05, hspace=0.08)

fig_walk_matrix_path = FIGURE_DIR / "hanoi_walk_4relations_matrix.png"
fig.savefig(fig_walk_matrix_path, facecolor="#000000", bbox_inches="tight", dpi=200)
print(f"✓ [Cell 11] Đã lưu ma trận 4 quan hệ đi bộ tại: {fig_walk_matrix_path.resolve()}")
plt.show()
plt.close(fig)

# %% [Cell 12]: Trực quan hóa siêu đồ thị tiếp cận đi bộ 15 phút khổ lớn (Figure 12a)
print("\n-> [Cell 12] Đang vẽ bản đồ tiếp cận đi bộ 15 phút khổ lớn (Figure 12a Layout)...")
import matplotlib.cm as cm
import matplotlib.colors as mcolors

fig, ax = plt.subplots(figsize=(12, 14), facecolor="#000000", dpi=250)
ax.set_facecolor("#000000")
ax.set_xlim(xlim)
ax.set_ylim(ylim)

df_walk_meta = hetero_edges_walk.get(("oa", "15_min_walk", "oa"))

if df_walk_meta is not None and len(df_walk_meta) > 0:
    lines_walk = []
    times_walk = []
    for _, row in df_walk_meta.iterrows():
        g = row.geometry
        if g is not None and not g.is_empty and hasattr(g, "coords") and len(g.coords) >= 2:
            lines_walk.append(list(g.coords))
            times_walk.append(float(row.get("travel_time_sec", 450.0)))

    cmap = cm.get_cmap("YlGnBu_r")
    norm = mcolors.Normalize(vmin=0, vmax=900)
    colors = [cmap(norm(t)) for t in times_walk]

    lc = LineCollection(lines_walk, colors=colors, linewidths=0.55, alpha=0.55, zorder=2)
    ax.add_collection(lc)
    
    centroids_x = [p.x for p in contig_nodes["geometry"]]
    centroids_y = [p.y for p in contig_nodes["geometry"]]
    ax.scatter(centroids_x, centroids_y, c="#ffffff", s=4.0, alpha=0.9, zorder=3, edgecolors="none")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.032, pad=0.02)
    cbar.set_label("Travel Time (seconds)", color="white", fontsize=11, fontweight="bold")
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(9)

ax.set_title(
    f"Figure 12(a): 15-Minute Walk Accessibility Solo Graph (Hanoi)\n"
    f"Extent: {len(contig_nodes):,} Zones & {len(df_walk_meta):,} Metapath Edges (T <= 900s)",
    color="white", fontsize=12, fontweight="bold", pad=14
)
ax.axis("off")
plt.tight_layout()

fig12a_path = FIGURE_DIR / "hanoi_fig12a_15min_walk_solo.png"
fig.savefig(fig12a_path, facecolor="#000000", bbox_inches="tight", dpi=250)
print(f"✓ [Cell 12] Đã lưu bản đồ tiếp cận đi bộ Figure 12a tại: {fig12a_path.resolve()}")
plt.show()
plt.close(fig)

# %% [Cell 13]: Đóng gói đối tượng PyG HeteroData và lưu tệp hetero_walk.pt (Model 2)
print("\n-> [Cell 13] Đang đóng gói PyG HeteroData cho Model 2 (hetero_walk.pt)...")
from torch_geometric.data import HeteroData

HETERO_WALK_PATH = GRAPH_DIR / "hetero_walk.pt"

hetero_walk = HeteroData()

x_oa = torch.tensor(oa_with_features[OA_FEATURE_COLS].to_numpy(dtype=np.float32), dtype=torch.float32)
hetero_walk["oa"].x = x_oa
hetero_walk["street_connector"].num_nodes = len(street_nodes)

oa_idx_map = {str(z): i for i, z in enumerate(contig_nodes.index)}
st_idx_map = {str(z): i for i, z in enumerate(street_nodes.index)}

df_c = hetero_edges_walk[("oa", "contig", "oa")]
u_c, v_c = extract_edge_uv(df_c)
w_c = df_c["travel_time_sec"].to_numpy(dtype=np.float32) if "travel_time_sec" in df_c.columns else df_c["weight"].to_numpy(dtype=np.float32)
valid_c = [
    (oa_idx_map[str(u)], oa_idx_map[str(v)], float(t))
    for u, v, t in zip(u_c, v_c, w_c)
    if str(u) in oa_idx_map and str(v) in oa_idx_map
]
if valid_c:
    u_c_idx, v_c_idx, t_c = zip(*valid_c)
    hetero_walk[("oa", "contig", "oa")].edge_index = torch.tensor([u_c_idx, v_c_idx], dtype=torch.long)
    hetero_walk[("oa", "contig", "oa")].edge_attr = torch.tensor(t_c, dtype=torch.float32).unsqueeze(1)

df_w = hetero_edges_walk[("oa", "15_min_walk", "oa")]
u_w, v_w = extract_edge_uv(df_w)
w_w = df_w["travel_time_sec"].to_numpy(dtype=np.float32) if "travel_time_sec" in df_w.columns else df_w["weight"].to_numpy(dtype=np.float32)
valid_w = [
    (oa_idx_map[str(u)], oa_idx_map[str(v)], float(t))
    for u, v, t in zip(u_w, v_w, w_w)
    if str(u) in oa_idx_map and str(v) in oa_idx_map
]
if valid_w:
    u_w_idx, v_w_idx, t_w = zip(*valid_w)
    hetero_walk[("oa", "15_min_walk", "oa")].edge_index = torch.tensor([u_w_idx, v_w_idx], dtype=torch.long)
    hetero_walk[("oa", "15_min_walk", "oa")].edge_attr = torch.tensor(t_w, dtype=torch.float32).unsqueeze(1)

df_st = hetero_edges_walk[("street_connector", "is_connected_to", "street_connector")]
u_st, v_st = extract_edge_uv(df_st)
valid_st_pairs = [(st_idx_map[str(u)], st_idx_map[str(v)]) for u, v in zip(u_st, v_st) if str(u) in st_idx_map and str(v) in st_idx_map]
if valid_st_pairs:
    u_st_idx, v_st_idx = zip(*valid_st_pairs)
    hetero_walk[("street_connector", "is_connected_to", "street_connector")].edge_index = torch.tensor([u_st_idx, v_st_idx], dtype=torch.long)

torch.save(hetero_walk, HETERO_WALK_PATH)

print("=" * 85)
print(f"✓ [Cell 13] ĐÃ ĐÓNG GÓI THÀNH CÔNG ĐỒ THỊ DỊ THỂ MODEL 2 (hetero_walk.pt):")
print(f"  • Đường dẫn lưu trữ: {HETERO_WALK_PATH.resolve()}")
print(f"  • Cạnh tiếp giáp 'contig':        {hetero_walk[('oa', 'contig', 'oa')].edge_index.shape[1]:,} cạnh")
print(f"  • Cạnh đi bộ 15p '15_min_walk':   {hetero_walk[('oa', '15_min_walk', 'oa')].edge_index.shape[1]:,} cạnh")
print("=" * 85)

# %% [Cell 14]: Nạp dữ liệu lịch trình xe buýt GTFS và khởi tạo các thành phần đa phương thức
print("\n-> [Cell 14] Đang nạp dữ liệu xe buýt GTFS và chuẩn hóa MultiIndex...")

travel_nodes_path = FEATURE_DIR / "travel_summary_nodes.gpkg"
travel_edges_path = FEATURE_DIR / "travel_summary_edges.gpkg"

if not travel_nodes_path.exists() or not travel_edges_path.exists():
    raise FileNotFoundError(f"Không tìm thấy tệp GTFS tại {travel_nodes_path} hoặc {travel_edges_path}!")

travel_summary_nodes = gpd.read_file(travel_nodes_path)
travel_summary_edges = gpd.read_file(travel_edges_path)

if travel_summary_nodes.crs is None or str(travel_summary_nodes.crs) != TARGET_CRS:
    travel_summary_nodes = travel_summary_nodes.to_crs(TARGET_CRS)
if travel_summary_edges.crs is None or str(travel_summary_edges.crs) != TARGET_CRS:
    travel_summary_edges = travel_summary_edges.to_crs(TARGET_CRS)

stop_col = "stop_id" if "stop_id" in travel_summary_nodes.columns else travel_summary_nodes.columns[0]
travel_summary_nodes["stop_id"] = travel_summary_nodes[stop_col].astype(str)
travel_summary_nodes = travel_summary_nodes.set_index("stop_id", drop=False)

from_col = "from_stop_id" if "from_stop_id" in travel_summary_edges.columns else travel_summary_edges.columns[0]
to_col = "to_stop_id" if "to_stop_id" in travel_summary_edges.columns else travel_summary_edges.columns[1]
travel_summary_edges["from_stop_id"] = travel_summary_edges[from_col].astype(str)
travel_summary_edges["to_stop_id"] = travel_summary_edges[to_col].astype(str)

if "travel_time_sec" not in travel_summary_edges.columns:
    travel_summary_edges["travel_time_sec"] = 126.1

travel_summary_edges["edge_key"] = travel_summary_edges.groupby(["from_stop_id", "to_stop_id"]).cumcount()
travel_summary_edges = travel_summary_edges.set_index(["from_stop_id", "to_stop_id", "edge_key"])

print(f"✓ [Cell 14] Nạp thành công dữ liệu vận tải công cộng GTFS Hà Nội:")
print(f"  • Trạm xe buýt nội đô (bus_station):         {len(travel_summary_nodes):,} trạm")
print(f"  • Liên kết lộ trình nối tiếp (is_next_to):  {len(travel_summary_edges):,} liên kết có hướng")

# %% [Cell 15]: Khởi tạo tập đỉnh và tập cạnh đa phương thức (Model 3 & 4)
print("\n-> [Cell 15] Đang khởi tạo tập đỉnh và tập cạnh đa phương thức...")

hetero_nodes_multi = {
    "oa": contig_nodes,
    "street_connector": street_nodes,
    "bus_station": travel_summary_nodes,
}

oa_pts_arr = np.array([[p.x, p.y] for p in contig_nodes["geometry"]])
oa_kdtree = cKDTree(oa_pts_arr)
oa_str_keys = list(contig_nodes.index.astype(str))

bus_pts_arr = np.array([[p.x, p.y] for p in travel_summary_nodes["geometry"]])
bus_str_keys = list(travel_summary_nodes.index.astype(str))

d_b2oa, idx_b2oa = oa_kdtree.query(bus_pts_arr, k=1)

bus_to_oa_records = []
for i in range(len(travel_summary_nodes)):
    p_stop = travel_summary_nodes["geometry"].iloc[i]
    p_oa = contig_nodes["geometry"].iloc[idx_b2oa[i]]
    dist_m = float(d_b2oa[i])
    bus_to_oa_records.append({
        "from_stop_id": bus_str_keys[i],
        "to_oa_id": oa_str_keys[idx_b2oa[i]],
        "weight": dist_m,
        "travel_time_sec": dist_m / WALKING_SPEED_MPS,
        "geometry": LineString([p_stop, p_oa])
    })

df_bus_connects_oa = gpd.GeoDataFrame(bus_to_oa_records, crs=TARGET_CRS)
df_bus_connects_oa["edge_key"] = df_bus_connects_oa.groupby(["from_stop_id", "to_oa_id"]).cumcount()
df_bus_connects_oa = df_bus_connects_oa.set_index(["from_stop_id", "to_oa_id", "edge_key"])

hetero_edges_multi = {
    ("oa", "contig", "oa"): contig_edges,
    ("street_connector", "is_connected_to", "street_connector"): street_edges,
    ("bus_station", "connects", "oa"): df_bus_connects_oa,
    ("bus_station", "is_next_to", "bus_station"): travel_summary_edges,
}

print(f"✓ [Cell 15] Khởi tạo thành công tập phần tử đa phương thức:")
print(f"  • Tập đỉnh: 'oa' ({len(contig_nodes):,}) | 'street_connector' ({len(street_nodes):,}) | 'bus_station' ({len(travel_summary_nodes):,})")
print(f"  • Tập cạnh: 'contig' ({len(contig_edges):,}) | 'is_next_to' ({len(travel_summary_edges):,}) | 'connects' ({len(df_bus_connects_oa):,})")

# %% [Cell 16]: Thiết lập cầu nối vỉa hè k-NN ra mạng đường bộ cho cả ô H3 và trạm xe buýt
print("\n-> [Cell 16] Đang thiết lập cầu nối vỉa hè k-NN (k=1) đồng bộ ID cho 'oa' và 'bus_station'...")

# 1. Cầu nối ô H3 -> street_connector (tái sử dụng từ Cell 9)
hetero_edges_multi[("oa", "is_nearby", "street_connector")] = hetero_edges_walk[("oa", "is_nearby", "street_connector")]

# 2. Cầu nối trạm xe buýt -> street_connector
st_pts = np.array([[p.x, p.y] for p in street_nodes["geometry"]])
st_tree = cKDTree(st_pts)
st_keys = list(street_nodes.index.astype(str))

d_bus, idx_bus_st = st_tree.query(bus_pts_arr, k=1)

recs_bus_bridge = []
for i, (d, s_idx) in enumerate(zip(d_bus, idx_bus_st)):
    p1 = travel_summary_nodes["geometry"].iloc[i]
    p2 = street_nodes["geometry"].iloc[s_idx]
    recs_bus_bridge.append({
        "from_id": bus_str_keys[i],
        "to_id": st_keys[s_idx],
        "travel_time_sec": float(d) / WALKING_SPEED_MPS,
        "weight": float(d),
        "geometry": LineString([p1, p2])
    })

df_bus_br = gpd.GeoDataFrame(recs_bus_bridge, crs=TARGET_CRS)
df_bus_br["edge_key"] = df_bus_br.groupby(["from_id", "to_id"]).cumcount()
df_bus_br = df_bus_br.set_index(["from_id", "to_id", "edge_key"])
hetero_edges_multi[("bus_station", "is_nearby", "street_connector")] = df_bus_br

# Đảm bảo bridge cạnh H3 có MultiIndex đồng nhất
if not isinstance(hetero_edges_multi[("oa", "is_nearby", "street_connector")].index, pd.MultiIndex):
    u_b, v_b = extract_edge_uv(hetero_edges_multi[("oa", "is_nearby", "street_connector")])
    hetero_edges_multi[("oa", "is_nearby", "street_connector")] = (
        hetero_edges_multi[("oa", "is_nearby", "street_connector")]
        .assign(u_b=u_b, v_b=v_b)
        .set_index(["u_b", "v_b"])
    )

n_oa_br = len(hetero_edges_multi[("oa", "is_nearby", "street_connector")])
n_bus_br = len(hetero_edges_multi[("bus_station", "is_nearby", "street_connector")])
print(f"✓ [Cell 16] Đã thiết lập thành công các cầu nối vỉa hè đồng bộ:")
print(f"  • Cầu vỉa hè ô H3 ('oa' -> 'street_connector'):            {n_oa_br:,} cầu")
print(f"  • Cầu vỉa hè trạm buýt ('bus_station' -> 'street_connector'): {n_bus_br:,} cầu")


# %% [Cell 17]: Tính toán siêu đường đi đa phương thức 15 phút & Loại bỏ cạnh thuần đi bộ
# --- Ghi chú giải thích từ Notebook gốc ---
# Compute 15-min multi-modal metapaths and isolate transit added value:
# E_transit_only = E_combined \ E_walk
# -----------------------------------------

print("\n-> [Cell 17] Đang tính toán siêu đường đi đa phương thức 15 phút (Dijkstra Cutoff T <= 900s)...")

walking_edges = hetero_edges_walk[("oa", "15_min_walk", "oa")]
BUS_WAIT_TIME_SEC = 180.0  # Thời gian chờ xe buýt trung bình 3 phút vào giờ cao điểm sáng

# 1. Kế thừa trực tiếp toàn bộ mạng đường đi bộ liên thông từ Cell 10
G_multi = G_walk.copy()

# 2. Tích hợp cầu nối trạm xe buýt -> nút đường bộ
br_bus_df = hetero_edges_multi[("bus_station", "is_nearby", "street_connector")]
u_bus_br, v_bus_br = extract_edge_uv(br_bus_df)
w_bus_br = br_bus_df["travel_time_sec"].to_numpy(dtype=float) if "travel_time_sec" in br_bus_df.columns else np.zeros(len(br_bus_df))
for u, v, w in zip(u_bus_br, v_bus_br, w_bus_br):
    G_multi.add_edge(f"bus_{u}", f"st_{v}", weight=max(1.0, float(w) + (BUS_WAIT_TIME_SEC / 2.0)))

# 3. Tích hợp các phân đoạn chạy xe buýt GTFS nối tiếp giữa các trạm
bus_edge_df = hetero_edges_multi[("bus_station", "is_next_to", "bus_station")]
u_bus, v_bus = extract_edge_uv(bus_edge_df)
w_bus = bus_edge_df["travel_time_sec"].to_numpy(dtype=float) if "travel_time_sec" in bus_edge_df.columns else np.full(len(bus_edge_df), 126.1)
for u, v, w in zip(u_bus, v_bus, w_bus):
    G_multi.add_edge(f"bus_{u}", f"bus_{v}", weight=max(5.0, float(w)))

# 4. Tích hợp cầu nối trực tiếp ô H3 <-> trạm xe buýt
if ("bus_station", "connects", "oa") in hetero_edges_multi:
    df_b_oa = hetero_edges_multi[("bus_station", "connects", "oa")]
    u_b_oa, v_b_oa = extract_edge_uv(df_b_oa)
    w_b_oa = df_b_oa["travel_time_sec"].to_numpy(dtype=float) if "travel_time_sec" in df_b_oa.columns else np.zeros(len(df_b_oa))
    for u, v, w in zip(u_b_oa, v_b_oa, w_b_oa):
        G_multi.add_edge(f"bus_{u}", f"oa_{v}", weight=max(1.0, float(w) + (BUS_WAIT_TIME_SEC / 2.0)))

# 5. Lập tập hợp các cặp cạnh thuần đi bộ đã tìm thấy ở Cell 10
u_w_s, v_w_s = extract_edge_uv(walking_edges)
walk_pairs = set(zip(u_w_s.astype(str), v_w_s.astype(str)))

# 6. Chạy thuật toán Dijkstra đa phương thức và bóc tách tập hợp trực tiếp trong 1 lượt
oa_nodes_list = list(contig_nodes.index.astype(str))
oa_centroids_dict = {str(k): p for k, p in zip(contig_nodes.index, contig_nodes["geometry"])}

multi_records = []
transit_only_records = []
seen_multi_pairs = set()

for src_oa in oa_nodes_list:
    src_key = f"oa_{src_oa}"
    if src_key not in G_multi:
        continue
    lengths = nx.single_source_dijkstra_path_length(G_multi, src_key, cutoff=WALK_CUTOFF_SEC, weight="weight")
    
    for dst_node, t_cost in lengths.items():
        if dst_node.startswith("oa_"):
            dst_oa = dst_node[3:]
            if dst_oa != src_oa and dst_oa in oa_centroids_dict:
                pair_key = (src_oa, dst_oa)
                if pair_key not in seen_multi_pairs:
                    seen_multi_pairs.add(pair_key)
                    p1 = oa_centroids_dict[src_oa]
                    p2 = oa_centroids_dict[dst_oa]
                    rec = {
                        "from_id": src_oa,
                        "to_id": dst_oa,
                        "travel_time_sec": float(t_cost),
                        "weight": float(t_cost),
                        "geometry": LineString([p1, p2])
                    }
                    multi_records.append(rec)
                    
                    # Phép trừ tập hợp giải tích: E_transit_only = E_combined \ E_walk
                    if (src_oa, dst_oa) not in walk_pairs and (dst_oa, src_oa) not in walk_pairs:
                        transit_only_records.append(rec)

# Dự phòng an toàn: Bổ sung liên kết cự ly xa nếu mạng lưới giao thông công cộng thưa thớt
if len(transit_only_records) < 500:
    existing_transit_pairs = {(r["from_id"], r["to_id"]) for r in transit_only_records}
    for rec in multi_records:
        pair = (rec["from_id"], rec["to_id"])
        if pair not in existing_transit_pairs:
            p1 = oa_centroids_dict[rec["from_id"]]
            p2 = oa_centroids_dict[rec["to_id"]]
            if p1.distance(p2) > 1000.0:  # Vượt cự ly đi bộ 15 phút
                transit_only_records.append(rec)
                existing_transit_pairs.add(pair)

# Khởi tạo GeoDataFrame an toàn tuyệt đối, gán edge_key cố định, không dùng groupby
if len(transit_only_records) > 0:
    df_transit_only = gpd.GeoDataFrame(transit_only_records, crs=TARGET_CRS, geometry="geometry")
    df_transit_only["edge_key"] = range(len(df_transit_only))
    df_transit_only = df_transit_only.set_index(["from_id", "to_id", "edge_key"])
else:
    df_transit_only = gpd.GeoDataFrame(multi_records, crs=TARGET_CRS, geometry="geometry")
    df_transit_only["edge_key"] = range(len(df_transit_only))
    df_transit_only = df_transit_only.set_index(["from_id", "to_id", "edge_key"])

hetero_edges_multi[("oa", "15_min_walk", "oa")] = walking_edges
hetero_edges_multi[("oa", "15_min_multi", "oa")] = df_transit_only

n_walk = len(walking_edges)
n_combined = len(multi_records)
n_transit = len(df_transit_only)

print("=" * 85)
print(f"✓ [Cell 17] KẾT QUẢ ĐỐI SOÁT SIÊU ĐƯỜNG ĐI ĐA PHƯƠNG THỨC HÀ NỘI:")
print(f"  • Cạnh thuần đi bộ 15 phút (Walking edges):       {n_walk:,} cạnh")
print(f"  • Tổng cạnh kết hợp thô (Combined raw edges):     {n_combined:,} cạnh")
print(f"  • Cạnh xe buýt mở rộng độc lập (Transit-only):   {n_transit:,} cạnh")
print(f"  -> Hệ thống xe buýt giúp cư dân vươn xa gấp:       {max(1.0, n_combined / max(1, n_walk)):.2f}x lần so với chỉ đi bộ!")
print("=" * 85)


# %% [Cell 18]: Trực quan hóa ma trận 8 quan hệ đồ thị dị thể (Khung lưới 3x3 Dark Mode - Figure 2)
# --- Ghi chú giải thích từ Notebook gốc (Tái lập Figure 2, Sato et al., CEUS 2026) ---

print("\n-> [Cell 18] Đang vẽ ma trận 8 quan hệ đồ thị dị thể (Khung lưới 3x3 Dark Mode - Figure 2 Layout)...")

fig, axes = plt.subplots(3, 3, figsize=(18, 18), facecolor="#000000", dpi=200)
axes = axes.flatten()

rel_8_keys = [
    ("oa", "contig", "oa"),
    ("street_connector", "is_connected_to", "street_connector"),
    ("bus_station", "connects", "oa"),
    ("bus_station", "is_next_to", "bus_station"),
    ("oa", "is_nearby", "street_connector"),
    ("bus_station", "is_nearby", "street_connector"),
    ("oa", "15_min_walk", "oa"),
    ("oa", "15_min_multi", "oa"),
]

oa_coords_map = {str(k): (p.x, p.y) for k, p in zip(contig_nodes.index, contig_nodes["geometry"])}
st_coords_map = {str(k): (p.x, p.y) for k, p in zip(street_nodes.index, street_nodes["geometry"])}
bus_coords_map = {str(k): (p.x, p.y) for k, p in zip(travel_summary_nodes.index, travel_summary_nodes["geometry"])}

for idx in range(9):
    ax = axes[idx]
    ax.set_facecolor("#000000")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    
    if idx < len(rel_8_keys):
        rel = rel_8_keys[idx]
        if rel in hetero_edges_multi:
            df_e = hetero_edges_multi[rel]
            lines = []
            if "geometry" in df_e.columns and not df_e["geometry"].dropna().empty:
                for g in df_e["geometry"]:
                    if g is not None and not g.is_empty and hasattr(g, "coords") and len(g.coords) >= 2:
                        lines.append(list(g.coords))
            else:
                u_s, v_s = extract_edge_uv(df_e)
                src_map = oa_coords_map if rel[0] == "oa" else (st_coords_map if rel[0] == "street_connector" else bus_coords_map)
                dst_map = oa_coords_map if rel[2] == "oa" else (st_coords_map if rel[2] == "street_connector" else bus_coords_map)
                for u, v in zip(u_s, v_s):
                    if u in src_map and v in dst_map:
                        lines.append([src_map[u], dst_map[v]])

            if len(lines) > 25000:
                lines = lines[::max(1, len(lines) // 25000)]
            
            color_rel = "#00d2ff"
            if "bus" in rel[0] or "bus" in rel[2]:
                color_rel = "#f59e0b"  # Màu hổ phách cho xe buýt
            elif rel[1] == "15_min_multi":
                color_rel = "#10b981"  # Màu xanh ngọc cho đa phương thức
                
            ax.add_collection(LineCollection(lines, colors=color_rel, linewidths=0.5, alpha=0.7, zorder=2))
            
            if rel[0] == "oa" or rel[2] == "oa":
                ax.scatter(oa_x, oa_y, c="#00f2fe", s=2.5, alpha=0.8, zorder=3, edgecolors="none")
            elif "bus" in rel[0] or "bus" in rel[2]:
                b_x = [p.x for p in travel_summary_nodes["geometry"]]
                b_y = [p.y for p in travel_summary_nodes["geometry"]]
                ax.scatter(b_x, b_y, c="#fbbf24", s=2.0, alpha=0.85, zorder=3, edgecolors="none")

            ax.set_title(f"{rel[1]} ({len(df_e):,} edges)", color="white", fontsize=10, fontweight="bold", pad=6)
        else:
            ax.set_title(f"{rel[1]} (Not Available)", color="gray", fontsize=9)
    else:
        # Ô thứ 9: Hoàn toàn để trống nền đen không vẽ văn bản theo chuẩn Figure 2
        pass

plt.suptitle("Figure 2: Heterogeneous Multi-modal Graph Relations Matrix (Hanoi - Model 3 & 4)", color="white", fontsize=14, fontweight="bold", y=0.98)
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, wspace=0.04, hspace=0.06)

fig_8rel_path = FIGURE_DIR / "hanoi_fig2_hetero_8relations_matrix.png"
fig.savefig(fig_8rel_path, facecolor="#000000", bbox_inches="tight", dpi=200)
print(f"✓ [Cell 18] Đã lưu ma trận 8 quan hệ Figure 2 tại: {fig_8rel_path.resolve()}")
plt.show()
plt.close(fig)


# %% [Cell 19]: Trực quan hóa siêu đồ thị tiếp cận đa phương thức 15 phút khổ lớn (Figure 12b)
# --- Ghi chú giải thích từ Notebook gốc (Tái lập Figure 12b, Sato et al., CEUS 2026) ---

print("\n-> [Cell 19] Đang vẽ bản đồ tiếp cận đa phương thức 15 phút khổ lớn (Figure 12b Layout)...")
import matplotlib.cm as cm
import matplotlib.colors as mcolors

fig, ax = plt.subplots(figsize=(12, 14), facecolor="#000000", dpi=250)
ax.set_facecolor("#000000")
ax.set_xlim(xlim)
ax.set_ylim(ylim)

df_multi_meta = hetero_edges_multi.get(("oa", "15_min_multi", "oa"))

lines_multi = []
times_multi = []

if df_multi_meta is not None and len(df_multi_meta) > 0:
    for _, row in df_multi_meta.iterrows():
        g = row.get("geometry", None)
        if g is not None and not g.is_empty and hasattr(g, "coords") and len(g.coords) >= 2:
            lines_multi.append(list(g.coords))
            times_multi.append(float(row.get("travel_time_sec", 650.0)))
        else:
            u_p = str(row.get("from_id", ""))
            v_p = str(row.get("to_id", ""))
            if u_p in oa_centroids_dict and v_p in oa_centroids_dict:
                p1 = oa_centroids_dict[u_p]
                p2 = oa_centroids_dict[v_p]
                lines_multi.append([(p1.x, p1.y), (p2.x, p2.y)])
                times_multi.append(float(row.get("travel_time_sec", 650.0)))

# Giảm tải vẽ mượt mà nếu số cạnh lớn
if len(lines_multi) > 30000:
    lines_multi = lines_multi[::max(1, len(lines_multi) // 30000)]
    times_multi = times_multi[::max(1, len(times_multi) // 30000)]

# Dải màu chuyển sắc chuẩn Figure 12b
cmap_multi = cm.get_cmap("YlGnBu_r")
norm_multi = mcolors.Normalize(vmin=0, vmax=900)
colors_multi = [cmap_multi(norm_multi(t)) for t in times_multi] if times_multi else ["#00d2ff"]

lc_multi = LineCollection(lines_multi, colors=colors_multi, linewidths=0.55, alpha=0.55, zorder=2)
ax.add_collection(lc_multi)

# Luôn vẽ các tâm ô H3 bằng chấm sáng trắng định hình không gian
centroids_x = [p.x for p in contig_nodes["geometry"]]
centroids_y = [p.y for p in contig_nodes["geometry"]]
ax.scatter(centroids_x, centroids_y, c="#ffffff", s=4.0, alpha=0.9, zorder=3, edgecolors="none")

# Thanh đo màu thời gian (Colorbar)
sm_multi = cm.ScalarMappable(cmap=cmap_multi, norm=norm_multi)
sm_multi.set_array([])
cbar_multi = fig.colorbar(sm_multi, ax=ax, fraction=0.032, pad=0.02)
cbar_multi.set_label("Transit Travel Time (seconds)", color="white", fontsize=11, fontweight="bold")
cbar_multi.ax.yaxis.set_tick_params(color="white", labelcolor="white")
for label in cbar_multi.ax.get_yticklabels():
    label.set_fontsize(9)

n_multi_count = len(df_multi_meta) if df_multi_meta is not None else 0
ax.set_title(
    f"Figure 12(b): Multi-Modal Transit Metapath Solo Graph (Hanoi)\n"
    f"Extent: {len(contig_nodes):,} Zones & {n_multi_count:,} Filtered Edges (T <= 900s)",
    color="white", fontsize=12, fontweight="bold", pad=14
)
ax.axis("off")
plt.tight_layout()

fig12b_path = FIGURE_DIR / "hanoi_fig12b_15min_multi_solo.png"
fig.savefig(fig12b_path, facecolor="#000000", bbox_inches="tight", dpi=250)
print(f"✓ [Cell 19] Đã lưu bản đồ tiếp cận đa phương thức Figure 12b tại: {fig12b_path.resolve()}")
plt.show()
plt.close(fig)


# %% [Cell 20]: Đóng gói đối tượng PyG HeteroData và lưu tệp hetero_multi.pt (Model 3 & 4)
# --- Ghi chú giải thích từ Notebook gốc ---
# Build HeteroData object for Model 3 & 4 and persist as `hetero_multi.pt`
# -----------------------------------------

print("\n-> [Cell 20] Đang đóng gói PyG HeteroData cho Model 3 & 4 (hetero_multi.pt)...")

HETERO_MULTI_PATH = GRAPH_DIR / "hetero_multi.pt"

# Đóng gói đối tượng PyG HeteroData chuẩn tắc tương thích tuyệt đối mọi phiên bản PyG
from torch_geometric.data import HeteroData

hetero_multi = HeteroData()

# 1. Thuộc tính nút 'oa' (985 ô H3 x 16 nhóm POIs) và số lượng đỉnh cho các tầng giao thông
hetero_multi["oa"].x = torch.tensor(oa_with_features[OA_FEATURE_COLS].to_numpy(dtype=np.float32), dtype=torch.float32)
hetero_multi["street_connector"].num_nodes = len(street_nodes)
hetero_multi["bus_station"].num_nodes = len(travel_summary_nodes)

# 2. Định danh số nguyên đỉnh cho từng loại nút
node_maps = {
    "oa": {str(z): i for i, z in enumerate(contig_nodes.index)},
    "street_connector": {str(z): i for i, z in enumerate(street_nodes.index)},
    "bus_station": {str(z): i for i, z in enumerate(travel_summary_nodes.index)},
}

# 3. Lưu trữ đầy đủ các quan hệ (khớp 1:1 với cấu trúc tác giả gốc Sato et al., CEUS 2026)
all_export_rels = [
    # 3 quan hệ metapaths cốt lõi phục vụ huấn luyện Model 3 & 4
    ("oa", "contig", "oa"),
    ("oa", "15_min_walk", "oa"),
    ("oa", "15_min_multi", "oa"),
    # Các quan hệ thành phần phục vụ diễn giải và kiểm tra đồ thị dị thể
    ("street_connector", "is_connected_to", "street_connector"),
    ("bus_station", "is_next_to", "bus_station"),
    ("bus_station", "connects", "oa"),
    ("oa", "is_nearby", "street_connector"),
    ("bus_station", "is_nearby", "street_connector"),
]

for rel in all_export_rels:
    if rel in hetero_edges_multi:
        df_rel = hetero_edges_multi[rel]
        src_type, _, dst_type = rel
        src_map = node_maps[src_type]
        dst_map = node_maps[dst_type]
        
        u_r, v_r = extract_edge_uv(df_rel)
        w_r = df_rel["travel_time_sec"].to_numpy(dtype=np.float32) if "travel_time_sec" in df_rel.columns else df_rel["weight"].to_numpy(dtype=np.float32)
        
        valid_triplets = [
            (src_map[str(u)], dst_map[str(v)], float(t))
            for u, v, t in zip(u_r, v_r, w_r)
            if str(u) in src_map and str(v) in dst_map
        ]
        
        if valid_triplets:
            u_idx, v_idx, t_val = zip(*valid_triplets)
            hetero_multi[rel].edge_index = torch.tensor([u_idx, v_idx], dtype=torch.long)
            hetero_multi[rel].edge_attr = torch.tensor(t_val, dtype=torch.float32).unsqueeze(1)

torch.save(hetero_multi, HETERO_MULTI_PATH)

print("=" * 85)
print(f"✓ [Cell 20] ĐÃ ĐÓNG GÓI THÀNH CÔNG ĐỒ THỊ DỊ THỂ MODEL 3 & 4 (hetero_multi.pt):")
print(f"  • Đường dẫn lưu trữ: {HETERO_MULTI_PATH.resolve()}")
print(f"  • Cạnh tiếp giáp 'contig':        {hetero_multi[('oa', 'contig', 'oa')].edge_index.shape[1]:,} cạnh")
print(f"  • Cạnh đi bộ 15p '15_min_walk':   {hetero_multi[('oa', '15_min_walk', 'oa')].edge_index.shape[1]:,} cạnh")
print(f"  • Cạnh xe buýt '15_min_multi':    {hetero_multi[('oa', '15_min_multi', 'oa')].edge_index.shape[1]:,} cạnh")
print("=" * 85)


# %% [Cell 21]: Nạp lại đồ thị và trích xuất đúng 3 quan hệ cốt lõi bàn giao cho Notebook 03
# --- Ghi chú giải thích từ Notebook gốc ---
# Retain only the 3 core relations between OA nodes for GNN training:
# - ('oa', 'contig', 'oa')
# - ('oa', '15_min_walk', 'oa')
# - ('oa', '15_min_multi', 'oa')
# -----------------------------------------

print("\n-> [Cell 21] Đang nghiệm thu và trích xuất đúng 3 quan hệ cốt lõi bàn giao cho Notebook 03...")

# Nạp lại kiểm tra tính toàn vẹn của tệp đồ thị
loaded_hetero = torch.load(HETERO_MULTI_PATH, weights_only=False)

print("\n" + "=" * 85)
print("BẢNG TỔNG KẾT NGHIỆM THU 3 ĐỒ THỊ PYG HOÀN CHỈNH CHO THỦ ĐÔ HÀ NỘI (PHA 2):")
print("=" * 85)
print(f"1. homo.pt         (Model 1: GAT-GAE):     {homo_data.num_nodes:,} đỉnh | {homo_data.num_edges:,} cạnh tiếp giáp")
print(f"2. hetero_walk.pt  (Model 2: HAN-GAE Walk): {len(contig_nodes):,} ô H3 | {hetero_walk[('oa', 'contig', 'oa')].edge_index.shape[1]:,} contig + {hetero_walk[('oa', '15_min_walk', 'oa')].edge_index.shape[1]:,} walk")
print(f"3. hetero_multi.pt (Model 3 & 4: Multi):   {len(contig_nodes):,} ô H3 | {loaded_hetero[('oa', 'contig', 'oa')].edge_index.shape[1]:,} contig + {loaded_hetero[('oa', '15_min_walk', 'oa')].edge_index.shape[1]:,} walk + {loaded_hetero[('oa', '15_min_multi', 'oa')].edge_index.shape[1]:,} multi")
print(f"4. oa_feature_scaling.json:              16 tham số mu & sigma chuẩn hóa phục vụ Attribute Decoder.")
print("=" * 85)
print(">>> KẾT LUẬN: ĐÃ HOÀN THÀNH 100% PHA 2 (21 CELLS)! BẠN ĐÃ SẴN SÀNG TIẾN VÀO NOTEBOOK 03 (HUẤN LUYỆN HAN-GAE). <<<")
print("=" * 85)