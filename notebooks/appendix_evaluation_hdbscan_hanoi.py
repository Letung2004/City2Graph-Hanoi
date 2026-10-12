# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH KIỂM ĐỊNH ĐỘ BỀN VỮNG PHÂN CỤM HDBSCAN HÀ NỘI (ĐẦY ĐỦ 10 CELLS)
Kế thừa & Chuyển đổi 1:1 từ: appendix_evaluation_hdbscan.py (Sato et al., CEUS 2026)
Đối tượng khảo sát: 985 ô lục giác Uber H3 Resolution 9 tại Thủ đô Hà Nội
Thuật toán: HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
Tối ưu hóa siêu tham số: Quét min_cluster_size (MCS) dựa trên chỉ số DBCV (Density-Based Clustering Validation)
Định dạng hỗ trợ: VS Code Interactive Window (# %%) hoặc python dòng lệnh
"""

from __future__ import annotations

# %% [Cell 1]: Thiết lập môi trường thực thi và tự động định vị thư mục gốc
# --- Ghi chú giải thích từ Notebook gốc ---
# # Appendix. HDBSCAN Robustness Check (Hanoi Case Study)
#
# This notebook evaluates the trained GAT-GAE/HAN-GAE models using HDBSCAN
# to ensure the clustering results are robust and not merely artifacts of K-Means.
# -----------------------------------------

import os
import sys
import random
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

print("=" * 95)
print(f"✓ [Cell 1] Đã định vị thư mục gốc dự án: {REPO_ROOT}")
print(f"✓ [Cell 1] Đã kết nối module mã nguồn lõi: {SRC_DIR}")
print("=" * 95)


# %% [Cell 2]: Khai báo thư viện, tham số mô hình và cấu hình đường dẫn Hà Nội
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 1. Setup & Configuration
# Import dependencies, define paths, set random seed, and specify model metadata.
# -----------------------------------------

import yaml
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import seaborn as sns
import pyproj
from shapely import wkt
from shapely.geometry import Point, LineString
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import networkx as nx

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("! CẢNH BÁO: Chưa tìm thấy thư viện 'hdbscan'. Hãy cài đặt bằng lệnh: pip install hdbscan")

try:
    import city2graph as c2g
except ImportError:
    c2g = None

TARGET_CRS = "EPSG:32648"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "hanoi"
PROCESSED_DIR = DATA_DIR / "processed" / "hanoi"
OUTPUT_DIR = DATA_DIR / "outputs" / "hanoi"

FEATURES_DIR = PROCESSED_DIR / "features"
GRAPHS_DIR = PROCESSED_DIR / "graphs"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
CLUSTERS_DIR = OUTPUT_DIR / "clusters"
MEM_DIR = str(REPO_ROOT / "tmp" / "hdbscan_cache")

for d in [FIGURES_DIR, TABLES_DIR, CLUSTERS_DIR, Path(MEM_DIR)]:
    d.mkdir(parents=True, exist_ok=True)

# Tải cấu hình siêu tham số thực nghiệm
config_path = REPO_ROOT / "configs" / "experiment_config.yaml"
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
else:
    config = {
        "seeds": {"global": 42, "tsne": 42},
        "model": {"out_dim": 32}
    }

# Cố định hạt giống ngẫu nhiên để tái lập thực nghiệm 100%
SEED = config.get("seeds", {}).get("global", 42)
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

pd.options.display.float_format = "{:,.2f}".format

# Danh mục 9 mô hình đối chuẩn
MODELS = {
    "Baseline (PCA)":              {"type": "baseline"},
    "Model 1 (GAT-GAE)":           {"key": "model1_gat_gae", "type": "homo"},
    "Model 2 (HAN-GAE: Walk)":     {"key": "model2_han_gae_walk", "type": "hetero"},
    "Model 3 (HAN-GAE: Multi)":    {"key": "model3_han_gae_multi", "type": "hetero"},
    "Model 4 (HAN-GAE: All)":      {"key": "model4_han_gae_all", "type": "hetero"},
    "Model 1S (GAT-GAE: Struct)":  {"key": "model1s_gat_gae_struct", "type": "homo_struct"},
    "Model 2S (HAN-GAE: Walk-S)":  {"key": "model2s_han_gae_walk_struct", "type": "hetero_struct"},
    "Model 3S (HAN-GAE: Multi-S)": {"key": "model3s_han_gae_multi_struct", "type": "hetero_struct"},
    "Model 4S (HAN-GAE: All-S)":   {"key": "model4s_han_gae_all_struct", "type": "hetero_struct"},
}

EMBEDDING_DIM = config.get("model", {}).get("out_dim", 32)
EMBEDDING_COLS = [f"embedding_{i}" for i in range(EMBEDDING_DIM)]

# Cột trọng số Semantic Attention (Layer 1 & Layer 2)
BETA_COLS = {
    "Model 2 (HAN-GAE: Walk)":  [f"beta1_{i}" for i in range(2)] + [f"beta2_{i}" for i in range(2)],
    "Model 3 (HAN-GAE: Multi)": [f"beta1_{i}" for i in range(2)] + [f"beta2_{i}" for i in range(2)],
    "Model 4 (HAN-GAE: All)":   [f"beta1_{i}" for i in range(3)] + [f"beta2_{i}" for i in range(3)],
}

RELATION_NAMES = {
    "Model 2 (HAN-GAE: Walk)":  ["contig", "15_min_walk"],
    "Model 3 (HAN-GAE: Multi)": ["contig", "15_min_multi"],
    "Model 4 (HAN-GAE: All)":   ["contig", "15_min_walk", "15_min_multi"],
}

# Điểm mốc tâm CBD Thủ đô Hà Nội: Hồ Hoàn Kiếm / Đông Kinh Nghĩa Thục
CITY_CENTRE = {"lat": 21.0285, "lon": 105.8542}

print(f"✓ [Cell 2] Thiết lập thành công: Không gian vector nhúng {EMBEDDING_DIM}D | Mốc CBD: Hồ Hoàn Kiếm.")


# %% [Cell 3]: Nạp đa giác 985 ô H3, gán vector nhúng và dựng 3 đồ thị NetworkX
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 2. Load Embeddings & Convert to GeoDataFrames
# Load 985 OA polygons, load embeddings into GeoDataFrames,
# and extract the 3 relation graphs (contig, walk, multi) for Modularity calculation.
# -----------------------------------------

print("\n-> [Cell 3] Đang nạp đa giác 985 ô H3 và vector nhúng cho 9 mô hình...")

zones_path = FEATURES_DIR / "oa_with_features.gpkg"
if not zones_path.exists():
    zones_path = FEATURES_DIR / "hanoi_normalized_features.parquet"

oa_polygons = gpd.read_file(zones_path) if zones_path.suffix == ".gpkg" else gpd.read_parquet(zones_path)
if oa_polygons.crs is None or str(oa_polygons.crs) != TARGET_CRS:
    oa_polygons = oa_polygons.to_crs(TARGET_CRS)

if "zone_id" not in oa_polygons.columns:
    oa_polygons["zone_id"] = [f"h3_res9_{i:04d}" for i in range(len(oa_polygons))]
oa_polygons = oa_polygons.set_index("zone_id", drop=False)

model_gdfs: dict[str, gpd.GeoDataFrame] = {}

for model_name, meta in MODELS.items():
    gdf_m = oa_polygons.copy()
    m_type = meta["type"]

    if m_type == "baseline":
        model_gdfs[model_name] = gdf_m
        print(f"  • Đã tải {len(gdf_m)} ô cho: {model_name}")
        continue

    m_key = meta["key"]
    npy_path = EMBEDDINGS_DIR / f"Z_{m_key}.npy"
    pt_path = EMBEDDINGS_DIR / f"{m_key}.pt"

    if npy_path.exists():
        Z_arr = np.load(npy_path)
    elif pt_path.exists():
        pt_obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        if hasattr(pt_obj, "node_types") and "oa" in pt_obj.node_types:
            z_keys = [f"embedding_{i}" for i in range(EMBEDDING_DIM)]
            Z_arr = torch.stack([pt_obj["oa"][k] for k in z_keys], dim=1).detach().cpu().numpy()
        elif hasattr(pt_obj, "embedding_0"):
            z_keys = [f"embedding_{i}" for i in range(EMBEDDING_DIM)]
            Z_arr = torch.stack([pt_obj[k] for k in z_keys], dim=1).detach().cpu().numpy()
        else:
            Z_arr = np.array(pt_obj)
    else:
        raise FileNotFoundError(f"Không tìm thấy tệp nhúng cho mô hình: {model_name}")

    for i in range(Z_arr.shape[1]):
        gdf_m[f"embedding_{i}"] = Z_arr[:, i]

    model_gdfs[model_name] = gdf_m
    print(f"  • Đã tải {len(gdf_m)} ô cho: {model_name} (Z shape: {Z_arr.shape})")

print("\n-> [Cell 3] Đang nạp đồ thị hetero_multi.pt và xây dựng 3 đồ thị NetworkX...")

graph_path = GRAPHS_DIR / "hetero_multi.pt"
if not graph_path.exists():
    raise FileNotFoundError(f"Không tìm thấy tệp đồ thị {graph_path}!")

data_multi = torch.load(graph_path, map_location="cpu", weights_only=False)

oa_nodes = list(oa_polygons.index.astype(str))
node_to_idx = {nid: idx for idx, nid in enumerate(oa_nodes)}
idx_to_node = {idx: nid for idx, nid in enumerate(oa_nodes)}

nx_graphs: dict[str, nx.Graph] = {}
relation_specs = [
    ("contig", ("oa", "contig", "oa")),
    ("15_min_walk", ("oa", "15_min_walk", "oa")),
    ("15_min_multi", ("oa", "15_min_multi", "oa")),
]

for rel_name, edge_type in relation_specs:
    G = nx.Graph()
    G.add_nodes_from(oa_nodes)
    if hasattr(data_multi, "edge_types") and edge_type in data_multi.edge_types:
        edge_index = data_multi[edge_type].edge_index.numpy()
        u_list = [idx_to_node[u] for u in edge_index[0]]
        v_list = [idx_to_node[v] for v in edge_index[1]]
        for u, v in zip(u_list, v_list):
            if u != v:
                G.add_edge(u, v)
    nx_graphs[rel_name] = G
    print(f"  • Đồ thị {rel_name:<12}: {G.number_of_nodes()} nút, {G.number_of_edges():,} cạnh")

print("✓ [Cell 3] Nạp toàn bộ dữ liệu nền và 3 mạng lưới thành công.")


# %% [Cell 4]: Cấu hình dải quét siêu tham số min_cluster_size (MCS_RANGE)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3. Clustering Pipeline
# Search for optimal HDBSCAN hyperparameter (min_cluster_size) via DBCV.
# -----------------------------------------

# Quét kích thước cụm tối thiểu từ 5 đến 30 ô H3 (mỗi bước tăng 1)
MCS_RANGE = list(range(5, 31, 1))
print(f"✓ [Cell 4] Dải quét siêu tham số min_cluster_size (MCS): {MCS_RANGE[0]} -> {MCS_RANGE[-1]} ({len(MCS_RANGE)} giá trị)")


# %% [Cell 5]: Định nghĩa hàm phân cụm HDBSCAN và thực thi quét siêu tham số qua DBCV
# --- Ghi chú giải thích từ Notebook gốc ---
# Extracts normalized embeddings, iterates over MCS_RANGE, computes DBCV & modularity,
# selects best configuration based on maximum DBCV, and saves cluster assignments.
# -----------------------------------------

def get_embeddings(gdf: gpd.GeoDataFrame, model_type: str) -> np.ndarray:
    """Trích xuất và chuẩn hóa mặt cầu L2 ma trận vector nhúng Z."""
    if model_type == "baseline":
        poi_cols = [
            "arts_culture", "automotive_facility", "consumer_service", "corporate_service",
            "education", "entertainment", "food_beverage", "healthcare",
            "hotel_lodging", "industrial_service", "park", "public_service",
            "religion", "retail", "sports_fitness", "transportation_facility"
        ]
        available_cols = [
            c for c in poi_cols
            if c in gdf.columns and pd.api.types.is_numeric_dtype(gdf[c])
        ]
        if not available_cols:
            available_cols = [
                c for c in gdf.columns
                if c not in ["geometry", "cluster", "zone_id", "cluster_label_rank"]
                and pd.api.types.is_numeric_dtype(gdf[c])
            ]
        X = gdf[available_cols].values.astype(float)
        # Giới hạn n_components không vượt quá số thuộc tính thực tế (16 POIs)
        n_comp = min(X.shape[1], EMBEDDING_DIM)
        pca = PCA(n_components=n_comp, random_state=SEED)
        Z = pca.fit_transform(X)
    else:
        Z = gdf[EMBEDDING_COLS].values

    return normalize(Z, norm="l2")

def optimal_k_clustering(
    Z: np.ndarray,
    nx_graphs: dict[str, nx.Graph],
    oa_nodes: list[str],
    mcs_range: list[int]
) -> dict:
    """Quét siêu tham số min_cluster_size, tìm cấu hình có DBCV cực đại."""
    if not HAS_HDBSCAN:
        raise RuntimeError("Cần cài đặt thư viện hdbscan: pip install hdbscan")

    results_by_k = {}

    for mcs in mcs_range:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            gen_min_span_tree=True,
            memory=MEM_DIR,
        )
        labels = clusterer.fit_predict(Z)

        # Lọc các điểm phân cụm hợp lệ (loại trừ điểm nhiễu Noise: label = -1)
        valid_mask = (labels != -1)
        valid_labels = labels[valid_mask]
        n_clusters = len(set(valid_labels))

        # Chỉ số đánh giá hợp lệ mật độ phân cụm DBCV
        dbcv = float(clusterer.relative_validity_) if hasattr(clusterer, "relative_validity_") else 0.0

        # Đo tính mô-đun Newman Modularity trên 3 mạng lưới (loại trừ điểm nhiễu)
        mods = {"contig": np.nan, "15_min_walk": np.nan, "15_min_multi": np.nan}

        if len(oa_nodes) == len(labels) and n_clusters >= 2:
            valid_nodes = set()
            cluster_dict = {}
            for i, c_id in enumerate(labels):
                if c_id != -1:
                    cluster_dict.setdefault(c_id, set()).add(oa_nodes[i])
                    valid_nodes.add(oa_nodes[i])

            communities = list(cluster_dict.values())

            for rel, G in nx_graphs.items():
                sub_G = G.subgraph(valid_nodes)
                if sub_G.number_of_edges() > 0:
                    try:
                        mods[rel] = float(nx.community.modularity(sub_G, communities))
                    except Exception:
                        mods[rel] = np.nan

        config_result = {
            "mcs": mcs,
            "labels": labels,
            "dbcv": dbcv,
            "modularity_contig": mods["contig"],
            "modularity_walk": mods["15_min_walk"],
            "modularity_multi": mods["15_min_multi"],
        }

        # Lưu cấu hình tốt nhất cho từng số lượng cụm K
        if n_clusters not in results_by_k or dbcv > results_by_k[n_clusters]["dbcv"]:
            results_by_k[n_clusters] = config_result

    # Dựng bảng lịch sử tìm kiếm k_df
    k_scores = []
    for k_val, res in sorted(results_by_k.items()):
        row = {"k": k_val, **res}
        del row["labels"]
        k_scores.append(row)

    k_df = pd.DataFrame(k_scores)

    # Chọn cấu hình tối ưu toàn cục tại điểm có DBCV cực đại
    if not k_df.empty and k_df["dbcv"].notna().any():
        best_k = int(k_df.loc[k_df["dbcv"].idxmax(), "k"])
        best = results_by_k[best_k]
    else:
        best_k = list(results_by_k.keys())[0] if results_by_k else 2
        best = list(results_by_k.values())[0] if results_by_k else {}

    return {
        "k": best_k,
        "mcs": best.get("mcs", 10),
        "dbcv": best.get("dbcv", 0.0),
        "labels": best.get("labels", np.zeros(len(oa_nodes), dtype=int)),
        "modularity_contig": best.get("modularity_contig", np.nan),
        "modularity_walk": best.get("modularity_walk", np.nan),
        "modularity_multi": best.get("modularity_multi", np.nan),
        "k_search": k_df,
    }

print("\n-> [Cell 5] Bắt đầu quét siêu tham số HDBSCAN cho toàn bộ 9 mô hình...")

clustering_results: dict[str, dict] = {}

for model_name, gdf in model_gdfs.items():
    print(f"\nPhân cụm HDBSCAN: {model_name}...")
    Z = get_embeddings(gdf, MODELS[model_name]["type"])
    result = optimal_k_clustering(Z, nx_graphs, oa_nodes, MCS_RANGE)

    # Gán nhãn phân cụm vào GeoDataFrame
    gdf["cluster"] = result["labels"]
    clustering_results[model_name] = result

    # Lưu tệp CSV phân cụm HDBSCAN
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "").replace("-", "_")
    out_cluster_csv = CLUSTERS_DIR / f"{safe_name}_hdbscan.csv"
    gdf[["cluster"]].to_csv(out_cluster_csv)

    n_noise = int(np.sum(result["labels"] == -1))
    noise_pct = (n_noise / len(result["labels"])) * 100
    print(f"  ✓ Tối ưu: K*={result['k']} cụm (mcs={result['mcs']}, DBCV={result['dbcv']:.2f}) | Nhiễu (Noise): {n_noise}/{len(result['labels'])} ô ({noise_pct:.1f}%)")

print("\n✓ [Cell 5] Hoàn thành phân cụm HDBSCAN và lưu trữ 9 tệp nhãn cụm.")


# %% [Cell 6]: Bảng thống kê phân cụm HDBSCAN (Phụ lục Bảng C.2)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3.5 Cluster Statistics (Appendix Table C.2)
# Computes Mean ± SD for DBCV and Modularity metrics across explored configurations.
# -----------------------------------------

print("\n-> [Cell 6] Đang tổng hợp Phụ lục Bảng C.2 (Table C.2 HDBSCAN Benchmark)...")

def mean_std_str(s: pd.Series) -> str:
    """Định dạng chuỗi Mean ± SD an toàn."""
    clean_s = s.dropna()
    if clean_s.empty:
        return "N/A"
    mean_val = clean_s.mean()
    std_val = 0.0 if len(clean_s) == 1 else clean_s.std()
    return f"{mean_val:.2f} ± {std_val:.2f}"

table_data = {}
for model_name, result in clustering_results.items():
    k_df = result["k_search"]
    dbcv_series = k_df["dbcv"]
    mc_series = k_df["modularity_contig"]
    mw_series = k_df["modularity_walk"]
    mm_series = k_df["modularity_multi"]

    table_data[model_name] = {
        ("DBCV", "Mean ± SD"): mean_std_str(dbcv_series),
        ("Modularity (Contig)", "Mean ± SD"): mean_std_str(mc_series),
        ("Modularity (Walk)", "Mean ± SD"): mean_std_str(mw_series),
        ("Modularity (Multi)", "Mean ± SD"): mean_std_str(mm_series),
    }

df_results_range = pd.DataFrame(table_data).T
df_results_range.columns.names = ["Metric", "Statistic"]

print("\n" + "=" * 95)
print("PHỤ LỤC BẢNG C.2: THỐNG KÊ PHÂN CỤM HDBSCAN (HÀ NỘI CASE STUDY)")
print("=" * 95)
print(df_results_range.to_string())
print("=" * 95)

out_table_hdbscan = TABLES_DIR / "cluster_stats_hdbscan.csv"
df_results_range.to_csv(out_table_hdbscan)
print(f"✓ [Cell 6] Đã lưu bảng thống kê tại: {out_table_hdbscan.resolve()}")


# %% [Cell 7]: Kiểm tra chẩn đoán bảng tìm kiếm siêu tham số của Model 1S
# --- Ghi chú giải thích từ Notebook gốc ---
# Inspect k_search dataframe for Model 1S (GAT-GAE: Struct)
# -----------------------------------------

print("\n-> [Cell 7] Bảng chẩn đoán quét siêu tham số HDBSCAN (Model 1S: Struct-Only):")
if "Model 1S (GAT-GAE: Struct)" in clustering_results:
    m1s_search = clustering_results["Model 1S (GAT-GAE: Struct)"]["k_search"]
    print(m1s_search.head(10).to_string(index=False))


# %% [Cell 8]: Thiết lập không gian, hàm khử va chạm nhãn và vẽ bản đồ đơn lẻ plot_cluster_map
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 6. Cluster Map Visualizations (HDBSCAN)
# Transforms CBD coordinates, defines label collision adjuster,
# and plots cluster map with noise points in light gray (#CCCCCC).
# -----------------------------------------

print("\n-> [Cell 8] Thiết lập hàm hiển thị bản đồ phân cụm HDBSCAN...")

# Chuyển đổi tọa độ Hồ Hoàn Kiếm sang EPSG:32648
transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
centre_x, centre_y = transformer.transform(CITY_CENTRE["lon"], CITY_CENTRE["lat"])

# Định vị ô lục giác chứa mốc Hồ Hoàn Kiếm
ref_gdf = model_gdfs["Model 1 (GAT-GAE)"]
centre_oa_idx = ref_gdf.geometry.distance(Point(centre_x, centre_y)).idxmin()
print(f"✓ Ô lục giác chứa tâm CBD Hồ Hoàn Kiếm: zone_id = '{centre_oa_idx}'")

# Khởi tạo bảng màu toàn cục xáo trộn ngẫu nhiên cố định hạt giống 42
base_palette = (
    sns.color_palette("bright", 10) +
    sns.color_palette("dark", 10) +
    sns.color_palette("husl", 15) +
    sns.color_palette("pastel", 15)
)
rng = np.random.RandomState(42)
GLOBAL_COLORS = [to_hex(c) for c in base_palette]
rng.shuffle(GLOBAL_COLORS)

proposed_models = [m for m in model_gdfs if "S (" not in m and m != "Baseline (PCA)"]
baseline_models = ["Baseline (PCA)"] + [m for m in model_gdfs if "S (" in m]

NO_RANK_LABELS = {
    "Baseline (PCA)",
    "Model 1S (GAT-GAE: Struct)", "Model 2S (HAN-GAE: Walk-S)",
    "Model 3S (HAN-GAE: Multi-S)", "Model 4S (HAN-GAE: All-S)",
}

def adjust_component_labels(ax, labels, padding: float = 2.0, max_iter: int = 500) -> None:
    """Tự động phân tách các nhãn số bị đè lên nhau trong hệ tọa độ hiển thị."""
    if len(labels) < 2:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    anchors = np.array([ax.transData.transform(anchor) for _, anchor in labels], dtype=float)
    positions = anchors.copy()
    extents = [text.get_window_extent(renderer=renderer) for text, _ in labels]
    widths = np.array([extent.width for extent in extents])
    heights = np.array([extent.height for extent in extents])
    axes_bbox = ax.get_window_extent(renderer=renderer)

    for _ in range(max_iter):
        moved = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                delta = positions[j] - positions[i]
                overlap_x = (widths[i] + widths[j]) / 2 + padding - abs(delta[0])
                overlap_y = (heights[i] + heights[j]) / 2 + padding - abs(delta[1])
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                moved = True
                shift_dir = 1.0 if delta[0] >= 0 else -1.0
                positions[i, 0] -= shift_dir * overlap_x / 2
                positions[j, 0] += shift_dir * overlap_x / 2
        positions[:, 0] = np.clip(positions[:, 0], axes_bbox.x0 + widths / 2, axes_bbox.x1 - widths / 2)
        positions[:, 1] = np.clip(positions[:, 1], axes_bbox.y0 + heights / 2, axes_bbox.y1 - heights / 2)
        if not moved:
            break

    inverse = ax.transData.inverted()
    for (text, anchor), display_pos in zip(labels, positions):
        data_pos = inverse.transform(display_pos)
        text.set_position(data_pos)

def plot_cluster_map(ax, gdf: gpd.GeoDataFrame, result: dict, model_name: str) -> None:
    """Vẽ bản đồ phân cụm HDBSCAN với điểm nhiễu màu xám #CCCCCC."""
    Z = get_embeddings(gdf, MODELS[model_name]["type"])

    # Xếp hạng cụm theo độ tương đồng Cosine từ cụm chứa Hồ Hoàn Kiếm
    centre_pos = list(gdf.index).index(centre_oa_idx)
    target_label = result["labels"][centre_pos]
    if target_label == -1:
        non_noise = [l for l in result["labels"] if l != -1]
        target_label = non_noise[0] if non_noise else 0

    centroids = pd.DataFrame(Z).assign(label=result["labels"])
    valid_centroids = centroids[centroids["label"] != -1].groupby("label").mean()

    if not valid_centroids.empty and target_label in valid_centroids.index:
        target_vec = valid_centroids.loc[target_label].values.reshape(1, -1)
        cluster_sims = dict(zip(valid_centroids.index, cosine_similarity(target_vec, valid_centroids.values)[0]))
        sorted_labels = sorted(cluster_sims, key=cluster_sims.get, reverse=True)
    else:
        sorted_labels = sorted(list(set(result["labels"]) - {-1}))

    label_to_rank = {label: i for i, label in enumerate(sorted_labels)}
    rank_color_map = {label: GLOBAL_COLORS[i % len(GLOBAL_COLORS)] for i, label in enumerate(sorted_labels)}
    node_colors = [rank_color_map[l] if l != -1 else "#CCCCCC" for l in result["labels"]]

    gdf.plot(ax=ax, color=node_colors, edgecolor="white", linewidth=0.2)

    # Đánh dấu sao tâm Hồ Hoàn Kiếm
    ax.scatter([centre_x], [centre_y], c="black", s=130, marker="*", zorder=12, edgecolors="white", linewidths=1.2)

    # Đánh số thứ hạng cụm lên từng mảng đa giác lớn (loại trừ điểm nhiễu)
    component_labels = []
    if model_name not in NO_RANK_LABELS and sorted_labels:
        valid_mask = (np.asarray(result["labels"]) != -1)
        if valid_mask.any():
            gdf_temp = gdf.loc[valid_mask].assign(
                cluster_label_rank=[label_to_rank[l] for l in np.asarray(result["labels"])[valid_mask]]
            )
            dissolved = gdf_temp.dissolve(by="cluster_label_rank").reset_index()
            for _, row in dissolved.iterrows():
                geom = row.geometry
                polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom] if geom.geom_type == "Polygon" else []
                large_polys = [p for p in polys if p.area >= 150000.0]
                if not large_polys and polys:
                    large_polys = [max(polys, key=lambda p: p.area)]
                for poly in large_polys:
                    anchor = poly.representative_point()
                    text = ax.text(
                        anchor.x, anchor.y, str(row["cluster_label_rank"]),
                        fontsize=14, fontweight="bold", ha="center", va="center",
                        color="black", zorder=10,
                        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")]
                    )
                    component_labels.append((text, (anchor.x, anchor.y)))
            adjust_component_labels(ax, component_labels)

    # Bảng chú giải Legend
    legend_elements = [
        Patch(facecolor=rank_color_map[ol], edgecolor="none", label=f"{i}")
        for i, ol in enumerate(sorted_labels)
    ]
    n_noise = int(np.sum(np.array(result["labels"]) == -1))
    if n_noise > 0:
        legend_elements.append(Patch(facecolor="#CCCCCC", edgecolor="none", label=f"Noise ({n_noise})"))

    n_leg_cols = 2 if len(legend_elements) > 10 else 1
    ax.legend(
        handles=legend_elements, loc="center left", bbox_to_anchor=(1.01, 0.5),
        title="Cluster", title_fontsize=14, frameon=False, prop={"size": 11},
        ncol=n_leg_cols, columnspacing=0.4
    )
    ax.set_axis_off()


# %% [Cell 9]: Ma trận 3x3 Bản đồ phân vùng chức năng HDBSCAN (Hình C.2 trong Phụ lục C)
# --- Ghi chú giải thích từ Notebook gốc ---
# 3x3 Combined Cluster Map for all 9 models (Figure C.2 in Sato et al., CEUS 2026)
# -----------------------------------------

print("\n-> [Cell 9] Đang kết xuất ma trận 3x3 bản đồ phân vùng HDBSCAN (Hình C.2)...")

all_models = list(MODELS.keys())
fig, axes = plt.subplots(3, 3, figsize=(27, 27), dpi=300)

for idx, model_name in enumerate(all_models):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    plot_cluster_map(ax, model_gdfs[model_name], clustering_results[model_name], model_name)

    res = clustering_results[model_name]
    ax.set_title(
        f"{model_name}\n(K*={res['k']}, mcs={res['mcs']}, DBCV={res['dbcv']:.2f})",
        fontsize=20, fontweight="bold", pad=8
    )

plt.tight_layout()
cluster_maps_hdbscan_path = FIGURES_DIR / "cluster_maps_hdbscan.jpg"
plt.savefig(cluster_maps_hdbscan_path, dpi=300, bbox_inches="tight")
print(f"✓ [Cell 9] Đã lưu ma trận bản đồ phân vùng HDBSCAN tại: {cluster_maps_hdbscan_path.resolve()}")
plt.show()
plt.close(fig)


# %% [Cell 10]: Trực quan hóa không gian tiềm ẩn t-SNE 2D theo cụm HDBSCAN (Hình C.3b)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 7. t-SNE Embeddings Visualization (HDBSCAN Clustering)
# Projects 32D embeddings onto 2D space colored by HDBSCAN clusters, with noise in gray.
# -----------------------------------------

print("\n-> [Cell 10] Đang chiếu t-SNE 2D và kết xuất ma trận không gian tiềm ẩn HDBSCAN...")

models_to_plot = list(MODELS.keys())
n_models = len(models_to_plot)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, n_rows * 6.8), dpi=300)
axes = axes.flatten()

tsne_seed = config.get("seeds", {}).get("tsne", 42)

for idx, model_name in enumerate(models_to_plot):
    ax = axes[idx]
    gdf = model_gdfs[model_name]
    result = clustering_results[model_name]
    Z = get_embeddings(gdf, MODELS[model_name]["type"])

    # Xếp hạng cụm theo độ tương đồng Cosine
    centre_pos = list(gdf.index).index(centre_oa_idx)
    target_label = result["labels"][centre_pos]
    if target_label == -1:
        non_noise = [l for l in result["labels"] if l != -1]
        target_label = non_noise[0] if non_noise else 0

    centroids = pd.DataFrame(Z).assign(label=result["labels"])
    valid_centroids = centroids[centroids["label"] != -1].groupby("label").mean()

    if not valid_centroids.empty and target_label in valid_centroids.index:
        target_vec = valid_centroids.loc[target_label].values.reshape(1, -1)
        cluster_sims = dict(zip(valid_centroids.index, cosine_similarity(target_vec, valid_centroids.values)[0]))
        sorted_labels = sorted(cluster_sims, key=cluster_sims.get, reverse=True)
    else:
        sorted_labels = sorted(list(set(result["labels"]) - {-1}))

    rank_color_map = {label: GLOBAL_COLORS[i % len(GLOBAL_COLORS)] for i, label in enumerate(sorted_labels)}

    # Chiếu t-SNE phi tuyến 32D -> 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=tsne_seed)
    Z_2d = tsne.fit_transform(Z)

    for rank, label in enumerate(sorted_labels):
        mask = (np.array(result["labels"]) == label)
        ax.scatter(
            Z_2d[mask, 0], Z_2d[mask, 1],
            c=rank_color_map[label], s=18, alpha=0.8, label=str(rank),
            edgecolors="none"
        )

    # Tô màu các điểm nhiễu Noise bằng màu xám nhạt #CCCCCC
    noise_mask = (np.array(result["labels"]) == -1)
    if noise_mask.any():
        ax.scatter(
            Z_2d[noise_mask, 0], Z_2d[noise_mask, 1],
            c="#CCCCCC", s=14, alpha=0.5, label="Noise",
            edgecolors="none"
        )

    ax.set_title(f"{model_name}", fontsize=17, fontweight="bold", pad=8)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

# Ẩn các ô trống nếu có
for i in range(len(models_to_plot), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
tsne_hdbscan_fig_path = FIGURES_DIR / "tsne_embeddings_hdbscan.jpg"
plt.savefig(tsne_hdbscan_fig_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 10] Đã lưu hình chiếu t-SNE HDBSCAN tại: {tsne_hdbscan_fig_path.resolve()}")
plt.show()
plt.close(fig)

print("\n" + "=" * 95)
print("✓ HOÀN THÀNH TOÀN BỘ 10 CELLS CỦA PHỤ LỤC ĐÁNH GIÁ ĐỘ BỀN VỮNG HDBSCAN CHO HÀ NỘI!")
print("==========================================================================================")