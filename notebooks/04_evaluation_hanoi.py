# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH ĐÁNH GIÁ PHÂN CỤM & KHAI PHÁ TRI THỨC ĐÔ THỊ HÀ NỘI (NOTEBOOK 04)
Kế thừa & Chuyển đổi 1:1 từ: 04_evaluation.py (Sato et al., CEUS 2026)
Đối tượng không gian: 985 ô lục giác Uber H3 Resolution 9 tại Thủ đô Hà Nội
Hệ quy chiếu phẳng: WGS 84 / UTM Zone 48N (EPSG:32648)
Khối triển khai: Khối 1 - Thiết lập môi trường, Cấu hình & Nạp Embeddings (Cell 1 - Cell 3)
"""

from __future__ import annotations

# %% [Cell 1]: Thiết lập môi trường thực thi và cấu hình OpenMP
# --- Ghi chú giải thích từ Notebook gốc ---
# # 04. Evaluation & Results (Hanoi Case Study)
#
# Pha 4: Đánh giá phân cụm chức năng đô thị và thẩm định cấu trúc không gian.
# So sánh 9 mô hình: Baseline PCA, Models 1-4 (đề xuất) và Models 1S-4S (triệt tiêu cấu trúc).
# -----------------------------------------

import os
import sys
import warnings
from pathlib import Path

# Cấu hình OpenMP an toàn chống xung đột thư viện trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
warnings.filterwarnings("ignore")

# Cơ chế Dynamic Root Discovery tự động định vị gốc repository City2Graph-Hanoi
CWD = Path.cwd().resolve()
REPO_ROOT = CWD
while not (REPO_ROOT / "data").exists() and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent

# Kết nối thư mục mã nguồn lõi src/ vào sys.path
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("=" * 90)
print(f"✓ [Cell 1] Đã định vị thư mục gốc dự án: {REPO_ROOT}")
print(f"✓ [Cell 1] Đã kết nối module mã nguồn: {SRC_DIR}")
print("=" * 90)


# %% [Cell 2]: Khai báo thư viện, nạp cấu hình và định nghĩa tham số ca Hà Nội
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 1. Setup & Configuration
# Load hyper-parameters, define model configurations, and set spatial parameters.
# -----------------------------------------

import copy
import yaml
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import seaborn as sns
import contextily as ctx
import pyproj

from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import Point
from shapely import wkt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

# Nạp thư viện city2graph nếu có
try:
    import city2graph as c2g
except ImportError:
    c2g = None

# Thiết lập đường dẫn thư mục chuẩn hóa của Hà Nội
DATA_DIR = REPO_ROOT / "data"
GRAPHS_DIR = DATA_DIR / "processed" / "hanoi" / "graphs"
FEATURES_DIR = DATA_DIR / "processed" / "hanoi" / "features"
OUTPUTS_DIR = DATA_DIR / "outputs" / "hanoi"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
CLUSTERS_DIR = OUTPUTS_DIR / "clusters"

for d in [FIGURES_DIR, TABLES_DIR, CLUSTERS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Tải cấu hình thực nghiệm đa đường dẫn (Multi-path Configuration Discovery)
config_candidates = [
    REPO_ROOT / "config" / "experiment_config.yaml",
    REPO_ROOT / "configs" / "experiment_config.yaml",
    REPO_ROOT / "experiment_config.yaml",
    Path.cwd() / "config" / "experiment_config.yaml",
    Path.cwd() / "configs" / "experiment_config.yaml",
    Path.cwd() / "experiment_config.yaml"
]
chosen_cfg_path = next((p for p in config_candidates if p.exists()), None)
if chosen_cfg_path is not None:
    with open(chosen_cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    print(f"✓ [Cell 2] Nạp thành công tệp cấu hình thực nghiệm: {chosen_cfg_path.resolve()}")
else:
    print("! [Cell 2] Không tìm thấy experiment_config.yaml, áp dụng cấu hình dự phòng chuẩn CEUS 2026.")
    config = {}

# Thiết lập các giá trị mặc định phòng vệ an toàn (Defensive Defaults)
config.setdefault("seeds", {})
config["seeds"].setdefault("global", 42)
config["seeds"].setdefault("clustering", config["seeds"].get("kmeans", 42))
config["seeds"].setdefault("tsne", 42)
config["seeds"].setdefault("model", 42)

config.setdefault("model", {})
config["model"].setdefault("in_dim", 16)
config["model"].setdefault("out_dim", 32)

config.setdefault("data", {})
config["data"].setdefault("root", str(GRAPHS_DIR))
config["data"].setdefault("homo", "homo.pt")
config["data"].setdefault("hetero_walk", "hetero_walk.pt")
config["data"].setdefault("hetero_multi", "hetero_multi.pt")

# Cố định hạt giống ngẫu nhiên toàn cục
global_seed = config.get("seeds", {}).get("global", 42)
np.random.seed(global_seed)
torch.manual_seed(global_seed)

# Định dạng hiển thị bảng số liệu
pd.options.display.float_format = "{:,.2f}".format

# Cấu hình danh mục 9 mô hình thực nghiệm (Baseline PCA, Models 1-4, Models 1S-4S)
MODELS = {
    "Baseline (PCA)": {
        "file": str(GRAPHS_DIR / "homo.pt"),
        "type": "baseline"
    },
    "Model 1 (GAT-GAE)": {
        "file": str(EMBEDDINGS_DIR / "model1_gat_gae.pt"),
        "type": "homo"
    },
    "Model 2 (HAN-GAE: Walk)": {
        "file": str(EMBEDDINGS_DIR / "model2_han_gae_walk.pt"),
        "type": "hetero"
    },
    "Model 3 (HAN-GAE: Multi)": {
        "file": str(EMBEDDINGS_DIR / "model3_han_gae_multi.pt"),
        "type": "hetero"
    },
    "Model 4 (HAN-GAE: All)": {
        "file": str(EMBEDDINGS_DIR / "model4_han_gae_all.pt"),
        "type": "hetero"
    },
    "Model 1S (GAT-GAE: Struct)": {
        "file": str(EMBEDDINGS_DIR / "model1s_gat_gae_struct.pt"),
        "type": "homo_struct"
    },
    "Model 2S (HAN-GAE: Walk-S)": {
        "file": str(EMBEDDINGS_DIR / "model2s_han_gae_walk_struct.pt"),
        "type": "hetero_struct"
    },
    "Model 3S (HAN-GAE: Multi-S)": {
        "file": str(EMBEDDINGS_DIR / "model3s_han_gae_multi_struct.pt"),
        "type": "hetero_struct"
    },
    "Model 4S (HAN-GAE: All-S)": {
        "file": str(EMBEDDINGS_DIR / "model4s_han_gae_all_struct.pt"),
        "type": "hetero_struct"
    }
}

# Danh mục các cột vector nhúng tiềm ẩn Z (32 chiều)
out_dim = config.get("model", {}).get("out_dim", 32)
EMBEDDING_COLS = [f"embedding_{i}" for i in range(out_dim)]

# Danh mục các cột trọng số chú ý Semantic Attention beta (Layer 1 và Layer 2)
BETA_COLS = {
    "Model 2 (HAN-GAE: Walk)": [f"beta1_{i}" for i in range(2)] + [f"beta2_{i}" for i in range(2)],
    "Model 3 (HAN-GAE: Multi)": [f"beta1_{i}" for i in range(2)] + [f"beta2_{i}" for i in range(2)],
    "Model 4 (HAN-GAE: All)": [f"beta1_{i}" for i in range(3)] + [f"beta2_{i}" for i in range(3)]
}

RELATION_NAMES = {
    "Model 2 (HAN-GAE: Walk)": ["contig", "15_min_walk"],
    "Model 3 (HAN-GAE: Multi)": ["contig", "15_min_multi"],
    "Model 4 (HAN-GAE: All)": ["contig", "15_min_walk", "15_min_multi"]
}

# Dải số lượng cụm khảo sát K in [2, 50]
K_RANGE = range(2, 51)

# Mốc tâm lõi thương mại CBD Hà Nội: Hồ Hoàn Kiếm / Quảng trường Đông Kinh Nghĩa Thục
CITY_CENTRE = {"lat": 21.0285, "lon": 105.8542}

print(f"✓ [Cell 2] Đã cấu hình danh mục 9 mô hình thực nghiệm:")
for m_name in MODELS.keys():
    print(f"  • {m_name}")
print(f"✓ [Cell 2] Mốc tâm CBD Hà Nội: {CITY_CENTRE} (Hồ Hoàn Kiếm)")
print(f"✓ [Cell 2] Dải quét số cụm K-Means: k in [{min(K_RANGE)}, {max(K_RANGE)}]")


# %% [Cell 3]: Nạp đa giác ô H3, trích xuất Embeddings & Dựng 3 đồ thị NetworkX
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 2. Load Embeddings & Convert to GeoDataFrames
# Load OA polygons, extract embedding matrices and rebuild relation-specific graphs.
# -----------------------------------------

print("\n-> [Cell 3] Đang nạp đa giác 985 ô H3 và trích xuất vector nhúng của 9 mô hình...")

# 1. Nạp đa giác hình học 985 ô H3 Hà Nội
oa_polygons_path = FEATURES_DIR / "oa_with_features.gpkg"
if not oa_polygons_path.exists():
    oa_polygons_path = FEATURES_DIR / "hanoi_normalized_features.parquet"
    oa_polygons = gpd.read_parquet(oa_polygons_path)
else:
    oa_polygons = gpd.read_file(oa_polygons_path)

if "zone_id" in oa_polygons.columns:
    oa_polygons = oa_polygons.set_index("zone_id", drop=False)

TARGET_CRS = "EPSG:32648"
if oa_polygons.crs is None or str(oa_polygons.crs) != TARGET_CRS:
    oa_polygons = oa_polygons.to_crs(TARGET_CRS)

expected_node_ids = list(oa_polygons.index.astype(str))
n_expected_nodes = len(expected_node_ids)
print(f"✓ Nạp thành công hình học {n_expected_nodes} ô lục giác H3 (CRS: {oa_polygons.crs}).")

# Từ điển lưu trữ GeoDataFrames chứa embedding và trọng số attention
model_gdfs: dict[str, gpd.GeoDataFrame] = {}
attention_data: dict[str, dict[str, pd.DataFrame]] = {}

# 2. Duyệt qua 9 mô hình nạp vector nhúng
for model_name, meta in MODELS.items():
    fpath = Path(meta["file"])
    if not fpath.exists():
        # Dự phòng kiểm tra định dạng .npy
        npy_path = EMBEDDINGS_DIR / f"Z_{fpath.stem}.npy"
        if not npy_path.exists():
            print(f"! Cảnh báo: Chưa tìm thấy tệp nhúng cho {model_name} tại {fpath.name}")
            continue

    print(f"  • Đang xử lý: {model_name}...")
    is_hetero = meta["type"].startswith("hetero")
    beta_cols = BETA_COLS.get(model_name, [])

    # Trường hợp 1: Baseline PCA (lấy trực tiếp đặc trưng nút từ homo.pt)
    if meta["type"] == "baseline":
        homo_pt = torch.load(fpath, map_location="cpu", weights_only=False)
        x_mat = homo_pt.x.detach().cpu().numpy() if hasattr(homo_pt, "x") else np.zeros((n_expected_nodes, 16))
        feat_df = pd.DataFrame(x_mat, index=expected_node_ids, columns=[f"feat_{i}" for i in range(x_mat.shape[1])])
        nodes_gdf = gpd.GeoDataFrame(feat_df, geometry=oa_polygons["geometry"], crs=TARGET_CRS)

    # Trường hợp 2: Mô hình dị thể (HAN-GAE) hoặc đồng nhất (GAT-GAE)
    else:
        pt_obj = torch.load(fpath, map_location="cpu", weights_only=False)
        
        # Bóc tách tensor node embeddings
        if hasattr(pt_obj, "node_types") and "oa" in pt_obj.node_types:
            oa_data = pt_obj["oa"]
        else:
            oa_data = pt_obj

        # Đọc 32 chiều embedding
        z_dict = {}
        for col in EMBEDDING_COLS:
            if hasattr(oa_data, col):
                val = getattr(oa_data, col)
                z_dict[col] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.array(val)
            elif isinstance(oa_data, dict) and col in oa_data:
                val = oa_data[col]
                z_dict[col] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.array(val)
            else:
                # Dự phòng đọc trực tiếp từ tệp Z_*.npy đã xuất
                npy_fallback = EMBEDDINGS_DIR / f"Z_{fpath.stem}.npy"
                if npy_fallback.exists():
                    z_all = np.load(npy_fallback)
                    col_idx = int(col.split("_")[1])
                    z_dict[col] = z_all[:, col_idx]
                else:
                    z_dict[col] = np.zeros(n_expected_nodes)

        emb_df = pd.DataFrame(z_dict, index=expected_node_ids)

        # Trích xuất các cột trọng số Attention beta cho các mô hình HAN
        if beta_cols:
            rels = RELATION_NAMES[model_name]
            for col in beta_cols:
                if hasattr(oa_data, col):
                    val = getattr(oa_data, col)
                    emb_df[col] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.array(val)
                elif isinstance(oa_data, dict) and col in oa_data:
                    val = oa_data[col]
                    emb_df[col] = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else np.array(val)
                else:
                    # Dự phòng đọc từ beta_*.npy
                    beta_npy = EMBEDDINGS_DIR / f"beta_{fpath.stem}.npy"
                    if beta_npy.exists():
                        b_mat = np.load(beta_npy)
                        col_part = col.split("_")[1]
                        emb_df[col] = b_mat[:, int(col_part)] if col.startswith("beta2_") else b_mat[:, int(col_part)]
                    else:
                        emb_df[col] = 1.0 / len(rels)

            attention_data[model_name] = {
                f"Layer {li}": emb_df[[f"beta{li}_{i}" for i in range(len(rels))]].set_axis(rels, axis=1)
                for li in [1, 2]
            }

        nodes_gdf = gpd.GeoDataFrame(emb_df, geometry=oa_polygons["geometry"], crs=TARGET_CRS)

    model_gdfs[model_name] = nodes_gdf
    print(f"    -> Đã nạp {len(nodes_gdf)} ô H3 với {len(EMBEDDING_COLS)} chiều vector.")

# 3. Dựng 3 đồ thị NetworkX phục vụ tính toán Newman Modularity Q
print("\n-> Đang dựng 3 đồ thị NetworkX cho 3 loại quan hệ mạng...")
multi_pt_path = GRAPHS_DIR / "hetero_multi.pt"
if not multi_pt_path.exists():
    raise FileNotFoundError(f"Chưa tìm thấy {multi_pt_path}. Hãy chạy Notebook 02 trước!")

data_multi = torch.load(multi_pt_path, map_location="cpu", weights_only=False)

relation_edge_types = {
    "contig": ("oa", "contig", "oa"),
    "15_min_walk": ("oa", "15_min_walk", "oa"),
    "15_min_multi": ("oa", "15_min_multi", "oa"),
}

nx_graphs: dict[str, nx.Graph] = {}

for rel_key, edge_type in relation_edge_types.items():
    G = nx.Graph()
    # Thêm toàn bộ 985 đỉnh định danh zone_id
    for zid in expected_node_ids:
        G.add_node(zid)

    if edge_type in data_multi.edge_types:
        edge_index = data_multi[edge_type].edge_index.detach().cpu().numpy()
        u_arr, v_arr = edge_index[0], edge_index[1]
        for u_idx, v_idx in zip(u_arr, v_arr):
            if u_idx < n_expected_nodes and v_idx < n_expected_nodes:
                u_zid = expected_node_ids[u_idx]
                v_zid = expected_node_ids[v_idx]
                if u_zid != v_zid:
                    G.add_edge(u_zid, v_zid)

    nx_graphs[rel_key] = G
    print(f"  ✓ Đồ thị '{rel_key:<12}': {G.number_of_nodes():>4} đỉnh, {G.number_of_edges():>6} cạnh vô hướng.")

print("=" * 90)
print("✓ HOÀN THÀNH KHỐI 1 (CELL 1 - CELL 3): HỆ THỐNG ĐÃ NẠP ĐẦY ĐỦ DỮ LIỆU ĐỂ PHÂN CỤM!")
print("=" * 90)

# %% [Cell 4]: Hàm get_embeddings, optimal_k_clustering & Thực thi phân cụm K-Means cho 9 mô hình
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3. Clustering Pipeline
#
# For each model:
# 1. L2-normalize embeddings
# 2. Search for optimal K using Silhouette score across k in [2, 50]
# 3. Run K-Means with multiple seeds (10 seeds) for stability at optimal k
# 4. Save best clustering results to data/outputs/clusters/
# -----------------------------------------

# STREAMING_CHUNK:Định nghĩa hàm get_embeddings và optimal_k_clustering...
try:
    from IPython.display import display
except ImportError:
    display = print

def get_embeddings(gdf: gpd.GeoDataFrame, model_type: str) -> np.ndarray:
    """Trích xuất ma trận vector nhúng và chuẩn hóa L2 về mặt cầu đơn vị."""
    if model_type == "baseline":
        # Đối với Baseline PCA: Giảm chiều tuyến tính từ 16 POIs ban đầu
        feat_cols = [
            c for c in gdf.columns
            if c not in ["geometry", "cluster", "cluster_label_rank"]
            and gdf[c].dtype in ["float64", "float32", "int64"]
        ]
        Z_raw = gdf[feat_cols].values
        n_comp = min(len(EMBEDDING_COLS), Z_raw.shape[1])
        pca = PCA(n_components=n_comp, random_state=config.get("seeds", {}).get("global", 42))
        Z = pca.fit_transform(Z_raw)
        if Z.shape[1] < len(EMBEDDING_COLS):
            pad_width = len(EMBEDDING_COLS) - Z.shape[1]
            Z = np.pad(Z, ((0, 0), (0, pad_width)), mode="constant")
    else:
        Z = gdf[EMBEDDING_COLS].values

    # Chuẩn hóa mặt cầu L2: ||z||_2 = 1 giúp khoảng cách Euclid tỷ lệ thuận với góc Cosine
    return normalize(Z, norm="l2")


def optimal_k_clustering(
    Z: np.ndarray,
    graphs: dict[str, nx.Graph],
    node_ids: list[str] | pd.Index,
    k_range: range,
    n_seeds: int = 10,
    base_seed: int = 42,
) -> dict:
    """Quét tìm số cụm k tối ưu theo Silhouette và chạy lặp 10 seeds để đo độ ổn định."""
    node_ids = list(node_ids)
    if len(node_ids) != len(Z):
        raise ValueError(f"Số lượng ID đỉnh ({len(node_ids)}) không khớp với số hàng vector nhúng ({len(Z)})")

    # Kiểm tra tính khớp 100% của danh sách đỉnh với 3 đồ thị NetworkX
    unmatched_by_rel = {
        rel: set(node_ids).symmetric_difference(G.nodes)
        for rel, G in graphs.items()
    }
    unmatched_by_rel = {rel: nodes for rel, nodes in unmatched_by_rel.items() if nodes}
    if unmatched_by_rel:
        details = ", ".join(f"{rel}: {len(nodes)}" for rel, nodes in unmatched_by_rel.items())
        raise ValueError(f"Tập đỉnh của vector nhúng và đồ thị NetworkX không khớp nhau ({details})")

    # 1. Quét dải số cụm k in [2, 50]
    k_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=base_seed, n_init=10)
        labels = km.fit_predict(Z)

        # Gom nhóm cộng đồng tính toán Newman Modularity Q trên 3 quan hệ
        communities = [set() for _ in range(k)]
        for node_id, cluster_id in zip(node_ids, labels):
            communities[cluster_id].add(node_id)

        mods = {}
        for rel, G in graphs.items():
            try:
                mods[rel] = nx.community.modularity(G, communities)
            except Exception:
                mods[rel] = np.nan

        k_scores.append({
            "k": k,
            "silhouette": silhouette_score(Z, labels),
            "calinski_harabasz": calinski_harabasz_score(Z, labels),
            "davies_bouldin": davies_bouldin_score(Z, labels),
            "modularity_contig": mods.get("contig", np.nan),
            "modularity_walk": mods.get("15_min_walk", np.nan),
            "modularity_multi": mods.get("15_min_multi", np.nan),
        })

    k_df = pd.DataFrame(k_scores)
    # Xác định k* tối ưu tại điểm cực đại của Silhouette Score
    best_k = int(k_df.loc[k_df["silhouette"].idxmax(), "k"])

    # 2. Khởi tạo đa hạt nhân (10 seeds) tại k* để đo độ ổn định thống kê
    results = []
    for i in range(n_seeds):
        km = KMeans(n_clusters=best_k, random_state=base_seed + i, n_init=10)
        labels = km.fit_predict(Z)

        communities = [set() for _ in range(best_k)]
        for node_id, cluster_id in zip(node_ids, labels):
            communities[cluster_id].add(node_id)

        mods = {}
        for rel, G in graphs.items():
            try:
                mods[rel] = nx.community.modularity(G, communities)
            except Exception:
                mods[rel] = np.nan

        results.append({
            "labels": labels,
            "silhouette": silhouette_score(Z, labels),
            "calinski_harabasz": calinski_harabasz_score(Z, labels),
            "davies_bouldin": davies_bouldin_score(Z, labels),
            "modularity_contig": mods.get("contig", np.nan),
            "modularity_walk": mods.get("15_min_walk", np.nan),
            "modularity_multi": mods.get("15_min_multi", np.nan),
        })

    # Chọn lượt chạy có Silhouette cao nhất làm nhãn cụm đại diện
    best = max(results, key=lambda x: x["silhouette"])

    return {
        "k": best_k,
        "labels": best["labels"],
        "silhouette": best["silhouette"],
        "silhouette_std": float(np.std([r["silhouette"] for r in results])),
        "calinski_harabasz": best["calinski_harabasz"],
        "calinski_harabasz_std": float(np.std([r["calinski_harabasz"] for r in results])),
        "davies_bouldin": best["davies_bouldin"],
        "davies_bouldin_std": float(np.std([r["davies_bouldin"] for r in results])),
        "modularity_contig": best["modularity_contig"],
        "modularity_contig_std": float(np.std([r["modularity_contig"] for r in results])),
        "modularity_walk": best["modularity_walk"],
        "modularity_walk_std": float(np.std([r["modularity_walk"] for r in results])),
        "modularity_multi": best["modularity_multi"],
        "modularity_multi_std": float(np.std([r["modularity_multi"] for r in results])),
        "k_search": k_df,
    }


# STREAMING_CHUNK:Thực thi vòng lặp phân cụm K-Means cho 9 mô hình...
print("\n" + "=" * 90)
print("-> [Cell 4] BẮT ĐẦU QUY TRÌNH PHÂN CỤM K-MEANS ĐỐI SOÁT 9 MÔ HÌNH HÀ NỘI...")
print("=" * 90)

clustering_results: dict[str, dict] = {}
clustering_seed = config.get("seeds", {}).get("clustering", config.get("seeds", {}).get("kmeans", 42))

for model_name, gdf in model_gdfs.items():
    print(f"\n• Đang phân cụm mô hình: {model_name}...")
    Z = get_embeddings(gdf, MODELS[model_name]["type"])
    result = optimal_k_clustering(
        Z,
        nx_graphs,
        gdf.index,
        K_RANGE,
        n_seeds=10,
        base_seed=clustering_seed,
    )

    # Gán nhãn cụm vào GeoDataFrame
    gdf["cluster"] = result["labels"]
    clustering_results[model_name] = result

    # Xuất tệp nhãn cụm CSV phục vụ kiểm tra
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(":", "")
    out_cluster_csv = CLUSTERS_DIR / f"{safe_name}_kmeans.csv"
    gdf[["cluster"]].to_csv(out_cluster_csv)

    print(
        f"  ✓ Optimal k*={result['k']:>2d} | "
        f"Silhouette: {result['silhouette']:.4f} ± {result['silhouette_std']:.4f} | "
        f"Q(contig): {result['modularity_contig']:.4f} | "
        f"Q(walk): {result['modularity_walk']:.4f} | "
        f"Q(multi): {result['modularity_multi']:.4f}"
    )

print("\n" + "=" * 90)
print("✓ [Cell 4] ĐÃ HOÀN TẤT PHÂN CỤM K-MEANS CHO TOÀN BỘ 9 MÔ HÌNH!")
print("=" * 90)


# %% [Cell 5]: Xuất bảng thống kê phân cụm cốt lõi (Bảng 3 trong bài báo Sato et al., CEUS 2026)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3.5 Cluster Statistics (Table 3 in Sato et al., CEUS 2026)
# Summary metrics: Silhouette Score and Newman Modularity (Contig, Walk, Multi)
# Formatted as Mean ± SD across the search range k in [2, 50].
# -----------------------------------------

# STREAMING_CHUNK:Tổng hợp và xuất Bảng 3 thống kê phân cụm cốt lõi...
print("\n-> [Cell 5] Đang tổng hợp Bảng 3: Chỉ số phân cụm cốt lõi (Table 3 Benchmark)...")

table3_data = {}
for model_name, result in clustering_results.items():
    k_df = result["k_search"]
    sil = k_df["silhouette"]
    mc = k_df["modularity_contig"].dropna()
    mw = k_df["modularity_walk"].dropna()
    mm = k_df["modularity_multi"].dropna()

    table3_data[model_name] = {
        ("Silhouette Score", "Mean ± SD"): f"{sil.mean():.2f} ± {sil.std():.2f}",
        ("Modularity (Contig)", "Mean ± SD"): f"{mc.mean():.2f} ± {mc.std():.2f}" if not mc.empty else "N/A",
        ("Modularity (Walk)", "Mean ± SD"): f"{mw.mean():.2f} ± {mw.std():.2f}" if not mw.empty else "N/A",
        ("Modularity (Multi)", "Mean ± SD"): f"{mm.mean():.2f} ± {mm.std():.2f}" if not mm.empty else "N/A",
    }

df_table3 = pd.DataFrame(table3_data).T
df_table3.columns.names = ["Metric", "Statistic"]

print("\n" + "=" * 90)
print("BẢNG 3: CHỈ SỐ PHÂN CỤM CỐT LÕI QUA DẢI k in [2, 50] (HÀ NỘI CASE STUDY)")
print("=" * 90)
print(df_table3.to_string())
print("=" * 90)

table3_path = TABLES_DIR / "cluster_stats_mean_std.csv"
df_table3.to_csv(table3_path)
print(f"✓ [Cell 5] Đã xuất Bảng 3 thành công tại: {table3_path.resolve()}")


# %% [Cell 6]: Xuất bảng thống kê phân cụm bổ sung (Phụ lục Bảng C.1 trong bài báo)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3.5b Supplemental Cluster Statistics (Appendix Table C.1)
# Calinski-Harabasz and Davies-Bouldin indices formatted as Mean ± SD across k in [2, 50].
# -----------------------------------------

# STREAMING_CHUNK:Tổng hợp và xuất Bảng C.1 chỉ số bổ sung Calinski-Harabasz và Davies-Bouldin...
print("\n-> [Cell 6] Đang tổng hợp Bảng C.1: Chỉ số bổ sung Calinski-Harabasz & Davies-Bouldin...")

appendix_data = {}
for model_name, result in clustering_results.items():
    k_df = result["k_search"]
    ch = k_df["calinski_harabasz"]
    db = k_df["davies_bouldin"]

    appendix_data[model_name] = {
        ("Calinski-Harabasz", "Mean ± SD"): f"{ch.mean():.2f} ± {ch.std():.2f}",
        ("Davies-Bouldin", "Mean ± SD"): f"{db.mean():.2f} ± {db.std():.2f}",
    }

df_appendix = pd.DataFrame(appendix_data).T
df_appendix.columns.names = ["Metric", "Statistic"]

print("\n" + "=" * 90)
print("PHỤ LỤC BẢNG C.1: CHỈ SỐ CALINSKI-HARABASZ & DAVIES-BOULDIN (HÀ NỘI CASE STUDY)")
print("=" * 90)
print(df_appendix.to_string())
print("=" * 90)

appendix_path = TABLES_DIR / "cluster_stats_mean_std_appendix.csv"
df_appendix.to_csv(appendix_path)
print(f"✓ [Cell 6] Đã xuất Phụ lục Bảng C.1 thành công tại: {appendix_path.resolve()}")
print("\n✓ HOÀN THÀNH KHỐI 2 (CELL 4 - CELL 6): ĐÃ HOÀN TẤT PHÂN CỤM VÀ XUẤT 2 BẢNG THỐNG KÊ!")

# %% [Cell 7]: Hàm dùng chung plot_cluster_metric vẽ đường cong chỉ số qua dải k in [2, 50]
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3.6 Cluster Charts Utility
# Plots performance metric across candidate k in [2, 50] for all 9 models.
# Color scheme:
# - Baseline (PCA): Black (#000000)
# - Proposed Models 1-4: Solid lines with Okabe-Ito palette
# - Structure-only Models 1S-4S: Dashed lines with open markers
# - Optimal k*: Marked with a star symbol on selected plots
# -----------------------------------------

# STREAMING_CHUNK:Định nghĩa hàm plot_cluster_metric vẽ biểu đồ 9 mô hình...
def plot_cluster_metric(metric_col: str, ylabel: str, save_filename: str, show_optimal_k: bool = False) -> None:
    """Vẽ đường cong biến thiên chỉ số phân cụm theo dải k cho 9 mô hình."""
    model_colors = {
        "Baseline (PCA)":              "#000000",  # Đen
        "Model 1 (GAT-GAE)":           "#009E73",  # Xanh lục (Bluish green)
        "Model 2 (HAN-GAE: Walk)":     "#E69F00",  # Cam hổ phách (Orange)
        "Model 3 (HAN-GAE: Multi)":    "#56B4E9",  # Xanh da trời (Sky blue)
        "Model 4 (HAN-GAE: All)":      "#0072B2",  # Xanh dương đậm (Blue)
        "Model 1S (GAT-GAE: Struct)":  "#009E73",
        "Model 2S (HAN-GAE: Walk-S)":  "#E69F00",
        "Model 3S (HAN-GAE: Multi-S)": "#56B4E9",
        "Model 4S (HAN-GAE: All-S)":   "#0072B2",
    }

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

    # STREAMING_CHUNK:Vẽ các mô hình triệt tiêu cấu trúc nét đứt...
    struct_models = [m for m in clustering_results if "S (" in m]
    other_models = [m for m in clustering_results if "S (" not in m]

    def _mark_optimal_k(m: str, k_df: pd.DataFrame, is_struct: bool) -> None:
        """Đánh dấu ngôi sao tại điểm k* tối ưu của mô hình."""
        opt_k = clustering_results[m]["k"]
        opt_val = k_df.loc[k_df["k"] == opt_k, metric_col]
        if opt_val.empty or pd.isna(opt_val.values[0]):
            return

        star_size = 11
        if is_struct:
            ax.plot(
                opt_k, opt_val.values[0], marker="*", markersize=star_size,
                markerfacecolor="white", markeredgecolor=model_colors[m],
                markeredgewidth=1.4, linestyle="none", zorder=6
            )
        else:
            ax.plot(
                opt_k, opt_val.values[0], marker="*", markersize=star_size,
                markerfacecolor=model_colors[m], markeredgecolor=model_colors[m],
                markeredgewidth=0.8, linestyle="none", zorder=6
            )

    # 1. Vẽ các mô hình chỉ có cấu trúc (1S - 4S) dạng nét đứt rỗng tâm
    for m in struct_models:
        k_df = clustering_results[m]["k_search"]
        if metric_col not in k_df.columns or k_df[metric_col].isna().all():
            continue

        ax.plot(
            k_df["k"], k_df[metric_col],
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgecolor=model_colors[m],
            markeredgewidth=1.2,
            color=model_colors[m],
            label=m,
            linewidth=0.8,
            linestyle="--",
            alpha=1.0,
        )
        if show_optimal_k:
            _mark_optimal_k(m, k_df, is_struct=True)

    # STREAMING_CHUNK:Vẽ các mô hình chính nét liền đặc tâm...
    # 2. Vẽ Baseline PCA và các mô hình đề xuất chính (Models 1 - 4) nét liền
    for m in other_models:
        k_df = clustering_results[m]["k_search"]
        if metric_col not in k_df.columns or k_df[metric_col].isna().all():
            continue

        ax.plot(
            k_df["k"], k_df[metric_col],
            marker="o",
            markersize=4.5,
            color=model_colors[m],
            label=m,
            linewidth=1.1,
            linestyle="-",
            alpha=1.0,
        )
        if show_optimal_k:
            _mark_optimal_k(m, k_df, is_struct=False)

    # STREAMING_CHUNK:Định dạng trục tọa độ và thiết lập chú giải 3 cột...
    ax.set_xlabel("Number of Clusters (k)", fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Chuẩn hóa bảng chú giải (Legend) bố cục 3 cột ở dưới đáy
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    proposed_names = [
        "Model 1 (GAT-GAE)",
        "Model 2 (HAN-GAE: Walk)",
        "Model 3 (HAN-GAE: Multi)",
        "Model 4 (HAN-GAE: All)"
    ]
    struct_names = [
        "Model 1S (GAT-GAE: Struct)",
        "Model 2S (HAN-GAE: Walk-S)",
        "Model 3S (HAN-GAE: Multi-S)",
        "Model 4S (HAN-GAE: All-S)"
    ]

    star_handle = Line2D(
        [], [], marker="*", color="none", markerfacecolor="#555555",
        markeredgecolor="black", markeredgewidth=0.8, markersize=12,
        linestyle="none"
    )

    ordered_handles = []
    ordered_labels = []

    col3 = [
        ("Baseline (PCA)", label_to_handle.get("Baseline (PCA)", Line2D([], [], linestyle="none"))),
        ("Optimal $k$" if show_optimal_k else "", star_handle if show_optimal_k else Line2D([], [], linestyle="none")),
        ("", Line2D([], [], linestyle="none")),
        ("", Line2D([], [], linestyle="none")),
    ]

    for i in range(4):
        ordered_handles.append(label_to_handle.get(proposed_names[i], Line2D([], [], linestyle="none")))
        ordered_labels.append(proposed_names[i])
    for i in range(4):
        ordered_handles.append(label_to_handle.get(struct_names[i], Line2D([], [], linestyle="none")))
        ordered_labels.append(struct_names[i])
    for i in range(4):
        ordered_handles.append(col3[i][1])
        ordered_labels.append(col3[i][0])

    legend = ax.legend(
        handles=ordered_handles, labels=ordered_labels,
        loc="lower center", bbox_to_anchor=(0.5, -0.42),
        ncol=3, fontsize=11, title_fontsize=13,
        frameon=True, facecolor="white", framealpha=1, edgecolor="lightgray",
        columnspacing=1.2, handletextpad=0.5
    )

    keep_texts = set(labels) | {"Baseline (PCA)"}
    if show_optimal_k:
        keep_texts.add("Optimal $k$")
    for text in legend.get_texts():
        if text.get_text() == "" or text.get_text() not in keep_texts:
            text.set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24)
    save_path = FIGURES_DIR / save_filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Đã lưu biểu đồ '{ylabel}' tại: {save_path.resolve()}")
    plt.show()
    plt.close(fig)


# %% [Cell 8]: Vẽ biểu đồ Silhouette Score qua dải k in [2, 50] (Hình 14a)
# --- Ghi chú giải thích từ Notebook gốc ---
# Plot Silhouette Score across candidate k in [2, 50] (with optimal k* marked)
# -----------------------------------------

# STREAMING_CHUNK:Vẽ biểu đồ Silhouette Score qua dải k...
print("\n-> [Cell 8] Đang kết xuất biểu đồ Silhouette Score qua dải k in [2, 50] (Hình 14a)...")
plot_cluster_metric("silhouette", "Silhouette Score", "silhouette.jpg", show_optimal_k=True)


# %% [Cell 9]: Vẽ biểu đồ Davies-Bouldin Index qua dải k in [2, 50] (Phụ lục C)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3.7 Cluster Chart (Davies-Bouldin Index - Lower is better)
# -----------------------------------------

# STREAMING_CHUNK:Vẽ biểu đồ Davies-Bouldin Index qua dải k...
print("\n-> [Cell 9] Đang kết xuất biểu đồ Davies-Bouldin Index (Phụ lục C)...")
plot_cluster_metric("davies_bouldin", "Davies-Bouldin Index", "davies_bouldin.jpg", show_optimal_k=False)


# %% [Cell 10]: Vẽ biểu đồ Calinski-Harabasz Index qua dải k in [2, 50] (Phụ lục C)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3.8 Cluster Chart (Calinski-Harabasz Index - Higher is better)
# -----------------------------------------

# STREAMING_CHUNK:Vẽ biểu đồ Calinski-Harabasz Index qua dải k...
print("\n-> [Cell 10] Đang kết xuất biểu đồ Calinski-Harabasz Index (Phụ lục C)...")
plot_cluster_metric("calinski_harabasz", "Calinski-Harabasz Index", "calinski_harabasz.jpg", show_optimal_k=False)


# %% [Cell 11]: Vẽ biểu đồ Newman Modularity - Quan hệ Tiếp giáp Contiguity (Hình 14b)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3.9 Cluster Chart (Modularity - Contiguity)
# -----------------------------------------

# STREAMING_CHUNK:Vẽ biểu đồ Modularity Contiguity qua dải k...
print("\n-> [Cell 11] Đang kết xuất biểu đồ Modularity (Contiguity) (Hình 14b)...")
plot_cluster_metric("modularity_contig", "Modularity", "modularity_contig.jpg")


# %% [Cell 12]: Vẽ biểu đồ Newman Modularity - Quan hệ Đi bộ 15 phút (Hình 14c)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3.10 Cluster Chart (Modularity - 15 Min Walk)
# -----------------------------------------

# STREAMING_CHUNK:Vẽ biểu đồ Modularity Walk qua dải k...
print("\n-> [Cell 12] Đang kết xuất biểu đồ Modularity (15-Min Walk) (Hình 14c)...")
plot_cluster_metric("modularity_walk", "Modularity", "modularity_walk.jpg")


# %% [Cell 13]: Vẽ biểu đồ Newman Modularity - Quan hệ Xe buýt Đa phương thức (Hình 14d)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3.11 Cluster Chart (Modularity - 15 Min Multi)
# -----------------------------------------

# STREAMING_CHUNK:Vẽ biểu đồ Modularity Multi qua dải k...
print("\n-> [Cell 13] Đang kết xuất biểu đồ Modularity (15-Min Multi) (Hình 14d)...")
plot_cluster_metric("modularity_multi", "Modularity", "modularity_multi.jpg")

print("\n" + "=" * 90)
print("✓ HOÀN THÀNH KHỐI 3 (CELL 7 - CELL 13): ĐÃ XUẤT ĐỦ 6 BIỂU ĐỒ CHỈ SỐ QUA DẢI k in [2, 50]!")
print("=" * 90)

# %% [Cell 14]: Thiết lập tham số không gian, tọa độ tâm CBD và bảng màu toàn cục
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 6. Shared Setup for Map Visualizations
# Transform CBD coordinates to projected CRS, locate reference CBD zone,
# and define the global shuffled categorical color palette.
# -----------------------------------------

# STREAMING_CHUNK:Cấu hình tham số không gian và tọa độ tâm CBD Hà Nội...
print("\n-> [Cell 14] Đang thiết lập tham số không gian và định vị ô tâm CBD Hà Nội...")

# Chuyển đổi tọa độ tâm CBD (Hồ Hoàn Kiếm) sang hệ tọa độ phẳng mét UTM Zone 48N (EPSG:32648)
transformer = pyproj.Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
centre_x, centre_y = transformer.transform(CITY_CENTRE["lon"], CITY_CENTRE["lat"])

# Định vị ô lục giác H3 gần mốc Hồ Hoàn Kiếm nhất
ref_gdf = model_gdfs["Model 1 (GAT-GAE)"]
centre_oa_idx = ref_gdf.geometry.distance(Point(centre_x, centre_y)).idxmin()
print(f"✓ Điểm mốc tâm CBD Hồ Hoàn Kiếm: X={centre_x:.1f} m, Y={centre_y:.1f} m (EPSG:32648)")
print(f"✓ Ô lục giác chứa tâm CBD (Reference Zone): zone_id = '{centre_oa_idx}'")

# Khởi tạo bảng màu phân loại toàn cục ngẫu nhiên có hạt giống cố định seed=42
base_palette = (
    sns.color_palette("bright", 10) +
    sns.color_palette("dark", 10) +
    sns.color_palette("husl", 15) +
    sns.color_palette("pastel", 15)
)
rng = np.random.RandomState(42)
GLOBAL_COLORS = [to_hex(c) for c in base_palette]
rng.shuffle(GLOBAL_COLORS)

# STREAMING_CHUNK:Phân tách danh sách các mô hình chính và mô hình chỉ có cấu trúc...
proposed_models = [m for m in model_gdfs if "S (" not in m and m != "Baseline (PCA)"]
baseline_models = [m for m in model_gdfs if "S (" in m]

print(f"✓ Đã thiết lập bảng màu toàn cục {len(GLOBAL_COLORS)} màu.")
print(f"✓ Mô hình đề xuất chính ({len(proposed_models)}): {proposed_models}")
print(f"✓ Mô hình bóc tách cấu trúc ({len(baseline_models)}): {baseline_models}")


# %% [Cell 15]: Khai báo danh mục thứ tự 9 mô hình vẽ bản đồ
# --- Ghi chú giải thích từ Notebook gốc ---
# Build all_models list: Baseline (PCA) first, then proposed models, then ablation variants.
# -----------------------------------------

# STREAMING_CHUNK:Khai báo danh mục 9 mô hình theo bố cục ma trận 3x3...
all_models = ["Baseline (PCA)"] + proposed_models + baseline_models
print(f"\n✓ [Cell 15] Danh mục 9 mô hình hiển thị ma trận 3x3: {all_models}")


# %% [Cell 16]: Ma trận bản đồ phân vùng chức năng đô thị (Hình 15a trong bài báo)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 6.1 Cluster Maps (Figure 15a in Sato et al., CEUS 2026)
# - Rank cluster labels by Cosine Similarity to the City-Centre cluster centroid (Rank 0 = Centre cluster)
# - Dissolve contiguous zones per cluster rank and place numeric labels
# - Filter out tiny fragmented slivers to eliminate label collisions
# - Adjust overlapping label annotations
# -----------------------------------------

# STREAMING_CHUNK:Định nghĩa hàm khử va chạm nhãn adjust_component_labels...
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
                if overlap_x < overlap_y:
                    direction = 1.0 if delta[0] > 0 or (delta[0] == 0 and j > i) else -1.0
                    shift = direction * overlap_x / 2
                    positions[i, 0] -= shift
                    positions[j, 0] += shift
                else:
                    direction = 1.0 if delta[1] > 0 or (delta[1] == 0 and j > i) else -1.0
                    shift = direction * overlap_y / 2
                    positions[i, 1] -= shift
                    positions[j, 1] += shift

        positions[:, 0] = np.clip(positions[:, 0], axes_bbox.x0 + widths / 2, axes_bbox.x1 - widths / 2)
        positions[:, 1] = np.clip(positions[:, 1], axes_bbox.y0 + heights / 2, axes_bbox.y1 - heights / 2)
        if not moved:
            break

    inverse = ax.transData.inverted()
    for (text, anchor), display_pos, original_pos in zip(labels, positions, anchors):
        data_pos = inverse.transform(display_pos)
        text.set_position(data_pos)
        if np.linalg.norm(display_pos - original_pos) > 4:
            ax.annotate(
                "", xy=anchor, xytext=data_pos,
                arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.6, "alpha": 0.75},
                zorder=text.get_zorder() - 0.1,
            )


# STREAMING_CHUNK:Vẽ ma trận 3x3 bản đồ phân vùng chức năng Hình 15a...
print("\n-> [Cell 16] Đang vẽ ma trận 3x3 bản đồ phân vùng chức năng đô thị (Hình 15a)...")

MIN_CLUSTER_AREA = 150000.0  # Lọc diện tích tối thiểu (>1.5 ô H3) để loại bỏ nhãn rác trên mảnh vụn

for model_list, n_rows, n_cols, save_name in [
    (all_models, 3, 3, "cluster_maps.jpg"),
]:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 10), dpi=300)
    axes = np.atleast_2d(axes)

    for idx, model_name in enumerate(model_list):
        ax = axes[idx // n_cols, idx % n_cols]
        gdf = model_gdfs[model_name]
        result = clustering_results[model_name]
        Z = get_embeddings(gdf, MODELS[model_name]["type"])

        # 1. Tính độ tương đồng Cosine giữa tâm cụm chứa mốc Hồ Gươm và tất cả các tâm cụm khác
        centre_pos = list(gdf.index).index(centre_oa_idx)
        target_label = result["labels"][centre_pos]
        centroids = pd.DataFrame(Z).assign(label=result["labels"]).groupby("label").mean()
        target_vec = centroids.loc[target_label].values.reshape(1, -1)
        cluster_sims = dict(zip(centroids.index, cosine_similarity(target_vec, centroids.values)[0]))

        # Sắp xếp thứ hạng nhãn cụm (Hạng 0 là cụm tâm trung tâm Hồ Gươm)
        sorted_labels = sorted(cluster_sims, key=cluster_sims.get, reverse=True)
        rank_color_map = {label: GLOBAL_COLORS[i % len(GLOBAL_COLORS)] for i, label in enumerate(sorted_labels)}
        label_to_rank = {label: i for i, label in enumerate(sorted_labels)}

        # Tô màu từng ô H3 theo thứ hạng cụm đã sắp xếp
        node_colors = [rank_color_map[l] for l in result["labels"]]

        # STREAMING_CHUNK:Vẽ các đa giác ô H3 và gắn nhãn phân vùng chức năng...
        if c2g is not None and hasattr(c2g, "plot_graph"):
            try:
                c2g.plot_graph(
                    nodes=gdf,
                    ax=ax,
                    node_color=node_colors,
                    node_edgecolor=node_colors,
                    node_alpha=1.0,
                    linewidth=0.1,
                    bgcolor="white",
                )
            except Exception:
                gdf.plot(ax=ax, color=node_colors, edgecolor="white", linewidth=0.2)
        else:
            gdf.plot(ax=ax, color=node_colors, edgecolor="white", linewidth=0.2)

        # 2. Gắn nhãn số thứ hạng cụm lên từng vùng không gian liên thông (Models 1 - 4)
        component_labels = []
        if model_name in proposed_models:
            gdf_temp = gdf.assign(cluster_label_rank=[label_to_rank[l] for l in result["labels"]])
            dissolved = gdf_temp.dissolve(by="cluster_label_rank").reset_index()

            for _, row in dissolved.iterrows():
                geom = row.geometry
                polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom] if geom.geom_type == "Polygon" else []
                # Lọc chỉ gắn nhãn cho các khối đa giác có diện tích đủ lớn để tránh rối bản đồ
                large_polys = [p for p in polys if p.area >= MIN_CLUSTER_AREA]
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

        # STREAMING_CHUNK:Thiết lập tiêu đề và bảng chú giải thích ứng cho bản đồ phân cụm...
        # Đánh dấu vị trí tâm Hồ Gươm bằng ngôi sao nhỏ viền trắng
        ax.scatter([centre_x], [centre_y], c="black", s=130, marker="*", zorder=12, edgecolors="white", linewidths=1.2)

        # Bảng chú giải (Legend) thích ứng linh hoạt số cột
        legend_elements = [
            Patch(facecolor=rank_color_map[ol], edgecolor="none", label=f"{i}")
            for i, ol in enumerate(sorted_labels)
        ]
        n_leg_cols = 3 if len(sorted_labels) > 18 else (2 if len(sorted_labels) > 6 else 1)
        leg = ax.legend(
            handles=legend_elements, loc="center left", bbox_to_anchor=(1.01, 0.5),
            title="Cluster", title_fontsize=16, frameon=False, prop={"size": 13},
            ncol=n_leg_cols, columnspacing=0.4, handletextpad=0.4
        )
        if leg.get_title():
            leg.get_title().set_fontweight("bold")

        ax.set_title(
            f"{model_name}\n(k*={result['k']}, Silhouette={result['silhouette']:.2f})",
            fontsize=22, fontweight="bold", pad=10
        )
        ax.set_axis_off()

    # Ẩn các ô trống nếu có
    for i in range(len(model_list), n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    cluster_maps_path = FIGURES_DIR / save_name
    plt.savefig(cluster_maps_path, dpi=300, bbox_inches="tight")
    print(f"✓ [Cell 16] Đã lưu ma trận bản đồ phân vùng chức năng tại: {cluster_maps_path.resolve()}")
    plt.show()
    plt.close(fig)


# %% [Cell 17]: Ma trận bản đồ tương đồng Cosine từ tâm CBD (Hình 15b trong bài báo)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 6.2 Cosine Similarity Maps (Figure 15b in Sato et al., CEUS 2026)
# Calculates cosine similarity between each OA's embedding and the city-centre OA embedding.
# Color scale: RdYlBu_r [-1.0 to +1.0].
# -----------------------------------------

# STREAMING_CHUNK:Vẽ ma trận 3x3 bản đồ tương đồng Cosine Hình 15b...
print("\n-> [Cell 17] Đang vẽ ma trận 3x3 bản đồ tương đồng Cosine so với tâm CBD (Hình 15b)...")

cmap_cos = plt.get_cmap("RdYlBu_r")
norm_cos = mcolors.Normalize(vmin=-1.0, vmax=1.0, clip=True)

for model_list, n_rows, n_cols, save_name in [
    (all_models, 3, 3, "similarity_maps.jpg"),
]:
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 10), dpi=300)
    axes = np.atleast_2d(axes)

    for idx, model_name in enumerate(model_list):
        ax = axes[idx // n_cols, idx % n_cols]
        gdf = model_gdfs[model_name]
        Z = get_embeddings(gdf, MODELS[model_name]["type"])

        # Tính độ tương đồng Cosine giữa vector từng ô và ô tâm CBD Hồ Hoàn Kiếm
        centre_pos = list(gdf.index).index(centre_oa_idx)
        sims = cosine_similarity(Z[centre_pos:centre_pos + 1], Z)[0]

        # Ánh xạ độ tương đồng sang màu sắc theo thang RdYlBu_r
        node_colors = [to_hex(cmap_cos(norm_cos(v))) for v in sims]

        # STREAMING_CHUNK:Vẽ bản đồ tương đồng Cosine và đánh dấu tâm Hồ Hoàn Kiếm...
        if c2g is not None and hasattr(c2g, "plot_graph"):
            try:
                c2g.plot_graph(
                    nodes=gdf,
                    ax=ax,
                    node_color=node_colors,
                    node_edgecolor=node_colors,
                    node_alpha=1.0,
                    linewidth=0.1,
                    bgcolor="white",
                )
            except Exception:
                gdf.plot(ax=ax, color=node_colors, edgecolor="white", linewidth=0.2)
        else:
            gdf.plot(ax=ax, color=node_colors, edgecolor="white", linewidth=0.2)

        # Đánh dấu vị trí mốc Hồ Hoàn Kiếm bằng ngôi sao đen viền trắng nổi bật
        ax.scatter(
            [centre_x], [centre_y], c="black", s=180, marker="*", zorder=10,
            edgecolors="white", linewidths=1.6
        )

        # Thêm thanh đo màu sắc (Colorbar)
        sm = plt.cm.ScalarMappable(cmap=cmap_cos, norm=norm_cos)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.72, pad=0.02)
        cbar.set_label("Cosine Similarity", fontsize=20, weight="bold")
        for label in cbar.ax.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight("bold")

        ax.set_title(f"{model_name}", fontsize=24, fontweight="bold", pad=12)
        ax.set_axis_off()

    # Ẩn các ô trống nếu có
    for i in range(len(model_list), n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    plt.tight_layout()
    sim_maps_path = FIGURES_DIR / save_name
    plt.savefig(sim_maps_path, dpi=300, bbox_inches="tight")
    print(f"✓ [Cell 17] Đã lưu ma trận bản đồ tương đồng Cosine tại: {sim_maps_path.resolve()}")
    plt.show()
    plt.close(fig)

print("\n" + "=" * 90)
print("✓ HOÀN THÀNH KHỐI 4 (CELL 14 - CELL 17): ĐÃ XUẤT ĐỦ 2 MA TRẬN BẢN ĐỒ LỚN (HÌNH 15a & 15b)!")
print("=" * 90)

# %% [Cell 18]: Biểu đồ cột trung bình trọng số chú ý ngữ nghĩa beta (attention_bar.jpg)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 7. Attention Summary & Bar Chart
# Computes average attention weight beta per relation for each HAN layer
# and plots faceted bar charts (Layer 1 vs Layer 2).
# -----------------------------------------

# STREAMING_CHUNK:Tính toán tổng hợp trọng số Attention beta và vẽ biểu đồ cột...
print("\n-> [Cell 18] Đang tính toán và kết xuất biểu đồ cột trọng số Attention beta...")

attention_summary = []
for model_name, layers_data in attention_data.items():
    for layer_name, beta_df in layers_data.items():
        for rel in beta_df.columns:
            attention_summary.append({
                "Model": model_name,
                "Layer": layer_name,
                "Relation": rel,
                "Mean β": float(beta_df[rel].mean()),
                "Std β": float(beta_df[rel].std())
            })

if attention_summary:
    df_att = pd.DataFrame(attention_summary)
    
    # Biểu đồ cột phân lớp Faceted Catplot
    g = sns.catplot(
        data=df_att,
        kind="bar",
        x="Model",
        y="Mean β",
        hue="Relation",
        col="Layer",
        palette="viridis",
        height=5.5,
        aspect=1.2,
        legend_out=True
    )

    g.set_axis_labels("", "Mean Attention Weight (β)", fontsize=13, fontweight="bold")
    g.set_titles("{col_name}", fontsize=15, fontweight="bold")
    g.set(ylim=(0, 1.05))

    for ax in g.axes.flat:
        ax.tick_params(axis="y", labelsize=11)
        for label in ax.get_xticklabels():
            label.set_fontsize(10)
            label.set_rotation(20)
            label.set_ha("right")
        ax.grid(True, linestyle="--", alpha=0.5)

    g.tight_layout()
    att_bar_path = FIGURES_DIR / "attention_bar.jpg"
    g.savefig(att_bar_path, dpi=200, bbox_inches="tight")
    print(f"✓ [Cell 18] Đã lưu biểu đồ cột trọng số Attention tại: {att_bar_path.resolve()}")
    plt.show()
    plt.close(g.fig)


# %% [Cell 19]: Ma trận bản đồ nhiệt trọng số Attention beta Layer 1 & Layer 2 (Hình 18)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 7.1 Attention Heatmaps across Space (Figure 18 in Sato et al., CEUS 2026)
# Maps the spatial variation of beta_{i,m} for each relation in Layer 1 and Layer 2.
# Demonstrates how the network prioritizes accessibility in Layer 1 and smoothing in Layer 2.
# -----------------------------------------

# STREAMING_CHUNK:Vẽ ma trận bản đồ nhiệt không gian trọng số Attention Layer 1 và Layer 2...
print("\n-> [Cell 19] Đang kết xuất ma trận bản đồ nhiệt không gian trọng số Attention (Hình 18)...")

# Sử dụng dải màu viridis chuẩn mực
try:
    from cmcrameri import cm as cmc
    cmap_att = cmc.davos
except ImportError:
    cmap_att = plt.get_cmap("viridis")

norm_att = mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)
han_models_list = ["Model 2 (HAN-GAE: Walk)", "Model 3 (HAN-GAE: Multi)", "Model 4 (HAN-GAE: All)"]
all_relations_list = ["contig", "15_min_walk", "15_min_multi"]
layers_list = ["Layer 1", "Layer 2"]

for layer in layers_list:
    fig, axes = plt.subplots(len(han_models_list), len(all_relations_list), figsize=(20, 18), dpi=300)
    
    for row_idx, model_name in enumerate(han_models_list):
        model_relations = RELATION_NAMES[model_name]
        gdf = model_gdfs[model_name].copy()
        beta_df = attention_data[model_name][layer]

        for col_idx, rel in enumerate(all_relations_list):
            ax = axes[row_idx, col_idx]

            if rel not in model_relations:
                # Nếu quan hệ không có trong mô hình, hiển thị ô placeholder xám nhạt
                ax.text(
                    0.5, 0.5, f"{rel}\n(Not Defined)",
                    ha="center", va="center", fontsize=18, color="gray",
                    transform=ax.transAxes
                )
                ax.axis("off")
                ax.set_facecolor("#f1f5f9")
                ax.set_title(f"{model_name}\n{rel}", fontsize=18, fontweight="bold", color="gray", pad=8)
                continue

            beta_values = beta_df[rel].values
            node_colors = [to_hex(cmap_att(norm_att(v))) for v in beta_values]

            if c2g is not None and hasattr(c2g, "plot_graph"):
                try:
                    c2g.plot_graph(
                        nodes=gdf,
                        ax=ax,
                        node_color=node_colors,
                        node_edgecolor=node_colors,
                        node_alpha=1.0,
                        bgcolor="white",
                    )
                except Exception:
                    gdf.plot(ax=ax, color=node_colors, edgecolor="none")
            else:
                gdf.plot(ax=ax, color=node_colors, edgecolor="none")

            sm = plt.cm.ScalarMappable(cmap=cmap_att, norm=norm_att)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.72, pad=0.02)
            cbar.set_label(f"β ({rel})", fontsize=16, weight="bold")
            cbar.ax.tick_params(labelsize=13)

            ax.set_title(f"{model_name}\n{rel}", fontsize=18, fontweight="bold", pad=8)
            ax.set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    layer_filename = f"attention_weight_{layer.lower().replace(' ', '_')}.jpg"
    att_map_path = FIGURES_DIR / layer_filename
    plt.savefig(att_map_path, dpi=300, bbox_inches="tight")
    print(f"✓ [Cell 19] Đã lưu bản đồ nhiệt Attention ({layer}) tại: {att_map_path.resolve()}")
    plt.show()
    plt.close(fig)


# %% [Cell 20]: Chiếu giảm chiều không gian tiềm ẩn bằng thuật toán phi tuyến t-SNE 2D (Hình C.3)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 8. t-SNE 2D Manifold Embedding Visualization (Figure C.3 in Appendix C)
# Projects 32D embeddings onto 2D manifold space using t-SNE (perplexity=30).
# Points are colored by cluster labels ranked relative to city centre.
# -----------------------------------------

# STREAMING_CHUNK:Thực hiện chiếu phi tuyến t-SNE 2D cho 9 mô hình...
print("\n-> [Cell 20] Đang chiếu giảm chiều t-SNE 2D và vẽ ma trận không gian tiềm ẩn (Hình C.3)...")

models_to_tsne = list(model_gdfs.keys())
n_models_tsne = len(models_to_tsne)
n_tsne_cols = 3
n_tsne_rows = (n_models_tsne + n_tsne_cols - 1) // n_tsne_cols

fig, axes = plt.subplots(n_tsne_rows, n_tsne_cols, figsize=(21, n_tsne_rows * 6.8), dpi=300)
axes = axes.flatten()

tsne_seed = config.get("seeds", {}).get("tsne", 42)

for idx, model_name in enumerate(models_to_tsne):
    ax = axes[idx]
    gdf = model_gdfs[model_name]
    result = clustering_results[model_name]
    Z = get_embeddings(gdf, MODELS[model_name]["type"])

    # Xếp hạng cụm theo độ tương đồng Cosine từ tâm CBD (đồng bộ 100% với Hình 15a)
    centre_pos = list(gdf.index).index(centre_oa_idx)
    target_label = result["labels"][centre_pos]
    centroids = pd.DataFrame(Z).assign(label=result["labels"]).groupby("label").mean()
    target_vec = centroids.loc[target_label].values.reshape(1, -1)
    cluster_sims = dict(zip(centroids.index, cosine_similarity(target_vec, centroids.values)[0]))
    sorted_labels = sorted(cluster_sims, key=cluster_sims.get, reverse=True)
    rank_color_map = {label: GLOBAL_COLORS[i % len(GLOBAL_COLORS)] for i, label in enumerate(sorted_labels)}

    # Chiếu t-SNE phi tuyến 32D -> 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=tsne_seed)
    Z_2d = tsne.fit_transform(Z)

    # STREAMING_CHUNK:Vẽ các điểm phân cụm trên không gian t-SNE 2D...
    for rank, label in enumerate(sorted_labels):
        mask = (np.array(result["labels"]) == label)
        ax.scatter(
            Z_2d[mask, 0], Z_2d[mask, 1],
            c=rank_color_map[label], s=18, alpha=0.8, label=str(rank),
            edgecolors="none"
        )

    ax.set_title(f"{model_name}", fontsize=17, fontweight="bold", pad=8)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)

# Ẩn các ô subplot thừa nếu có
for i in range(len(models_to_tsne), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
tsne_fig_path = FIGURES_DIR / "tsne_embeddings.jpg"
plt.savefig(tsne_fig_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 20] Đã lưu hình chiếu không gian tiềm ẩn t-SNE tại: {tsne_fig_path.resolve()}")
plt.show()
plt.close(fig)

print("\n" + "=" * 90)
print("✓ HOÀN THÀNH KHỐI 5 (CELL 18 - CELL 20): ĐÃ KẾT XUẤT ĐỦ CÁC ẤN PHẨM GIẢI THÍCH ATTENTION & t-SNE!")
print("=" * 90)

# %% [Cell 21]: Nạp đồ thị nền và thiết lập tham số vận tốc cho phân tích Isochrone
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 9. Isochrone Analysis & Network Setup
# Khai báo vận tốc đi bộ và xe buýt chuẩn quốc tế:
# - Vận tốc đi bộ: 4.8 km/h = 1.3333 m/s
# - Vận tốc xe buýt trung bình nội đô: 15.0 km/h = 4.1667 m/s
# -----------------------------------------

print("\n-> [Cell 21] Đang nạp đồ thị nền và thiết lập tham số vận tốc cho Isochrone...")

WALKING_SPEED_MPS = 4.8 / 3.6  # 1.3333 m/s (~4.8 km/h)
BUS_SPEED_MPS = 15.0 / 3.6      # 4.1667 m/s (~15.0 km/h)

# Nạp đồ thị dị thể hetero_multi.pt
data_iso_pt = torch.load(GRAPHS_DIR / "hetero_multi.pt", map_location="cpu", weights_only=False)

if c2g is not None and hasattr(c2g, "pyg_to_gdf"):
    try:
        iso_nodes_dict, iso_edges_dict = c2g.pyg_to_gdf(data_iso_pt)
    except Exception:
        iso_nodes_dict = {"oa": model_gdfs["Model 4 (HAN-GAE: All)"]}
        iso_edges_dict = {}
else:
    iso_nodes_dict = {"oa": model_gdfs["Model 4 (HAN-GAE: All)"]}
    iso_edges_dict = {}

print(f"✓ [Cell 21] Đã nạp mạng lưới hạ tầng. Vận tốc đi bộ = {WALKING_SPEED_MPS:.4f} m/s, Xe buýt = {BUS_SPEED_MPS:.4f} m/s.")


# %% [Cell 22]: Tính toán các đa giác vùng đẳng thời (5, 10, 15, 20, 25, 30 phút) từ Hồ Hoàn Kiếm
# --- Ghi chú giải thích từ Notebook gốc ---
# Generates travel-time isochrones from city centre (Hoan Kiem Lake)
# across [5, 10, 15, 20, 25, 30] minutes for Walk-only and Multimodal.
# -----------------------------------------

print("\n-> [Cell 22] Đang tính toán đa giác vùng đẳng thời 5p - 30p từ mốc Hồ Hoàn Kiếm...")

from shapely.ops import unary_union

# Tọa độ điểm tâm Hồ Hoàn Kiếm
center_pt_geom = Point(centre_x, centre_y)

# Dải thời gian khảo sát: 5, 10, 15, 20, 25, 30 phút
travel_times_mins = list(range(5, 35, 5))
travel_times_sorted = sorted(travel_times_mins, reverse=True)
thresholds_sec = [m * 60 for m in travel_times_sorted]

iso_gdfs_walk = {}
iso_gdfs_multi = {}

oa_geoms = model_gdfs["Model 4 (HAN-GAE: All)"]["geometry"]

# Xây dựng vùng đẳng thời qua bán kính di chuyển thực tế của 985 ô H3
for mins in travel_times_sorted:
    t_sec = mins * 60.0

    # 1. Thuần đi bộ đường phố: Bán kính cự ly R = v_walk * t
    r_walk_m = WALKING_SPEED_MPS * t_sec
    walk_mask = oa_geoms.centroid.distance(center_pt_geom) <= r_walk_m
    if hasattr(oa_geoms, "union_all"):
        walk_poly = oa_geoms[walk_mask].union_all()
    else:
        walk_poly = unary_union(oa_geoms[walk_mask])

    if walk_poly is None or walk_poly.is_empty:
        walk_poly = center_pt_geom.buffer(max(150.0, r_walk_m))

    iso_gdfs_walk[mins] = gpd.GeoDataFrame([{"geometry": walk_poly, "threshold": t_sec}], crs=TARGET_CRS)

    # 2. Đa phương thức xe buýt: Xe buýt giúp mở rộng phạm vi di chuyển cự ly xa
    r_multi_m = min(12000.0, (WALKING_SPEED_MPS * 300.0) + (BUS_SPEED_MPS * max(0.0, t_sec - 300.0)))
    multi_mask = oa_geoms.centroid.distance(center_pt_geom) <= r_multi_m
    if hasattr(oa_geoms, "union_all"):
        multi_poly = oa_geoms[multi_mask].union_all()
    else:
        multi_poly = unary_union(oa_geoms[multi_mask])

    if multi_poly is None or multi_poly.is_empty:
        multi_poly = center_pt_geom.buffer(max(300.0, r_multi_m))

    iso_gdfs_multi[mins] = gpd.GeoDataFrame([{"geometry": multi_poly, "threshold": t_sec}], crs=TARGET_CRS)

print(f"✓ [Cell 22] Đã tính xong các đa giác đẳng thời ({len(travel_times_sorted)} dải thời gian: {travel_times_sorted} phút).")


# %% [Cell 23]: Trực quan hóa bản đồ vùng đẳng thời Isochrones (Hình 16 trong bài báo)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 9.1 Isochrone Visualization (Figure 16 in Sato et al., CEUS 2026)
# Compares Walk-based vs Multimodal accessibility rings from Hoan Kiem Lake.
# -----------------------------------------

print("\n-> [Cell 23] Đang kết xuất bản đồ vùng đẳng thời Isochrones (Hình 16)...")

try:
    from matplotlib_scalebar.scalebar import ScaleBar
    has_scalebar_lib = True
except ImportError:
    has_scalebar_lib = False

# Sử dụng dải màu lapaz chuẩn của bài báo hoặc YlGnBu_r
try:
    from cmcrameri import cm as cmc
    palette_iso = [cmc.lapaz(i) for i in np.linspace(0.1, 0.9, len(travel_times_mins))]
except Exception:
    cmap_base_iso = plt.get_cmap("YlGnBu_r")
    palette_iso = [cmap_base_iso(i) for i in np.linspace(0.15, 0.85, len(travel_times_mins))]

fig, axes = plt.subplots(1, 2, figsize=(16, 10), dpi=300)

study_boundary = oa_polygons.union_all() if hasattr(oa_polygons, "union_all") else unary_union(oa_polygons.geometry)
study_boundary_gs = gpd.GeoSeries([study_boundary], crs=TARGET_CRS)

iso_modes = [
    (axes[0], iso_gdfs_walk, "Walk-based (streets)"),
    (axes[1], iso_gdfs_multi, "Multimodal (streets + bus transits)"),
]

for ax, iso_dict, title in iso_modes:
    # 1. Ranh giới nghiên cứu 985 ô H3
    study_boundary_gs.plot(
        ax=ax, facecolor="none", edgecolor="#1e293b",
        linewidth=1.0, linestyle=(0, (6, 4)), zorder=5,
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white", alpha=0.8)]
    )

    # 2. Vẽ các dải đẳng thời từ lớn đến nhỏ (30p -> 5p)
    legend_elements = []
    for i, mins in enumerate(travel_times_sorted):
        geom_df = iso_dict[mins]
        geom_df.plot(ax=ax, color=palette_iso[i], alpha=0.9, edgecolor="none", zorder=2)
        legend_elements.append(Patch(facecolor=palette_iso[i], edgecolor="none", label=f"{mins} min"))

    # 3. Đánh dấu mốc Hồ Hoàn Kiếm bằng ngôi sao đen
    ax.scatter([centre_x], [centre_y], c="black", s=140, marker="*", zorder=10, edgecolors="white", linewidths=1.2)
    legend_elements.append(
        Line2D([0], [0], marker="*", color="w", label="Hoan Kiem Lake (CBD)",
               markerfacecolor="black", markersize=14, markeredgecolor="white")
    )

    # Thêm bản đồ nền mở Esri
    try:
        ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas, attribution=False)
    except Exception:
        pass

    # Thước tỷ lệ ScaleBar
    if has_scalebar_lib:
        scalebar = ScaleBar(
            1.0, units="m", location="lower right", frameon=True,
            box_alpha=0.6, font_properties={"size": 11}, border_pad=0.4
        )
        ax.add_artist(scalebar)

    ax.legend(handles=legend_elements, loc="upper right", fontsize=11, frameon=True, framealpha=0.92)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.set_axis_off()

plt.tight_layout()
isochrones_path = FIGURES_DIR / "isochrones.jpg"
plt.savefig(isochrones_path, dpi=300, bbox_inches="tight")
print(f"✓ [Cell 23] Đã lưu bản đồ vùng đẳng thời Hình 16 tại: {isochrones_path.resolve()}")
plt.show()
plt.close(fig)


# %% [Cell 24]: Phân khoảng thời gian di chuyển (Travel Time Bins: 0-5, 5-10, ..., >30 min)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 10. Travel Time Binning
# Assigns each OA centroid to 7 discrete travel time bins [0-5, 5-10, ..., >30].
# -----------------------------------------

print("\n-> [Cell 24] Đang phân khoảng thời gian di chuyển (Travel time bins) cho 985 ô H3...")

oa_centroids_gs = model_gdfs["Model 4 (HAN-GAE: All)"].geometry.centroid

bins = [0, 5, 10, 15, 20, 25, 30, np.inf]
bin_labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", ">30"]

t_walk_bin_list = []
t_multi_bin_list = []

for iso_dict, out_list in [(iso_gdfs_walk, t_walk_bin_list), (iso_gdfs_multi, t_multi_bin_list)]:
    min_time_series = pd.Series(np.inf, index=oa_centroids_gs.index, dtype=float)
    for t in sorted(iso_dict.keys()):
        geom_val = iso_dict[t].geometry.iloc[0] if len(iso_dict[t]) > 0 else None
        if geom_val is not None and not geom_val.is_empty:
            mask_inside = oa_centroids_gs.within(geom_val) & (min_time_series > t)
            min_time_series[mask_inside] = float(t)
    out_list.append(pd.cut(min_time_series, bins=bins, labels=bin_labels, include_lowest=True))

t_walk_bin = t_walk_bin_list[0]
t_multi_bin = t_multi_bin_list[0]

print("✓ [Cell 24] Đã phân loại 985 ô H3 vào 7 khoảng thời gian tiếp cận thành công:")
print("  • Đi bộ (Walk):")
print(t_walk_bin.value_counts(sort=False).to_string())
print("  • Đa phương thức (Multimodal):")
print(t_multi_bin.value_counts(sort=False).to_string())


# %% [Cell 25]: Đường cong suy giảm tương đồng Cosine theo khoảng cách thời gian (Hình 17)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 10.1 Distance Decay Curves (Figure 17 in Sato et al., CEUS 2026)
# Calculates average cosine similarity per travel-time bin from Hoan Kiem Lake.
# Demonstrates that HAN-GAE preserves Distance Decay, whereas structure-only models fail.
# -----------------------------------------

print("\n-> [Cell 25] Đang vẽ đường cong suy giảm tương đồng Cosine (Hình 17)...")

decay_records = []

for model_name in model_gdfs:
    gdf = model_gdfs[model_name]
    Z = get_embeddings(gdf, MODELS[model_name]["type"])
    centre_pos = list(gdf.index).index(centre_oa_idx)
    sims = cosine_similarity(Z[centre_pos:centre_pos + 1], Z)[0]

    for mode_name, t_bin in [("Walk-based (streets)", t_walk_bin), ("Multimodal (streets + bus transits)", t_multi_bin)]:
        decay_records.append(pd.DataFrame({
            "Model": model_name,
            "Mode": mode_name,
            "time_bin": t_bin.loc[gdf.index].values,
            "similarity": sims,
        }))

decay_df = pd.concat(decay_records, ignore_index=True)
decay_df["time_bin"] = pd.Categorical(decay_df["time_bin"], categories=bin_labels, ordered=True)

# Tính giá trị trung bình độ tương đồng theo từng khoảng thời gian
decay_summary = (
    decay_df.groupby(["Model", "Mode", "time_bin"], observed=False)["similarity"]
    .mean()
    .reset_index()
    .dropna()
)

# Khai báo biến danh mục mô hình cục bộ an toàn chống lỗi NameError
proposed_names = [
    "Model 1 (GAT-GAE)",
    "Model 2 (HAN-GAE: Walk)",
    "Model 3 (HAN-GAE: Multi)",
    "Model 4 (HAN-GAE: All)"
]
struct_names = [
    "Model 1S (GAT-GAE: Struct)",
    "Model 2S (HAN-GAE: Walk-S)",
    "Model 3S (HAN-GAE: Multi-S)",
    "Model 4S (HAN-GAE: All-S)"
]
other_models_list = [m for m in model_gdfs if "S (" not in m]
struct_models_list = [m for m in model_gdfs if "S (" in m]

model_colors_decay = {
    "Baseline (PCA)":              "#000000",
    "Model 1 (GAT-GAE)":           "#009E73",
    "Model 2 (HAN-GAE: Walk)":     "#E69F00",
    "Model 3 (HAN-GAE: Multi)":    "#56B4E9",
    "Model 4 (HAN-GAE: All)":      "#0072B2",
    "Model 1S (GAT-GAE: Struct)":  "#009E73",
    "Model 2S (HAN-GAE: Walk-S)":  "#E69F00",
    "Model 3S (HAN-GAE: Multi-S)": "#56B4E9",
    "Model 4S (HAN-GAE: All-S)":   "#0072B2",
}

struct_dashes = {
    "Model 1S (GAT-GAE: Struct)":  (3, 1),
    "Model 2S (HAN-GAE: Walk-S)":  (5, 2),
    "Model 3S (HAN-GAE: Multi-S)": (1, 1),
    "Model 4S (HAN-GAE: All-S)":   (5, 1, 1, 1),
}

fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), sharey=True, dpi=300)

for ax, mode in zip(axes, ["Walk-based (streets)", "Multimodal (streets + bus transits)"]):
    sub = decay_summary[decay_summary["Mode"] == mode]

    # Vẽ các mô hình chính bằng nét liền
    for m in other_models_list:
        sub_m = sub[sub["Model"] == m]
        ax.plot(
            sub_m["time_bin"], sub_m["similarity"],
            marker="o", markersize=6.5,
            color=model_colors_decay[m], label=m,
            linewidth=1.2, linestyle="-", alpha=0.95
        )

    # Vẽ các mô hình chỉ có cấu trúc bằng nét đứt rỗng tâm
    for m in struct_models_list:
        sub_m = sub[sub["Model"] == m]
        ax.plot(
            sub_m["time_bin"], sub_m["similarity"],
            marker="o", markersize=7.5, fillstyle="none", markeredgewidth=1.4,
            color=model_colors_decay[m], label=m,
            linewidth=1.4, linestyle="--", dashes=struct_dashes[m], alpha=0.9
        )

    ax.set_title(f"{mode}", fontweight="bold", fontsize=16)
    ax.set_xlabel("Interval of isochrone bin (min)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean cosine similarity to centre", fontsize=14, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)

axes[1].tick_params(labelleft=True)

# Chú giải 3 cột đặt ở dưới đáy
handles, labels = axes[0].get_legend_handles_labels()
label_to_handle = dict(zip(labels, handles))

ordered_handles = []
ordered_labels = []

col3 = [
    ("Baseline (PCA)", label_to_handle.get("Baseline (PCA)", Line2D([], [], linestyle="none"))),
    ("", Line2D([], [], linestyle="none")),
    ("", Line2D([], [], linestyle="none")),
    ("", Line2D([], [], linestyle="none")),
]

for k in range(4):
    ordered_handles.append(label_to_handle.get(proposed_names[k], Line2D([], [], linestyle="none")))
    ordered_labels.append(proposed_names[k])
for k in range(4):
    ordered_handles.append(label_to_handle.get(struct_names[k], Line2D([], [], linestyle="none")))
    ordered_labels.append(struct_names[k])
for k in range(4):
    ordered_handles.append(col3[k][1])
    ordered_labels.append(col3[k][0])

legend = fig.legend(
    handles=ordered_handles, labels=ordered_labels,
    loc="lower center", bbox_to_anchor=(0.5, -0.04),
    ncol=3, fontsize=12, title_fontsize=14,
    frameon=True, facecolor="white", framealpha=1, edgecolor="lightgray", columnspacing=1.2
)

for text in legend.get_texts():
    if text.get_text() == "" or (text.get_text() not in labels and text.get_text() != "Baseline (PCA)"):
        text.set_visible(False)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

decay_fig_path = FIGURES_DIR / "similarity_decay_by_time_bin.jpg"
plt.savefig(decay_fig_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 25] Đã lưu biểu đồ đường cong suy giảm Cosine tại: {decay_fig_path.resolve()}")
plt.show()
plt.close(fig)

# Xuất bảng số liệu CSV
decay_csv_path = TABLES_DIR / "similarity_decay_by_time_bin.csv"
decay_summary.to_csv(decay_csv_path, index=False)
print(f"✓ [Cell 25] Đã xuất dữ liệu đường cong suy giảm tại: {decay_csv_path.resolve()}")

print("\n" + "=" * 90)
print("✓ HOÀN THÀNH TOÀN BỘ 25 CELLS CỦA NOTEBOOK 04 ĐÁNH GIÁ PHÂN CỤM HÀ NỘI!")
print("==========================================================================================")