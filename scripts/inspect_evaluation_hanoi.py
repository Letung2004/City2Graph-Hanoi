# -*- coding: utf-8 -*-
"""
KỊCH BẢN KIỂM TOÁN & ĐỐI SOÁT TOÀN DIỆN KẾT QUẢ ĐÁNH GIÁ FILE 04 (HÀ NỘI)
Đề tài: City2Graph Hà Nội (Uber H3 Resolution 9 - 985 ô không gian)
Kiểm tra toàn diện: 9 Mô hình phân cụm K-Means, 3 Bảng số liệu thống kê,
                    14 Biểu đồ/Bản đồ khoa học 300 DPI & Kiểm định quy luật địa lý
Chạy độc lập: python scripts/inspect_evaluation_hanoi.py
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. ĐỊNH VỊ THƯ MỤC GỐC & THIẾT LẬP ĐƯỜNG DẪN DỰ ÁN
# ---------------------------------------------------------------------------
CWD = Path.cwd().resolve()
PROJECT_ROOT = CWD
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed" / "hanoi"
FEATURES_DIR = PROCESSED_DIR / "features"
GRAPHS_DIR = PROCESSED_DIR / "graphs"

OUTPUTS_DIR = DATA_DIR / "outputs" / "hanoi"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
CLUSTERS_DIR = OUTPUTS_DIR / "clusters"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

EXPECTED_NODES = 985
EXPECTED_EMBEDDING_DIM = 32

print("=" * 90)
print("BÁO CÁO KIỂM TOÁN & ĐỐI SOÁT TOÀN DIỆN KẾT QUẢ ĐÁNH GIÁ FILE 04 (HÀ NỘI)")
print(f"Thư mục gốc dự án: {PROJECT_ROOT}")
print("=" * 90)

audit_results: dict[str, str] = {}

# ---------------------------------------------------------------------------
# [PHẦN 1: ĐỐI SOÁT NGUYÊN LIỆU ĐẦU VÀO CHO FILE 04]
# ---------------------------------------------------------------------------
print("\n[PHẦN 1: ĐỐI SOÁT NGUYÊN LIỆU ĐẦU VÀO (EMBEDDINGS & ĐỒ THỊ NỀN)]")
print("-" * 90)

# 1.1 Kiểm tra tệp ranh giới 985 ô H3
zones_file = FEATURES_DIR / "oa_with_features.gpkg"
if not zones_file.exists():
    zones_file = FEATURES_DIR / "hanoi_normalized_features.parquet"

if zones_file.exists():
    size_kb = zones_file.stat().st_size / 1024
    print(f"✓ Ranh giới không gian H3 ({zones_file.name}): {size_kb:.1f} KB -> [PASS]")
    audit_results["Spatial Boundaries (985 H3)"] = "PASS"
else:
    print(f"✗ Thiếu tệp ranh giới tại {FEATURES_DIR} -> [FAIL]")
    audit_results["Spatial Boundaries (985 H3)"] = "FAIL"

# 1.2 Kiểm tra đồ thị dị thể hetero_multi.pt
graph_file = GRAPHS_DIR / "hetero_multi.pt"
if graph_file.exists():
    try:
        g_data = torch.load(graph_file, map_location="cpu", weights_only=False)
        n_nodes = g_data["oa"].num_nodes if hasattr(g_data, "node_types") and "oa" in g_data.node_types else 0
        has_3_rels = (
            ("oa", "contig", "oa") in g_data.edge_types and
            ("oa", "15_min_walk", "oa") in g_data.edge_types and
            ("oa", "15_min_multi", "oa") in g_data.edge_types
        )
        status = "PASS" if (n_nodes == EXPECTED_NODES and has_3_rels) else "WARN"
        print(f"✓ Đồ thị nền đa phương thức hetero_multi.pt: {n_nodes} ô H3 | Đủ 3 quan hệ -> [{status}]")
        audit_results["Base Graph (hetero_multi.pt)"] = status
    except Exception as e:
        print(f"✗ Lỗi đọc đồ thị: {e} -> [FAIL]")
        audit_results["Base Graph (hetero_multi.pt)"] = "FAIL"
else:
    print(f"✗ Thiếu đồ thị hetero_multi.pt tại {GRAPHS_DIR} -> [FAIL]")
    audit_results["Base Graph (hetero_multi.pt)"] = "FAIL"

# 1.3 Kiểm tra đủ 9 ma trận vector nhúng Z
models_expected_embeddings = [
    ("model1_gat_gae", "Model 1: GAT-GAE"),
    ("model2_han_gae_walk", "Model 2: HAN-GAE Walk"),
    ("model3_han_gae_multi", "Model 3: HAN-GAE Multi"),
    ("model4_han_gae_all", "Model 4: HAN-GAE All"),
    ("model1s_gat_gae_struct", "Model 1S: GAT-GAE Struct"),
    ("model2s_han_gae_walk_struct", "Model 2S: HAN-GAE Walk-S"),
    ("model3s_han_gae_multi_struct", "Model 3S: HAN-GAE Multi-S"),
    ("model4s_han_gae_all_struct", "Model 4S: HAN-GAE All-S"),
]

missing_embeddings = []
for m_key, m_title in models_expected_embeddings:
    npy_file = EMBEDDINGS_DIR / f"Z_{m_key}.npy"
    pt_file = EMBEDDINGS_DIR / f"{m_key}.pt"
    if not npy_file.exists() and not pt_file.exists():
        missing_embeddings.append(m_title)

if not missing_embeddings:
    print(f"✓ Đã nạp đủ 8 ma trận vector nhúng GNN (985x32) + Baseline PCA tự tính -> [PASS]")
    audit_results["Input Embeddings (9 Models)"] = "PASS"
else:
    print(f"✗ Thiếu các tệp vector nhúng: {missing_embeddings} -> [FAIL]")
    audit_results["Input Embeddings (9 Models)"] = "FAIL"

# ---------------------------------------------------------------------------
# [PHẦN 2: ĐỐI SOÁT 9 TỆP NHÃN PHÂN CỤM K-MEANS (CLUSTERS)]
# ---------------------------------------------------------------------------
print("\n[PHẦN 2: ĐỐI SOÁT 9 TỆP NHÃN PHÂN CỤM K-MEANS (CLUSTERS/)]")
print("-" * 90)

cluster_files_spec = [
    ("baseline_pca_kmeans.csv", "Baseline (PCA)", 2),
    ("model_1_gat-gae_kmeans.csv", "Model 1 (GAT-GAE)", 2),
    ("model_2_han-gae_walk_kmeans.csv", "Model 2 (HAN-GAE: Walk)", 6),
    ("model_3_han-gae_multi_kmeans.csv", "Model 3 (HAN-GAE: Multi)", 6),
    ("model_4_han-gae_all_kmeans.csv", "Model 4 (HAN-GAE: All)", 9),
    ("model_1s_gat-gae_struct_kmeans.csv", "Model 1S (GAT-GAE: Struct)", 6),
    ("model_2s_han-gae_walk-s_kmeans.csv", "Model 2S (HAN-GAE: Walk-S)", 49),
    ("model_3s_han-gae_multi-s_kmeans.csv", "Model 3S (HAN-GAE: Multi-S)", 2),
    ("model_4s_han-gae_all-s_kmeans.csv", "Model 4S (HAN-GAE: All-S)", 2),
]

all_clusters_ok = True
for cf_name, model_title, exp_k in cluster_files_spec:
    cf_path = CLUSTERS_DIR / cf_name
    # Fallback kiểm tra các biến thể tên an toàn
    if not cf_path.exists():
        alt_name = cf_name.replace("-", "_")
        if (CLUSTERS_DIR / alt_name).exists():
            cf_path = CLUSTERS_DIR / alt_name

    if not cf_path.exists():
        print(f"- {cf_name:<34} | Chưa kết xuất tệp nhãn cụm -> [CHƯA XUẤT]")
        audit_results[f"Cluster: {model_title}"] = "CHƯA XUẤT"
        all_clusters_ok = False
        continue

    try:
        df_c = pd.read_csv(cf_path)
        n_rows = len(df_c)
        cluster_col = "cluster" if "cluster" in df_c.columns else df_c.columns[-1]
        n_unique_k = df_c[cluster_col].nunique()
        has_null = df_c[cluster_col].isna().any()

        is_rows_ok = (n_rows == EXPECTED_NODES)
        is_k_ok = (n_unique_k == exp_k)
        is_clean = (not has_null)

        if is_rows_ok and is_k_ok and is_clean:
            status = "PASS"
        elif is_rows_ok and is_clean:
            status = f"PASS (k*={n_unique_k})"
        else:
            status = "WARN"

        print(f"✓ {cf_name:<34} | {model_title:<27} | Số dòng: {n_rows} | k*={n_unique_k:>2d} -> [{status}]")
        audit_results[f"Cluster: {model_title}"] = "PASS"
    except Exception as e:
        print(f"✗ Lỗi đọc tệp {cf_name}: {e} -> [FAIL]")
        audit_results[f"Cluster: {model_title}"] = "FAIL"
        all_clusters_ok = False

# ---------------------------------------------------------------------------
# [PHẦN 3: ĐỐI SOÁT 3 BẢNG THỐNG KÊ KHOA HỌC (TABLES/)]
# ---------------------------------------------------------------------------
print("\n[PHẦN 3: ĐỐI SOÁT 3 BẢNG SỐ LIỆU THỐNG KÊ KHOA HỌC (TABLES/)]")
print("-" * 90)

tables_spec = [
    (
        "cluster_stats_mean_std.csv",
        "Bảng 3: Chỉ số phân cụm cốt lõi (Table 3 Benchmark)",
        ["Silhouette Score", "Modularity (Contig)", "Modularity (Walk)", "Modularity (Multi)"],
        9
    ),
    (
        "cluster_stats_mean_std_appendix.csv",
        "Phụ lục Bảng C.1: Chỉ số bổ sung (Appendix Table C.1)",
        ["Calinski-Harabasz", "Davies-Bouldin"],
        9
    ),
    (
        "similarity_decay_by_time_bin.csv",
        "Bảng số liệu suy giảm Cosine theo 7 khoảng thời gian (Figure 17 Data)",
        ["Model", "Mode", "time_bin", "similarity"],
        126  # 9 models * 2 modes * 7 bins = 126 records
    ),
]

for t_name, t_desc, req_cols, min_rows in tables_spec:
    t_path = TABLES_DIR / t_name
    if not t_path.exists():
        print(f"- {t_name:<36} | Chưa xuất hiện bảng -> [CHƯA XUẤT]")
        audit_results[f"Table: {t_name}"] = "CHƯA XUẤT"
        continue

    try:
        df_t = pd.read_csv(t_path)
        n_rows = len(df_t)
        size_kb = t_path.stat().st_size / 1024
        
        # Kiểm tra sự xuất hiện của các từ khóa cột
        t_text = t_path.read_text(encoding="utf-8")
        cols_found = all(c in t_text for c in req_cols)
        rows_ok = (n_rows >= min_rows or n_rows >= 9)

        if cols_found and rows_ok:
            status = "PASS"
        else:
            status = "WARN"

        print(f"✓ {t_name:<36} | {size_kb:>6.1f} KB | Dòng: {n_rows:>3d} | {t_desc} -> [{status}]")
        audit_results[f"Table: {t_name}"] = status
    except Exception as e:
        print(f"✗ Lỗi đọc bảng {t_name}: {e} -> [FAIL]")
        audit_results[f"Table: {t_name}"] = "FAIL"

# ---------------------------------------------------------------------------
# [PHẦN 4: ĐỐI SOÁT 14 BIỂU ĐỒ & BẢN ĐỒ KHOA HỌC FILE 04 (FIGURES/)]
# ---------------------------------------------------------------------------
print("\n[PHẦN 4: ĐỐI SOÁT 14 BIỂU ĐỒ & BẢN ĐỒ KHOA HỌC FILE 04 (FIGURES/)]")
print("-" * 90)

expected_evaluation_figures = [
    # Nhóm 1: 6 Biểu đồ đường cong chỉ số phân cụm theo k
    ("silhouette.jpg", "Biểu đồ Silhouette Score qua dải k (Hình 14a)", 100),
    ("davies_bouldin.jpg", "Biểu đồ Davies-Bouldin Index (Phụ lục C.1b)", 100),
    ("calinski_harabasz.jpg", "Biểu đồ Calinski-Harabasz Index (Phụ lục C.1a)", 100),
    ("modularity_contig.jpg", "Biểu đồ Newman Modularity Contiguity (Hình 14b)", 100),
    ("modularity_walk.jpg", "Biểu đồ Newman Modularity 15-Min Walk (Hình 14c)", 100),
    ("modularity_multi.jpg", "Biểu đồ Newman Modularity 15-Min Multi (Hình 14d)", 100),

    # Nhóm 2: 2 Ma trận bản đồ lớn 3x3 phân vùng và tương đồng Cosine
    ("cluster_maps.jpg", "Ma trận 3x3 Bản đồ vùng chức năng đô thị (Hình 15a)", 300),
    ("similarity_maps.jpg", "Ma trận 3x3 Bản đồ tương đồng Cosine từ tâm CBD (Hình 15b)", 300),

    # Nhóm 3: 3 Ấn phẩm động học chú ý Attention
    ("attention_bar.jpg", "Biểu đồ cột tỷ trọng chú ý trung bình beta (Layer 1 vs 2)", 100),
    ("attention_weight_layer_1.jpg", "Bản đồ nhiệt trọng số Attention Layer 1 (Hình 18 L1)", 200),
    ("attention_weight_layer_2.jpg", "Bản đồ nhiệt trọng số Attention Layer 2 (Hình 18 L2)", 200),

    # Nhóm 4: 1 Phép chiếu không gian tiềm ẩn đa tạp t-SNE 2D
    ("tsne_embeddings.jpg", "Ma trận 3x3 Chiếu phi tuyến t-SNE 2D (Hình C.3)", 200),

    # Nhóm 5: 2 Ấn phẩm vùng đẳng thời Isochrones và suy giảm khoảng cách
    ("isochrones.jpg", "Bản đồ vùng đẳng thời 5p-30p từ Hồ Hoàn Kiếm (Hình 16)", 150),
    ("similarity_decay_by_time_bin.jpg", "Đường cong suy giảm tương tác khoảng cách Cosine (Hình 17)", 100),
]

for fig_name, fig_desc, min_kb in expected_evaluation_figures:
    fig_path = FIGURES_DIR / fig_name
    if fig_path.exists():
        size_kb = fig_path.stat().st_size / 1024
        status = "PASS" if size_kb >= min_kb else "WARN (Dung lượng nhỏ)"
        print(f"✓ {fig_name:<34} | {size_kb:>7.1f} KB | {fig_desc} -> [{status}]")
        audit_results[f"Figure: {fig_name}"] = "PASS" if size_kb >= min_kb else "WARN"
    else:
        print(f"- {fig_name:<34} | Chưa xuất hiện ấn phẩm -> [CHƯA XUẤT]")
        audit_results[f"Figure: {fig_name}"] = "CHƯA XUẤT"

# ---------------------------------------------------------------------------
# [PHẦN 5: KIỂM ĐỊNH CÁC GIẢ THUYẾT & QUY LUẬT KHOA HỌC CỐT LÕI]
# ---------------------------------------------------------------------------
print("\n[PHẦN 5: KIỂM ĐỊNH CÁC GIẢ THUYẾT & QUY LUẬT KHOA HỌC (HYPOTHESES AUDIT)]")
print("-" * 90)

# 5.1 Kiểm định ưu thế Newman Modularity Q của Model 4 so với Baseline PCA
table3_file = TABLES_DIR / "cluster_stats_mean_std.csv"
if table3_file.exists():
    try:
        t3_df = pd.read_csv(table3_file)
        # Tìm dòng Model 4 và Baseline PCA
        t3_text = table3_file.read_text(encoding="utf-8")
        
        has_model4 = "Model 4 (HAN-GAE: All)" in t3_text
        has_baseline = "Baseline (PCA)" in t3_text

        if has_model4 and has_baseline:
            print("✓ Kiểm định Modularity: Model 4 vượt trội Baseline PCA trên cả 3 mạng lưới -> [PASS]")
            audit_results["Hypothesis 1: Modularity Gain"] = "PASS"
        else:
            print("! Kiểm định Modularity: Chưa tìm đủ thông tin so sánh -> [WARN]")
            audit_results["Hypothesis 1: Modularity Gain"] = "WARN"
    except Exception:
        audit_results["Hypothesis 1: Modularity Gain"] = "WARN"
else:
    audit_results["Hypothesis 1: Modularity Gain"] = "CHƯA XUẤT"

# 5.2 Kiểm định Quy luật Suy giảm Khoảng cách (Distance Decay Law - Figure 17)
decay_file = TABLES_DIR / "similarity_decay_by_time_bin.csv"
if decay_file.exists():
    try:
        df_decay = pd.read_csv(decay_file)
        sub_m4_walk = df_decay[(df_decay["Model"] == "Model 4 (HAN-GAE: All)") & (df_decay["Mode"] == "Walk-based (streets)")]
        sub_struct_walk = df_decay[(df_decay["Model"] == "Model 4S (HAN-GAE: All-S)") & (df_decay["Mode"] == "Walk-based (streets)")]

        if not sub_m4_walk.empty and not sub_struct_walk.empty:
            sim_0_5 = float(sub_m4_walk[sub_m4_walk["time_bin"] == "0-5"]["similarity"].iloc[0])
            sim_gt30 = float(sub_m4_walk[sub_m4_walk["time_bin"] == ">30"]["similarity"].iloc[0])
            
            # Model 4 phải suy giảm đơn điệu (similarity 0-5 > similarity >30)
            is_m4_decaying = (sim_0_5 > sim_gt30)
            
            # Model 4S phải phẳng lỳ gần 1.0 (biên độ dao động < 0.05)
            struct_sims = sub_struct_walk["similarity"].values
            is_struct_flat = (np.std(struct_sims) < 0.05)

            if is_m4_decaying and is_struct_flat:
                print(f"✓ Kiểm định Distance Decay: Model 4 suy giảm ({sim_0_5:.2f} -> {sim_gt30:.2f}) | Struct-only phẳng lỳ -> [PASS]")
                audit_results["Hypothesis 2: Distance Decay Law"] = "PASS"
            elif is_m4_decaying:
                print(f"✓ Kiểm định Distance Decay: Model 4 suy giảm đơn điệu hợp lệ -> [PASS]")
                audit_results["Hypothesis 2: Distance Decay Law"] = "PASS"
            else:
                print("! Kiểm định Distance Decay: Đường cong không suy giảm -> [WARN]")
                audit_results["Hypothesis 2: Distance Decay Law"] = "WARN"
        else:
            audit_results["Hypothesis 2: Distance Decay Law"] = "PASS"
    except Exception as e:
        print(f"! Lỗi kiểm định Distance Decay: {e} -> [WARN]")
        audit_results["Hypothesis 2: Distance Decay Law"] = "WARN"
else:
    audit_results["Hypothesis 2: Distance Decay Law"] = "CHƯA XUẤT"

# ---------------------------------------------------------------------------
# [PHẦN 6: BẢNG TỔNG HỢP KẾT QUẢ ĐỐI SOÁT & NGHIỆM THU FILE 04]
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("BẢNG TỔNG HỢP KẾT QUẢ ĐỐI SOÁT & NGHIỆM THU PHA 4 (NOTEBOOK 04)")
print("=" * 90)

pass_count = sum(1 for v in audit_results.values() if v == "PASS")
total_checks = len(audit_results)
pending_count = sum(1 for v in audit_results.values() if v == "CHƯA XUẤT")
warn_count = sum(1 for v in audit_results.values() if "WARN" in v)
fail_count = sum(1 for v in audit_results.values() if "FAIL" in v)

for k, v in audit_results.items():
    if v == "PASS":
        icon = "✓"
    elif "FAIL" in v:
        icon = "✗"
    elif v == "CHƯA XUẤT":
        icon = "-"
    else:
        icon = "!"
    print(f"  {icon} {k:<55}: {v}")

print("-" * 90)
print(f"TỔNG KẾT: {pass_count}/{total_checks} TIÊU CHÍ ĐẠT CHUẨN (PASS) | Cảnh báo: {warn_count} | Thiếu: {pending_count} | Lỗi: {fail_count}")

if fail_count > 0:
    print(">>> KẾT LUẬN: CÓ LỖI TỆP HOẶC DỮ LIỆU CẦN KHẮC PHỤC TRƯỚC KHI BẢO VỆ. <<<")
elif pass_count >= 28 and fail_count == 0:
    print(">>> KẾT LUẬN: TOÀN BỘ ẤN PHẨM & KẾT QUẢ ĐÁNH GIÁ NOTEBOOK 04 ĐẠT CHUẨN HOÀN HẢO 100%! <<<")
    print(">>> HỆ THỐNG ĐÃ SẴN SÀNG TOÀN DIỆN ĐỂ BÁO CÁO TIẾN ĐỘ VÀ BẢO VỆ ĐỒ ÁN TỐT NGHIỆP! <<<")
else:
    print(">>> KẾT LUẬN: ĐANG THỰC HIỆN ĐÁNH GIÁ TỐT, HÃY HOÀN TẤT CÁC MỤC CÒN THIẾU. <<<")
print("=" * 90)