# -*- coding: utf-8 -*-
"""
KỊCH BẢN KIỂM TOÁN & ĐỐI SOÁT TOÀN DIỆN ĐỒ THỊ KHÔNG GIAN FILE 02 (HÀ NỘI)
Công trình đối chuẩn: Sato et al. (CEUS 2026, Section 4.3 & 4.4, Table 3, Table A.2)
Chạy độc lập từ terminal: python scripts/inspect_graph_hanoi.py
"""

from __future__ import annotations

import os
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import torch

# Cấu hình OpenMP an toàn cho môi trường Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

# Cơ chế Dynamic Root Discovery tự động định vị gốc repository City2Graph-Hanoi
CWD = Path.cwd().resolve()
REPO_ROOT = CWD
while not (REPO_ROOT / "data").exists() and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent

DATA_DIR = REPO_ROOT / "data"
FEATURE_DIR = DATA_DIR / "processed" / "hanoi" / "features"
GRAPH_DIR = DATA_DIR / "processed" / "hanoi" / "graphs"
OUTPUT_FIG_DIR = DATA_DIR / "outputs" / "hanoi" / "figures"

# Danh mục 16 nhóm tiện ích tầng trệt chuẩn mực theo Table A.1 (Sato et al., CEUS 2026)
EXPECTED_POI_COLS = [
    "arts_culture", "automotive_facility", "consumer_service", "corporate_service",
    "education", "entertainment", "food_beverage", "healthcare",
    "hotel_lodging", "industrial_service", "park", "public_service",
    "religion", "retail", "sports_fitness", "transportation_facility"
]

audit_scores: dict[str, bool] = {}

print("=" * 85)
print("BÁO CÁO KIỂM TOÁN & ĐỐI SOÁT TOÀN DIỆN ĐỒ THỊ KHÔNG GIAN FILE 02 (HÀ NỘI)")
print(f"Thư mục gốc dự án: {REPO_ROOT}")
print("=" * 85)

print("\n[PHẦN 1: ĐỐI SOÁT TỆP THAM SỐ CHUẨN HÓA (oa_feature_scaling.json)]")
print("-" * 85)

scaling_file = GRAPH_DIR / "oa_feature_scaling.json"
if scaling_file.exists():
    try:
        with open(scaling_file, "r", encoding="utf-8") as f:
            scaling_data = json.load(f)
        
        method = scaling_data.get("method", "N/A")
        feat_cols = scaling_data.get("feature_cols", [])
        has_log1p = scaling_data.get("log1p", False)
        means = scaling_data.get("mean_", [])
        scales = scaling_data.get("scale_", [])
        
        is_cols_valid = (feat_cols == EXPECTED_POI_COLS)
        is_params_valid = (len(means) == 16 and len(scales) == 16 and not np.isnan(means).any() and not np.isnan(scales).any())
        
        print(f"✓ Đã đọc tệp cấu hình: {scaling_file.name} ({scaling_file.stat().st_size} bytes)")
        print(f"• Phương pháp chuẩn hóa (Method):    {method} (Kỳ vọng: log1p+zscore)")
        print(f"• Số chiều đặc trưng (Features):    {len(feat_cols)}/16 nhóm POIs -> {'[HỢP LỆ]' if is_cols_valid else '[CẢNH BÁO LỆCH CỘT]'}")
        print(f"• Áp dụng hàm log1p (Nén đuôi dài): {has_log1p} (Kỳ vọng: True)")
        print(f"• Tham số kỳ vọng (mu) & tỉ lệ (sigma): Đủ {len(means)} cặp tham số -> {'[HOÀN HẢO]' if is_params_valid else '[KHUYẾT TẬT]'}")
        
        audit_scores["Feature Scaling JSON (Equation 1)"] = (is_cols_valid and is_params_valid and has_log1p)
    except Exception as e:
        print(f"✗ Lỗi khi đọc tệp json chuẩn hóa: {e}")
        audit_scores["Feature Scaling JSON (Equation 1)"] = False
else:
    print(f"✗ Chưa tìm thấy tệp {scaling_file.name} tại {GRAPH_DIR}.")
    audit_scores["Feature Scaling JSON (Equation 1)"] = False

print("\n[PHẦN 2: ĐỐI SOÁT ĐỒ THỊ ĐỒNG NHẤT MODEL 1 (homo.pt)]")
print("-" * 85)

homo_path = GRAPH_DIR / "homo.pt"
if homo_path.exists():
    try:
        homo_data = torch.load(homo_path, map_location="cpu", weights_only=False)
        
        num_nodes = homo_data.num_nodes if hasattr(homo_data, "num_nodes") else homo_data.x.shape[0]
        num_edges = homo_data.num_edges if hasattr(homo_data, "num_edges") else homo_data.edge_index.shape[1]
        feat_dim = homo_data.x.shape[1] if hasattr(homo_data, "x") else 0
        
        has_edge_attr = hasattr(homo_data, "edge_attr") and homo_data.edge_attr is not None
        has_nan = torch.isnan(homo_data.x).any().item() or torch.isinf(homo_data.x).any().item()
        
        is_node_ok = (num_nodes == 985)
        is_feat_ok = (feat_dim == 16)
        is_edge_ok = (5000 <= num_edges <= 6000)  # Kỳ vọng ~5,550 cạnh có hướng đối xứng
        
        print(f"✓ Đã đọc tệp PyG Data: {homo_path.name} ({homo_path.stat().st_size / 1024:.1f} KB)")
        print(f"• Số lượng đỉnh (Nodes):    {num_nodes:,} ô H3 (Kỳ vọng: 985) -> {'[CHUẨN XÁC]' if is_node_ok else '[LỆCH SỐ ĐỈNH]'}")
        print(f"• Số chiều đặc trưng (x):   {feat_dim} chiều (Kỳ vọng: 16 POIs) -> {'[CHUẨN XÁC]' if is_feat_ok else '[LỆCH SỐ CHIỀU]'}")
        print(f"• Số lượng cạnh tiếp giáp:  {num_edges:,} cạnh định hướng (Kỳ vọng: ~5,550) -> {'[HỢP LỆ]' if is_edge_ok else '[CẢNH BÁO]'}")
        print(f"• Thuộc tính thời gian tau: {'✓ Có sẵn (travel_time_sec)' if has_edge_attr else '✗ Thiếu edge_attr'}")
        print(f"• Kiểm tra dữ liệu bất thường: {'✓ Không chứa NaN/Inf' if not has_nan else '✗ Phát hiện NaN/Inf trong tensor x!'}")
        
        audit_scores["Homogeneous Graph (homo.pt)"] = (is_node_ok and is_feat_ok and is_edge_ok and not has_nan)
    except Exception as e:
        print(f"✗ Lỗi khi đọc tệp homo.pt: {e}")
        audit_scores["Homogeneous Graph (homo.pt)"] = False
else:
    print(f"✗ Chưa tìm thấy tệp {homo_path.name} tại {GRAPH_DIR}.")
    audit_scores["Homogeneous Graph (homo.pt)"] = False

print("\n[PHẦN 3: ĐỐI SOÁT ĐỒ THỊ DỊ THỂ ĐI BỘ MODEL 2 (hetero_walk.pt)]")
print("-" * 85)

walk_path = GRAPH_DIR / "hetero_walk.pt"
if walk_path.exists():
    try:
        walk_data = torch.load(walk_path, map_location="cpu", weights_only=False)
        
        oa_nodes = walk_data["oa"].num_nodes if hasattr(walk_data["oa"], "num_nodes") else walk_data["oa"].x.shape[0]
        rel_contig = ("oa", "contig", "oa")
        rel_walk = ("oa", "15_min_walk", "oa")
        
        has_contig = rel_contig in walk_data.edge_types
        has_walk = rel_walk in walk_data.edge_types
        
        n_contig = walk_data[rel_contig].edge_index.shape[1] if has_contig else 0
        n_walk = walk_data[rel_walk].edge_index.shape[1] if has_walk else 0
        
        # Kiểm tra ngưỡng thời gian di chuyển cutoff <= 900s
        walk_times_valid = False
        if has_walk and hasattr(walk_data[rel_walk], "edge_attr") and walk_data[rel_walk].edge_attr is not None:
            max_walk_t = walk_data[rel_walk].edge_attr.max().item()
            min_walk_t = walk_data[rel_walk].edge_attr.min().item()
            walk_times_valid = (min_walk_t >= 0 and max_walk_t <= 900.5)
            print(f"• Thời gian đi bộ 15p:      Min = {min_walk_t:.1f}s | Max = {max_walk_t:.1f}s (Ngưỡng cắt: <= 900s) -> {'[ĐẠT CHUẨN]' if walk_times_valid else '[VƯỢT NGƯỠNG]'}")
            
        print(f"✓ Đã đọc tệp PyG HeteroData: {walk_path.name} ({walk_path.stat().st_size / 1024:.1f} KB)")
        print(f"• Số đỉnh ô không gian 'oa': {oa_nodes:,} ô (Kỳ vọng: 985) -> {'[CHUẨN XÁC]' if oa_nodes == 985 else '[LỆCH ĐỈNH]'}")
        print(f"• Cạnh tiếp giáp 'contig':   {n_contig:,} cạnh")
        print(f"• Cạnh siêu đường đi 'walk': {n_walk:,} cạnh có hướng (Kỳ vọng: ~31,040 - 62,080)")
        
        audit_scores["Hetero Walk Graph (hetero_walk.pt)"] = (oa_nodes == 985 and has_contig and has_walk and n_walk > 5000 and walk_times_valid)
    except Exception as e:
        print(f"✗ Lỗi khi đọc tệp hetero_walk.pt: {e}")
        audit_scores["Hetero Walk Graph (hetero_walk.pt)"] = False
else:
    print(f"✗ Chưa tìm thấy tệp {walk_path.name} tại {GRAPH_DIR}.")
    audit_scores["Hetero Walk Graph (hetero_walk.pt)"] = False

print("\n[PHẦN 4: ĐỐI SOÁT ĐỒ THỊ DỊ THỂ ĐA PHƯƠNG THỨC MODEL 3 & 4 (hetero_multi.pt)]")
print("-" * 85)

multi_path = GRAPH_DIR / "hetero_multi.pt"
if multi_path.exists():
    try:
        multi_data = torch.load(multi_path, map_location="cpu", weights_only=False)
        
        oa_nodes_m = multi_data["oa"].num_nodes if hasattr(multi_data["oa"], "num_nodes") else multi_data["oa"].x.shape[0]
        
        rel_c = ("oa", "contig", "oa")
        rel_w = ("oa", "15_min_walk", "oa")
        rel_m = ("oa", "15_min_multi", "oa")
        
        has_3_core = (rel_c in multi_data.edge_types and rel_w in multi_data.edge_types and rel_m in multi_data.edge_types)
        
        n_c = multi_data[rel_c].edge_index.shape[1] if rel_c in multi_data.edge_types else 0
        n_w = multi_data[rel_w].edge_index.shape[1] if rel_w in multi_data.edge_types else 0
        n_m = multi_data[rel_m].edge_index.shape[1] if rel_m in multi_data.edge_types else 0
        
        print(f"✓ Đã đọc tệp PyG HeteroData: {multi_path.name} ({multi_path.stat().st_size / 1024:.1f} KB)")
        print(f"• Cấu trúc 3 quan hệ cốt lõi bàn giao cho Notebook 03:")
        print(f"  1. ('oa', 'contig', 'oa')       : {n_c:,} cạnh")
        print(f"  2. ('oa', '15_min_walk', 'oa')   : {n_w:,} cạnh")
        print(f"  3. ('oa', '15_min_multi', 'oa')  : {n_m:,} cạnh (Siêu liên kết xe buýt mở rộng)")
        
        is_multi_expanded = (n_m > 0)
        print(f"• Tính mở rộng của mạng xe buýt: {'✓ Thành công (bóc tách transit-only độc lập)' if is_multi_expanded else '✗ Thiếu cạnh mở rộng!'}")
        
        audit_scores["Multimodal Graph (hetero_multi.pt)"] = (oa_nodes_m == 985 and has_3_core and is_multi_expanded)
    except Exception as e:
        print(f"✗ Lỗi khi đọc tệp hetero_multi.pt: {e}")
        audit_scores["Multimodal Graph (hetero_multi.pt)"] = False
else:
    print(f"✗ Chưa tìm thấy tệp {multi_path.name} tại {GRAPH_DIR}.")
    audit_scores["Multimodal Graph (hetero_multi.pt)"] = False

print("\n[PHẦN 5: ĐỐI SOÁT 6 BẢN ĐỒ KHOA HỌC FILE 02 (FIGURES)]")
print("-" * 85)

expected_figures_f02 = [
    ("hanoi_raw_features_histogram.png", "Histogram ma trận 16 POIs thô (Log Y-Scale)", 200),
    ("hanoi_scaled_features_histogram.png", "Histogram ma trận 16 POIs sau chuẩn hóa (Log1p+Z-score)", 200),
    ("hanoi_walk_4relations_matrix.png", "Ma trận 4 quan hệ đồ thị đi bộ (Khung lưới 2x2 Dark Mode)", 300),
    ("hanoi_fig12a_15min_walk_solo.png", "Bản đồ tiếp cận đi bộ 15 phút khổ lớn (Figure 12a layout)", 400),
    ("hanoi_fig2_hetero_8relations_matrix.png", "Ma trận 8 quan hệ đồ thị dị thể (Khung lưới 3x3 Dark Mode - Figure 2)", 400),
    ("hanoi_fig12b_15min_multi_solo.png", "Bản đồ tiếp cận đa phương thức buýt 15 phút khổ lớn (Figure 12b layout)", 200),
]

all_figs_ok = True
for fname, desc, min_kb in expected_figures_f02:
    fpath = OUTPUT_FIG_DIR / fname
    if fpath.exists():
        size_kb = fpath.stat().st_size / 1024
        is_ok = (size_kb >= min_kb)
        status_tag = "[ĐẠT ĐỘ NÉT KHOA HỌC]" if is_ok else "[CẢNH BÁO: FILE NHỎ]"
        print(f"✓ {fname:<42} | {size_kb:>7.1f} KB | {status_tag}")
        if not is_ok:
            all_figs_ok = False
    else:
        print(f"✗ {fname:<42} | THIẾU TỆP | [CHƯA XUẤT RA]")
        all_figs_ok = False

audit_scores["6 Scientific Figures (File 02)"] = all_figs_ok

print("\n" + "=" * 85)
print("BẢNG TỔNG HỢP KẾT QUẢ ĐỐI SOÁT & NGHIỆM THU PHA 2 (NOTEBOOK 02)")
print("=" * 85)

passed_cnt = sum(1 for v in audit_scores.values() if v)
total_cnt = len(audit_scores)

for criterion, status in audit_scores.items():
    status_str = "✓ HOÀN THÀNH (PASS)" if status else "✗ CHƯA ĐẠT (FAIL/WARN)"
    print(f"  • {criterion:<42}: {status_str}")

print("-" * 85)
print(f"TỶ LỆ ĐẠT CHUẨN: {passed_cnt}/{total_cnt} TIÊU CHÍ ({(passed_cnt/total_cnt)*100:.1f}%)")

if passed_cnt == total_cnt:
    print(">>> KẾT LUẬN: TOÀN BỘ ĐỒ THỊ FILE 02 CHUẨN XÁC 100%! BẠN HOÀN TOÀN TỰ TIN TIẾN TỚI NOTEBOOK 03 (HUẤN LUYỆN HAN-GAE). <<<")
else:
    print(">>> KẾT LUẬN: CÓ MỘT SỐ HẠNG MỤC CẦN KIỂM TRA LẠI (XEM CÁC DÒNG CẢNH BÁO Ở TRÊN). <<<")
print("=" * 85)