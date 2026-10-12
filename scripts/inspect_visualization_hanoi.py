"""
========================================================================================
CITY2GRAPH-HANOI: HỆ THỐNG KIỂM TOÁN & ĐỐI SOÁT KẾT QUẢ TRỰC QUAN HÓA FILE 05
Vị trí: D:\DATN\City2Graph-Hanoi\scripts\inspect_visualization_hanoi.py
Mục tiêu:
  1. Kiểm toán tài nguyên đầu vào: 985 ô H3, 16 POIs, mạng đường bộ Overture và đồ thị hetero_multi.pt.
  2. Đối soát 6 ấn phẩm hình ảnh khoa học chuẩn 300 DPI xuất ra tại figures/ (poi.jpg, land_use.jpg,
     hanoi_contig_oa.jpg, hanoi_contig_edges.jpg, hanoi_metapaths_walk.jpg, hanoi_metapaths_multi.jpg).
  3. Thẩm định Phụ lục Bảng A.2 (edge_relation_descriptive_stats.csv) tại tables/: số lượng cạnh,
     thời gian di chuyển tau (min, mean, max, std) của 5 loại quan hệ cạnh cốt lõi.
  4. Đánh giá tính nhất quán tô-pô mạng lưới và xuất Bảng điểm Nghiệm thu Pha 5.
========================================================================================
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------------
# 1. Tự động định vị thư mục gốc dự án (Dynamic Root Discovery)
# ------------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if not (REPO_ROOT / "data").exists():
    search = CURRENT_DIR
    while search.parent != search:
        if (search / "data").exists() and (search / "notebooks").exists():
            REPO_ROOT = search
            break
        search = search.parent

# Cấu hình các đường dẫn tiêu chuẩn
FEATURES_DIR = REPO_ROOT / "data" / "processed" / "hanoi" / "features"
GRAPHS_DIR = REPO_ROOT / "data" / "processed" / "hanoi" / "graphs"
FIGURES_DIR = REPO_ROOT / "data" / "outputs" / "hanoi" / "figures"
TABLES_DIR = REPO_ROOT / "data" / "outputs" / "hanoi" / "tables"

# Danh mục 16 nhóm tiện ích tầng trệt Overture Bảng A.1
POI_COLS = [
    "arts_culture", "automotive_facility", "consumer_service", "corporate_service",
    "education", "entertainment", "food_beverage", "healthcare",
    "hotel_lodging", "industrial_service", "park", "public_service",
    "religion", "retail", "sports_fitness", "transportation_facility"
]

# Danh mục 6 ấn phẩm khoa học của File 05
EXPECTED_FIGURES = {
    "poi.jpg": "Ma trận 16 bản đồ POIs tầng trệt (Figure 10b)",
    "land_use.jpg": "Bản đồ mặt phủ chức năng ưu thế đô thị (Figure 10a)",
    "hanoi_contig_oa.jpg": "Đa giác 985 ô H3 & Trọng tâm trên nền đen (Figure 1A)",
    "hanoi_contig_edges.jpg": "Mạng lưới cạnh tiếp giáp Queen Contiguity nền đen (Figure 1B)",
    "hanoi_metapaths_walk.jpg": "Siêu đường đi bộ 15 phút 15_min_walk nền đen (Figure 2)",
    "hanoi_metapaths_multi.jpg": "Siêu đường đi buýt đa phương thức 15_min_multi nền đen (Figure 2)"
}

# 5 loại quan hệ cạnh chuẩn mực trong Phụ lục Bảng A.2
EXPECTED_EDGE_RELATIONS = [
    "('street_connector', 'is_connected_to', 'street_connector')",
    "('bus_station', 'is_next_to', 'bus_station')",
    "('oa', 'contig', 'oa')",
    "('oa', '15_min_walk', 'oa')",
    "('oa', '15_min_multi', 'oa')"
]


def print_header(title: str):
    print(f"\n{'-' * 85}")
    print(f" {title}")
    print(f"{'-' * 85}")


def audit_visualization():
    print("=" * 85)
    print(" BÁO CÁO KIỂM TOÁN & ĐỐI SOÁT KẾT QUẢ TRỰC QUAN HÓA FILE 05 (HÀ NỘI)")
    print(f" Thư mục gốc dự án: {REPO_ROOT}")
    print("=" * 85)

    scores = []

    # --------------------------------------------------------------------------
    # PHẦN 1: ĐỐI SOÁT NGUYÊN LIỆU ĐẦU VÀO
    # --------------------------------------------------------------------------
    print_header("[PHẦN 1: ĐỐI SOÁT TÀI NGUYÊN ĐẦU VÀO CHO FILE 05]")
    input_status = True

    # 1.1 Kiểm tra oa_with_features.gpkg
    oa_path = FEATURES_DIR / "oa_with_features.gpkg"
    if oa_path.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(oa_path)
            num_nodes = len(gdf)
            feat_count = sum(1 for c in POI_COLS if c in gdf.columns)
            
            if num_nodes == 985 and feat_count == 16:
                print(f" ✓ Ranh giới 985 ô H3 (oa_with_features.gpkg) : Đủ {num_nodes} ô | Đủ 16/16 nhóm POIs -> [PASS]")
            else:
                print(f" ! oa_with_features.gpkg : Có {num_nodes} ô (Kỳ vọng 985), {feat_count}/16 nhóm POIs -> [WARN]")
                input_status = False
        except Exception as e:
            print(f" ✗ Lỗi đọc oa_with_features.gpkg: {e} -> [FAIL]")
            input_status = False
    else:
        print(f" ✗ Không tìm thấy tệp ranh giới: {oa_path} -> [FAIL]")
        input_status = False

    # 1.2 Kiểm tra đồ thị hetero_multi.pt
    graph_path = GRAPHS_DIR / "hetero_multi.pt"
    if graph_path.exists():
        try:
            import torch
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            has_contig = ('oa', 'contig', 'oa') in graph.edge_types
            has_walk = ('oa', '15_min_walk', 'oa') in graph.edge_types
            has_multi = ('oa', '15_min_multi', 'oa') in graph.edge_types
            
            if has_contig and has_walk and has_multi:
                print(f" ✓ Đồ thị nền hetero_multi.pt              : {graph['oa'].num_nodes} đỉnh | Đủ 3 quan hệ chính -> [PASS]")
            else:
                print(f" ! hetero_multi.pt                         : Thiếu một trong các quan hệ cốt lõi -> [WARN]")
                input_status = False
        except Exception as e:
            print(f" ✗ Lỗi đọc đồ thị hetero_multi.pt: {e} -> [FAIL]")
            input_status = False
    else:
        print(f" ✗ Không tìm thấy đồ thị hetero_multi.pt: {graph_path} -> [FAIL]")
        input_status = False

    scores.append(("Tài nguyên đầu vào (Features & Graphs)", input_status))

    # --------------------------------------------------------------------------
    # PHẦN 2: ĐỐI SOÁT 6 ẤN PHẨM HÌNH ẢNH KHOA HỌC (FIGURES/)
    # --------------------------------------------------------------------------
    print_header("[PHẦN 2: ĐỐI SOÁT 6 ẤN PHẨM HÌNH ẢNH KHOA HỌC FILE 05 (FIGURES/)]")
    figs_passed = 0

    for fig_name, fig_desc in EXPECTED_FIGURES.items():
        fig_path = FIGURES_DIR / fig_name
        if fig_path.exists():
            size_kb = fig_path.stat().st_size / 1024
            if size_kb >= 50.0:  # Ảnh độ nét cao phải >= 50 KB
                print(f" ✓ {fig_name:<28} | {size_kb:>8.1f} KB | {fig_desc} -> [PASS]")
                figs_passed += 1
            else:
                print(f" ! {fig_name:<28} | {size_kb:>8.1f} KB | Dung lượng quá nhỏ (<50KB), có thể bị lỗi vẽ -> [WARN]")
        else:
            print(f" ✗ {fig_name:<28} | Chưa tồn tại trên đĩa cứng -> [FAIL]")

    figs_all_ok = (figs_passed == len(EXPECTED_FIGURES))
    scores.append(("6 Ấn phẩm trực quan khoa học (Figures)", figs_all_ok))

    # --------------------------------------------------------------------------
    # PHẦN 3: ĐỐI SOÁT BẢNG THỐNG KÊ PHỤ LỤC A.2 (TABLES/)
    # --------------------------------------------------------------------------
    print_header("[PHẦN 3: ĐỐI SOÁT PHỤ LỤC BẢNG A.2 (TABLES/edge_relation_descriptive_stats.csv)]")
    table_path = TABLES_DIR / "edge_relation_descriptive_stats.csv"
    table_ok = False

    if table_path.exists():
        try:
            df_table = pd.read_csv(table_path)
            size_kb = table_path.stat().st_size / 1024
            
            # Kiểm tra các cột bắt buộc: edge_type, count, min, mean, max, std
            req_cols = ["count", "min", "mean", "max", "std"]
            col_check = all(any(rc in c.lower() for c in df_table.columns) for rc in req_cols)
            
            first_col = df_table.columns[0]
            edge_vals = df_table[first_col].astype(str).tolist()
            
            matched_rels = 0
            for exp_rel in EXPECTED_EDGE_RELATIONS:
                # Kiểm tra tương đối chuỗi
                rel_clean = exp_rel.replace("'", "").replace(" ", "").replace("(", "").replace(")", "")
                if any(rel_clean in str(v).replace("'", "").replace(" ", "").replace("(", "").replace(")", "") for v in edge_vals):
                    matched_rels += 1

            print(f" ✓ Tệp Phụ lục Bảng A.2: {table_path.name} ({size_kb:.1f} KB) | Số dòng: {len(df_table)}")
            print(f"   • Cấu trúc cột thống kê (count, min, mean, max, std) : {'Đầy đủ' if col_check else 'Chưa chuẩn'}")
            print(f"   • Khớp danh mục quan hệ mạng cốt lõi                 : {matched_rels}/5 quan hệ")

            # In chi tiết nội dung bảng để người dùng đối chiếu
            print("\n   [Chi tiết thông số Phụ lục Bảng A.2]:")
            for _, r in df_table.iterrows():
                etype = str(r[first_col])
                c_val = r.get("count", r.get(df_table.columns[1], "N/A"))
                mean_val = r.get("mean", "N/A")
                max_val = r.get("max", "N/A")
                print(f"     - {etype:<55} | count: {c_val:>6} | mean: {mean_val:>6} | max: {max_val:>6}")

            if col_check and matched_rels >= 3:
                table_ok = True
                print(" -> [PASS]")
            else:
                print(" -> [WARN]")

        except Exception as e:
            print(f" ✗ Lỗi phân tích Bảng A.2: {e} -> [FAIL]")
    else:
        print(f" ✗ Không tìm thấy tệp Bảng A.2 tại: {table_path} -> [FAIL]")

    scores.append(("Phụ lục Bảng A.2 (Table A.2 Benchmark)", table_ok))

    # --------------------------------------------------------------------------
    # PHẦN 4: KIỂM ĐỊNH TÍNH NHẤT QUÁN TOÁN HỌC & NGƯỠNG DI CHUYỂN
    # --------------------------------------------------------------------------
    print_header("[PHẦN 4: KIỂM ĐỊNH QUY LUẬT TOÁN HỌC & NGƯỠNG DI CHUYỂN (T <= 900s)]")
    consistency_ok = True

    if table_path.exists():
        try:
            df = pd.read_csv(table_path)
            first_col = df.columns[0]
            
            # Lấy dòng 15_min_walk và 15_min_multi
            for _, r in df.iterrows():
                etype = str(r[first_col])
                if "15_min_walk" in etype or "walk" in etype:
                    max_t = float(r.get("max", 0))
                    min_t = float(r.get("min", 0))
                    if max_t <= 900.0:
                        print(f" ✓ Quan hệ đi bộ 15 phút ('15_min_walk'): T_max = {max_t:.1f}s <= 900s (Tuân thủ ngưỡng cắt) -> [PASS]")
                    else:
                        print(f" ! Quan hệ đi bộ 15 phút: T_max = {max_t:.1f}s vượt ngưỡng 900s -> [WARN]")
                        consistency_ok = False
                        
                elif "15_min_multi" in etype or "multi" in etype:
                    count_m = int(r.get("count", 0))
                    max_m = float(r.get("max", 0))
                    if max_m <= 900.0:
                        print(f" ✓ Quan hệ buýt đa phương thức ('15_min_multi'): T_max = {max_m:.1f}s <= 900s, {count_m:,} cạnh -> [PASS]")
                    else:
                        print(f" ! Quan hệ đa phương thức: T_max = {max_m:.1f}s vượt ngưỡng 900s -> [WARN]")
                        consistency_ok = False
        except Exception as e:
            print(f" ✗ Không thể kiểm định ngưỡng di chuyển: {e}")
            consistency_ok = False

    scores.append(("Tính nhất quán ngưỡng di chuyển (Cutoff T <= 900s)", consistency_ok))

    # --------------------------------------------------------------------------
    # BẢNG TỔNG HỢP NGHIỆM THU PHA 5
    # --------------------------------------------------------------------------
    print_header("BẢNG TỔNG HỢP KẾT QUẢ ĐỐI SOÁT & NGHIỆM THU PHA 5 (NOTEBOOK 05)")
    passed_count = sum(1 for _, ok in scores if ok)
    total_criteria = len(scores)

    for name, ok in scores:
        status_str = "✓ HOÀN THÀNH (PASS)" if ok else "✗ CHƯA ĐẠT (WARN/FAIL)"
        print(f"  • {name:<50} : {status_str}")

    pct = (passed_count / total_criteria) * 100
    print(f"\nTỶ LỆ ĐẠT CHUẨN: {passed_count}/{total_criteria} TIÊU CHÍ ({pct:.1f}%)")

    if pct == 100.0:
        print("\n>>> KẾT LUẬN: TOÀN BỘ KẾT QUẢ FILE 05 ĐẠT CHUẨN KHOA HỌC 100%! BẢO VỆ THÀNH CÔNG ĐỒ ÁN! <<<")
    elif pct >= 75.0:
        print("\n>>> KẾT LUẬN: KẾT QUẢ ĐÃ CƠ BẢN HOÀN TẤT. KIỂM TRA LẠI CÁC MỤC CẢNH BÁO TRƯỚC KHI NỘP. <<<")
    else:
        print("\n>>> KẾT LUẬN: CẦN THỰC THI LẠI NOTEBOOK 05 ĐỂ XUẤT ĐẦY ĐỦ CÁC TỆP CÒN THIẾU. <<<")
    print("=" * 85)


if __name__ == "__main__":
    audit_visualization()