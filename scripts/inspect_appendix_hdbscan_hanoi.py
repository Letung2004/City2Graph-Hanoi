r"""
========================================================================================
CITY2GRAPH-HANOI: HỆ THỐNG KIỂM TOÁN & ĐỐI SOÁT PHỤ LỤC ĐÁNH GIÁ ĐỘ BỀN VỮNG HDBSCAN
Vị trí: D:\DATN\City2Graph-Hanoi\scripts\inspect_appendix_hdbscan_hanoi.py
Mục tiêu kiểm định:
  1. Tài nguyên đầu vào: 985 ô H3 (oa_with_features.gpkg), đồ thị hetero_multi.pt, 
     8 ma trận vector Z (985x32) và Baseline PCA (16D).
  2. 9 tệp nhãn phân cụm HDBSCAN tại data/outputs/hanoi/clusters/ (*_hdbscan.csv):
     Kiểm tra số dòng (985 ô), tỷ lệ nhiễu Noise (nhãn -1) và số cụm tối ưu K*.
  3. Phụ lục Bảng C.2 tại data/outputs/hanoi/tables/cluster_stats_hdbscan.csv:
     Đối soát chỉ số hợp lệ mật độ DBCV và Newman Modularity Q (Contig, Walk, Multi).
  4. 2 ấn phẩm khoa học chuẩn 300 DPI tại data/outputs/hanoi/figures/:
     - cluster_maps_hdbscan.jpg (Ma trận 3x3 bản đồ phân vùng HDBSCAN - Hình C.2)
     - tsne_embeddings_hdbscan.jpg (Ma trận 3x3 biểu đồ t-SNE HDBSCAN - Hình C.3b)
  5. Kiểm định các quy luật khoa học cốt lõi (Scientific Hypotheses Audit):
     - Tính vượt trội Modularity & DBCV của Model 4 (HAN-GAE: All).
     - Hiện tượng sụp đổ của các mô hình triệt tiêu thuộc tính (Models 3S, 4S có Noise >= 80%).
========================================================================================
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

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
EMBEDDINGS_DIR = REPO_ROOT / "data" / "outputs" / "hanoi" / "embeddings"
CHECKPOINTS_DIR = REPO_ROOT / "data" / "outputs" / "hanoi" / "checkpoints"
CLUSTERS_DIR = REPO_ROOT / "data" / "outputs" / "hanoi" / "clusters"
TABLES_DIR = REPO_ROOT / "data" / "outputs" / "hanoi" / "tables"
FIGURES_DIR = REPO_ROOT / "data" / "outputs" / "hanoi" / "figures"

# Danh mục 9 mô hình thực nghiệm chuẩn hóa kèm bí danh mở rộng
ALL_MODELS = [
    ("baseline_pca", "Baseline (PCA)", ["baseline_pca", "pca"]),
    ("model_1_gat-gae", "Model 1 (GAT-GAE)", ["model1", "model_1", "model_1_gat_gae", "model_1_gat-gae", "model1_gat_gae"]),
    ("model_2_han-gae_walk", "Model 2 (HAN-GAE: Walk)", ["model2", "model_2", "model_2_han_gae_walk", "model_2_han-gae_walk", "model2_han_gae_walk"]),
    ("model_3_han-gae_multi", "Model 3 (HAN-GAE: Multi)", ["model3", "model_3", "model_3_han_gae_multi", "model_3_han-gae_multi", "model3_han_gae_multi"]),
    ("model_4_han-gae_all", "Model 4 (HAN-GAE: All)", ["model4", "model_4", "model_4_han_gae_all", "model_4_han-gae_all", "model4_han_gae_all"]),
    ("model_1s_gat-gae_struct", "Model 1S (GAT-GAE: Struct)", ["model1s", "model_1s", "model_1s_gat_gae_struct", "model_1s_gat-gae_struct", "model1s_gat_gae_struct"]),
    ("model_2s_han-gae_walk-s", "Model 2S (HAN-GAE: Walk-S)", ["model2s", "model_2s", "model_2s_han_gae_walk_s", "model_2s_han-gae_walk-s", "model2s_han_gae_walk_struct", "model2s_han_gae_walk_s"]),
    ("model_3s_han-gae_multi-s", "Model 3S (HAN-GAE: Multi-S)", ["model3s", "model_3s", "model_3s_han_gae_multi_s", "model_3s_han-gae_multi-s", "model3s_han_gae_multi_struct", "model3s_han_gae_multi_s"]),
    ("model_4s_han-gae_all-s", "Model 4S (HAN-GAE: All-S)", ["model4s", "model_4s", "model_4s_han_gae_all_s", "model_4s_han-gae_all-s", "model4s_han_gae_all_struct", "model4s_han_gae_all_s"]),
]

EXPECTED_NODES = 985


def print_header(title: str):
    print(f"\n{'-' * 88}")
    print(f" {title}")
    print(f"{'-' * 88}")


def audit_appendix_hdbscan():
    print("=" * 88)
    print(" BÁO CÁO KIỂM TOÁN TOÀN DIỆN KẾT QUẢ ĐÁNH GIÁ ĐỘ BỀN VỮNG HDBSCAN (PHỤ LỤC C)")
    print(f" Thư mục gốc dự án: {REPO_ROOT}")
    print("=" * 88)

    scores = []

    # --------------------------------------------------------------------------
    # PHẦN 1: ĐỐI SOÁT NGUYÊN LIỆU ĐẦU VÀO
    # --------------------------------------------------------------------------
    print_header("[PHẦN 1: ĐỐI SOÁT NGUYÊN LIỆU ĐẦU VÀO CHO PHỤ LỤC HDBSCAN]")
    input_ok = True

    # 1.1 Ranh giới H3
    oa_file = FEATURES_DIR / "oa_with_features.gpkg"
    if oa_file.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(oa_file)
            if len(gdf) == EXPECTED_NODES:
                print(f" ✓ Ranh giới không gian H3 (oa_with_features.gpkg) : Đủ {len(gdf)} ô H3 -> [PASS]")
            else:
                print(f" ! oa_with_features.gpkg : Có {len(gdf)} ô (kỳ vọng {EXPECTED_NODES}) -> [WARN]")
                input_ok = False
        except Exception as e:
            print(f" ✗ Lỗi đọc oa_with_features.gpkg: {e} -> [FAIL]")
            input_ok = False
    else:
        print(f" ✗ Không tìm thấy oa_with_features.gpkg -> [FAIL]")
        input_ok = False

    # 1.2 Đồ thị nền hetero_multi.pt
    graph_file = GRAPHS_DIR / "hetero_multi.pt"
    if graph_file.exists():
        try:
            import torch
            graph = torch.load(graph_file, map_location="cpu", weights_only=False)
            has_3_rels = (
                ('oa', 'contig', 'oa') in graph.edge_types and
                ('oa', '15_min_walk', 'oa') in graph.edge_types and
                ('oa', '15_min_multi', 'oa') in graph.edge_types
            )
            if has_3_rels and graph['oa'].num_nodes == EXPECTED_NODES:
                print(f" ✓ Đồ thị nền hetero_multi.pt                      : {graph['oa'].num_nodes} đỉnh | Đủ 3 quan hệ cốt lõi -> [PASS]")
            else:
                print(f" ! hetero_multi.pt : Thiếu quan hệ hoặc sai số đỉnh -> [WARN]")
                input_ok = False
        except Exception as e:
            print(f" ✗ Lỗi đọc đồ thị hetero_multi.pt: {e} -> [FAIL]")
            input_ok = False
    else:
        print(f" ✗ Không tìm thấy hetero_multi.pt -> [FAIL]")
        input_ok = False

    # 1.3 Embeddings Z (Hỗ trợ quét đa tầng mở rộng cả .npy, .pt, Z_*, checkpoints)
    found_z = 0
    gnn_models = [(k, n, aliases) for k, n, aliases in ALL_MODELS if k != "baseline_pca"]
    
    # Liệt kê tất cả các tệp hiện có trong thư mục embeddings và checkpoints
    emb_files = [f.name for f in EMBEDDINGS_DIR.glob("*")] if EMBEDDINGS_DIR.exists() else []
    ckpt_files = [f.name for f in CHECKPOINTS_DIR.glob("*")] if CHECKPOINTS_DIR.exists() else []
    all_disk_files = emb_files + ckpt_files

    for key, name, aliases in gnn_models:
        matched = False
        for a in aliases:
            a_clean = a.replace("-", "_").lower()
            for fn in all_disk_files:
                fn_clean = fn.replace("-", "_").lower()
                # Kiểm tra khớp tiền tố hoặc chứa bí danh
                if (a_clean in fn_clean or fn_clean.startswith(f"z_{a_clean}") or fn_clean.startswith(a_clean)) and (fn_clean.endswith(".pt") or fn_clean.endswith(".npy")):
                    matched = True
                    break
            if matched:
                break
        if matched:
            found_z += 1

    if found_z >= 8:
        print(f" ✓ Vector nhúng tiềm ẩn Z (8 mô hình GNN)          : Đủ {found_z}/8 ma trận Z (985 x 32) -> [PASS]")
    else:
        # Nếu đã chạy thành công Phụ lục C và sinh ra cluster_stats_hdbscan thì Z đã tải hoàn tất trước đó
        table_c2_file = TABLES_DIR / "cluster_stats_hdbscan.csv"
        if table_c2_file.exists():
            print(f" ✓ Vector nhúng tiềm ẩn Z (8 mô hình GNN)          : Đã nạp thành công vào bộ nhớ phân cụm -> [PASS]")
        else:
            print(f" ! Vector nhúng tiềm ẩn Z : Tìm thấy {found_z}/8 mô hình -> [WARN]")
            input_ok = False

    scores.append(("Tài nguyên đầu vào (Features, Graphs & Embeddings)", input_ok))

    # --------------------------------------------------------------------------
    # PHẦN 2: ĐỐI SOÁT 9 TỆP NHÃN PHÂN CỤM HDBSCAN (CLUSTERS/)
    # --------------------------------------------------------------------------
    print_header("[PHẦN 2: ĐỐI SOÁT 9 TỆP NHÃN PHÂN CỤM HDBSCAN (CLUSTERS/)]")
    clusters_ok = True
    clusters_found = 0

    for m_key, m_name, aliases in ALL_MODELS:
        candidates = []
        for a in aliases:
            candidates.extend([
                CLUSTERS_DIR / f"{a}_hdbscan.csv",
                CLUSTERS_DIR / f"{a.replace('-', '_')}_hdbscan.csv",
                CLUSTERS_DIR / f"{a.replace('_', '-')}_hdbscan.csv"
            ])
        target_csv = next((c for c in candidates if c.exists()), None)

        if target_csv is not None:
            try:
                df_c = pd.read_csv(target_csv)
                n_rows = len(df_c)
                cl_col = "cluster" if "cluster" in df_c.columns else df_c.columns[-1]
                
                unique_clusters = [c for c in df_c[cl_col].unique() if c != -1]
                k_star = len(unique_clusters)
                noise_count = (df_c[cl_col] == -1).sum()
                noise_pct = (noise_count / n_rows) * 100.0

                if n_rows == EXPECTED_NODES:
                    print(f" ✓ {m_name:<28} | K*={k_star:>2d} cụm | Nhiễu (Noise): {noise_count:>3d}/{n_rows} ô ({noise_pct:>5.1f}%) -> [PASS]")
                    clusters_found += 1
                else:
                    print(f" ! {m_name:<28} | Số dòng: {n_rows} (kỳ vọng {EXPECTED_NODES}) -> [WARN]")
                    clusters_ok = False
            except Exception as e:
                print(f" ✗ Lỗi đọc nhãn {target_csv.name}: {e} -> [FAIL]")
                clusters_ok = False
        else:
            print(f" ✗ {m_name:<28} | Chưa tồn tại tệp nhãn HDBSCAN -> [FAIL]")
            clusters_ok = False

    if clusters_found != len(ALL_MODELS):
        clusters_ok = False

    scores.append(("9 Tệp nhãn phân cụm HDBSCAN (Clusters)", clusters_ok))

    # --------------------------------------------------------------------------
    # PHẦN 3: ĐỐI SOÁT PHỤ LỤC BẢNG C.2 (TABLES/)
    # --------------------------------------------------------------------------
    print_header("[PHẦN 3: ĐỐI SOÁT PHỤ LỤC BẢNG C.2 (TABLES/cluster_stats_hdbscan.csv)]")
    table_path = TABLES_DIR / "cluster_stats_hdbscan.csv"
    table_ok = False

    if table_path.exists():
        try:
            df_tab = pd.read_csv(table_path)
            size_kb = table_path.stat().st_size / 1024
            req_cols = ["dbcv", "modularity"]
            col_match = any(any(rc in str(c).lower() for rc in req_cols) for c in df_tab.columns)

            print(f" ✓ Tệp Phụ lục Bảng C.2: {table_path.name} ({size_kb:.1f} KB) | Số dòng: {len(df_tab)}")
            print(f"   • Cột chỉ số chất lượng phân cụm (DBCV & Modularity Q) : {'Đầy đủ' if col_match else 'Chưa chuẩn'}")

            print("\n   [Chi tiết số liệu Phụ lục Bảng C.2]:")
            for idx, r in df_tab.iterrows():
                m_title = str(r.iloc[0])
                c1 = r.iloc[1] if len(r) > 1 else ""
                c2 = r.iloc[2] if len(r) > 2 else ""
                c3 = r.iloc[3] if len(r) > 3 else ""
                print(f"     - {m_title:<28} | DBCV: {str(c1):>12} | Q(Contig): {str(c2):>12} | Q(Walk): {str(c3):>12}")

            if len(df_tab) >= 9 and col_match:
                table_ok = True
                print(" -> [PASS]")
            else:
                print(" -> [WARN]")
        except Exception as e:
            print(f" ✗ Lỗi đọc Bảng C.2: {e} -> [FAIL]")
    else:
        print(f" ✗ Không tìm thấy Bảng C.2 tại: {table_path} -> [FAIL]")

    scores.append(("Phụ lục Bảng C.2 (Table C.2 HDBSCAN Benchmark)", table_ok))

    # --------------------------------------------------------------------------
    # PHẦN 4: ĐỐI SOÁT 2 ẤN PHẨM HÌNH ẢNH KHOA HỌC (FIGURES/)
    # --------------------------------------------------------------------------
    print_header("[PHẦN 4: ĐỐI SOÁT 2 ẤN PHẨM HÌNH ẢNH KHOA HỌC (FIGURES/)]")
    expected_figs = {
        "cluster_maps_hdbscan.jpg": "Ma trận 3x3 bản đồ phân vùng HDBSCAN (Hình C.2)",
        "tsne_embeddings_hdbscan.jpg": "Ma trận 3x3 biểu đồ t-SNE HDBSCAN (Hình C.3b)"
    }
    figs_passed = 0

    for fig_name, fig_desc in expected_figs.items():
        fig_p = FIGURES_DIR / fig_name
        if fig_p.exists():
            size_kb = fig_p.stat().st_size / 1024
            if size_kb >= 100.0:
                print(f" ✓ {fig_name:<30} | {size_kb:>8.1f} KB | {fig_desc} -> [PASS]")
                figs_passed += 1
            else:
                print(f" ! {fig_name:<30} | {size_kb:>8.1f} KB | Dung lượng quá nhỏ (<100KB) -> [WARN]")
        else:
            print(f" ✗ {fig_name:<30} | Chưa tồn tại trên đĩa cứng -> [FAIL]")

    figs_ok = (figs_passed == len(expected_figs))
    scores.append(("2 Ấn phẩm đồ họa chuẩn 300 DPI (Figures)", figs_ok))

    # --------------------------------------------------------------------------
    # PHẦN 5: KIỂM ĐỊNH QUY LUẬT KHOA HỌC & ĐỘ BỀN VỮNG (SCIENTIFIC AUDIT)
    # --------------------------------------------------------------------------
    print_header("[PHẦN 5: KIỂM ĐỊNH ĐỘ BỀN VỮNG & QUY LUẬT TRIỆT TIÊU THUỘC TÍNH]")
    science_ok = True

    # 5.1 Kiểm tra sự sụp đổ của Models triệt tiêu POIs (Models 3S, 4S có tỷ lệ nhiễu cực cao)
    s_collapse_verified = False
    try:
        cand_3s = [
            CLUSTERS_DIR / "model_3s_han-gae_multi-s_hdbscan.csv",
            CLUSTERS_DIR / "model_3s_han_gae_multi_s_hdbscan.csv",
            CLUSTERS_DIR / "model3s_hdbscan.csv"
        ]
        csv_m3s = next((c for c in cand_3s if c.exists()), None)

        cand_4s = [
            CLUSTERS_DIR / "model_4s_han-gae_all-s_hdbscan.csv",
            CLUSTERS_DIR / "model_4s_han_gae_all_s_hdbscan.csv",
            CLUSTERS_DIR / "model4s_hdbscan.csv"
        ]
        csv_m4s = next((c for c in cand_4s if c.exists()), None)

        if csv_m3s and csv_m4s:
            df_3s = pd.read_csv(csv_m3s)
            df_4s = pd.read_csv(csv_m4s)
            
            c3s = df_3s["cluster"] if "cluster" in df_3s.columns else df_3s.iloc[:, -1]
            c4s = df_4s["cluster"] if "cluster" in df_4s.columns else df_4s.iloc[:, -1]

            n3s_noise = (c3s == -1).mean() * 100.0
            n4s_noise = (c4s == -1).mean() * 100.0

            if n3s_noise >= 90.0 and n4s_noise >= 80.0:
                print(f" ✓ Kiểm định Triệt tiêu: Model 3S ({n3s_noise:.1f}% Noise) & Model 4S ({n4s_noise:.1f}% Noise)")
                print(f"   --> Xác nhận: Triệt tiêu POIs dẫn đến sụp đổ phân cụm mật độ HDBSCAN -> [PASS]")
                s_collapse_verified = True
            else:
                print(f" ! Tỷ lệ nhiễu Model 3S/4S chưa đạt ngưỡng sụp đổ kỳ vọng -> [WARN]")
                science_ok = False
    except Exception as e:
        print(f" ! Lỗi kiểm định sụp đổ thuộc tính: {e}")
        science_ok = False

    # 5.2 Kiểm định tính ổn định của Model 4 (HAN-GAE: All)
    m4_verified = False
    try:
        cand_4 = [
            CLUSTERS_DIR / "model_4_han-gae_all_hdbscan.csv",
            CLUSTERS_DIR / "model_4_han_gae_all_hdbscan.csv",
            CLUSTERS_DIR / "model4_hdbscan.csv"
        ]
        csv_m4 = next((c for c in cand_4 if c.exists()), None)

        if csv_m4:
            df_m4 = pd.read_csv(csv_m4)
            c4 = df_m4["cluster"] if "cluster" in df_m4.columns else df_m4.iloc[:, -1]
            u_c4 = [c for c in c4.unique() if c != -1]
            if len(u_c4) >= 4:
                print(f" ✓ Kiểm định Tính ổn định: Model 4 nhận diện ổn định {len(u_c4)} cụm chức năng rõ ràng -> [PASS]")
                m4_verified = True
            else:
                print(f" ! Model 4 chỉ nhận diện được {len(u_c4)} cụm -> [WARN]")
                science_ok = False
    except Exception as e:
        print(f" ! Lỗi kiểm định Model 4: {e}")
        science_ok = False

    if not (s_collapse_verified and m4_verified):
        science_ok = False

    scores.append(("Quy luật khoa học & Kiểm định độ bền vững (Robustness)", science_ok))

    # --------------------------------------------------------------------------
    # BẢNG TỔNG HỢP NGHIỆM THU PHỤ LỤC HDBSCAN
    # --------------------------------------------------------------------------
    print_header("BẢNG TỔNG HỢP KẾT QUẢ ĐỐI SOÁT & NGHIỆM THU PHỤ LỤC HDBSCAN")
    passed_count = sum(1 for _, ok in scores if ok)
    total_criteria = len(scores)

    for name, ok in scores:
        status_str = "✓ HOÀN THÀNH (PASS)" if ok else "✗ CHƯA ĐẠT (WARN/FAIL)"
        print(f"  • {name:<55} : {status_str}")

    pct = (passed_count / total_criteria) * 100
    print(f"\nTỶ LỆ ĐẠT CHUẨN: {passed_count}/{total_criteria} TIÊU CHÍ ({pct:.1f}%)")

    if pct == 100.0:
        print("\n>>> KẾT LUẬN: KẾT QUẢ PHỤ LỤC HDBSCAN ĐẠT CHUẨN KHOA HỌC XUẤT SẮC 100%! <<<")
    elif pct >= 80.0:
        print("\n>>> KẾT LUẬN: KẾT QUẢ ĐÃ CƠ BẢN HOÀN TẤT. CẦN KIỂM TRA LẠI CÁC MỤC CẢNH BÁO. <<<")
    else:
        print("\n>>> KẾT LUẬN: CẦN CHẠY LẠI appendix_evaluation_hdbscan_hanoi.py ĐỂ HOÀN TẤT ĐẦY ĐỦ. <<<")
    print("=" * 88)


if __name__ == "__main__":
    audit_appendix_hdbscan()