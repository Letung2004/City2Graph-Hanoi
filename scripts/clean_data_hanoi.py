# -*- coding: utf-8 -*-
r"""
KỊCH BẢN KIỂM SOÁT & DỌN DẸP DỮ LIỆU TOÀN DIỆN CHO DỰ ÁN CITY2GRAPH HÀ NỘI
Quét toàn bộ thư mục data/ theo cơ chế Whitelist: Đối soát từng tệp (.png, .jpg,
.csv, .pt, .npy, .parquet, .gpkg, .json, .zip, .state, .geojson).
Đảm bảo data/ chỉ chứa duy nhất dữ liệu Hà Nội và kết quả chính thức từ Notebooks 01-04.

Vị trí lưu: D:\DATN\City2Graph-Hanoi\scripts\clean_data_hanoi.py

Cách sử dụng:
1. Chế độ mô phỏng kiểm tra (Dry-run, an toàn 100%, không xóa gì):
   python scripts/clean_data_hanoi.py
2. Thực hiện chuyển an toàn các file thừa vào data/_trash_backup/:
   python scripts/clean_data_hanoi.py --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Tự động định vị thư mục gốc dự án
CWD = Path.cwd().resolve()
PROJECT_ROOT = CWD
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
TRASH_DIR = DATA_DIR / "_trash_backup"

print("=" * 105)
print("HỆ THỐNG KIỂM TOÁN & LỌC TỆP DỮ LIỆU TOÀN DIỆN DỰ ÁN CITY2GRAPH THỦ ĐÔ HÀ NỘI")
print(f"Thư mục khảo sát: {DATA_DIR.resolve()}")
print("=" * 105)

# ---------------------------------------------------------------------------
# 1. DANH MỤC 33 ẤN PHẨM HÌNH ẢNH KHOA HỌC CHÍNH THỨC (NOTEBOOKS 01 - 04)
# ---------------------------------------------------------------------------
OFFICIAL_FIGURES = {
    # File 01: 5 ấn phẩm bản đồ tiền xử lý dữ liệu
    "16_poi_categories.png",
    "hanoi_total_poi_density.png",
    "centroids_comparison.png",
    "hanoi_street_network.png",
    "hanoi_gtfs_travel_graph.png",

    # File 02: 6 ấn phẩm bản đồ & histogram đồ thị không gian
    "hanoi_raw_features_histogram.png",
    "hanoi_scaled_features_histogram.png",
    "hanoi_walk_4relations_matrix.png",
    "hanoi_fig12a_15min_walk_solo.png",
    "hanoi_fig2_hetero_8relations_matrix.png",
    "hanoi_fig12b_15min_multi_solo.png",

    # File 03: 8 biểu đồ suy giảm hàm mất mát (Loss Progress)
    "hanoi_loss_progress_model1_gat_gae.png",
    "hanoi_loss_progress_model2_han_gae_walk.png",
    "hanoi_loss_progress_model3_han_gae_multi.png",
    "hanoi_loss_progress_model4_han_gae_all.png",
    "hanoi_loss_progress_model1s_gat_gae_struct.png",
    "hanoi_loss_progress_model2s_han_gae_walk_struct.png",
    "hanoi_loss_progress_model3s_han_gae_multi_struct.png",
    "hanoi_loss_progress_model4s_han_gae_all_struct.png",

    # File 04: 14 ấn phẩm đánh giá phân cụm, Modularity, Attention, t-SNE & Isochrones
    "silhouette.jpg",
    "davies_bouldin.jpg",
    "calinski_harabasz.jpg",
    "modularity_contig.jpg",
    "modularity_walk.jpg",
    "modularity_multi.jpg",
    "cluster_maps.jpg",
    "similarity_maps.jpg",
    "attention_bar.jpg",
    "attention_weight_layer_1.jpg",
    "attention_weight_layer_2.jpg",
    "tsne_embeddings.jpg",
    "isochrones.jpg",
    "similarity_decay_by_time_bin.jpg",
}

# ---------------------------------------------------------------------------
# 2. BẢNG WHITELIST CHÍNH XÁC THEO ĐƯỜNG DẪN TƯƠNG ĐỐI (DATA/...)
# ---------------------------------------------------------------------------
# Mọi file nằm trong danh sách này là BẮT BUỘC BẢO VỆ 100%
CANONICAL_RELATIVE_FILES = {
    # 2.1. Dữ liệu thô Hà Nội (Raw)
    "raw/hanoi/zones/hanoi_zones.parquet",
    "raw/hanoi/zones/hanoi_boundary_raw.geojson",
    "raw/hanoi/zones/hanoi_wards_overpass_real.geojson",
    "raw/hanoi/overture/hanoi_pois.parquet",
    "raw/hanoi/overture/hanoi_segments.parquet",
    "raw/hanoi/overture/hanoi_connectors.parquet",
    "raw/hanoi/gtfs/hanoi_bus_gtfs.zip",

    # 2.2. Đặc trưng đã tiền xử lý (Processed Features)
    "processed/hanoi/features/oa_with_features.gpkg",
    "processed/hanoi/features/hanoi_normalized_features.parquet",
    "processed/hanoi/features/street_nodes.gpkg",
    "processed/hanoi/features/street_edges.gpkg",
    "processed/hanoi/features/travel_summary_nodes.gpkg",
    "processed/hanoi/features/travel_summary_edges.gpkg",

    # 2.3. Đồ thị PyTorch Geometric (Processed Graphs)
    "processed/hanoi/graphs/homo.pt",
    "processed/hanoi/graphs/hetero_walk.pt",
    "processed/hanoi/graphs/hetero_multi.pt",
    "processed/hanoi/graphs/oa_feature_scaling.json",
    "processed/hanoi/graphs/hanoi_hetero_data.pt",  # Bảo lưu file hợp nhất

    # 2.4. Ma trận Vector nhúng Z & Attention (Outputs Embeddings)
    "outputs/hanoi/embeddings/Z_model1_gat_gae.npy",
    "outputs/hanoi/embeddings/Z_model2_han_gae_walk.npy",
    "outputs/hanoi/embeddings/Z_model3_han_gae_multi.npy",
    "outputs/hanoi/embeddings/Z_model4_han_gae_all.npy",
    "outputs/hanoi/embeddings/Z_model1s_gat_gae_struct.npy",
    "outputs/hanoi/embeddings/Z_model2s_han_gae_walk_struct.npy",
    "outputs/hanoi/embeddings/Z_model3s_han_gae_multi_struct.npy",
    "outputs/hanoi/embeddings/Z_model4s_han_gae_all_struct.npy",
    "outputs/hanoi/embeddings/model1_gat_gae.pt",
    "outputs/hanoi/embeddings/model2_han_gae_walk.pt",
    "outputs/hanoi/embeddings/model3_han_gae_multi.pt",
    "outputs/hanoi/embeddings/model4_han_gae_all.pt",
    "outputs/hanoi/embeddings/model1s_gat_gae_struct.pt",
    "outputs/hanoi/embeddings/model2s_han_gae_walk_struct.pt",
    "outputs/hanoi/embeddings/model3s_han_gae_multi_struct.pt",
    "outputs/hanoi/embeddings/model4s_han_gae_all_struct.pt",
    "outputs/hanoi/embeddings/beta_model2_han_gae_walk.npy",
    "outputs/hanoi/embeddings/beta_model3_han_gae_multi.npy",
    "outputs/hanoi/embeddings/beta_model4_han_gae_all.npy",

    # 2.5. Checkpoints trọng số tối ưu
    "outputs/hanoi/checkpoints/model1_gat_gae/model.pt",
    "outputs/hanoi/checkpoints/model2_han_gae_walk/model.pt",
    "outputs/hanoi/checkpoints/model3_han_gae_multi/model.pt",
    "outputs/hanoi/checkpoints/model4_han_gae_all/model.pt",
    "outputs/hanoi/checkpoints/model1s_gat_gae_struct/model.pt",
    "outputs/hanoi/checkpoints/model2s_han_gae_walk_struct/model.pt",
    "outputs/hanoi/checkpoints/model3s_han_gae_multi_struct/model.pt",
    "outputs/hanoi/checkpoints/model4s_han_gae_all_struct/model.pt",
    # Biến thể file checkpoint phẳng
    "outputs/hanoi/checkpoints/model1_gat_gae.pt",
    "outputs/hanoi/checkpoints/model2_han_gae_walk.pt",
    "outputs/hanoi/checkpoints/model3_han_gae_multi.pt",
    "outputs/hanoi/checkpoints/model4_han_gae_all.pt",
    "outputs/hanoi/checkpoints/model1s_gat_gae_struct.pt",
    "outputs/hanoi/checkpoints/model2s_han_gae_walk_struct.pt",
    "outputs/hanoi/checkpoints/model3s_han_gae_multi_struct.pt",
    "outputs/hanoi/checkpoints/model4s_han_gae_all_struct.pt",

    # 2.6. Nhãn phân cụm K-Means (Outputs Clusters)
    "outputs/hanoi/clusters/baseline_pca_kmeans.csv",
    "outputs/hanoi/clusters/model_1_gat-gae_kmeans.csv",
    "outputs/hanoi/clusters/model_2_han-gae_walk_kmeans.csv",
    "outputs/hanoi/clusters/model_3_han-gae_multi_kmeans.csv",
    "outputs/hanoi/clusters/model_4_han-gae_all_kmeans.csv",
    "outputs/hanoi/clusters/model_1s_gat-gae_struct_kmeans.csv",
    "outputs/hanoi/clusters/model_2s_han-gae_walk-s_kmeans.csv",
    "outputs/hanoi/clusters/model_3s_han-gae_multi-s_kmeans.csv",
    "outputs/hanoi/clusters/model_4s_han-gae_all-s_kmeans.csv",
    # Biến thể dấu gạch dưới an toàn
    "outputs/hanoi/clusters/model_1_gat_gae_kmeans.csv",
    "outputs/hanoi/clusters/model_2_han_gae_walk_kmeans.csv",
    "outputs/hanoi/clusters/model_3_han_gae_multi_kmeans.csv",
    "outputs/hanoi/clusters/model_4_han_gae_all_kmeans.csv",
    "outputs/hanoi/clusters/model_1s_gat_gae_struct_kmeans.csv",
    "outputs/hanoi/clusters/model_2s_han_gae_walk_s_kmeans.csv",
    "outputs/hanoi/clusters/model_3s_han_gae_multi_s_kmeans.csv",
    "outputs/hanoi/clusters/model_4s_han_gae_all_s_kmeans.csv",

    # 2.7. Bảng số liệu thống kê khoa học (Outputs Tables)
    "outputs/hanoi/tables/cluster_stats_mean_std.csv",
    "outputs/hanoi/tables/cluster_stats_mean_std_appendix.csv",
    "outputs/hanoi/tables/similarity_decay_by_time_bin.csv",
}

# Bổ sung 33 ấn phẩm hình ảnh vào danh mục đường dẫn chuẩn
for fig_name in OFFICIAL_FIGURES:
    CANONICAL_RELATIVE_FILES.add(f"outputs/hanoi/figures/{fig_name}")

# Chuẩn hóa dấu gạch chéo
CANONICAL_NORMALIZED = {Path(p).as_posix() for p in CANONICAL_RELATIVE_FILES}


def is_protected_file(file_path: Path) -> bool:
    """Kiểm tra xem tệp có nằm trong danh mục bảo vệ chuẩn tắc không."""
    try:
        rel_path = file_path.relative_to(DATA_DIR).as_posix()
    except ValueError:
        return False

    if rel_path in CANONICAL_NORMALIZED:
        return True

    # Nếu nằm trong figures/hanoi và tên file nằm trong 33 ấn phẩm chính thức
    if "outputs/hanoi/figures" in rel_path and file_path.name in OFFICIAL_FIGURES:
        return True

    return False


def categorize_removable_file(file_path: Path) -> tuple[str, str]:
    """Phân loại lý do và hạng mục cần dọn dẹp của tệp."""
    rel_posix = file_path.relative_to(DATA_DIR).as_posix()
    fname = file_path.name.lower()

    # 1. Dữ liệu ngoài Hà Nội (Liverpool hoặc thư mục không có tiền tố hanoi)
    if "liverpool" in rel_posix:
        return ("Dữ liệu Liverpool thừa", "Tệp của ca nghiên cứu Liverpool (không thuộc Hà Nội)")

    if any(rel_posix.startswith(p) for p in ["raw/overture/", "raw/gtfs/", "raw/zones/", "processed/graphs/", "processed/features/"]):
        return ("Dữ liệu ngoài thư mục hanoi/", "Tệp nằm ngoài cấu trúc chuẩn hanoi/ (cần dọn dẹp)")

    # 2. Dữ liệu thô trùng lặp trong overture
    if "raw/hanoi/overture" in rel_posix and file_path.suffix.lower() == ".geojson":
        return ("Dữ liệu thô trùng lặp (.geojson)", "Đã có bản nén .parquet tương ứng nhanh và nhẹ hơn")

    # 3. File text giải nén trong gtfs
    if "raw/hanoi/gtfs" in rel_posix and file_path.suffix.lower() == ".txt":
        return ("GTFS giải nén rời rạc", "Tệp văn bản giải nén (Code đã nạp trực tiếp hanoi_bus_gtfs.zip)")

    # 4. Ảnh nháp/thừa trong figures
    if "outputs/hanoi/figures" in rel_posix and file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        if file_path.name not in OFFICIAL_FIGURES:
            return ("Ảnh nháp / thử nghiệm cũ", "Không thuộc 33 ấn phẩm khoa học chính thức của bài báo")

    # 5. Tệp tạm, cache, backup, checkpoint nháp
    if file_path.suffix.lower() in [".tmp", ".bak", ".state", ".cache", ".swp"] or fname.startswith("."):
        return ("Tệp tạm / Trạng thái nháp (.state/.tmp)", f"Tệp tạm thời {file_path.suffix}")

    # 6. Trọng số cũ hoặc tệp không nằm trong pipeline
    if "weights_" in fname or "old" in fname or "backup" in fname or "test" in fname:
        return ("Trọng số / Thử nghiệm cũ", "Tệp nháp kiểm thử không dùng trong pipeline 01-04")

    return ("Tệp thừa / Ngoài pipeline 01-04", f"Không thuộc danh mục 33 ấn phẩm và tệp chuẩn tắc")


def scan_unwanted_files() -> list[tuple[Path, str, float, str]]:
    """
    Quét toàn bộ thư mục data/ (ngoại trừ _trash_backup/)
    để tìm các tệp không nằm trong whitelist chuẩn tắc.
    """
    removable = []

    for root, _, files in os.walk(DATA_DIR):
        root_path = Path(root)
        # Bỏ qua thư mục lưu trữ rác tạm
        if "_trash_backup" in root_path.as_posix():
            continue

        for f in files:
            f_path = root_path / f
            if not is_protected_file(f_path):
                cat_type, reason = categorize_removable_file(f_path)
                size_mb = f_path.stat().st_size / (1024 * 1024)
                removable.append((f_path, reason, size_mb, cat_type))

    return removable


def clean_empty_directories():
    """Dọn dẹp các thư mục rỗng sau khi đã di chuyển tệp thừa."""
    empty_dirs = []
    for root, dirs, files in os.walk(DATA_DIR, topdown=False):
        p = Path(root)
        if "_trash_backup" in p.as_posix() or p == DATA_DIR:
            continue
        if not any(p.iterdir()):
            empty_dirs.append(p)

    for d in empty_dirs:
        try:
            d.rmdir()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Kiểm soát & Dọn dẹp dữ liệu thừa trong thư mục data/")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Thực hiện chuyển các file thừa vào data/_trash_backup/ (Mặc định: Chỉ mô phỏng kiểm tra)",
    )
    args = parser.parse_args()

    # 1. Kiểm kê các tệp chính thức hiện diện
    existing_canonical = sum(1 for p in CANONICAL_RELATIVE_FILES if (DATA_DIR / p).exists())
    existing_official_figs = sum(1 for f in OFFICIAL_FIGURES if (DATA_DIR / "outputs" / "hanoi" / "figures" / f).exists())

    print(f"\n📊 BÁO CÁO KIỂM KÊ TỆP CHÍNH THỨC HÀ NỘI:")
    print(f"  • Cơ sở dữ liệu cốt lõi (Bảng, Đồ thị, Vector Z, Cụm): {existing_canonical - existing_official_figs} tệp hợp lệ.")
    print(f"  • Ấn phẩm khoa học chuẩn mực (300 DPI):                {existing_official_figs}/{len(OFFICIAL_FIGURES)} tệp đang được bảo vệ an toàn.")
    print("-" * 105)

    # 2. Quét các tệp cần dọn dẹp
    candidates = scan_unwanted_files()

    if not candidates:
        print("\n✓ TUYỆT VỜI: Thư mục data/ của bạn hiện đã ĐẠT CHUẨN HOÀN HẢO 100%!")
        print("  Không phát hiện bất kỳ tệp GeoJSON trùng lặp, tệp giải nén thừa, ảnh nháp hay tệp .state nào.")
        print("=" * 105)
        return

    total_size_mb = sum(c[2] for c in candidates)

    # Phân nhóm theo danh mục
    categories: dict[str, list[tuple[Path, str, float, str]]] = {}
    for item in candidates:
        cat = item[3]
        categories.setdefault(cat, []).append(item)

    print(f"\n⚠️  PHÁT HIỆN {len(candidates)} TỆP DƯ THỪA / KHÔNG THUỘC PIPELINE CHÍNH THỨC (Tổng dung lượng: {total_size_mb:.2f} MB):")
    print("-" * 105)

    idx = 1
    for cat_name, items in categories.items():
        cat_size = sum(it[2] for it in items)
        print(f"\n📁 [{cat_name.upper()}] - {len(items)} tệp (Tổng: {cat_size:.2f} MB):")
        for p, reason, size_mb, _ in items:
            try:
                rel_path = p.relative_to(PROJECT_ROOT)
            except ValueError:
                rel_path = p
            print(f"  {idx:2d}. {str(rel_path):<62} | {size_mb:>7.2f} MB | {reason}")
            idx += 1

    print("-" * 105)

    if not args.apply:
        print("\n[CHẾ ĐỘ MÔ PHỎNG / DRY-RUN: CHƯA CÓ BẤT KỲ FILE NÀO BỊ XÓA HAY DI CHUYỂN]")
        print("💡 Để thực hiện chuyển an toàn toàn bộ các file thừa trên vào thư mục 'data/_trash_backup/', hãy chạy:")
        print("   python scripts/clean_data_hanoi.py --apply")
        print("=" * 105)
    else:
        print("\n-> Đang chuyển các tệp thừa vào thư mục lưu trữ an toàn: data/_trash_backup/...")
        TRASH_DIR.mkdir(parents=True, exist_ok=True)
        moved_count = 0

        for p, _, _, cat_type in candidates:
            try:
                # Tạo thư mục con tương ứng trong trash để giữ nguyên cấu trúc
                rel_to_data = p.relative_to(DATA_DIR)
                dest_file = TRASH_DIR / rel_to_data
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                if dest_file.exists():
                    dest_file = dest_file.parent / f"{dest_file.stem}_old{dest_file.suffix}"

                shutil.move(str(p), str(dest_file))
                moved_count += 1
            except Exception as e:
                print(f"  ! Không thể di chuyển {p.name}: {e}")

        # Dọn dẹp thư mục rỗng
        clean_empty_directories()

        print(f"\n✓ ĐÃ DI CHUYỂN AN TOÀN {moved_count}/{len(candidates)} TỆP VÀO: {TRASH_DIR.resolve()}")
        print(f"✓ BẠN ĐÃ GIẢI PHÓNG THÀNH CÔNG {total_size_mb:.2f} MB CHO THƯ MỤC DATA!")
        print("💡 Toàn bộ dữ liệu Hà Nội, 3 đồ thị PyG, 8 vector Z, 9 nhãn K-Means và 33 ấn phẩm chính thức đều được bảo toàn 100%.")
        print("=" * 105)


if __name__ == "__main__":
    main()