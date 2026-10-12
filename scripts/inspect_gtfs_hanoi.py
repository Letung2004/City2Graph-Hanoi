# -*- coding: utf-8 -*-
"""
KỊCH BẢN KIỂM TOÁN & ĐỐI SOÁT TOÀN DIỆN DỮ LIỆU FILE 01 (TIỀN XỬ LÝ HÀ NỘI)
Công trình đối chuẩn: Sato et al. (CEUS 2026, Section 4.1 - 4.3)
Chạy độc lập từ terminal: python scripts/inspect_gtfs_hanoi.py
"""

import os
import sys
import zipfile
from pathlib import Path
import geopandas as gpd
import pandas as pd
import numpy as np

# 1. Tự động định vị thư mục gốc của repository (Dynamic Root Discovery)
CWD = Path.cwd().resolve()
REPO_ROOT = CWD
while not (REPO_ROOT / "data").exists() and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "hanoi"
PROCESSED_FEAT_DIR = DATA_DIR / "processed" / "hanoi" / "features"
OUTPUT_FIG_DIR = DATA_DIR / "outputs" / "hanoi" / "figures"

# Danh mục 16 nhóm tiện ích chuẩn mực theo Table A.1 (Sato et al., CEUS 2026)
EXPECTED_POI_CATEGORIES = [
    'retail', 'food_beverage', 'corporate_service', 'industrial_service',
    'consumer_service', 'healthcare', 'education', 'public_service',
    'transportation_facility', 'automotive_facility', 'hotel_lodging',
    'sports_fitness', 'park', 'arts_culture', 'entertainment', 'religion'
]

# Bộ cờ ghi nhận kết quả kiểm toán tổng thể
audit_results = {}

print("=" * 85)
print("BÁO CÁO KIỂM TOÁN & ĐỐI SOÁT TOÀN DIỆN DỮ LIỆU ĐÔ THỊ FILE 01 (HÀ NỘI)")
print(f"Thư mục gốc dự án: {REPO_ROOT}")
print("=" * 85)

print("\n[PHẦN 1: ĐỐI SOÁT ĐƠN VỊ PHÂN VÙNG H3 & MA TRẬN 16 NHÓM POIs]")
print("-" * 85)

gpkg_path = PROCESSED_FEAT_DIR / "oa_with_features.gpkg"
parquet_path = PROCESSED_FEAT_DIR / "hanoi_normalized_features.parquet"

oa_features = None
if gpkg_path.exists():
    try:
        oa_features = gpd.read_file(gpkg_path)
        print(f"✓ Đã đọc tệp GeoPackage: {gpkg_path.name} ({gpkg_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"✗ Lỗi khi đọc GeoPackage: {e}")

if oa_features is None and parquet_path.exists():
    try:
        oa_features = gpd.read_parquet(parquet_path)
        print(f"✓ Đã đọc tệp Parquet dự phòng: {parquet_path.name} ({parquet_path.stat().st_size / 1024:.1f} KB)")
    except Exception as e:
        print(f"✗ Lỗi khi đọc Parquet: {e}")

if oa_features is not None:
    # 1.1 Kiểm tra số lượng đơn vị không gian
    n_zones = len(oa_features)
    crs_str = str(oa_features.crs)
    is_count_valid = (n_zones == 985)
    is_crs_valid = ("32648" in crs_str)
    
    print(f"• Số lượng ô không gian:    {n_zones:,} ô lục giác (Kỳ vọng: 985 ô) -> {'[CHẨN ĐOÁN: HỢP LỆ]' if is_count_valid else '[CẢNH BÁO: LỆCH SỐ Ô]'}")
    print(f"• Hệ quy chiếu không gian:  {crs_str} (Kỳ vọng: EPSG:32648) -> {'[CHẨN ĐOÁN: CHUẨN MÉT]' if is_crs_valid else '[CẢNH BÁO: CHƯA PHẢI UTM 48N]'}")
    
    audit_results["Spatial Units (985 H3)"] = is_count_valid
    audit_results["Projected CRS (EPSG:32648)"] = is_crs_valid

    # 1.2 Kiểm tra diện tích và tính đẳng hướng hình học
    if "area_km2" in oa_features.columns:
        mean_area = oa_features["area_km2"].mean()
        total_area = oa_features["area_km2"].sum()
        print(f"• Diện tích trung bình/ô:   {mean_area:.4f} km² (~105,000 m²)")
        print(f"• Tổng diện tích nghiên cứu: {total_area:.2f} km² (Kỳ vọng: ~113.22 km²)")
    
    # 1.3 Kiểm tra chi tiết 16 cột POIs
    print("\n• Bảng đối soát phân bố số học 16 nhóm chức năng tầng trệt:")
    missing_cats = [c for c in EXPECTED_POI_CATEGORIES if c not in oa_features.columns]
    
    if not missing_cats:
        poi_stats = []
        total_all_pois = 0
        zero_cat_count = 0
        
        for cat in EXPECTED_POI_CATEGORIES:
            col_sum = oa_features[cat].sum()
            col_max = oa_features[cat].max()
            covered_zones = (oa_features[cat] > 0).sum()
            coverage_pct = (covered_zones / n_zones) * 100
            total_all_pois += col_sum
            if col_sum == 0:
                zero_cat_count += 1
                
            poi_stats.append({
                "Nhóm POI": cat,
                "Tổng số": f"{col_sum:,}",
                "Cực đại/ô": f"{col_max:,}",
                "Số ô phủ": f"{covered_zones}/{n_zones}",
                "Tỷ lệ phủ": f"{coverage_pct:.1f}%",
                "Đánh giá": "✓ Đạt" if col_sum > 0 else "✗ Rỗng (0)"
            })
            
        df_poi_audit = pd.DataFrame(poi_stats)
        print(df_poi_audit.to_string(index=False))
        print(f"\n-> Tổng số tiện ích tầng trệt ghi nhận: {total_all_pois:,} cơ sở.")
        
        has_diversity = (zero_cat_count == 0)
        audit_results["16 POI Categories Diversity"] = has_diversity
        if not has_diversity:
            print(f"⚠ CẢNH BÁO: Có {zero_cat_count} nhóm POI có tổng số lượng bằng 0! Cần kiểm tra lại hàm classify_poi.")
        else:
            print("✓ Toàn bộ 16 nhóm chức năng đô thị đều có dữ liệu phân bố, không có nhóm nào bị rỗng.")
    else:
        print(f"✗ CẢNH BÁO: Thiếu các cột đặc trưng POI: {missing_cats}")
        audit_results["16 POI Categories Diversity"] = False

    # 1.4 Kiểm tra các thuộc tính mạng và tâm đối xứng
    has_centroid = "pop_weighted_centroid" in oa_features.columns
    has_centrality = all(c in oa_features.columns for c in ['degree_centrality', 'betweenness_centrality', 'closeness_centrality'])
    print(f"\n• Tâm đối xứng (pop_weighted_centroid): {'✓ Có sẵn' if has_centroid else '✗ Thiếu'}")
    print(f">• Chỉ số Centrality mạng lưới:         {'✓ Có đủ (Degree, Betweenness, Closeness)' if has_centrality else '✗ Thiếu'}")
    audit_results["Centroids & Centrality"] = (has_centroid and has_centrality)
else:
    print(f"✗ Không tìm thấy tệp đặc trưng nút tại {PROCESSED_FEAT_DIR}.")
    audit_results["Spatial Units (985 H3)"] = False
    audit_results["16 POI Categories Diversity"] = False

print("\n[PHẦN 2: ĐỐI SOÁT MẠNG LƯỚI ĐƯỜNG BỘ NGƯỜI ĐI BỘ (OVERTURE)]")
print("-" * 85)

street_nodes_path = PROCESSED_FEAT_DIR / "street_nodes.gpkg"
street_edges_path = PROCESSED_FEAT_DIR / "street_edges.gpkg"

if street_nodes_path.exists() and street_edges_path.exists():
    try:
        street_nodes = gpd.read_file(street_nodes_path)
        street_edges = gpd.read_file(street_edges_path)
        
        n_st_nodes = len(street_nodes)
        n_st_edges = len(street_edges)
        print(f"✓ Đã đọc đồ thị đường bộ thành công:")
        print(f"  • Số lượng nút giao (street_nodes): {n_st_nodes:,} nút")
        print(f"  • Số lượng phân đoạn (street_edges): {n_st_edges:,} cạnh")
        
        # Kiểm tra trọng số thời gian đi bộ tau = length / 1.3333
        if "travel_time_sec" in street_edges.columns and "length" in street_edges.columns:
            implied_speed = (street_edges["length"] / street_edges["travel_time_sec"]).mean()
            print(f"  • Vận tốc đi bộ thực tế tính toán:  {implied_speed:.4f} m/s (~{implied_speed * 3.6:.1f} km/h)")
            is_speed_valid = np.isclose(implied_speed, 1.3333, atol=0.05)
            audit_results["Street Walking Speed (4.8km/h)"] = is_speed_valid
        else:
            print("  ⚠ Thiếu cột travel_time_sec hoặc length trong street_edges.")
            audit_results["Street Walking Speed (4.8km/h)"] = False
            
        audit_results["Street Network Extraction"] = (n_st_nodes > 10000 and n_st_edges > 10000)
    except Exception as e:
        print(f"✗ Lỗi khi đọc tệp đường bộ: {e}")
        audit_results["Street Network Extraction"] = False
else:
    print(f"✗ Chưa tìm thấy tệp street_nodes.gpkg hoặc street_edges.gpkg.")
    audit_results["Street Network Extraction"] = False

print("\n[PHẦN 3: ĐỐI SOÁT LỊCH TRÌNH XE BUÝT CÔNG CỘNG GTFS (HÀ NỘI)]")
print("-" * 85)

gtfs_zip = RAW_DIR / "gtfs" / "hanoi_bus_gtfs.zip"
if not gtfs_zip.exists():
    fallback_zips = list((RAW_DIR / "gtfs").glob("*.zip"))
    if fallback_zips:
        gtfs_zip = fallback_zips[0]

raw_stops_count = 0
if gtfs_zip.exists():
    with zipfile.ZipFile(gtfs_zip, "r") as z:
        if "stops.txt" in z.namelist():
            stops_df = pd.read_csv(z.open("stops.txt"))
            raw_stops_count = len(stops_df)
            print(f"• Tổng số trạm xe buýt thô trong toàn mạng lưới GTFS: {raw_stops_count:,} trạm (File: {gtfs_zip.name})")

travel_nodes_path = PROCESSED_FEAT_DIR / "travel_summary_nodes.gpkg"
travel_edges_path = PROCESSED_FEAT_DIR / "travel_summary_edges.gpkg"

if travel_nodes_path.exists() and travel_edges_path.exists():
    try:
        travel_nodes = gpd.read_file(travel_nodes_path)
        travel_edges = gpd.read_file(travel_edges_path)
        
        valid_stops = len(travel_nodes)
        valid_edges = len(travel_edges)
        
        print(f"• Số trạm buýt hợp lệ lọt vào nội đô (travel_nodes):     {valid_stops:,} trạm")
        if raw_stops_count > 0:
            print(f"  -> Đã loại bỏ ngoại thành xa:                      {raw_stops_count - valid_stops:,} trạm ({((raw_stops_count - valid_stops)/raw_stops_count)*100:.1f}%)")
        print(f"• Số cạnh liên kết lộ trình nối tiếp (travel_edges):       {valid_edges:,} liên kết")
        
        # Kiểm tra tính toàn vẹn của mã trạm kết nối
        col_src = "from_stop_id" if "from_stop_id" in travel_edges.columns else travel_edges.columns[0]
        col_dst = "to_stop_id" if "to_stop_id" in travel_edges.columns else travel_edges.columns[1]
        active_stops = set(travel_edges[col_src].astype(str)).union(set(travel_edges[col_dst].astype(str)))
        print(f"• Số trạm thực tế có tuyến buýt lưu thông:                 {len(active_stops):,} trạm ({len(active_stops)/valid_stops * 100:.1f}% tổng trạm nội đô)")
        
        # Kiểm tra thời gian xe buýt chạy trung bình
        if "travel_time_sec" in travel_edges.columns:
            mean_bus_time = travel_edges["travel_time_sec"].mean()
            print(f"• Thời gian xe buýt chạy trung bình giữa 2 trạm:           {mean_bus_time:.1f} giây (~{mean_bus_time/60:.2f} phút)")
        
        audit_results["GTFS Bus Graph"] = (valid_stops > 2000 and valid_edges > 1500)
    except Exception as e:
        print(f"✗ Lỗi khi đọc tệp GTFS đã xử lý: {e}")
        audit_results["GTFS Bus Graph"] = False
else:
    print(f"✗ Chưa tìm thấy travel_summary_nodes.gpkg hoặc travel_summary_edges.gpkg.")
    audit_results["GTFS Bus Graph"] = False

print("\n[PHẦN 4: ĐỐI SOÁT BỘ 5 BẢN ĐỒ KHOA HỌC XUẤT RA (FIGURES)]")
print("-" * 85)

expected_figures = [
    ("16_poi_categories.png", "Ma trận 16 bản đồ POIs tầng trệt (Figure 10b)", 500),
    ("hanoi_total_poi_density.png", "Bản đồ tổng mật độ POI có ScaleBar (Figure 10a Layout)", 300),
    ("centroids_comparison.png", "Bản đồ đối sánh phân vùng & Centroids (Section 4.2)", 200),
    ("hanoi_street_network.png", "Mạng lưới đường bộ phân cấp Dark Mode (Figure 9)", 500),
    ("hanoi_gtfs_travel_graph.png", "Đồ thị lộ trình xe buýt GTFS nền tối", 400),
]

all_figures_ok = True
for fname, desc, min_kb in expected_figures:
    fpath = OUTPUT_FIG_DIR / fname
    if fpath.exists():
        size_kb = fpath.stat().st_size / 1024
        is_size_ok = (size_kb >= min_kb)
        print(f"✓ {fname:<30} | {size_kb:>7.1f} KB | {desc} -> {'[ĐẠT CHUẨN ĐỒ HỌA]' if is_size_ok else '[CẢNH BÁO: DUNG LƯỢNG NHỎ]'}")
        if not is_size_ok:
            all_figures_ok = False
    else:
        print(f"✗ {fname:<30} | THIẾU FILE | {desc}")
        all_figures_ok = False

audit_results["5 Scientific Figures"] = all_figures_ok

print("\n" + "=" * 85)
print("BẢNG TỔNG HỢP KẾT QUẢ ĐỐI SOÁT & NGHIỆM THU PHA 1 (NOTEBOOK 01)")
print("=" * 85)

passed_count = sum(1 for v in audit_results.values() if v)
total_criteria = len(audit_results)

for criterion, status in audit_results.items():
    status_str = "✓ HOÀN THÀNH (PASS)" if status else "✗ CHƯA ĐẠT (FAIL/WARN)"
    print(f"  • {criterion:<35}: {status_str}")

print("-" * 85)
print(f"TỶ LỆ ĐẠT CHUẨN: {passed_count}/{total_criteria} TIÊU CHÍ ({(passed_count/total_criteria)*100:.1f}%)")

if passed_count == total_criteria:
    print(">>> KẾT LUẬN: TOÀN BỘ DỮ LIỆU FILE 01 CHUẨN XÁC 100%! BẠN HOÀN TOÀN TỰ TIN TIẾN TỚI PHA 2 (NOTEBOOK 02). <<<")
else:
    print(">>> KẾT LUẬN: CÓ MỘT SỐ HẠNG MỤC CẦN KIỂM TRA LẠI (XEM CÁC DÒNG CẢNH BÁO Ở TRÊN). <<<")
print("=" * 85)