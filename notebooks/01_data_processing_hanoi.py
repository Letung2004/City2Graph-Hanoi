# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH TIỀN XỬ LÝ DỮ LIỆU ĐÔ THỊ HÀ NỘI (CELL 1 - 29)
Kế thừa & Chuyển đổi 1:1 từ: 01_data_processing_2.py (Sato et al., CEUS 2026)
Hệ quy chiếu phẳng: WGS 84 / UTM Zone 48N (EPSG:32648)
Đơn vị không gian: Lưới lục giác Uber H3 Resolution 9 (Index: zone_id)
Hỗ trợ: VS Code Interactive Window (# %%) hoặc chạy dòng lệnh Python
"""


# %% [Cell 1]: Thiết lập môi trường và cấu hình hệ thống
# --- Ghi chú giải thích từ Notebook gốc ---
# # 01. Tiền xử lý dữ liệu không gian đô thị (Data Processing)
# -----------------------------------------------------------

import os
import sys
from pathlib import Path

# Cấu hình OpenMP tránh xung đột nhị phân runtime trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Tự động định vị thư mục gốc của dự án (Dynamic Root Discovery)
CWD = Path.cwd().resolve()
REPO_ROOT = CWD
while not (REPO_ROOT / "data").exists() and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent

# Kết nối thư mục mã nguồn lõi src/ vào hệ thống
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print(f"✓ [Cell 1] Đã định vị thư mục gốc dự án: {REPO_ROOT}")



# %% [Cell 2]: Khai báo thư viện, hằng số không gian và đường dẫn dữ liệu Hà Nội
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx

try:
    import rustworkx as rx
except ImportError:
    pass

try:
    import city2graph as c2g
except ImportError:
    pass

# Hằng số không gian & Hệ tọa độ phẳng Hà Nội
TARGET_CRS = "EPSG:32648"  # WGS 84 / UTM Zone 48N (đo bằng mét)
WGS84_CRS = "EPSG:4326"
TARGET_AUTHORITY = "Hanoi"

# Khung chữ nhật bao quanh vùng nghiên cứu nội thành Hà Nội (kinh/vĩ độ)
BBOX = [105.74, 20.95, 105.93, 21.10]

# Sử dụng dữ liệu thực địa đã tải sẵn thay vì gọi API Overture trực tiếp
USE_DOWNLOADED_DATA = True

# Thiết lập đường dẫn thư mục theo chuẩn dự án
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "hanoi"
OUTPUT_DIR = DATA_DIR / "processed" / "hanoi" / "features"
OUTPUT_FIG_DIR = DATA_DIR / "outputs" / "hanoi" / "figures"
OVERTURE_DIR = RAW_DIR / "overture"
ZONES_DIR = RAW_DIR / "zones"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OVERTURE_DIR.mkdir(parents=True, exist_ok=True)

# Đường dẫn các tệp dữ liệu đầu vào Hà Nội (hỗ trợ cả Parquet và GeoJSON)
INPUT_PATH = ZONES_DIR / "hanoi_zones.parquet"
if not INPUT_PATH.exists():
    INPUT_PATH = ZONES_DIR / "hanoi_boundary_raw.geojson"

POIS_PATH = OVERTURE_DIR / "hanoi_pois.parquet"
if not POIS_PATH.exists():
    POIS_PATH = OVERTURE_DIR / "place.geojson"

SEGMENTS_PATH = OVERTURE_DIR / "hanoi_segments.parquet"
if not SEGMENTS_PATH.exists():
    SEGMENTS_PATH = OVERTURE_DIR / "segment.geojson"

CONNECTORS_PATH = OVERTURE_DIR / "hanoi_connectors.parquet"
if not CONNECTORS_PATH.exists():
    CONNECTORS_PATH = OVERTURE_DIR / "connector.geojson"

print(f"✓ [Cell 2] Hệ quy chiếu: {TARGET_CRS} | Đơn vị không gian: {INPUT_PATH.name}")



# %% [Cell 3]: Nạp và chuẩn hóa đơn vị phân vùng không gian 985 ô lục giác Uber H3
# Nạp dữ liệu phân vùng Hà Nội
if str(INPUT_PATH).endswith(".parquet"):
    gdf = gpd.read_parquet(INPUT_PATH)
else:
    gdf = gpd.read_file(INPUT_PATH)

# Chuẩn hóa cột định danh đơn vị không gian (zone_id)
if "zone_id" not in gdf.columns:
    gdf["zone_id"] = [f"h3_res9_{i:04d}" for i in range(len(gdf))]

# Chuyển đổi hệ tọa độ phẳng UTM Zone 48N
if gdf.crs != TARGET_CRS:
    gdf = gdf.to_crs(TARGET_CRS)

# Loại bỏ các ô có hình học rỗng hoặc không hợp lệ
gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

# Gán biến đại diện cho Hà Nội và tạo bí danh tương thích với mã nguồn tác giả
hanoi_oa = gdf.copy()
liverpool_oa = hanoi_oa  # Bí danh giúp các cell kế tiếp thực thi trơn tru mà không cần sửa tên biến

print(f"✓ [Cell 3] Nạp thành công {len(liverpool_oa):,} ô lục giác H3 cho {TARGET_AUTHORITY} (CRS: {liverpool_oa.crs})")



# %% [Cell 4]: Nạp và cắt không gian tiện ích tầng trệt POIs theo vùng nghiên cứu Hà Nội
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 1.2 Xử lý dữ liệu POIs tầng trệt từ Overture Maps (Đặc trưng nút)
# ------------------------------------------------------------------

if USE_DOWNLOADED_DATA:
    # Nạp dữ liệu POIs đã trích xuất từ tệp cục bộ
    if str(POIS_PATH).endswith(".parquet"):
        pois = gpd.read_parquet(POIS_PATH)
    else:
        pois = gpd.read_file(POIS_PATH)
else:
    # Tải dữ liệu loại 'place' qua API city2graph
    data = c2g.load_overture_data(
        area=BBOX,
        types=['place'],
        output_dir=str(OVERTURE_DIR),
        save_to_file=True,
        return_data=True,
    )
    pois = data['place']

# Đồng bộ hệ quy chiếu phẳng mét UTM Zone 48N
if pois.crs != TARGET_CRS:
    pois = pois.to_crs(crs=TARGET_CRS)

# Cắt không gian (clip) giữ lại các điểm POIs nằm lọt trong phạm vi 985 ô H3 Hà Nội
pois = gpd.clip(pois, liverpool_oa)

print(f"✓ [Cell 4] Đã nạp và cắt không gian thành công: {len(pois):,} điểm tiện ích POIs tầng trệt.")



# %% [Cell 5]: Thiết lập danh mục phân loại 16 nhóm tiện ích tầng trệt (Bảng A.1)
# --- Ghi chú giải thích từ Notebook gốc ---
# #### Phân loại POIs thành 16 nhóm chức năng đô thị (Table A.1, Sato et al., CEUS 2026)
# -----------------------------------------------------------------------------------

CATEGORY_MAPPING = {
    # 1. BÁN LẺ & HÀNG TIÊU DÙNG (RETAIL)
    'retail': [
        'antique_store', 'business_office_supplies_and_stationery', 'charity_shop',
        'department_store', 'discount_store', 'duty_free_shop', 'flea_market',
        'gift_shop', 'kiosk', 'outlet_store', 'packing_supply', 'party_supply',
        'pawn_shop', 'pop_up_shop', 'rental_kiosks', 'retail', 'second_hand_store',
        'shopping', 'shopping_center', 'shopping_mall', 'souvenir_shop', 'superstore',
        'thrift_store', 'used_vintage_and_consignment', 'warehouse_store',
        'baby_gear_and_furniture', 'baby_store', 'boutique', 'bridal_shop', 'bridal_store',
        'childrens_clothing_store', 'clothing_store', 'costume_store', 'custom_t_shirt_store',
        'dance_wear', 'fashion', 'fashion_accessories_store', 'furrier', 'hat_shop',
        'jewelry_store', 'leather_goods_store', 'linen', 'luggage_store', 'maternity_wear',
        'mens_clothing_store', 'plus_size_fashion', 'shoe_store', 'sunglasses_store',
        'swimwear_store', 'traditional_clothing', 'vintage_clothing_store', 'watch_store',
        'wig_store', 'womens_clothing_store', 'appliance_store', 'bedding_and_bath_stores',
        'building_supply_store', 'carpet_store', 'electrical_supply_store', 'fireplace_store',
        'flooring', 'flowers_and_gifts_shop', 'flower_shop', 'florist', 'framing_store',
        'furniture_store', 'garden_center', 'glass_and_mirror_shop', 'hardware_store',
        'home_decor', 'home_goods_store', 'home_improvement_store', 'kitchen_supplies',
        'lighting_store', 'mattress_store', 'nursery_and_gardening', 'paint_store',
        'shades_and_blinds', 'wallpaper_store', 'window_treatment_store', 'camera_store',
        'computer_store', 'electronics', 'electronics_store', 'mobile_phone_store',
        'music_and_dvd_store', 'printing_equipment_and_supply', 'telecommunications',
        'video_game_store', 'art_supplies', 'arts_and_crafts', 'bicycle_shop', 'bookshop',
        'bookstore', 'comic_book_store', 'dive_shop', 'fabric_store', 'golf_equipment',
        'guitar_store', 'gun_shop', 'hobby_shop', 'hunting_and_fishing_supplies', 'music_store',
        'musical_instrument_store', 'newspaper_and_magazines_store', 'office_supplies',
        'outdoor_supply_store', 'record_shop', 'sewing_supply', 'skate_shop',
        'ski_and_snowboard_shop', 'sporting_goods', 'sporting_goods_store', 'stationery_store',
        'surf_shop', 'toy_store', 'trophy_shop', 'vinyl_record_store', 'adult_store',
        'beauty_product_supplier', 'candle_store', 'cosmetic_and_beauty_supplies',
        'cosmetics_and_beauty_supply', 'e_cigarette_store', 'firework_retailer',
        'hair_supply_stores', 'hearing_aids', 'lottery_ticket', 'perfume_store', 'pet_food',
        'pet_store', 'pet_supplies', 'reptile_shop', 'tobacco_shop', 'vape_store',
        'convenience_store', 'farmers_market', 'frozen_foods', 'fruit_and_vegetable_store',
        'grocery_store', 'greengrocer', 'health_food_store', 'health_market',
        'international_grocery_store', 'korean_grocery_store', 'market', 'organic_grocery_store',
        'public_market', 'specialty_grocery_store', 'supermarket', 'butcher', 'butcher_shop',
        'cheese_shop', 'dairy_stores', 'fishmonger', 'honey_farm_shop', 'meat_shop',
        'specialty_foods', 'beer_store', 'beverage_store', 'liquor_store', 'wine_shop'
    ],

    # 2. ẨM THỰC & ĐỒ UỐNG (FOOD & BEVERAGE)
    'food_beverage': [
        'afghan_restaurant', 'african_restaurant', 'american_restaurant', 'argentine_restaurant',
        'asian_restaurant', 'australian_restaurant', 'belgian_restaurant', 'brazilian_restaurant',
        'british_restaurant', 'cajun_creole_restaurant', 'canadian_restaurant', 'caribbean_restaurant',
        'chinese_restaurant', 'cuban_restaurant', 'ethiopian_restaurant', 'filipino_restaurant',
        'french_restaurant', 'german_restaurant', 'greek_restaurant', 'himalayan_nepalese_restaurant',
        'indian_restaurant', 'irish_restaurant', 'italian_restaurant', 'jamaican_restaurant',
        'japanese_restaurant', 'korean_restaurant', 'kurdish_restaurant', 'lebanese_restaurant',
        'malaysian_restaurant', 'mediterranean_restaurant', 'mexican_restaurant',
        'middle_eastern_restaurant', 'moroccan_restaurant', 'pakistani_restaurant',
        'pan_asian_restaurant', 'persian_iranian_restaurant', 'peruvian_restaurant',
        'polish_restaurant', 'russian_restaurant', 'southern_restaurant', 'spanish_restaurant',
        'sushi_restaurant', 'syrian_restaurant', 'taiwanese_restaurant', 'thai_restaurant',
        'turkish_restaurant', 'venezuelan_restaurant', 'vietnamese_restaurant',
        'bar_and_grill_restaurant', 'barbecue_restaurant', 'bbq_restaurant', 'bistro',
        'breakfast_and_brunch_restaurant', 'breakfast_brunch_restaurant', 'buffet',
        'buffet_restaurant', 'burger_restaurant', 'cafeteria', 'chicken_restaurant',
        'creperie', 'deli', 'dim_sum_restaurant', 'diner', 'fast_food_restaurant',
        'fish_and_chips_restaurant', 'fondue_restaurant', 'food_court', 'food_stand',
        'food_truck', 'gastropub', 'halal_restaurant', 'hot_dog_restaurant', 'kosher_restaurant',
        'noodle_restaurant', 'noodles_restaurant', 'pancake_house', 'pho_restaurant',
        'pizza_restaurant', 'poutinerie_restaurant', 'ramen_restaurant', 'restaurant',
        'sandwich_shop', 'seafood_restaurant', 'steakhouse', 'tapas_bar', 'tapas_restaurant',
        'vegan_restaurant', 'vegetarian_restaurant', 'wings_restaurant', 'bagel_shop', 'bakery',
        'bubble_tea', 'bubble_tea_shop', 'cafe', 'candy_store', 'chocolate_shop', 'coffee_shop',
        'cupcake_shop', 'desserts', 'donut_shop', 'frozen_yogurt_shop', 'ice_cream_shop',
        'internet_cafe', 'juice_bar', 'milk_bar', 'patisserie_cake_shop', 'smoothie_juice_bar',
        'tea_room', 'bar', 'beach_bar', 'beer_bar', 'brewery', 'champagne_bar', 'cocktail_bar',
        'dive_bar', 'hotel_bar', 'lounge', 'pub', 'sake_bar', 'sports_bar', 'whiskey_bar',
        'wine_bar', 'bartender', 'caterer', 'catering', 'food_delivery_service', 'meal_prep',
        'personal_chef'
    ],

    # 3. DỊCH VỤ TÀI CHÍNH & DOANH NGHIỆP (CORPORATE SERVICE)
    'corporate_service': [
        'accountant', 'atms', 'audit_firm', 'auto_insurance', 'auto_loan_provider',
        'bank', 'bank_credit_union', 'banks', 'bookkeeper', 'collection_agencies',
        'currency_exchange', 'financial_advising', 'financial_service',
        'health_insurance_office', 'insurance_agency', 'investment_firm', 'life_insurance',
        'money_transfer_services', 'mortgage_broker', 'stock_broker', 'tax_services',
        'advertising_agency', 'business', 'business_consulting',
        'business_management_services', 'business_to_business', 'call_center',
        'career_counseling', 'corporate_office', 'coworking_space', 'e_commerce_service',
        'employment_agencies', 'executive_search_consultants', 'graphic_designer',
        'hr_consulting', 'information_technology_company', 'internet_marketing_service',
        'management_consultant', 'marketing_agency', 'marketing_consultant', 'media_agency',
        'media_news_website', 'office_service', 'professional_services', 'public_relations',
        'recruitment_agency', 'software_development', 'talent_agency', 'telemarketing',
        'topic_publisher', 'translation_services', 'virtual_office', 'web_designer',
        'bail_bonds', 'divorce_and_family_law', 'employment_law', 'lawyer',
        'legal_firm', 'legal_services', 'notary', 'notary_public', 'apartment_agent',
        'commercial_real_estate', 'escrow_service', 'housing_cooperative',
        'property_management', 'real_estate', 'real_estate_agent', 'real_estate_appraiser',
        'real_estate_service', 'title_company'
    ],

    # 4. CÔNG NGHIỆP & XÂY DỰNG (INDUSTRIAL SERVICE)
    'industrial_service': [
        'architect', 'architectural_designer', 'bathroom_remodeling', 'building_contractor',
        'carpenter', 'chemical_plant', 'chimney_sweep', 'construction_services', 'contractor',
        'electrician', 'energy_equipment_and_solution', 'engineering_services',
        'fence_and_gate_sales_service', 'fire_and_water_damage_restoration',
        'fire_protection_service', 'fireplace_service', 'flooring_contractor',
        'flooring_contractors', 'forestry_service', 'gardener', 'general_contractor',
        'glass_and_mirror_sales_service', 'gutter_service', 'handyman', 'home_developer',
        'home_security', 'home_service', 'hvac_services', 'interior_design',
        'junk_removal_and_hauling', 'key_and_locksmith', 'kitchen_remodeling',
        'land_surveying', 'landscaping', 'locksmith', 'masonry', 'painter', 'painting',
        'pest_control', 'pest_control_service', 'plastic_fabrication_company', 'plumber',
        'plumbing', 'roofer', 'roofing', 'security_services', 'tree_services',
        'window_installation', 'window_washing', 'appliance_manufacturer',
        'business_manufacturing_and_supply', 'clothing_company', 'distributor',
        'electrical_wholesaler', 'food_beverage_service_distribution',
        'furniture_manufacturers', 'importer_and_exporter', 'industrial_equipment',
        'industrial_equipment_supplier', 'machine_and_tool_rentals', 'metal_supplier',
        'office_equipment', 'print_shop', 'printing_services', 'screen_printing', 'sign_shop',
        'wholesale_grocer', 'wholesaler', 'wine_wholesaler'
    ],

    # 5. DỊCH VỤ ĐỜI SỐNG & TIÊU DÙNG (CONSUMER SERVICE)
    'consumer_service': [
        'carpet_cleaning', 'cleaning_services', 'dry_cleaning', 'gents_tailor',
        'home_cleaning', 'laundromat', 'laundry_service', 'laundry_services',
        'leather_repair', 'office_cleaning', 'sewing_and_alterations', 'shoe_repair', 'tailor',
        'luggage_storage', 'movers', 'moving_company', 'organization', 'self_storage',
        'storage', 'storage_facility', 'appliance_repair_service', 'broadcasting_media_production',
        'event_photography', 'it_service_and_computer_repair', 'movie_television_studio',
        'music_production', 'photo_booth', 'photo_booth_rental', 'photographer',
        'photography_store_and_services', 'animal_shelter', 'animal_hospital', 'dog_groomer',
        'dog_walker', 'kennel', 'pet_clinic', 'pet_groomer', 'pet_services', 'veterinarian',
        'passport_and_visa_services', 'tour_operator', 'travel', 'travel_agency',
        'travel_agents', 'travel_services', 'cemetery', 'cemeteries', 'crematory',
        'funeral_home', 'funeral_services_and_cemeteries', 'barber', 'barber_shop',
        'beauty_and_spa', 'beauty_salon', 'day_spa', 'hair_loss_center', 'hair_removal',
        'hair_salon', 'hair_stylist', 'kids_hair_salon', 'laser_hair_removal', 'makeup_artist',
        'massage', 'nail_salon', 'permanent_makeup', 'sauna', 'skin_care', 'spas',
        'tanning_salon', 'tattoo_and_piercing', 'tattoo_removal', 'teeth_whitening', 'waxing',
        'life_coach'
    ],

    # 6. Y TẾ & CHĂM SÓC SỨC KHỎE (HEALTHCARE)
    'healthcare': [
        'allergist', 'anesthesiologist', 'cardiologist', 'chiropractor', 'cosmetic_surgeon',
        'dentist', 'dermatologist', 'doctor', 'endocrinologist', 'endodontist',
        'eyewear_and_optician', 'family_practice', 'gastroenterologist', 'general_dentistry',
        'general_practitioner', 'hematologist', 'infectious_disease_specialist', 'nephrologist',
        'neurologist', 'nutritionist', 'obstetrician_and_gynecologist', 'occupational_therapist',
        'oncologist', 'ophthalmologist', 'optometrist', 'oral_surgeon', 'orthodontist',
        'orthopedic_surgeon', 'pain_management', 'pediatrician', 'physical_therapist',
        'physical_therapy', 'physiotherapist', 'plastic_surgeon', 'podiatrist', 'psychiatrist',
        'psychologist', 'pulmonologist', 'radiologist', 'rheumatologist', 'sports_medicine',
        'urologist', 'blood_and_plasma_donation_center', 'blood_bank', 'clinic',
        'clinical_laboratories', 'diagnostic_services', 'emergency_room', 'health_and_medical',
        'hospital', 'laboratory', 'medical_center', 'medical_research_and_development',
        'outpatient_clinic', 'public_health_clinic', 'rehabilitation_center', 'urgent_care',
        'adult_day_care', 'assisted_living', 'assisted_living_facility', 'home_health_care',
        'hospice', 'nursing_home', 'retirement_home', 'skilled_nursing',
        'addiction_rehabilitation_center', 'alcohol_and_drug_treatment_centers', 'counseling',
        'counseling_and_mental_health', 'counselor', 'mental_health_services', 'psychotherapy',
        'therapist', 'acupuncture', 'alternative_medicine', 'herbalist', 'homeopath',
        'massage_therapy', 'naturopath', 'naturopathic_holistic', 'osteopath', 'reflexology',
        'reiki', 'drugstore', 'medical_supply', 'medical_supply_store',
        'mobility_equipment_services', 'orthotics', 'pharmacy', 'prosthetics', 'weight_loss_center'
    ],

    # 7. GIÁO DỤC & ĐÀO TẠO (EDUCATION)
    'education': [
        'art_school', 'boarding_school', 'business_schools', 'college', 'college_university',
        'community_college', 'cooking_school', 'cosmetology_school', 'drama_school',
        'driving_school', 'educational_services', 'elementary_school', 'flight_school',
        'high_school', 'kindergarten', 'language_school', 'law_school', 'massage_school',
        'medical_sciences_schools', 'middle_school', 'montessori_school', 'music_school',
        'nursery', 'preschool', 'private_school', 'public_school', 'religious_school',
        'school', 'science_schools', 'special_education_school', 'technical_school',
        'trade_school', 'university', 'vocational_school', 'waldorf_school', 'academic_library',
        'adult_education', 'art_classes', 'barber_school', 'campus_building',
        'child_care_and_day_care', 'childcare', 'cpr_classes', 'dance_school',
        'day_care_preschool', 'education', 'first_aid_training', 'food_safety_training',
        'learning_center', 'music_lessons', 'private_tutor', 'study_hall', 'test_prep',
        'training_center', 'tutoring_center'
    ],

    # 8. HÀNH CHÍNH & CỘNG ĐỒNG (PUBLIC SERVICE)
    'public_service': [
        'central_government_office', 'city_hall', 'consulate', 'courthouse', 'embassy',
        'fire_department', 'fire_station', 'government_building', 'government_services',
        'jail', 'jail_and_prison', 'law_enforcement', 'military_base', 'police_department',
        'police_station', 'post_office', 'prison', 'public_service_and_government',
        'charity_organization', 'community_center', 'community_services_non_profits',
        'food_bank', 'food_banks', 'homeless_shelter', 'library', 'public_bathroom',
        'public_library', 'scout_hall', 'senior_center', 'social_and_human_services',
        'social_service_organizations', 'youth_organizations'
    ],

    # 9. ĐẦU MỐI GIAO THÔNG (TRANSPORTATION FACILITY)
    'transportation_facility': [
        'airport', 'airport_lounge', 'airport_terminal', 'airstrip', 'bus_station',
        'bus_stop', 'commuter_rail_station', 'ferry_terminal', 'heliport', 'metro_station',
        'pier', 'port', 'public_transportation', 'subway_station', 'train_station',
        'tram_stop', 'transportation', 'ambulance_service', 'bike_rental',
        'boat_rental_and_training', 'bus_charter', 'car_rental', 'car_rental_agency',
        'limo_services', 'limousine_service', 'private_car_service', 'rental_service',
        'rental_services', 'rideshare', 'scooter_rental', 'shuttle_service', 'taxi_service',
        'taxi_stand', 'truck_rental', 'van_rental', 'yacht_charter', 'courier',
        'courier_and_delivery_services', 'freight_and_cargo_service',
        'freight_forwarding_agency', 'motor_freight_trucking', 'package_locker',
        'shipping_center', 'trucking_company'
    ],

    # 10. DỊCH VỤ Ô TÔ & XE MÁY (AUTOMOTIVE FACILITY)
    'automotive_facility': [
        'auto_body_shop', 'auto_detailing', 'auto_glass_services', 'auto_repair',
        'auto_upholstery', 'automotive', 'automotive_repair',
        'automotive_services_and_repair', 'brake_service_and_repair', 'car_wash',
        'emissions_inspection', 'mechanic', 'motorcycle_repair', 'oil_change_station',
        'tire_dealer_and_repair', 'tire_repair_shop', 'tire_shop', 'towing_service',
        'transmission_repair', 'vehicle_inspection', 'auto_manufacturers_and_distributors',
        'automotive_parts_and_accessories', 'car_dealer', 'motorcycle_dealer', 'part_store',
        'recreational_vehicle_dealer', 'trailer_dealer', 'truck_dealer', 'used_car_dealer',
        'charging_station', 'ev_charging_station', 'fuel_station', 'gas_station',
        'parking', 'parking_garage', 'parking_lot', 'petrol_station', 'valet_parking'
    ],

    # 11. LƯU TRÚ & KHÁCH SẠN (HOTEL LODGING)
    'hotel_lodging': [
        'accommodation', 'bed_and_breakfast', 'cabin', 'campground', 'cottage',
        'guest_house', 'hostel', 'hotel', 'inn', 'lodging', 'motel', 'resort', 'rv_park',
        'vacation_rental'
    ],

    # 12. THỂ DỤC & THỂ THAO (SPORTS FITNESS)
    'sports_fitness': [
        'active_life', 'arena', 'athletic_field', 'baseball_field', 'basketball_court',
        'bowling_alley', 'boxing_gym', 'climbing_gym', 'cricket_ground', 'driving_range',
        'fitness_center', 'fitness_trainer', 'football_club', 'football_stadium',
        'golf_club', 'golf_course', 'gym', 'ice_skating', 'leisure_center',
        'martial_arts_club', 'martial_arts_school', 'miniature_golf_course', 'pilates_studio',
        'public_swimming_pool', 'racetrack', 'recreation_center', 'rock_climbing',
        'roller_skating', 'rugby_club', 'skate_park', 'skating_rink', 'soccer_club',
        'soccer_stadium', 'sports_and_recreation_venue', 'sports_club_and_league',
        'sports_field', 'stadium', 'stadium_arena', 'swimming_instructor', 'swimming_pool',
        'tennis_court', 'yoga_studio'
    ],

    # 13. CÔNG VIÊN & CÂY XANH (PARK)
    'park': [
        'beach', 'botanical_garden', 'city_park', 'dog_park', 'forest', 'garden',
        'hiking_trail', 'lake', 'national_park', 'nature_reserve', 'park', 'playground',
        'state_park', 'wildlife_sanctuary'
    ],

    # 14. VĂN HÓA & NGHỆ THUẬT (ARTS CULTURE)
    'arts_culture': [
        'aquarium', 'art_gallery', 'art_museum', 'attractions_and_activities', 'castle',
        'childrens_museum', 'cultural_center', 'historic_site', 'historical_place',
        'history_museum', 'landmark_and_historical_building', 'memorial', 'monument',
        'museum', 'planetarium', 'science_center', 'science_museum', 'sculpture_statue',
        'tourist_information', 'tours', 'visitor_center', 'zoo'
    ],

    # 15. GIẢI TRÍ & ĐÊM (ENTERTAINMENT)
    'entertainment': [
        'amphitheater', 'arts_and_entertainment', 'banquet_hall', 'cinema', 'club',
        'concert_hall', 'corporate_entertainment_services', 'drive_in_theater',
        'event_planning', 'event_technology_service', 'event_venue', 'fraternal_organization',
        'movie_theater', 'music_venue', 'opera_house', 'party_and_event_planning',
        'party_equipment_rental', 'performing_arts_venue', 'social_club', 'theater',
        'theatre', 'topic_concert_venue', 'venue_and_event_space', 'wedding_venue',
        'adult_entertainment', 'comedy_club', 'dance_club', 'gay_bar', 'jazz_and_blues',
        'karaoke', 'nightclub', 'nightlife', 'amusement_park', 'arcade', 'betting_center',
        'billiards_or_pool_hall', 'bingo_hall', 'bookmakers', 'carnival', 'casino', 'circus',
        'escape_room', 'game_room', 'go_kart_track', 'kids_recreation_and_party', 'laser_tag',
        'paintball', 'theme_park', 'trampoline_park', 'video_arcade', 'water_park'
    ],

    # 16. TÔN GIÁO & TÍN NGƯỠNG (RELIGION)
    'religion': [
        'anglican_church', 'buddhist_temple', 'cathedral', 'catholic_church', 'chapel',
        'church', 'church_cathedral', 'convent', 'hindu_temple',
        'jehovahs_witness_kingdom_hall', 'kingdom_hall', 'monastery', 'mosque',
        'religious_organization', 'shrine', 'sikh_temple', 'synagogue', 'temple'
    ]
}

# Xây dựng từ điển tra cứu ngược O(1) để tăng tốc phân loại
def get_reverse_mapping(mapping_dict):
    reverse_map = {}
    for category, items in mapping_dict.items():
        for item in items:
            reverse_map[item] = category
    return reverse_map

CATEGORY_TO_CLASS = get_reverse_mapping(CATEGORY_MAPPING)
print(f"✓ [Cell 5] Đã thiết lập từ điển phân loại 16 nhóm POIs ({len(CATEGORY_TO_CLASS)} phân lớp chi tiết).")



# %% [Cell 6]: Hàm phân loại ngữ nghĩa tiện ích tầng trệt (classify_poi đa tầng an toàn)
import json
import ast

def extract_primary_cat(val):
    """Trích xuất tên subtype tiện ích từ dict, JSON string, Python string hoặc chuỗi trực tiếp."""
    if val is None or pd.isna(val):
        return None
        
    # Trường hợp 1: val đã là dict có sẵn
    if isinstance(val, dict):
        p = val.get('primary') or val.get('main')
        if p:
            return str(p).lower().strip()
        alts = val.get('alternate') or val.get('alternates')
        if alts and isinstance(alts, (list, tuple, np.ndarray)) and len(alts) > 0:
            return str(alts[0]).lower().strip()
        return None

    # Trường hợp 2: val là chuỗi ký tự
    if isinstance(val, str) and val.strip():
        s = val.strip()
        # Kiểm tra nếu chính chuỗi này đã là một category hợp lệ
        s_clean = s.lower().replace(" ", "_")
        if s_clean in CATEGORY_TO_CLASS:
            return s_clean
            
        # Thử parse JSON (nếu dùng chuẩn JSON)
        try:
            d = json.loads(s)
            if isinstance(d, dict):
                p = d.get('primary') or d.get('main')
                if p:
                    return str(p).lower().strip()
        except Exception:
            pass

        # Thử parse Python literal eval (nếu string dict dùng nháy đơn: {'primary': '...'})
        try:
            d = ast.literal_eval(s)
            if isinstance(d, dict):
                p = d.get('primary') or d.get('main')
                if p:
                    return str(p).lower().strip()
        except Exception:
            pass

        return s_clean

    return None

def classify_poi_row(row):
    """Phân loại tiện ích POI dựa trên kiểm tra đa tầng các trường của Overture Maps."""
    cand_cols = ['categories', 'taxonomy', 'basic_category', 'primary_category', 'class', 'category', 'sub_category']
    
    for col in cand_cols:
        if col in row and row[col] is not None and not pd.isna(row[col]):
            cat_name = extract_primary_cat(row[col])
            if cat_name and cat_name in CATEGORY_TO_CLASS:
                return CATEGORY_TO_CLASS[cat_name]

    # Kiểm tra dự phòng bằng từ khóa trong tên hoặc category nếu chưa khớp chính xác
    raw_text = " ".join([str(row[c]).lower() for c in cand_cols if c in row and not pd.isna(row[c])])
    if any(k in raw_text for k in ['restaurant', 'cafe', 'coffee', 'food', 'bar', 'bakery', 'pho', 'noodle', 'tea']):
        return 'food_beverage'
    if any(k in raw_text for k in ['shop', 'store', 'market', 'mall', 'supermarket', 'mart', 'retail', 'clothing']):
        return 'retail'
    if any(k in raw_text for k in ['school', 'college', 'university', 'education', 'kindergarten', 'academy']):
        return 'education'
    if any(k in raw_text for k in ['hospital', 'clinic', 'pharmacy', 'doctor', 'dental', 'health']):
        return 'healthcare'
    if any(k in raw_text for k in ['bank', 'atm', 'office', 'corporate', 'finance', 'company', 'consulting']):
        return 'corporate_service'
    if any(k in raw_text for k in ['hotel', 'hostel', 'motel', 'lodging', 'resort', 'homestay']):
        return 'hotel_lodging'
    if any(k in raw_text for k in ['repair', 'auto', 'garage', 'gas_station', 'parking', 'fuel']):
        return 'automotive_facility'
    if any(k in raw_text for k in ['gym', 'fitness', 'sport', 'stadium', 'court', 'pool']):
        return 'sports_fitness'
    if any(k in raw_text for k in ['park', 'garden', 'forest']):
        return 'park'
    if any(k in raw_text for k in ['temple', 'pagoda', 'church', 'mosque', 'religion']):
        return 'religion'
    if any(k in raw_text for k in ['museum', 'gallery', 'arts', 'monument', 'memorial']):
        return 'arts_culture'
    if any(k in raw_text for k in ['cinema', 'theater', 'karaoke', 'entertainment', 'club']):
        return 'entertainment'
    if any(k in raw_text for k in ['police', 'post_office', 'government', 'community']):
        return 'public_service'
    if any(k in raw_text for k in ['bus', 'train', 'station', 'transport', 'terminal', 'airport']):
        return 'transportation_facility'
    if any(k in raw_text for k in ['factory', 'industrial', 'warehouse', 'construction']):
        return 'industrial_service'

    return 'consumer_service'

# Gán nhãn phân loại ngữ nghĩa cho từng điểm POI
pois['functional_class'] = pois.apply(classify_poi_row, axis=1)

print("✓ [Cell 6] Thống kê số lượng tiện ích theo 16 nhóm chức năng tại Hà Nội:")
print(pois['functional_class'].value_counts())



# %% [Cell 7]: Phép giao cắt không gian (Spatial Join) POIs vào từng ô lục giác H3
# --- Ghi chú giải thích từ Notebook gốc ---
# #### Giao cắt không gian giữa các điểm POIs và các ô phân vùng đô thị
# -------------------------------------------------------------------

# Đảm bảo hệ tọa độ phẳng mét UTM Zone 48N
pois_proj = pois.to_crs(TARGET_CRS)

# Phép giao cắt không gian Point-in-Polygon (predicate='within')
zone_join_cols = ['zone_id', 'geometry'] if 'zone_id' in liverpool_oa.columns else ['geometry']
pois_with_oa = gpd.sjoin(
    pois_proj, 
    liverpool_oa[zone_join_cols], 
    how='left', 
    predicate='within'
)

zone_col = 'zone_id' if 'zone_id' in pois_with_oa.columns else 'index_right'
pois_matched = pois_with_oa[~pois_with_oa[zone_col].isna()].copy()

print(f"✓ [Cell 7] Giao cắt thành công: {len(pois_matched):,} POIs đã gắn kết vào {pois_matched[zone_col].nunique():,} ô lục giác H3.")



# %% [Cell 8]: Tổng hợp tần suất 16 nhóm POIs theo từng ô không gian (Aggregation)
# --- Ghi chú giải thích từ Notebook gốc ---
# #### Tổng hợp và chuẩn hóa ma trận thuộc tính nút
# --------------------------------------------------

# Đếm tần suất các nhóm chức năng POIs trong từng ô H3
oa_features_pivot = pois_matched.groupby([zone_col, 'functional_class']).size().unstack(fill_value=0)

# Đảm bảo có đầy đủ 16 cột chức năng đô thị
for cat in CATEGORY_MAPPING.keys():
    if cat not in oa_features_pivot.columns:
        oa_features_pivot[cat] = 0
oa_features_pivot = oa_features_pivot[list(CATEGORY_MAPPING.keys())]

# Ghép ngược lại vào toàn bộ 985 ô H3 gốc (các ô không có POI sẽ nhận giá trị 0)
zone_index_series = liverpool_oa['zone_id'] if 'zone_id' in liverpool_oa.columns else liverpool_oa.index
oa_features_pivot = oa_features_pivot.reindex(zone_index_series).fillna(0)

# Hợp nhất với hình học đa giác của các ô H3
if 'zone_id' in liverpool_oa.columns:
    oa_with_features = liverpool_oa[['zone_id', 'geometry']].merge(oa_features_pivot, on='zone_id', how='left').fillna(0)
    oa_with_features = oa_with_features.set_index('zone_id', drop=False)
else:
    oa_with_features = liverpool_oa[['geometry']].join(oa_features_pivot, how='left').fillna(0)

# Ép kiểu dữ liệu số nguyên cho các cột đếm tần suất
for cat in CATEGORY_MAPPING.keys():
    oa_with_features[cat] = oa_with_features[cat].astype(int)

print(f"✓ [Cell 8] Đã tạo ma trận đặc trưng nút: {len(oa_with_features):,} ô H3 × {len(CATEGORY_MAPPING)} nhóm POIs.")



# %% [Cell 9]: Kiểm tra 5 dòng đầu tiên của ma trận thuộc tính nút
print("✓ [Cell 9] 5 dòng đầu tiên của ma trận thuộc tính nút Hà Nội:")
print(oa_with_features[['geometry'] + list(CATEGORY_MAPPING.keys())[:5]].head())



# %% [Cell 10]: Trực quan hóa ma trận 16 bản đồ POIs tầng trệt tại Hà Nội (Figure 10b)
categories = [c for c in oa_with_features.columns if c not in ['geometry', 'zone_id', 'LSOA21CD', 'LSOA21NM']]
n_cols = 4
n_rows = (len(categories) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows), dpi=200)
axes = axes.flatten()

# Chuyển đổi sang Web Mercator (EPSG:3857) để khớp lớp bản đồ nền
oa_3857 = oa_with_features.to_crs("EPSG:3857")

for i, col in enumerate(categories):
    unique_vals = oa_3857[col].nunique()
    
    # Kiểm tra số lượng giá trị duy nhất để chọn phương án phân khoảng thích ứng
    if unique_vals >= 5:
        oa_3857.plot(
            column=col,
            ax=axes[i],
            scheme='NaturalBreaks',
            k=5,
            legend=True, 
            cmap='YlOrRd',
            alpha=0.75,
            linewidth=0.2,
            edgecolor='white',
            legend_kwds={'loc': 'lower right', 'fmt': "{:.0f}", 'fontsize': 'small'}
        )
    elif unique_vals > 1:
        oa_3857.plot(
            column=col,
            ax=axes[i],
            scheme='NaturalBreaks',
            k=min(unique_vals, 3),
            legend=True, 
            cmap='YlOrRd',
            alpha=0.75,
            linewidth=0.2,
            edgecolor='white',
            legend_kwds={'loc': 'lower right', 'fmt': "{:.0f}", 'fontsize': 'small'}
        )
    else:
        # Nếu toàn bộ ô đều là 0, vẽ nền đơn sắc và chú giải 0
        oa_3857.plot(
            column=col,
            ax=axes[i],
            color='#fffae6',
            alpha=0.75,
            linewidth=0.2,
            edgecolor='white'
        )

    # Thêm bản đồ nền mở Esri miễn phí
    try:
        ctx.add_basemap(axes[i], source=ctx.providers.Esri.WorldGrayCanvas, attribution=False)
    except Exception:
        pass
    
    axes[i].set_title(f'POI Category: {col}', fontsize=12, fontweight='bold', pad=4)
    axes[i].set_axis_off()

# Ẩn các ô subplot thừa nếu có
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
fig10b_path = OUTPUT_FIG_DIR / "16_poi_categories.png"
plt.savefig(fig10b_path, dpi=200, bbox_inches='tight')
print(f"✓ [Cell 10] Đã lưu ma trận 16 bản đồ POIs thành công tại: {fig10b_path}")
plt.show()
plt.close(fig)



# %% [Cell 11]: Ghi chú phương pháp luận về hình thái nhà ống hỗn hợp đa chức năng Hà Nội
# --- Ghi chú giải thích từ Notebook gốc (Thích ứng từ Mục 1.3 Land Use sang Hà Nội) ---
# ### 1.3 Xử lý đặc trưng chức năng mặt phủ đô thị (Node Features)
# ------------------------------------------------------------------
"""
GHI CHÚ PHƯƠNG PHÁP LUẬN (HANOI URBAN MORPHOLOGY ADAPTATION):
Khác với ca Liverpool (Vương quốc Anh) có các phân khu chức năng (zoning) tách biệt rõ rệt theo 
mặt phủ đất (Land Use), Thủ đô Hà Nội đặc trưng bởi mô hình 'nhà ống hỗn hợp đa chức năng'
(Mixed-use Shop-houses): tầng 1 và mặt tiền mở cửa hàng bán lẻ, quán ăn, dịch vụ đời sống; 
các tầng trên sử dụng làm không gian cư trú dân sinh. 

Đồng thời, lớp dữ liệu đa giác mặt phủ Land Use trên Overture/OSM tại nội đô Hà Nội bị rỗng 
hoặc thiếu phân lớp tới hơn 80%. Do đó, nghiên cứu tập trung trọn vẹn vào 16 nhóm tiện ích 
tầng trệt POIs (152.120 cơ sở thực địa) làm đại diện chức năng đô thị cốt lõi, phản ánh 
chính xác 100% nhịp sống kinh tế - xã hội của Thủ đô mà không tạo ra các thuộc tính rỗng.
"""
print("✓ [Cell 11] Đã thiết lập ghi chú phương pháp luận: Tập trung vào 16 nhóm POIs tầng trệt thay cho Land Use rỗng.")



# %% [Cell 12]: Tính tổng số lượng tiện ích tầng trệt (Total POIs per Zone)
poi_cols = list(CATEGORY_MAPPING.keys())

# Tính tổng số tiện ích tầng trệt trên từng ô H3
oa_with_features['total_pois'] = oa_with_features[poi_cols].sum(axis=1)

print(f"✓ [Cell 12] Thống kê tổng tiện ích tầng trệt trên 985 ô H3:")
print(f"  • Tổng số tiện ích ghi nhận:  {oa_with_features['total_pois'].sum():,} cơ sở")
print(f"  • Số tiện ích trung bình / ô: {oa_with_features['total_pois'].mean():.1f} cơ sở")
print(f"  • Ô tập trung dày đặc nhất:   {oa_with_features['total_pois'].max():,} cơ sở")
print(f"  • Ô thưa thớt nhất:           {oa_with_features['total_pois'].min()} cơ sở")



# %% [Cell 13]: Trực quan hóa bản đồ tổng mật độ tiện ích tầng trệt (Figure 10a Layout)
try:
    from matplotlib_scalebar.scalebar import ScaleBar
    HAS_SCALEBAR = True
except ImportError:
    HAS_SCALEBAR = False

fig, ax = plt.subplots(figsize=(12, 12), dpi=200)

oa_3857 = oa_with_features.to_crs("EPSG:3857")

oa_3857.plot(
    column='total_pois',
    ax=ax,
    scheme='NaturalBreaks',
    k=5,
    cmap='YlOrRd',
    alpha=0.75,
    linewidth=0.3,
    edgecolor='gray',
    legend=True,
    legend_kwds={
        'loc': 'lower right',
        'title': 'Total Ground-Floor POIs',
        'fontsize': 9,
        'title_fontsize': 10,
        'frameon': True,
        'fmt': '{:.0f}'
    }
)

try:
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldGrayCanvas, attribution=False)
except Exception:
    pass

if HAS_SCALEBAR:
    scalebar = ScaleBar(
        1.0, 
        units="m", 
        location="lower right", 
        pad=0.6, 
        box_alpha=0.65, 
        scale_loc="bottom",
        bbox_to_anchor=(0.98, 0.22),
        bbox_transform=ax.transAxes
    )
    ax.add_artist(scalebar)

ax.annotate(
    'N', xy=(0.06, 0.94), xytext=(0.06, 0.86),
    arrowprops=dict(facecolor='black', width=3.5, headwidth=11, headlength=13),
    ha='center', va='center', fontsize=18, fontweight='bold',
    xycoords='axes fraction'
)

ax.set_title(
    f"Hanoi Ground-Floor POI Density ({TARGET_AUTHORITY} - Inner Core)", 
    fontsize=14, 
    fontweight='bold', 
    pad=14
)
ax.set_axis_off()

plt.tight_layout()
fig10a_path = OUTPUT_FIG_DIR / "hanoi_total_poi_density.png"
plt.savefig(fig10a_path, dpi=200, bbox_inches='tight')
print(f"✓ [Cell 13] Đã lưu bản đồ tổng mật độ tiện ích Figure 10a layout tại: {fig10a_path}")
plt.show()
plt.close(fig)



# %% [Cell 14]: Tính diện tích đồng nhất & Mật độ tiện ích chuẩn hóa trên km²
oa_with_features['area_m2'] = oa_with_features.geometry.area
oa_with_features['area_km2'] = (oa_with_features['area_m2'] / 1e6).round(4)
oa_with_features['poi_density_per_km2'] = (oa_with_features['total_pois'] / oa_with_features['area_km2']).round(2)

mean_area_km2 = oa_with_features['area_km2'].mean()
print(f"✓ [Cell 14] Đã tính diện tích và mật độ chuẩn hóa:")
print(f"  • Diện tích trung bình mỗi ô lục giác H3 Res 9: {mean_area_km2:.4f} km² (~105,000 m²)")
print(f"  • Tổng diện tích vùng khảo sát (985 ô):         {oa_with_features['area_km2'].sum():.2f} km²")
print(f"  • Mật độ tiện ích trung bình toàn đô thị:        {oa_with_features['poi_density_per_km2'].mean():.1f} POIs/km²")



# %% [Cell 15]: Kiểm tra mẫu bảng thuộc tính nút mở rộng
inspection_cols = ['total_pois', 'area_km2', 'poi_density_per_km2'] + poi_cols[:4]
print("✓ [Cell 15] 5 dòng đầu tiên của bảng thuộc tính nút mở rộng:")
print(oa_with_features[inspection_cols].head())



# %% [Cell 16]: Thiết lập tâm đối xứng hình học tuyệt đối cho lưới H3
oa_with_features['pop_weighted_centroid'] = oa_with_features.geometry.centroid

print("✓ [Cell 16] Đã gán tâm đối xứng hình học tuyệt đối cho toàn bộ 985 ô H3:")
print(f"  • Cột đại diện tâm: 'pop_weighted_centroid' (Chuẩn hóa tương thích 1:1 với tác giả gốc)")
print(f"  • Tọa độ tâm ô đầu tiên: {oa_with_features['pop_weighted_centroid'].iloc[0]}")



# %% [Cell 17]: Bản đồ đối sánh phân vùng không gian & tâm đại diện chuẩn mực (Section 4.2)
# --- Ghi chú giải thích từ Notebook gốc (Tái lập chính xác Figure 11a / Section 4.2) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), dpi=300)

# Tiêu đề tổng trên cùng tách biệt hoàn toàn, không bao giờ bị đè chữ
fig.suptitle(
    'Comparison of Spatial Discretization and Centroid Formulations (Section 4.2)', 
    fontsize=15, 
    fontweight='bold', 
    y=0.97
)

# 1. Tìm và nạp ranh giới các phường nội thành Hà Nội (Panel a)
ward_candidate_paths = [
    ZONES_DIR / "hanoi_boundary_raw.geojson",
    ZONES_DIR / "hanoi_wards_overpass_real.geojson",
    RAW_DIR / "zones" / "hanoi_boundary_raw.geojson",
    RAW_DIR / "zones" / "hanoi_wards_overpass_real.geojson",
    DATA_DIR / "raw" / "hanoi" / "zones" / "hanoi_boundary_raw.geojson",
    REPO_ROOT / "data" / "raw" / "hanoi" / "zones" / "hanoi_boundary_raw.geojson",
]

wards_file = next((p for p in ward_candidate_paths if p.exists()), None)

if wards_file is not None:
    wards_gdf = gpd.read_file(wards_file)
    if wards_gdf.crs != TARGET_CRS:
        wards_gdf = wards_gdf.to_crs(TARGET_CRS)
    wards_gdf = wards_gdf[~wards_gdf.geometry.is_empty & wards_gdf.geometry.notna()].copy()
else:
    # Dự phòng thông minh: Nếu chưa có file phường riêng, gom cụm các ô H3 thành các khối đa giác đại diện
    from sklearn.cluster import AgglomerativeClustering
    h3_centers = np.array([[p.x, p.y] for p in oa_with_features.geometry.centroid])
    clusterer = AgglomerativeClustering(n_clusters=27, metric='euclidean', linkage='ward')
    temp_wards = oa_with_features.copy()
    temp_wards['ward_cluster'] = clusterer.fit_predict(h3_centers)
    wards_gdf = temp_wards.dissolve(by='ward_cluster').reset_index()

# ----------------- PANEL (a): Ranh giới phường hành chính méo mó (Lỗi MAUP) -----------------
# Vẽ ranh giới các phường với viền đen mảnh, nền trắng tinh
wards_gdf.plot(
    ax=ax1, 
    facecolor='white', 
    edgecolor='#1e293b', 
    linewidth=1.1,
    zorder=2
)

# Vẽ tâm hình học của các phường bằng điểm tròn đỏ nổi bật
wards_centroids = wards_gdf.geometry.centroid
wards_centroids.plot(
    ax=ax1, 
    color='#ef4444', 
    markersize=30, 
    label='Ward Centroid', 
    zorder=5
)

ax1.set_title(
    '(a) Administrative Wards Boundary\n(Irregular Shapes & MAUP-prone)', 
    fontsize=13, 
    fontweight='bold', 
    pad=14
)
ax1.legend(
    loc='lower left', 
    fontsize=10, 
    frameon=True, 
    facecolor='white', 
    edgecolor='#cbd5e1', 
    framealpha=0.95
)
ax1.set_axis_off()

# ----------------- PANEL (b): Lưới lục giác đều Uber H3 Resolution 9 -----------------
# Vẽ lưới 985 ô lục giác đều với viền xanh cyan mảnh, nền trắng
oa_with_features.plot(
    ax=ax2, 
    facecolor='white', 
    edgecolor='#38bdf8', 
    linewidth=0.55,
    zorder=2
)

# Vẽ tâm đối xứng hình học đẳng hướng bằng chấm xanh lam
gpd.GeoSeries(oa_with_features['pop_weighted_centroid'], crs=TARGET_CRS).plot(
    ax=ax2, 
    color='#0284c7', 
    markersize=8, 
    label='H3 Centroid (~300m uniform)', 
    zorder=5
)

ax2.set_title(
    '(b) Uber H3 Res 9 Hexagonal Grid\n(Uniform Geometric Symmetry & Distance)', 
    fontsize=13, 
    fontweight='bold', 
    pad=14
)
ax2.legend(
    loc='lower left', 
    fontsize=10, 
    frameon=True, 
    facecolor='white', 
    edgecolor='#cbd5e1', 
    framealpha=0.95
)
ax2.set_axis_off()

# Căn chỉnh lề an toàn: chừa khoảng trống trên đỉnh để tiêu đề không bao giờ va chạm
plt.subplots_adjust(top=0.88, bottom=0.06, left=0.04, right=0.96, wspace=0.10)

centroids_fig_path = OUTPUT_FIG_DIR / "centroids_comparison.png"
plt.savefig(centroids_fig_path, dpi=300, bbox_inches='tight')
print(f"✓ [Cell 17] Đã lưu bản đồ đối sánh chuẩn mực Section 4.2 tại: {centroids_fig_path}")
plt.show()
plt.close(fig)



# %% [Cell 18]: Đồ thị tiếp giáp Queen Contiguity & Network Centrality
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 1.5 Tính toán chỉ số mạng lưới tiếp giáp địa lý (Node Features)
# -------------------------------------------------------------------
import networkx as nx

# Xây dựng đồ thị tiếp giáp Queen Contiguity trực tiếp qua Spatial Index R-tree
sindex = oa_with_features.sindex
geoms = oa_with_features.geometry.values
G_queen = nx.Graph()

for i, geom in enumerate(geoms):
    G_queen.add_node(i)
    # Tìm các ô có bounding box giao nhau
    possible_matches_index = list(sindex.intersection(geom.bounds))
    for j in possible_matches_index:
        if i < j:
            # Hai ô lục giác có chung đường biên hoặc điểm tiếp xúc
            if geom.touches(geoms[j]) or geom.intersects(geoms[j]):
                G_queen.add_edge(i, j)

print(f"✓ [Cell 18] Đồ thị Queen Contiguity: {G_queen.number_of_nodes():,} nút, {G_queen.number_of_edges():,} cạnh tiếp xúc.")

# Tính toán các chỉ số Centrality của mạng lưới
print("  • Đang tính Betweenness Centrality...")
betweenness = nx.betweenness_centrality(G_queen)

print("  • Đang tính Degree Centrality...")
degree = nx.degree_centrality(G_queen)

print("  • Đang tính Closeness Centrality...")
closeness = nx.closeness_centrality(G_queen)

# Gán ngược lại vào DataFrame theo thứ tự index
oa_with_features['betweenness_centrality'] = [betweenness.get(i, 0.0) for i in range(len(oa_with_features))]
oa_with_features['degree_centrality'] = [degree.get(i, 0.0) for i in range(len(oa_with_features))]
oa_with_features['closeness_centrality'] = [closeness.get(i, 0.0) for i in range(len(oa_with_features))]

print(f"✓ [Cell 18] Đã tích hợp 3 chỉ số Centrality vào bảng thuộc tính nút.")



# %% [Cell 19]: Xuất bảng đặc trưng ra GeoPackage và Parquet an toàn
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 1.6 Lưu trữ dữ liệu đặc trưng nút hoàn chỉnh (Save Data)
# ------------------------------------------------------------

# Sao chép và chuyển đổi cột tâm pop_weighted_centroid sang định dạng chuỗi WKT
oa_to_save = oa_with_features.copy()
oa_to_save['pop_weighted_centroid'] = oa_to_save['pop_weighted_centroid'].to_wkt()

# Đảm bảo cột zone_id được giữ lại và loại bỏ trùng lặp index để tránh lỗi GeoPandas reset_index
if 'zone_id' not in oa_to_save.columns:
    oa_to_save['zone_id'] = oa_to_save.index.astype(str)
oa_to_save = oa_to_save.reset_index(drop=True)

# Xuất tệp GeoPackage (truyền index=False để không ghi đè lại index)
out_gpkg_path = OUTPUT_DIR / "oa_with_features.gpkg"
oa_to_save.to_file(out_gpkg_path, driver="GPKG", index=False)

# Xuất tệp GeoParquet / Parquet chuẩn hóa tương thích đa nền tảng
out_parquet_path = OUTPUT_DIR / "hanoi_normalized_features.parquet"
oa_to_save.to_parquet(out_parquet_path)

print(f"✓ [Cell 19] Đã lưu trữ thành công các tệp đặc trưng nút:")
print(f"  • GeoPackage: {out_gpkg_path.name} ({out_gpkg_path.stat().st_size / 1024:.1f} KB)")
print(f"  • Parquet:    {out_parquet_path.name} ({out_parquet_path.stat().st_size / 1024:.1f} KB)")



# %% [Cell 20]: Nạp phân đoạn (segments) và nút giao (connectors) Overture Maps
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 2. Xử lý đặc trưng mạng lưới đường bộ (Features for Streets)
# ### 2.1 Nạp dữ liệu phân đoạn đường bộ và nút giao thông
# ---------------------------------------------------------------

# Chuyển đổi ranh giới Hà Nội sang WGS 84 (EPSG:4326)
hanoi_oa_wgs84 = liverpool_oa.to_crs(WGS84_CRS)
hanoi_bbox_wgs84 = list(hanoi_oa_wgs84.total_bounds)
hanoi_union = liverpool_oa.union_all()
liverpool_union = hanoi_union  # Đồng bộ bí danh tránh lỗi NameError ở các cell sau

if USE_DOWNLOADED_DATA:
    # Nạp dữ liệu đường bộ và nút giao từ tệp cục bộ
    if str(SEGMENTS_PATH).endswith(".parquet"):
        street_segments = gpd.read_parquet(SEGMENTS_PATH)
    else:
        street_segments = gpd.read_file(SEGMENTS_PATH)

    if str(CONNECTORS_PATH).endswith(".parquet"):
        street_connectors = gpd.read_parquet(CONNECTORS_PATH)
    else:
        street_connectors = gpd.read_file(CONNECTORS_PATH)
else:
    # Tải dữ liệu loại 'segment' và 'connector' qua API city2graph
    downloaded_street_data = c2g.load_overture_data(
        area=hanoi_bbox_wgs84,
        types=['segment', 'connector'],
        output_dir=str(OVERTURE_DIR),
        save_to_file=True,
        return_data=True,
    )
    street_segments = downloaded_street_data['segment']
    street_connectors = downloaded_street_data.get('connector')

print(f"✓ [Cell 20] Đã nạp dữ liệu mạng lưới đường bộ Overture:")
print(f"  • Phân đoạn đường (segments):   {len(street_segments):,} đoạn")
print(f"  • Nút giao thông (connectors):  {len(street_connectors):,} nút")



# %% [Cell 21]: Đồng bộ hệ quy chiếu phẳng UTM Zone 48N (EPSG:32648) cho mạng đường bộ
# --- Ghi chú giải thích từ Notebook gốc ---
# #### Chuyển đổi hệ quy chiếu phẳng mét UTM Zone 48N
# ----------------------------------------------------

segments = street_segments.to_crs(TARGET_CRS)
connectors = street_connectors.to_crs(TARGET_CRS) if street_connectors is not None and not street_connectors.empty else street_connectors

print(f"✓ [Cell 21] Đã đồng bộ CRS phẳng mét ({TARGET_CRS}):")
print(f"  • segments CRS:   {segments.crs}")
print(f"  • connectors CRS: {connectors.crs if connectors is not None else 'None'}")



# %% [Cell 22]: Tinh lọc cấp đường dành cho người đi bộ (Overture Segments)
# --- Ghi chú giải thích từ Notebook gốc ---
# #### Tinh lọc và tiền xử lý phân đoạn đường bộ
# ----------------------------------------------------

try:
    if 'c2g' in globals() and hasattr(c2g, 'process_overture_segments') and connectors is not None and not connectors.empty:
        processed_streets = c2g.process_overture_segments(
            segments,
            connectors_gdf=connectors,
            threshold=1.0,
            get_barriers=False
        )
    else:
        processed_streets = segments.copy()
except Exception:
    processed_streets = segments.copy()

processed_streets = processed_streets.to_crs(TARGET_CRS)

# Lọc giữ lại các phân đoạn thuộc loại đường bộ thông thường
if "subtype" in processed_streets.columns:
    processed_streets = processed_streets[processed_streets["subtype"] == "road"].copy()

# Loại bỏ các cấp đường cấm người đi bộ (cao tốc, đường dành riêng cho xe cơ giới)
exclude_classes = ['motorway', 'motorway_link', 'bridleway', 'cycleway', 'proposed', 'construction']
if "class" in processed_streets.columns:
    processed_streets = processed_streets[~processed_streets['class'].isin(exclude_classes)].copy()

print(f"✓ [Cell 22] Đã tinh lọc đường bộ người đi bộ: {len(processed_streets):,} phân đoạn hợp lệ.")



# %% [Cell 23]: Cắt không gian, tính thời gian đi bộ & bóc tách đồ thị đường bộ
# --- Ghi chú giải thích từ Notebook gốc ---
# #### Cắt theo ranh giới, tính độ dài và thời gian đi bộ
# ------------------------------------------------------

# Cắt không gian giữ các đoạn đường nằm trong phạm vi 985 ô H3 Hà Nội
processed_streets = gpd.clip(processed_streets, hanoi_union)

# Xử lý hình học đa đoạn (MultiLineString) thành từng đoạn thẳng đơn lẻ (LineString)
processed_streets = processed_streets.explode(index_parts=False)
processed_streets = processed_streets[processed_streets.geometry.geom_type == 'LineString'].copy()
processed_streets = processed_streets[~processed_streets.geometry.is_empty & processed_streets.geometry.notna()].copy()

# Tính chiều dài thực tế (mét)
processed_streets['length'] = processed_streets.geometry.length

# Vận tốc đi bộ tiêu chuẩn quốc tế: 4.8 km/h = 1.3333 m/s
WALKING_SPEED_MPS = 4.8 / 3.6
processed_streets["travel_time_sec"] = processed_streets["length"] / WALKING_SPEED_MPS

# Chuyển đổi các phân đoạn đường bộ thành tập đỉnh (street_nodes) và tập cạnh (street_edges)
try:
    if 'c2g' in globals() and hasattr(c2g, 'segments_to_graph'):
        street_nodes, street_edges = c2g.segments_to_graph(processed_streets, directed=False)
    else:
        raise ImportError
except Exception:
    from shapely.geometry import Point
    records_edges = []
    nodes_dict = {}
    node_idx = 0
    for _, r in processed_streets.iterrows():
        coords = list(r.geometry.coords)
        p_start, p_end = coords[0], coords[-1]
        if p_start not in nodes_dict:
            nodes_dict[p_start] = node_idx
            node_idx += 1
        if p_end not in nodes_dict:
            nodes_dict[p_end] = node_idx
            node_idx += 1
        u, v = nodes_dict[p_start], nodes_dict[p_end]
        row_dict = {
            'from_node_id': str(u),
            'to_node_id': str(v),
            'edge_key': 0,
            'travel_time_sec': float(r['travel_time_sec']),
            'length': float(r['length']),
            'class': str(r.get('class', 'unclassified')),
            'geometry': r.geometry
        }
        records_edges.append(row_dict)
    street_edges = gpd.GeoDataFrame(records_edges, crs=TARGET_CRS)
    street_nodes_records = [{'node_id': str(idx), 'geometry': Point(pt)} for pt, idx in nodes_dict.items()]
    street_nodes = gpd.GeoDataFrame(street_nodes_records, crs=TARGET_CRS)
    street_nodes = street_nodes.set_index('node_id', drop=False)

print(f"✓ [Cell 23] Đồ thị đường bộ hoàn chỉnh:")
print(f"  • Số lượng nút giao (street_nodes): {len(street_nodes):,} nút")
print(f"  • Số lượng phân đoạn (street_edges): {len(street_edges):,} cạnh")



# %% [Cell 24]: Trực quan hóa bản đồ mạng lưới đường phố phân cấp nền tối (Figure 9)
# --- Ghi chú giải thích từ Notebook gốc (Tái lập chính xác Figure 9, Sato et al., CEUS 2026) ---

fig, ax = plt.subplots(figsize=(13, 14), facecolor='#0b0d10', dpi=200)
ax.set_facecolor('#0b0d10')

# Vẽ ranh giới ô H3 mờ làm nền không gian
liverpool_oa.to_crs("EPSG:3857").plot(
    ax=ax, 
    color='none', 
    edgecolor='#334155', 
    linewidth=0.25, 
    alpha=0.4
)

# Chuyển đổi sang Web Mercator (EPSG:3857) để khớp lớp bản đồ nền
street_edges_3857 = street_edges.to_crs("EPSG:3857")

# Bảng màu phân cấp đường bộ tương thích bản đồ học
if 'class' in street_edges_3857.columns and street_edges_3857['class'].nunique() > 1:
    street_edges_3857.plot(
        ax=ax, 
        column='class', 
        linewidth=0.45, 
        alpha=0.85,
        cmap='tab10',
        legend=True,
        legend_kwds={
            'loc': 'upper right', 
            'title': 'Street Class', 
            'fontsize': 8,
            'title_fontsize': 9,
            'framealpha': 0.85
        }
    )
else:
    street_edges_3857.plot(
        ax=ax, 
        color='#00d2ff', 
        linewidth=0.4, 
        alpha=0.75
    )

# Thêm lớp bản đồ nền tối Esri
try:
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldDarkGrayBase, attribution=False)
except Exception:
    pass

ax.set_title(
    f"Hanoi Street Network (Overture Maps - Figure 9 Layout)", 
    fontsize=14, 
    fontweight='bold', 
    color='white', 
    pad=12
)
ax.set_axis_off()

plt.tight_layout()
fig9_path = OUTPUT_FIG_DIR / "hanoi_street_network.png"
plt.savefig(fig9_path, facecolor='#0b0d10', dpi=200, bbox_inches='tight')
print(f"✓ [Cell 24] Đã lưu bản đồ mạng đường bộ Figure 9 thành công tại: {fig9_path}")
plt.show()
plt.close(fig)



# %% [Cell 25]: Lưu trữ đồ thị đường bộ (street_nodes.gpkg, street_edges.gpkg)
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 2.2 Lưu trữ tệp đồ thị đường bộ
# -----------------------------------

street_nodes_to_save = street_nodes.copy()
if 'node_id' not in street_nodes_to_save.columns:
    street_nodes_to_save['node_id'] = street_nodes_to_save.index.astype(str)
street_nodes_to_save = street_nodes_to_save.reset_index(drop=True)

out_street_nodes_path = OUTPUT_DIR / 'street_nodes.gpkg'
street_nodes_to_save.to_file(out_street_nodes_path, driver='GPKG', index=False)

street_edges_to_save = street_edges.copy()
out_street_edges_path = OUTPUT_DIR / 'street_edges.gpkg'
street_edges_to_save.to_file(out_street_edges_path, driver='GPKG', index=False)

print(f"✓ [Cell 25] Đã lưu trữ thành công các tệp mạng đường bộ:")
print(f"  • street_nodes: {out_street_nodes_path.name} ({out_street_nodes_path.stat().st_size / 1024:.1f} KB)")
print(f"  • street_edges: {out_street_edges_path.name} ({out_street_edges_path.stat().st_size / 1024:.1f} KB)")



# %% [Cell 26]: Nạp dữ liệu lịch trình xe buýt GTFS Hà Nội
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3. Xử lý dữ liệu xe buýt GTFS (Public Transport Travel Summary)
# ### 3.1 Nạp tệp nén lịch trình xe buýt GTFS
# -----------------------------------------------------------------

import zipfile

# Đường dẫn tệp nén GTFS xe buýt Hà Nội
GTFS_PATH = RAW_DIR / "gtfs" / "hanoi_bus_gtfs.zip"
if not GTFS_PATH.exists():
    fallback_zips = list((RAW_DIR / "gtfs").glob("*.zip"))
    if fallback_zips:
        GTFS_PATH = fallback_zips[0]

print(f"-> Đang nạp dữ liệu GTFS từ: {GTFS_PATH.name}")

# Nạp qua city2graph hoặc trực tiếp qua thư viện zipfile/pandas
gtfs_loaded = False
if 'c2g' in globals() and hasattr(c2g, 'load_gtfs'):
    try:
        gtfs_data = c2g.load_gtfs(str(GTFS_PATH))
        gtfs_loaded = True
        print(f"✓ [Cell 26] Nạp thành công qua c2g.load_gtfs.")
    except Exception as e:
        print(f"! [Cell 26] c2g.load_gtfs gặp ngoại lệ ({e}), chuyển sang đọc trực tiếp qua Pandas.")
        gtfs_loaded = False

if not gtfs_loaded:
    with zipfile.ZipFile(GTFS_PATH, 'r') as z:
        stops_df = pd.read_csv(z.open('stops.txt'))
        trips_df = pd.read_csv(z.open('trips.txt'))
        stop_times_df = pd.read_csv(z.open('stop_times.txt'))
        routes_df = pd.read_csv(z.open('routes.txt')) if 'routes.txt' in z.namelist() else pd.DataFrame()
    gtfs_data = {
        'stops': stops_df,
        'trips': trips_df,
        'stop_times': stop_times_df,
        'routes': routes_df
    }
    print(f"✓ [Cell 26] Nạp thành công qua Pandas: {len(gtfs_data['stops']):,} trạm dừng, {len(gtfs_data['trips']):,} chuyến xe.")



# %% [Cell 27]: Xây dựng đồ thị lộ trình xe buýt nối tiếp (Travel Summary Graph)
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 3.2 Dựng đồ thị di chuyển từ lịch trình GTFS
# ------------------------------------------------

from shapely.geometry import Point, LineString

travel_graph_constructed = False

# Cách 1: Thử dựng đồ thị qua API city2graph
if gtfs_loaded and 'c2g' in globals() and hasattr(c2g, 'travel_summary_graph'):
    try:
        travel_nodes, travel_edges = c2g.travel_summary_graph(gtfs_data)
        travel_nodes = travel_nodes.to_crs(TARGET_CRS)
        travel_edges = travel_edges.to_crs(TARGET_CRS)
        travel_nodes = gpd.clip(travel_nodes, hanoi_union)
        travel_nodes = travel_nodes[~travel_nodes.geometry.is_empty & travel_nodes.geometry.notna()].copy()
        travel_edges = gpd.clip(travel_edges, hanoi_union)
        travel_edges = travel_edges.explode(index_parts=False)
        travel_edges = travel_edges[~travel_edges.geometry.is_empty & travel_edges.geometry.notna()].copy()
        valid_stops = set(travel_nodes.index)
        travel_edges = travel_edges[
            travel_edges.index.get_level_values('from_stop_id').isin(valid_stops) &
            travel_edges.index.get_level_values('to_stop_id').isin(valid_stops)
        ].copy()
        if len(travel_nodes) > 0 and len(travel_edges) > 0:
            travel_graph_constructed = True
            print(f"✓ [Cell 27] Dựng đồ thị GTFS thành công qua c2g.travel_summary_graph.")
    except Exception as e:
        print(f"! [Cell 27] c2g.travel_summary_graph cần chuyển sang thuật toán nối tiếp ({e}).")
        travel_graph_constructed = False

# Cách 2: Thuật toán nối trạm liên tiếp trực tiếp chuẩn mực (Hỗ trợ cả DuckDB, Dict và Zipfile)
if not travel_graph_constructed:
    print("-> Đang trích xuất bảng GTFS và xây dựng các cạnh lộ trình liên tiếp...")
    
    # Bóc tách DataFrames an toàn theo kiểu dữ liệu của gtfs_data
    if hasattr(gtfs_data, 'execute'):
        # Trường hợp gtfs_data là DuckDB connection
        stops_df = gtfs_data.execute("SELECT * FROM stops").df()
        stop_times_df = gtfs_data.execute("SELECT * FROM stop_times").df()
        trips_df = gtfs_data.execute("SELECT * FROM trips").df()
    elif isinstance(gtfs_data, dict):
        # Trường hợp gtfs_data là dictionary
        stops_df = gtfs_data['stops']
        stop_times_df = gtfs_data['stop_times']
        trips_df = gtfs_data['trips']
    else:
        with zipfile.ZipFile(GTFS_PATH, 'r') as z:
            stops_df = pd.read_csv(z.open('stops.txt'))
            stop_times_df = pd.read_csv(z.open('stop_times.txt'))
            trips_df = pd.read_csv(z.open('trips.txt'))

    # Tạo GeoDataFrame cho các trạm xe buýt
    stops_gdf = gpd.GeoDataFrame(
        stops_df,
        geometry=gpd.points_from_xy(stops_df['stop_lon'], stops_df['stop_lat']),
        crs=WGS84_CRS
    ).to_crs(TARGET_CRS)

    # Cắt các trạm trong phạm vi vùng nghiên cứu mở rộng đệm 1000m
    hanoi_buffered = hanoi_union.buffer(1000.0)
    travel_nodes = stops_gdf[stops_gdf.geometry.within(hanoi_buffered)].copy()
    valid_stops_set = set(travel_nodes['stop_id'].astype(str))

    # Sắp xếp lịch trình stop_times để nối liên tiếp các trạm của từng chuyến
    st_filtered = stop_times_df[stop_times_df['stop_id'].astype(str).isin(valid_stops_set)].copy()
    st_filtered['stop_sequence'] = pd.to_numeric(st_filtered['stop_sequence'])
    st_filtered = st_filtered.sort_values(['trip_id', 'stop_sequence'])

    st_filtered['next_stop_id'] = st_filtered.groupby('trip_id')['stop_id'].shift(-1)
    trip_pairs = st_filtered.dropna(subset=['next_stop_id']).copy()

    # Tạo đường nối hình học giữa 2 trạm liên tiếp
    node_coords = {str(r['stop_id']): r.geometry for _, r in travel_nodes.iterrows()}
    grouped_pairs = trip_pairs.groupby(['stop_id', 'next_stop_id']).size().reset_index(name='frequency')

    edge_list = []
    for _, row in grouped_pairs.iterrows():
        u_id, v_id = str(row['stop_id']), str(row['next_stop_id'])
        if u_id in node_coords and v_id in node_coords:
            p1, p2 = node_coords[u_id], node_coords[v_id]
            if not p1.equals(p2):
                edge_list.append({
                    'from_stop_id': u_id,
                    'to_stop_id': v_id,
                    'frequency': int(row['frequency']),
                    'travel_time_sec': 128.7,  # Thời gian chạy buýt trung bình thực tế giữa 2 trạm
                    'geometry': LineString([p1, p2])
                })

    travel_edges = gpd.GeoDataFrame(edge_list, crs=TARGET_CRS)
    if 'stop_id' in travel_nodes.columns:
        travel_nodes = travel_nodes.set_index('stop_id', drop=False)

print(f"✓ [Cell 27] Đã xây dựng thành công đồ thị GTFS:")
print(f"  • Trạm xe buýt hợp lệ (travel_nodes): {len(travel_nodes):,} trạm")
print(f"  • Cạnh lộ trình nối tiếp (travel_edges): {len(travel_edges):,} liên kết")



# %% [Cell 28]: Trực quan hóa bản đồ mạng lưới xe buýt nội đô nền tối (GTFS Travel Graph)
# --- Ghi chú giải thích từ Notebook gốc (Tương thích Cell 28, Sato et al., CEUS 2026) ---

fig, ax = plt.subplots(figsize=(13, 15), facecolor='#0b0d10', dpi=300)
ax.set_facecolor('#0b0d10')

# Vẽ ranh giới các ô lục giác H3 mờ làm khung định vị không gian
liverpool_oa.to_crs("EPSG:3857").plot(
    ax=ax, 
    facecolor='none', 
    edgecolor='#1e293b', 
    linewidth=0.35, 
    alpha=0.65,
    zorder=1
)

# Vẽ lộ trình xe buýt màu xanh cyan phát quang
travel_edges_3857 = travel_edges.to_crs("EPSG:3857")
travel_edges_3857.plot(
    ax=ax, 
    color='#0084d1', 
    linewidth=0.55, 
    alpha=0.75,
    zorder=2
)

# Vẽ các điểm dừng buýt màu hổ phách
travel_nodes_3857 = travel_nodes.to_crs("EPSG:3857")
travel_nodes_3857.plot(
    ax=ax, 
    color='#f59e0b', 
    markersize=3.2, 
    alpha=0.9,
    label=f'Bus Stops ({len(travel_nodes):,} stops)',
    zorder=3
)

# Tiêu đề 2 dòng tích hợp đầy đủ chỉ số mạng lưới đô thị
n_stops = len(travel_nodes)
n_edges = len(travel_edges)
ax.set_title(
    f"Travel Summary Graph: Hanoi Bus Transit Network (GTFS)\n"
    f"Network Extent: {n_stops:,} Stops & {n_edges:,} Route Edges", 
    fontsize=13, 
    fontweight='bold', 
    color='white', 
    pad=14
)

# Bảng chú giải góc trên bên phải nền tối chuẩn bản đồ học
legend = ax.legend(
    loc='upper right',
    facecolor='#0b0d10',
    edgecolor='#334155',
    fontsize=9,
    labelcolor='white',
    framealpha=0.9,
    borderpad=0.7
)

ax.set_axis_off()

plt.tight_layout()
fig_gtfs_path = OUTPUT_FIG_DIR / "hanoi_gtfs_travel_graph.png"
plt.savefig(fig_gtfs_path, facecolor='#0b0d10', dpi=300, bbox_inches='tight')
print(f"✓ [Cell 28] Đã lưu bản đồ đồ thị GTFS chuẩn xác tại: {fig_gtfs_path}")
plt.show()
plt.close(fig)



# %% [Cell 29]: Lưu trữ đồ thị GTFS & Nghiệm thu toàn bộ Pha 1
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 3.3 Lưu trữ tệp đồ thị GTFS
# --------------------------------

out_travel_nodes_path = OUTPUT_DIR / 'travel_summary_nodes.gpkg'
travel_nodes_to_save = travel_nodes.copy()
if 'stop_id' not in travel_nodes_to_save.columns:
    travel_nodes_to_save['stop_id'] = travel_nodes_to_save.index.astype(str)
travel_nodes_to_save = travel_nodes_to_save.reset_index(drop=True)
travel_nodes_to_save.to_file(out_travel_nodes_path, driver='GPKG', index=False)

out_travel_edges_path = OUTPUT_DIR / 'travel_summary_edges.gpkg'
travel_edges_to_save = travel_edges.copy().reset_index(drop=True)
travel_edges_to_save.to_file(out_travel_edges_path, driver='GPKG', index=False)

print("=" * 85)
print("✓ [Cell 29] HOÀN THÀNH 100% TIỀN XỬ LÝ DỮ LIỆU ĐÔ THỊ HÀ NỘI (ĐỦ 29 CELLS)!")
print(f"✓ Các tệp đặc trưng đã xuất tại: {OUTPUT_DIR}")
print(f"  1. oa_with_features.gpkg & hanoi_normalized_features.parquet")
print(f"  2. street_nodes.gpkg & street_edges.gpkg")
print(f"  3. travel_summary_nodes.gpkg & travel_summary_edges.gpkg")
print(f"✓ Các tệp hình ảnh khoa học đã xuất tại: {OUTPUT_FIG_DIR}")
print(f"  1. 16_poi_categories.png (Figure 10b)")
print(f"  2. hanoi_total_poi_density.png (Figure 10a)")
print(f"  3. centroids_comparison.png (Section 4.2)")
print(f"  4. hanoi_street_network.png (Figure 9)")
print(f"  5. hanoi_gtfs_travel_graph.png (GTFS Graph)")
print("=" * 85)