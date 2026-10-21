# GIAI ĐOẠN 2

## TỔNG QUAN LÝ THUYẾT VÀ TÀI LIỆU DỮ LIỆU

**Tên dự án:** City2Graph-Hanoi  
**Chủ đề:** *Literature Review, Baseline Specification & Data Dictionary*  
**Đơn vị phân tích:** 985 ô H3 Resolution 9 trong 27 phường nội đô Hà Nội

---

## 1. Mục đích và phạm vi tài liệu

Tài liệu này chuẩn hóa hai lớp thông tin nền tảng của dự án:

1. Cơ sở lý thuyết, mô hình đối chuẩn và thiết kế thí nghiệm dùng để diễn giải kết quả.
2. Từ điển dữ liệu, schema manifest và quy ước biến đổi dữ liệu từ nguồn thô đến đồ thị học sâu.

Tài liệu tập trung vào cách triển khai trong City2Graph-Hanoi. Các mô hình được trình bày ở mức đủ để kiểm tra giả định, tái lập pipeline và giải thích trong báo cáo tốt nghiệp; không xem đây là bản thay thế cho đặc tả toán học đầy đủ của bài báo gốc.

### 1.1. Tài liệu nền

Dự án kế thừa và mở rộng hướng tiếp cận City2Graph của Sato et al., *Computers, Environment and Urban Systems*, Vol. 130, Article 102492 (2026), trên địa bàn Hà Nội. Trong dự án này, đối tượng học là các ô không gian H3 và các quan hệ đô thị giữa chúng, thay vì các nút giao thông hoặc POI được xem xét tách rời.

### 1.2. Ký hiệu chung

| Ký hiệu | Ý nghĩa |
|---|---|
| $V$ | Tập 985 nút không gian/ô H3 |
| $E_r$ | Tập cạnh của quan hệ $r$ |
| $X \in \mathbb{R}^{985\times16}$ | Ma trận đặc trưng POI đã chuẩn hóa |
| $Z \in \mathbb{R}^{985\times32}$ | Ma trận embedding đầu ra |
| $r$ | Một loại quan hệ đồ thị: `contig`, `15_min_walk`, `15_min_multi` |
| $\boldsymbol{\beta}$ | Trọng số semantic attention giữa các quan hệ |

---

## 2. Tổng quan lý thuyết và khung đối chuẩn

### 2.1. Graph Autoencoder và học biểu diễn tự giám sát

Graph Autoencoder (GAE) học một ánh xạ từ đặc trưng và cấu trúc đồ thị sang không gian ẩn có kích thước thấp. Encoder tạo embedding $Z$; decoder cố gắng tái tạo một phần hoặc toàn bộ thuộc tính đầu vào và liên kết đồ thị. Trong City2Graph-Hanoi, đây là học tự giám sát: không cần nhãn vùng chức năng được gán thủ công.

Hàm mất mát triển khai gồm hai thành phần:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{feat}}+\lambda_{\mathrm{struct}}\mathcal{L}_{\mathrm{struct}}.
$$

Trong đó, $\mathcal{L}_{\mathrm{feat}}$ là lỗi tái tạo thuộc tính bằng Smooth L1 khi mô hình có sử dụng POI; $\mathcal{L}_{\mathrm{struct}}$ là lỗi tái tạo liên kết bằng DistMult và binary cross-entropy với negative sampling.

### 2.2. GAT-GAE: mô hình đồ thị đồng nhất

GAT-GAE sử dụng Graph Attention Network trên đồ thị đồng nhất. Mỗi nút tổng hợp thông tin từ các nút lân cận bằng trọng số attention học được. Với một head attention, dạng khái quát là:

$$
\mathbf{h}'_i=\sigma\left(\sum_{j\in\mathcal{N}(i)}\alpha_{ij}W\mathbf{h}_j\right),
$$

trong đó $\alpha_{ij}$ thể hiện mức đóng góp tương đối của nút lân cận $j$ đối với nút $i$. Mô hình dùng hai lớp GAT:

- lớp thứ nhất học biểu diễn ẩn kích thước 64 với 4 attention heads;
- lớp thứ hai nén thành embedding 32 chiều;
- decoder thuộc tính là MLP;
- decoder cấu trúc là DistMult cho quan hệ `contig`.

GAT-GAE là đối chuẩn đồ thị đầu tiên để đo phần cải thiện khi bổ sung nhiều loại quan hệ đô thị.

### 2.3. HAN-GAE: mô hình đồ thị dị thể

HAN-GAE (Heterogeneous Attention Network Graph Autoencoder) xử lý nhiều loại quan hệ bằng hai tầng attention.

#### Node-level attention

Mỗi quan hệ hoặc metapath được xử lý bởi một nhánh GAT riêng. Nhánh `contig`, `15_min_walk` và `15_min_multi` vì vậy có thể học các kiểu lan truyền thông tin khác nhau.

#### Semantic-level attention

Các embedding theo quan hệ được xếp chồng và tổng hợp bằng trọng số ngữ nghĩa:

$$
\mathbf{z}_i=\sum_{r\in\mathcal{R}}\beta_{i,r}\mathbf{z}_{i,r},
\qquad
\sum_{r\in\mathcal{R}}\beta_{i,r}=1,
$$

với $\boldsymbol{\beta}_i$ được tạo bởi softmax trên điểm tương thích giữa biểu diễn quan hệ và một vector attention học được. Vì $\beta$ phụ thuộc vào nút, mô hình có thể sử dụng quan hệ đi bộ ở khu vực này và quan hệ xe buýt hoặc tiếp giáp ở khu vực khác.

Trong artefact của dự án, `beta1_*` và `beta2_*` lần lượt là trọng số semantic attention ở hai lớp HAN. Các trọng số này phục vụ diễn giải mô hình, không phải là xác suất giao thông hoặc quan hệ nhân quả.

### 2.4. Đối chuẩn PCA

PCA là baseline phi đồ thị. Mười sáu đặc trưng POI được chiếu tuyến tính sang không gian có số chiều không vượt quá 16, sau đó chuẩn hóa L2 trước khi phân cụm. PCA trả lời câu hỏi:

> Nếu chỉ dùng cấu trúc tương quan của các nhóm POI, không dùng topology hoặc khả năng tiếp cận, chất lượng phân vùng đạt mức nào?

PCA không có encoder đồ thị, attention hoặc decoder cấu trúc.

### 2.5. Các mô hình triệt tiêu cấu trúc (Structure-only)

Models 1S-4S giữ nguyên cấu hình quan hệ tương ứng của Models 1-4 nhưng loại bỏ tái tạo thuộc tính POI (`use_features=False`). Mô hình sử dụng node identity embedding học được và chỉ tối ưu cấu trúc liên kết.

| Mô hình | Quan hệ | Thành phần bị loại bỏ |
|---|---|---|
| Model 1S | `contig` | Attribute decoder và feature loss |
| Model 2S | `contig` + `15_min_walk` | Attribute decoder và feature loss |
| Model 3S | `contig` + `15_min_multi` | Attribute decoder và feature loss |
| Model 4S | cả ba quan hệ | Attribute decoder và feature loss |

Structure-only không phải mô hình triển khai chính. Đây là ablation để kiểm tra liệu topology đơn thuần có đủ để tạo ra vùng chức năng ổn định hay không. Việc một số mô hình 1S-4S có Silhouette thấp hoặc nhiều điểm HDBSCAN bị gán noise là kết quả nghiên cứu cần được diễn giải, không tự động là lỗi phần mềm.

### 2.6. Đặc tả thực nghiệm chuẩn

| Tham số | Giá trị thực tế |
|---|---:|
| Learning rate | 0.005 |
| Epoch tối đa | 200 |
| Số lớp GNN | 2 |
| Hidden dimension | 64 |
| Embedding dimension | 32 |
| Attention heads | 4 |
| Dropout | 0.6 |
| Weight decay | 0.0001 |
| Seed chính | 42 |
| K-Means range | $k=2..50$ |
| K-Means `n_init` | 10 |
| HDBSCAN `min_cluster_size` | 5..30 |

---

## 3. Data Dictionary / Schema Manifest

### 3.1. Quy ước dữ liệu chung

| Thuộc tính | Quy ước |
|---|---|
| Định danh ô | `zone_id`/H3 cell identifier, duy trì nhất quán giữa GeoDataFrame và PyG |
| Đơn vị không gian | Ô lục giác H3 Resolution 9 |
| Số nút kỳ vọng | 985 |
| CRS phân tích | EPSG:32648, WGS 84 / UTM zone 48N |
| Hệ tọa độ nguồn | Có thể là WGS84 hoặc CRS gốc của từng lớp; phải chuyển về EPSG:32648 trước phép đo mét |
| Đơn vị khoảng cách | mét |
| Đơn vị thời gian cạnh | giây |
| Giá trị thiếu | xử lý trong pha tiền xử lý; các cột đặc trưng đầu ra phải là số |
| Mã noise HDBSCAN | `-1` |

### 3.2. Dữ liệu nguồn sơ cấp

| Dataset | Đường dẫn chuẩn | Định dạng | Nội dung | Vai trò |
|---|---|---|---|---|
| H3 zones | `data/raw/hanoi/zones/hanoi_zones.parquet` | Parquet | ranh giới và mã 985 ô H3 | không gian nghiên cứu |
| Phường nội đô | `data/raw/hanoi/zones/hanoi_wards_overpass_real.geojson` | GeoJSON | ranh giới hành chính tham chiếu | xác định phạm vi 27 phường |
| Overture POIs | `data/raw/hanoi/overture/hanoi_pois.parquet` | Parquet | khoảng 152.000 POI và phân loại nguồn | đặc trưng tiện ích |
| Đoạn đường | `data/raw/hanoi/overture/hanoi_segments.parquet` | Parquet | các đoạn mạng đường | tính kết nối và thời gian đi bộ |
| Connector đường | `data/raw/hanoi/overture/hanoi_connectors.parquet` | Parquet | liên kết điểm/nút với mạng đường | nối POI/ô H3 vào mạng |
| GTFS xe buýt | `data/raw/hanoi/gtfs/hanoi_bus_gtfs.zip` | GTFS ZIP | tuyến, điểm dừng, chuyến và lịch trình | quan hệ đa phương thức |

### 3.3. Dữ liệu đã xử lý

| Dataset | Đường dẫn chuẩn | Schema/ý nghĩa |
|---|---|---|
| Features GeoPackage | `data/processed/hanoi/features/oa_with_features.gpkg` | mỗi dòng là một ô H3, geometry đa giác và các cột đặc trưng |
| Features Parquet | `data/processed/hanoi/features/hanoi_normalized_features.parquet` | phiên bản bảng của đặc trưng đã chuẩn hóa |
| Scaling parameters | `data/processed/hanoi/graphs/oa_feature_scaling.json` | $\mu$ và $\sigma$ dùng trong z-score |
| Street nodes/edges | `data/processed/hanoi/features/street_nodes.gpkg`, `street_edges.gpkg` | nút/đoạn đường và thuộc tính thời gian |
| Transit nodes/edges | `travel_summary_nodes.gpkg`, `travel_summary_edges.gpkg` | điểm dừng và liên kết xe buýt đã tổng hợp |
| Homogeneous graph | `data/processed/hanoi/graphs/homo.pt` | đồ thị PyG với quan hệ `contig` |
| Heterogeneous walk graph | `data/processed/hanoi/graphs/hetero_walk.pt` | `contig` + `15_min_walk` |
| Heterogeneous multimodal graph | `data/processed/hanoi/graphs/hetero_multi.pt` | cả `contig`, `15_min_walk`, `15_min_multi` |

### 3.4. Các cột định danh và hình học

| Cột | Kiểu dự kiến | Ý nghĩa |
|---|---|---|
| `zone_id` | string | mã định danh duy nhất của ô H3 |
| `geometry` | Polygon/MultiPolygon | biên hình học của ô H3 |
| `pop_weighted_centroid` | Point, nếu có | điểm đại diện dùng để nối ô vào mạng và tính khoảng cách |
| `total_pois` | numeric | tổng số POI được gán vào ô |
| `area_km2` | numeric | diện tích ô sau chuyển CRS |
| `poi_density_per_km2` | numeric | mật độ POI trên diện tích |

Các cột `total_pois`, `area_km2` và `poi_density_per_km2` là biến mô tả/kiểm tra. Ma trận đầu vào chính của mô hình sử dụng 16 cột nhóm POI bên dưới.

### 3.5. Mười sáu nhóm tiện ích tầng trệt

| Cột đặc trưng | Ý nghĩa nghiệp vụ |
|---|---|
| `arts_culture` | nghệ thuật, văn hóa, bảo tàng, di sản và hoạt động sáng tạo |
| `automotive_facility` | sửa chữa, bãi đỗ, trạm nhiên liệu và dịch vụ ô tô |
| `consumer_service` | dịch vụ tiêu dùng cá nhân và hộ gia đình |
| `corporate_service` | văn phòng, doanh nghiệp và dịch vụ chuyên môn |
| `education` | trường học, cơ sở đào tạo và giáo dục |
| `entertainment` | giải trí, rạp chiếu phim, câu lạc bộ và hoạt động ban đêm |
| `food_beverage` | nhà hàng, quán ăn, đồ uống và dịch vụ ẩm thực |
| `healthcare` | bệnh viện, phòng khám, nhà thuốc và chăm sóc sức khỏe |
| `hotel_lodging` | khách sạn, nhà nghỉ, hostel và lưu trú |
| `industrial_service` | sản xuất nhẹ, kho, bán buôn và dịch vụ công nghiệp |
| `park` | công viên, vườn, hồ và không gian xanh |
| `public_service` | cơ quan công, bưu điện, cảnh sát và dịch vụ hành chính |
| `religion` | chùa, nhà thờ và cơ sở tôn giáo |
| `retail` | cửa hàng, chợ, siêu thị, trung tâm thương mại và bán lẻ |
| `sports_fitness` | thể thao, phòng tập và cơ sở rèn luyện |
| `transportation_facility` | bến, điểm trung chuyển, dịch vụ vận tải và hạ tầng giao thông |

Mỗi giá trị trước chuẩn hóa đại diện cho số lượng POI thuộc nhóm tương ứng trong một ô. Việc gán nhóm dựa trên phân loại Overture và quy tắc ánh xạ được thực hiện trong pha xử lý dữ liệu; không nên diễn giải một nhóm là biến đo trực tiếp dân số hoặc doanh thu.

### 3.6. Schema quan hệ đồ thị

| Quan hệ | Nút nguồn/đích | Điều kiện tạo cạnh | Thuộc tính cạnh |
|---|---|---|---|
| `contig` | `oa -> oa` | hai ô có tiếp giáp Queen | `travel_time_sec`, `weight` |
| `15_min_walk` | `oa -> oa` | đường đi ngắn nhất trên mạng đi bộ $\le 900$ giây | thời gian đi bộ |
| `15_min_multi` | `oa -> oa` | đường đi ngắn nhất kết hợp đường bộ và xe buýt $\le 900$ giây; lưu phần transit-only theo pipeline | thời gian đa phương thức |

Trong bước đánh giá Newman Modularity, các cạnh được chiếu sang `networkx.Graph` vô hướng để dùng cùng một định nghĩa cộng đồng. Đây là quy ước đánh giá, không phủ nhận rằng dữ liệu giao thông gốc có thể mang hướng hoặc lịch trình.

---

## 4. Công thức chuẩn hóa dữ liệu

### 4.1. Biến đổi Log1p

Với số lượng POI thô $x_{i,c}$ của ô $i$ và nhóm tiện ích $c$:

$$
\tilde{x}_{i,c}=\log(1+x_{i,c}).
$$

Biến đổi này giữ được giá trị bằng 0, đồng thời giảm độ lệch phải do một số ô có quá nhiều POI.

### 4.2. Z-score

Sau Log1p, tính trung bình và độ lệch chuẩn theo từng cột trên 985 ô:

$$
z_{i,c}=\frac{\tilde{x}_{i,c}-\mu_c}{\sigma_c},
$$

trong đó:

- $\mu_c$ là trung bình của nhóm $c$;
- $\sigma_c$ là độ lệch chuẩn của nhóm $c$;
- $z_{i,c}$ là giá trị đưa vào encoder.

Các tham số $(\mu_c,\sigma_c)$ được lưu trong `oa_feature_scaling.json` để bảo đảm cùng một phép biến đổi có thể được áp dụng cho dữ liệu bổ sung hoặc nghiên cứu chuyển miền.

### 4.3. Chuẩn hóa embedding trước phân cụm

Sau khi mô hình sinh $Z$, mỗi vector nút được chuẩn hóa L2:

$$
\hat{\mathbf{z}}_i=\frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2}.
$$

Điều này làm cho khoảng cách Euclid giữa các vector đã chuẩn hóa có quan hệ trực tiếp với góc cosine, phù hợp với phân tích tương đồng trong không gian embedding.

---

## 5. Quy ước chất lượng và kiểm tra schema

Trước khi huấn luyện, tối thiểu phải xác nhận:

- số ô H3 bằng 985 và không có `zone_id` trùng;
- CRS hình học cuối là EPSG:32648;
- 16 cột POI đều tồn tại và có kiểu số;
- không có NaN/Inf trong ma trận $X$ đưa vào mô hình;
- số nút `oa` trong ba đồ thị bằng nhau;
- mỗi quan hệ có `edge_index` hợp lệ và chỉ số nút nằm trong $[0,984]$;
- các ma trận embedding có kích thước `985 x 32`;
- nhãn HDBSCAN `-1` được giữ nguyên và được tách riêng khi tính các chỉ số trên tập non-noise.

Các kiểm tra này là kiểm tra tính toàn vẹn dữ liệu, không phải đánh giá chất lượng mô hình. Việc đạt tất cả kiểm tra schema không chứng minh POI hoặc GTFS hoàn toàn đầy đủ về mặt thế giới thực.

---

## 6. Tài liệu tham khảo tối thiểu

1. Sato, Y. et al. (2026). *City2Graph: A Python library for urban heterogeneous graph neural networks across spatial and temporal scales*. *Computers, Environment and Urban Systems*, 130, 102492.
2. Velickovic, P. et al. (2018). *Graph Attention Networks*. ICLR.
3. Wang, X. et al. (2019). *Heterogeneous Graph Attention Network*. WWW.
4. Van der Maaten, L. and Hinton, G. (2008). *Visualizing Data using t-SNE*. Journal of Machine Learning Research.
5. Campello, R. J. G. B. et al. (2015). *Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection*. ACM TKDD.

---

## Kết luận

City2Graph-Hanoi kết hợp hai nguồn tín hiệu: **thuộc tính chức năng** từ 16 nhóm POI và **cấu trúc tiếp cận** từ ba quan hệ đồ thị. PCA cung cấp mốc phi đồ thị; GAT-GAE đo tác động của một mạng quan hệ; HAN-GAE học sự kết hợp giữa nhiều quan hệ; còn Models 1S-4S kiểm tra mức đóng góp riêng của topology. Data Dictionary này xác định rõ đơn vị phân tích, CRS, schema, công thức chuẩn hóa và giả định diễn giải, qua đó tạo nền tảng thống nhất cho các giai đoạn phân tích và viết báo cáo tiếp theo.
