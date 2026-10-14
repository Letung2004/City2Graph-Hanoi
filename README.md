# City2Graph-Hanoi: Mô Hình Hóa Tri Thức Đô Thị Đa Phương Thức Bằng Đồ Thị Dị Thể & Học Sâu Không Gian

Dự án tái lập và mở rộng nghiên cứu khoa học từ công trình của **Sato et al. (*Computers, Environment and Urban Systems - CEUS*, 2026, Vol. 130, 102492)** trên địa bàn Thủ đô Hà Nội. Hệ thống tích hợp mạng nơ-ron đồ thị tích chập dị thể (**HAN-GAE**) và tự chú ý (**GAT-GAE**) trên lưới phân giải không gian Uber H3 Resolution 9 ($985$ ô phân tích, diện tích $113.22\text{ km}^2$, đối chuẩn tỷ lệ $1:1$ với Liverpool $111.84\text{ km}^2$). Mô hình kết hợp 16 nhóm tiện ích tầng trệt ($152,120$ POIs), mạng lưới vỉa hè người đi bộ ($50,771$ nút giao, $67,748$ đoạn đường) và hệ thống vận tải hành khách công cộng xe buýt GTFS ($2,299$ liên kết trạm dừng).

## 📂 Cấu Trúc Thư Mục Dự Án

Cấu trúc thư mục được tinh giản, chuẩn hóa theo nguyên tắc quản lý mã nguồn khoa học độc lập:

```text
City2Graph-Hanoi/
├── config/                                    # Cấu hình thực nghiệm & siêu tham số
│   └── experiment_config.yaml                 # Seeds, epochs, learning rate, dimensions
├── data/
│   ├── raw/hanoi/                             # Dữ liệu nguồn sơ cấp
│   ├── processed/hanoi/                       # Dữ liệu đặc trưng & đồ thị PyG
│   └── outputs/hanoi/                         # Checkpoints, clusters, embeddings, figures, tables
├── notebooks/                                 # 6 Notebooks thực nghiệm (.py & .ipynb)
├── scripts/                                   # Bộ công cụ tự động hóa & kiểm toán
├── src/                                       # Mã nguồn lõi (city2graph, models, baselines)
├── environment.yml                            # Cấu hình môi trường Conda (khuyến nghị)
├── requirements.txt                           # Danh sách phụ thuộc cài đặt qua Pip/venv/Colab
└── README.md                                  # Hướng dẫn chi tiết dự án
```

## 🛠️ Hướng Dẫn Cài Đặt Môi Trường

Kho lưu trữ đã được tích hợp sẵn package `city2graph` bên trong thư mục `src/city2graph/`, hệ thống tự động ưu tiên nạp module cục bộ thông qua cơ chế tự định vị thư mục gốc (`sys.path.insert(0, str(SRC_DIR))`).

### Lựa chọn 1: Cài đặt qua Conda (Khuyến nghị nếu máy đã có Anaconda/Miniconda)

```powershell
# 1. Khởi tạo môi trường ảo Python 3.10
conda env create -f environment.yml

# 2. Kích hoạt môi trường
conda activate c2g_cpu
```

### Lựa chọn 2: Cài đặt bằng Python `venv` (Dành cho máy chưa cài Conda)

Trong trường hợp máy tính chỉ cài đặt Python tiêu chuẩn (Python 3.10 – 3.12) mà không có Conda, có thể khởi tạo môi trường ảo độc lập bằng `venv` tích hợp sẵn của Python:

**Trên Windows (PowerShell):**
```powershell
# 1. Tạo môi trường ảo tên là .venv tại thư mục gốc dự án
python -m venv .venv

# 2. Kích hoạt môi trường ảo
# (Nếu PowerShell báo lỗi ExecutionPolicy, chạy trước lệnh: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.\.venv\Scripts\Activate.ps1

# 3. Nâng cấp pip và cài đặt toàn bộ thư viện phụ thuộc
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Trên macOS / Linux (Terminal):**
```bash
# 1. Tạo môi trường ảo
python3 -m venv .venv

# 2. Kích hoạt môi trường ảo
source .venv/bin/activate

# 3. Nâng cấp pip và cài đặt toàn bộ thư viện phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

### Lựa chọn 3: Cài đặt trực tiếp qua Pip / Google Colab

Nếu đang đứng sẵn trong môi trường Python đích hoặc chạy trên Google Colab:

```bash
# Cài đặt toàn bộ danh mục thư viện phụ thuộc
pip install -r requirements.txt
```

## 🚀 Quy Trình Thực Thi Thực Nghiệm (Pipeline)

Toàn bộ các pha phân tích trong thư mục `notebooks/` đều được cấu hình song song ở hai định dạng: **Interactive Python Script (`.py`)** và **Jupyter Notebook (`.ipynb`)**. Người dùng có thể thực thi tuần tự từ terminal hoặc môi trường tương tác:

```text
[01_data_processing]  ──>  [02_graph_construction]  ──>  [03_model_training]
         │                            │                           │
         ▼                            ▼                           ▼
Features & Street GDF         homo.pt & hetero_*.pt         Embeddings Z & Attn β
                                                                  │
                                      ┌───────────────────────────┴───────────────────────────┐
                                      ▼                                                       ▼
                            [04_evaluation_hanoi]                                   [appendix_hdbscan]
                           K-Means, Modularity Q,                                  HDBSCAN Density, DBCV,
                           Isochrones, Distance Decay                              Robustness Verification
                                      │
                                      ▼
                            [05_visualization_hanoi]
                           Metapaths, Land Use, Table A.2
```

### 1. Pha 1: Tiền xử lý dữ liệu (`01_data_processing_hanoi`)

* **Không gian phân tích:** Khởi tạo lưới lục giác Uber H3 Resolution 9 ($985$ ô, bán kính $174\text{ m}$, diện tích bao phủ $113.22\text{ km}^2$, hệ tọa độ phẳng mét chuẩn quốc gia EPSG:32648).
* **Tiện ích tầng trệt (POIs):** Gán không gian $152,120$ điểm POIs Overture Maps vào 16 nhóm chức năng (Table A.1).
* **Mạng lưới hạ tầng:** Trích xuất đồ thị đường bộ người đi bộ ($50,771$ nút giao, $67,748$ phân đoạn, vận tốc $v_{\text{walk}} = 1.3333\text{ m/s}$) và đồ thị hành trình xe buýt GTFS ($2,299$ liên kết).

### 2. Pha 2: Khởi tạo đồ thị đa quan hệ (`02_graph_construction_hanoi`)

* **Chuẩn hóa phi tuyến:** Chuẩn hóa ma trận đặc trưng $16$ chiều theo công thức $\text{Log}(1 + x) + \text{Z-score}$ (Equation 1), lưu tham số $(\mu, \sigma)$ vào `oa_feature_scaling.json`.
* **Đóng gói PyG HeteroData:**
  * Quan hệ tiếp giáp địa lý `contig`: $5,550$ cạnh có hướng Queen contiguity.
  * Siêu đường đi đi bộ 15 phút `15_min_walk`: $31,040$ cạnh định hướng ($T \le 900\text{ s}$).
  * Siêu đường đi buýt đa phương thức `15_min_multi`: $4,898$ cạnh bóc tách độc lập (`transit_only`).
  * Xuất các tệp đồ thị: `homo.pt`, `hetero_walk.pt`, `hetero_multi.pt`.

### 3. Pha 3: Huấn luyện mạng học sâu đồ thị (`03_model_training_hanoi`)

* **Kiến trúc mô hình:**
  * **Model 1 (GAT-GAE):** Đồ thị đồng nhất với 2 tầng Edge-Aware Graph Attention Network.
  * **Model 2 – 4 (HAN-GAE):** Heterogeneous Graph Attention Network với 2 cấp độ chú ý: *Node-level Attention* (nội bộ từng quan hệ) và *Semantic-level Attention* (tổng hợp liên quan hệ qua ma trận $\boldsymbol{\beta}$).
  * **Mô hình triệt tiêu cấu trúc (Ablation Models):** Models 1S, 2S, 3S, 4S (chỉ huấn luyện Structure Decoder, đóng băng thuộc tính).
* **Hàm mất mát tự giám sát liên hiệp (Joint Loss):**

$$
\mathcal{L} = \mathcal{L}_{\text{struct}} + \gamma \mathcal{L}_{\text{attr}} = \frac{1}{\vert{}\mathcal{E}\vert{}}\sum_{(i,j)\in \mathcal{E}} \text{BCE}(\hat{A}_{ij}, A_{ij}) + \gamma \frac{1}{\vert{}V\vert{}}\sum_{i \in V} \Vert{}\mathbf{x}_i - \hat{\mathbf{x}}_i\Vert{}_2^2
$$

* **Đầu ra:** 8 ma trận vector nhúng tiềm ẩn $\mathbf{Z} \in \mathbb{R}^{985 \times 32}$ và trọng số chú ý ngữ nghĩa $\boldsymbol{\beta}$.

### 4. Pha 4: Khai phá & Đánh giá phân cụm (`04_evaluation_hanoi`)

* **Định vị mốc CBD:** Hồ Hoàn Kiếm / Quảng trường Đông Kinh Nghĩa Thục ($X=588761.6\text{ m}, Y=2325539.3\text{ m}$, Zone ID: `89415cb4e53ffff`).
* **Đánh giá K-Means:** Quét số cụm $k \in [2, 50]$, đo Silhouette Score, Calinski-Harabasz, Davies-Bouldin và kiểm tra tính liên thông cấu trúc bằng Newman Modularity $Q$ trên 3 mạng lưới.
* **Phân tích suy giảm không gian:** Tính vùng đẳng thời Isochrones $5 - 30$ phút (Hình 16) và đường cong suy giảm độ tương đồng Cosine theo khoảng cách thời gian (Hình 17).

### 5. Pha 5: Ấn phẩm đồ họa bổ sung (`05_visualization_hanoi`)

* Kết xuất ma trận bản đồ 16 nhóm tiện ích (`poi.jpg`) và hiện trạng mặt phủ chức năng ưu thế (`land_use.jpg`).
* Minh họa cấu trúc đồ thị dị thể nền tối Metapaths tách rời: `hanoi_contig_oa.jpg`, `hanoi_contig_edges.jpg`, `hanoi_metapaths_walk.jpg`, `hanoi_metapaths_multi.jpg` (đối chuẩn Hình 1, Hình 2).
* Xuất **Phụ lục Bảng A.2** thống kê mô tả thời gian di chuyển các quan hệ cạnh.

### 6. Phụ lục: Đánh giá độ bền vững HDBSCAN (`appendix_evaluation_hdbscan_hanoi`)

* Phân cụm dựa trên mật độ phi tham số HDBSCAN, quét siêu tham số $MCS \in [5, 30]$ qua chỉ số DBCV (Density-Based Clustering Validation).
* Xuất **Phụ lục Bảng C.2**, bản đồ phân vùng HDBSCAN (Hình C.2) và không gian tiềm ẩn t-SNE (Hình C.3b).

## 📊 Kết Quả Nghiên Cứu Thực Nghiệm Tiêu Biểu

### 1. Hiệu năng phân cụm & Tính liên thông mạng lưới (Đối chuẩn Table 3)

Các mô hình dị thể (**HAN-GAE**) thể hiện tính ưu việt vượt trội so với phân tích thành phần chính truyền thống (Baseline PCA) và đồ thị đồng nhất đơn lẻ (Model 1):

| Mô hình | Cấu trúc quan hệ đồ thị | Số cụm tối ưu ($k^*$) | Silhouette Score | Modularity $Q$ (Contiguity) | Modularity $Q$ (15-min Walk) | Modularity $Q$ (15-min Multi) |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| **Baseline (PCA)** | Không dùng đồ thị | 2 | $0.45 \pm 0.04$ | $0.25 \pm 0.03$ | $0.21 \pm 0.03$ | $0.18 \pm 0.02$ |
| **Model 1 (GAT-GAE)** | Contiguity | 2 | $0.43 \pm 0.03$ | $0.57 \pm 0.04$ | $0.39 \pm 0.05$ | $0.29 \pm 0.06$ |
| **Model 2 (HAN-GAE: Walk)** | Contiguity + 15-min Walk | 6 | $0.41 \pm 0.02$ | $0.61 \pm 0.07$ | $0.39 \pm 0.09$ | $0.28 \pm 0.10$ |
| **Model 3 (HAN-GAE: Multi)** | Contiguity + 15-min Multi | 6 | $0.40 \pm 0.03$ | $0.60 \pm 0.06$ | $0.39 \pm 0.09$ | $0.28 \pm 0.10$ |
| **Model 4 (HAN-GAE: All)** | Contiguity + Walk + Multi | **9** | $\mathbf{0.42 \pm 0.03}$ | $\mathbf{0.67 \pm 0.07}$ | $\mathbf{0.45 \pm 0.10}$ | $\mathbf{0.33 \pm 0.12}$ |

### 2. Chỉ số cô đặc và phân tách cụm (Phụ lục Bảng C.1)

* **Davies-Bouldin Index:** Model 4 đạt giá trị nhỏ nhất ($0.87 \pm 0.10$), chứng minh các cụm chức năng có tính cô đặc nội bộ và phân tách ngoại vi tối ưu nhất.
* **Calinski-Harabasz Index:** Các mô hình dị thể HAN-GAE đạt giá trị cao ($702.73 \pm 58.70$), vượt trội gấp đôi so với Baseline PCA ($360.81 \pm 331.95$).

### 3. Thống kê mô tả các quan hệ cạnh không gian (Phụ lục Bảng A.2)

| Loại quan hệ cạnh | Số lượng cạnh | Thời gian di chuyển tối thiểu ($\text{s}$) | Thời gian di chuyển trung bình ($\text{s}$) | Thời gian di chuyển tối đa ($\text{s}$) | Độ lệch chuẩn $\sigma$ ($\text{s}$) |
| ----- | ----- | ----- | ----- | ----- | ----- |
| `('street_connector', 'is_connected_to', 'street_connector')` | 67,748 | 0.01 | 28.54 | 1,682.59 | 36.09 |
| `('bus_station', 'is_next_to', 'bus_station')` | 2,299 | 2.00 | 126.14 | 811.00 | 63.86 |
| `('oa', 'contig', 'oa')` | 5,550 | 269.49 | 273.32 | 277.58 | 3.25 |
| `('oa', '15_min_walk', 'oa')` | 31,040 | 269.49 | 629.25 | 832.75 | 206.10 |
| `('oa', '15_min_multi', 'oa')` | 4,898 | 808.50 | 820.05 | 832.75 | 9.77 |

### 4. Phát hiện quy luật đô thị Hà Nội từ cơ chế Attention và Isochrones

1. **Quy trình tổng hợp hai bước (Two-step Aggregation):**
   * *Layer 1:* Mô hình dồn trọng số chú ý ($\beta_{\text{walk}} = 71.8\%$) vào mạng lưới vỉa hè người đi bộ để nhận diện các trung tâm sinh hoạt cộng đồng.
   * *Layer 2:* Mô hình chuyển dịch trọng số sang quan hệ tiếp giáp vật lý `contig` ($\beta_{\text{contig}} = 65.2\%$) nhằm thực hiện cơ chế làm mịn không gian (Spatial Smoothing), hợp nhất các ô lân cận thành phân vùng chức năng liền khoảnh.
2. **Năng lực vươn xa của hệ thống xe buýt:**
   * Trong ngưỡng đẳng thời $30\text{ phút}$ xuất phát từ Hồ Hoàn Kiếm, người đi bộ chỉ tiếp cận được $90$ ô H3 ($R \approx 2.4\text{ km}$).
   * Mạng lưới xe buýt mở rộng tầm với lên tới $560$ ô H3 ($R \approx 8 - 10\text{ km}$, **tăng gấp** $6.2$ **lần diện tích tiếp cận**), kết nối trực tiếp lõi di sản Hoàn Kiếm tới các trung tâm hành chính mới Cầu Giấy, Tây Hồ và Thanh Xuân.

## 🔧 Bộ Công Cụ Tự Động Hóa & Kiểm Toán (`scripts/`)

Hệ thống cung cấp các kịch bản kiểm tra chất lượng dữ liệu độc lập:

```powershell
# 1. Kiểm toán dữ liệu mạng giao thông công cộng GTFS (Pha 1)
python scripts/inspect_gtfs_hanoi.py

# 2. Kiểm toán cấu trúc đồ thị đồng nhất & dị thể PyG (Pha 2)
python scripts/inspect_graph_hanoi.py

# 3. Kiểm toán tiến trình huấn luyện & ma trận vector nhúng Z (Pha 3)
python scripts/inspect_training_hanoi.py

# 4. Kiểm toán 31/31 tiêu chí đánh giá phân cụm K-Means (Pha 4)
python scripts/inspect_evaluation_hanoi.py

# 5. Kiểm toán 6 ấn phẩm đồ họa & Phụ lục Bảng A.2 (Pha 5)
python scripts/inspect_visualization_hanoi.py

# 6. Kiểm toán kết quả phân cụm mật độ HDBSCAN (Phụ lục C)
python scripts/inspect_appendix_hdbscan_hanoi.py
```

## 📜 Trích Dẫn & Bản Quyền

Nghiên cứu được kế thừa và phát triển từ công trình khoa học:

> Sato, Y., et al. (2026). *City2Graph: A Python library for urban heterogeneous graph neural networks across spatial and temporal scales*. Computers, Environment and Urban Systems, 130, 102492.