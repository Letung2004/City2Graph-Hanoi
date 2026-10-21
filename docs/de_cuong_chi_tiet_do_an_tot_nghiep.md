# ĐỀ CƯƠNG CHI TIẾT ĐỒ ÁN TỐT NGHIỆP

## PROJECT PROPOSAL & OUTLINE

**Lĩnh vực:** Trí tuệ nhân tạo / Khoa học dữ liệu, chuyên sâu GeoAI và Graph Deep Learning  
**Địa bàn nghiên cứu:** Khu vực nội đô Hà Nội  
**Quy mô phân tích:** 985 ô lưới lục giác H3 Resolution 9 thuộc 27 phường nội đô

---

## 1. Thông tin chung

### 1.1. Tên đề tài

**Tên tiếng Việt:** Xây dựng mô hình biểu diễn tri thức đô thị Hà Nội bằng đồ thị dị thể và học sâu trên không gian địa lý

**Tên tiếng Anh:** *City2Graph-Hanoi: Urban Knowledge Representation of Hanoi Using Heterogeneous Graphs and Spatial Graph Deep Learning*

### 1.2. Đối tượng nghiên cứu

Đối tượng nghiên cứu là cấu trúc chức năng và khả năng tiếp cận không gian của khu vực nội đô Hà Nội, được biểu diễn dưới dạng đồ thị đô thị đa quan hệ. Mỗi nút không gian là một ô H3 Resolution 9; các nút được mô tả bởi 16 nhóm tiện ích đô thị và liên kết với nhau thông qua quan hệ tiếp giáp, đi bộ và tiếp cận đa phương thức bằng xe buýt.

### 1.3. Phạm vi nghiên cứu

- **Không gian:** 985 ô H3 Resolution 9 trong phạm vi 27 phường nội đô Hà Nội.
- **Đối tượng dữ liệu:** điểm tiện ích đô thị, mạng đường bộ và mạng vận tải công cộng bằng xe buýt.
- **Đầu ra phân tích:** vector nhúng 32 chiều cho từng ô H3, phân cụm vùng chức năng, chỉ số liên kết mạng, phân tích vùng tiếp cận và trực quan hóa không gian.
- **Phạm vi thời gian:** dữ liệu giao thông công cộng được sử dụng ở dạng GTFS tĩnh theo thời điểm thu thập; đề tài chưa mô phỏng biến động theo giờ hoặc theo tình trạng giao thông thực tế.
- **Phạm vi ứng dụng:** hỗ trợ phân tích đô thị và nghiên cứu GeoAI; không thay thế ranh giới hành chính, quy hoạch pháp lý hoặc hệ thống điều hành giao thông thời gian thực.

---

## 2. Tính cấp thiết và mục tiêu nghiên cứu

### 2.1. Tính cấp thiết

Đô thị Hà Nội có cấu trúc không gian phức tạp, trong đó các khu vực chức năng không chỉ được hình thành bởi sự gần nhau về địa lý mà còn bởi mức độ tiếp cận qua mạng đường bộ và vận tải công cộng. Các phương pháp phân tích dựa thuần túy trên ranh giới hành chính hoặc các biến thống kê độc lập khó mô tả đầy đủ mối quan hệ này.

Trong bối cảnh dữ liệu bản đồ mở, POI và GTFS ngày càng phong phú, việc kết hợp dữ liệu không gian với đồ thị dị thể và Graph Neural Network tạo ra hướng tiếp cận có khả năng học đồng thời đặc trưng thuộc tính và cấu trúc kết nối. Đề tài kế thừa và mở rộng hướng nghiên cứu City2Graph của Sato et al. trên một trường hợp nghiên cứu cụ thể tại Hà Nội, qua đó xây dựng một quy trình có thể tái lập cho phân tích đô thị dựa trên GeoAI.

### 2.2. Mục tiêu tổng quát

Xây dựng và đánh giá một hệ thống biểu diễn tri thức đô thị cho khu vực nội đô Hà Nội bằng đồ thị dị thể, trong đó vector nhúng học sâu được sử dụng để phát hiện các vùng chức năng và phân tích quan hệ giữa cấu trúc đô thị, khả năng đi bộ và khả năng tiếp cận xe buýt.

### 2.3. Mục tiêu cụ thể

1. Chuẩn hóa dữ liệu POI, mạng đường bộ và GTFS về cùng hệ quy chiếu, đơn vị phân tích và mã định danh H3.
2. Tạo 16 nhóm đặc trưng tiện ích đô thị cho 985 ô H3 bằng biến đổi `log1p` và z-score.
3. Xây dựng đồ thị đồng nhất và đồ thị dị thể với ba quan hệ chính: `contig`, `15_min_walk` và `15_min_multi`.
4. Huấn luyện, so sánh và phân tích các mô hình PCA, GAT-GAE, HAN-GAE và các mô hình ablation structure-only.
5. Sinh vector nhúng 32 chiều, phân cụm bằng K-Means và HDBSCAN, đồng thời đánh giá bằng Silhouette, Calinski-Harabasz, Davies-Bouldin, DBCV và Newman Modularity.
6. Diễn giải vai trò của các quan hệ đồ thị thông qua semantic attention, isochrone và phân tích suy giảm tương đồng theo thời gian tiếp cận.
7. Đóng gói dữ liệu, mô hình, bảng biểu, hình ảnh và tài liệu để bảo đảm khả năng kiểm tra và tái lập ở cấp độ đồ án tốt nghiệp.

---

## 3. Phương pháp nghiên cứu và cơ sở khoa học

### 3.1. Khung phương pháp tổng thể

Quy trình nghiên cứu gồm năm pha:

1. Tiền xử lý và kiểm định dữ liệu nguồn.
2. Tạo đặc trưng và xây dựng đồ thị đô thị.
3. Huấn luyện mô hình graph autoencoder.
4. Phân cụm, đánh giá và kiểm định độ bền.
5. Trực quan hóa, diễn giải và tổng hợp kết quả.

### 3.2. Chuẩn hóa dữ liệu

- Quy đổi dữ liệu không gian về EPSG:32648 để tính khoảng cách và thời gian trên hệ mét.
- Gán POI vào ô H3 tương ứng và tổng hợp thành 16 nhóm tiện ích.
- Áp dụng biến đổi `log1p` để giảm ảnh hưởng của các ô có mật độ POI cực lớn.
- Áp dụng z-score theo từng nhóm đặc trưng; lưu các tham số trung bình và độ lệch chuẩn để phục vụ tái lập.
- Tính thời gian đi bộ từ khoảng cách mạng với vận tốc đi bộ giả định; xây dựng quan hệ tiếp cận trong ngưỡng 15 phút.
- Tạo quan hệ tiếp cận đa phương thức dựa trên mạng đường bộ kết hợp các liên kết GTFS xe buýt trong cùng ngưỡng thời gian.

### 3.3. Biểu diễn đồ thị dị thể

Đồ thị có 985 nút loại `oa` và các quan hệ giữa các ô không gian:

- **`contig`:** hai ô H3 tiếp giáp theo tiêu chuẩn Queen contiguity.
- **`15_min_walk`:** hai ô có thể tiếp cận trong tối đa 900 giây bằng mạng đi bộ.
- **`15_min_multi`:** quan hệ tiếp cận đa phương thức, bao gồm kết nối đường bộ và xe buýt.

Các quan hệ được lưu dưới dạng PyTorch Geometric `HeteroData`, kèm thuộc tính cạnh là thời gian di chuyển hoặc trọng số tương ứng.

### 3.4. Các mô hình thực nghiệm

| Mô hình | Vai trò | Đầu vào cấu trúc |
|---|---|---|
| **Baseline PCA** | Mốc so sánh phi đồ thị | 16 nhóm POI |
| **Model 1: GAT-GAE** | Graph autoencoder đồng nhất với attention | `contig` |
| **Model 2: HAN-GAE Walk** | HAN-GAE dị thể | `contig` + `15_min_walk` |
| **Model 3: HAN-GAE Multi** | HAN-GAE dị thể có vận tải công cộng | `contig` + `15_min_multi` |
| **Model 4: HAN-GAE All** | Mô hình đầy đủ | cả ba quan hệ |
| **Models 1S-4S** | Ablation structure-only | tương ứng từng cấu hình, loại bỏ tái tạo thuộc tính POI |

GAT-GAE học biểu diễn nút thông qua hai lớp Graph Attention. HAN-GAE bổ sung attention cấp nút và attention ngữ nghĩa giữa các loại quan hệ. Bộ giải mã gồm tái tạo thuộc tính và tái tạo cấu trúc bằng DistMult với negative sampling.

### 3.5. Cấu hình thực nghiệm đã sử dụng

- Learning rate: **0.005**.
- Số epoch tối đa: **200**.
- Số lớp GNN: **L = 2**.
- Hidden dimension: **64**.
- Embedding dimension: **32**.
- Số attention heads: **4**.
- Dropout: **0.6**.
- Weight decay: **0.0001**.
- Random seed chính: **42**.
- K-Means: quét `k = 2..50`, `n_init = 10`.
- HDBSCAN: quét `min_cluster_size = 5..30` và chọn cấu hình theo DBCV.

### 3.6. Cơ sở đánh giá

- **Silhouette Score:** mức độ cô đọng và phân tách trong không gian embedding.
- **Calinski-Harabasz:** tỷ lệ phân tán giữa cụm và trong cụm.
- **Davies-Bouldin:** mức độ tương đồng giữa các cụm, càng thấp càng tốt.
- **Newman Modularity:** mức độ phù hợp giữa phân cụm và cấu trúc của từng mạng quan hệ.
- **DBCV:** chất lượng phân cụm theo mật độ cho HDBSCAN.
- **Isochrone và cosine decay:** diễn giải phạm vi tiếp cận và sự suy giảm tương đồng theo thời gian di chuyển.

Các chỉ số trên là đánh giá nội tại và cấu trúc, không được diễn giải như độ chính xác dự báo có nhãn.

---

## 4. Dữ liệu và công cụ thực hiện

### 4.1. Dữ liệu

| Nhóm dữ liệu | Nguồn / nội dung | Vai trò |
|---|---|---|
| Ranh giới không gian | 985 ô H3 Resolution 9 trong 27 phường nội đô | Đơn vị phân tích và nút đồ thị |
| POI | Overture Maps, khoảng 152.000 điểm | Tạo 16 nhóm đặc trưng tiện ích |
| Mạng đường bộ | Overture road segments và connectors | Tính kết nối và thời gian đi bộ |
| GTFS xe buýt | Dữ liệu tuyến, điểm dừng và thời gian hành trình Hà Nội | Tạo quan hệ đa phương thức |
| Dữ liệu trung gian | GeoPackage, Parquet, PyTorch Geometric | Phục vụ kiểm toán và tái lập |

### 4.2. Công cụ và hệ sinh thái thư viện

- Python 3.10 trở lên, môi trường Conda hoặc `venv`.
- PyTorch và PyTorch Geometric cho huấn luyện GNN.
- GeoPandas, Shapely, PyProj và NetworkX cho xử lý không gian và mạng.
- HDBSCAN và scikit-learn cho phân cụm, PCA và các chỉ số đánh giá.
- NumPy, Pandas, Matplotlib và Seaborn cho tính toán, bảng biểu và trực quan hóa.
- Jupyter Notebook / VS Code Interactive Window cho quy trình thực nghiệm.
- Git để quản lý phiên bản mã nguồn và artefact.

### 4.3. Sản phẩm dữ liệu trung gian chính

`oa_with_features.gpkg`, `homo.pt`, `hetero_walk.pt`, `hetero_multi.pt`, các tệp embedding `Z_*.npy`, checkpoint mô hình, tệp nhãn cụm, bảng CSV và hình ảnh khoa học.

---

## 5. Kế hoạch thực hiện và tiến độ 14 tuần

| Tuần | Nội dung chính | Sản phẩm / mốc kiểm tra |
|---:|---|---|
| 1 | Khảo sát bài báo nền, xác định câu hỏi và phạm vi | Đề cương sơ bộ, danh mục tài liệu |
| 2 | Thiết lập môi trường, cấu trúc repository và cấu hình thí nghiệm | Môi trường chạy được, cấu hình thống nhất |
| 3 | Kiểm tra ranh giới 27 phường và 985 ô H3 | Bộ vùng nghiên cứu được xác nhận |
| 4 | Nạp, làm sạch và gán POI Overture vào H3 | Bảng 16 nhóm POI |
| 5 | Tiền xử lý mạng đường bộ và GTFS | Các lớp nút/cạnh giao thông |
| 6 | Chuẩn hóa `log1p + z-score` và kiểm tra phân phối | `oa_with_features.gpkg`, scaling parameters |
| 7 | Xây dựng đồ thị `contig` và `15_min_walk` | `homo.pt`, `hetero_walk.pt` |
| 8 | Xây dựng đồ thị đa phương thức `15_min_multi` | `hetero_multi.pt`, kiểm toán edge statistics |
| 9 | Huấn luyện Baseline, GAT-GAE và HAN-GAE Models 2-4 | Checkpoint và loss curves |
| 10 | Huấn luyện Ablation Models 1S-4S | Embedding structure-only và so sánh ablation |
| 11 | Phân cụm K-Means, quét `k=2..50` | Nhãn cụm, bảng chỉ số và k* |
| 12 | Phân cụm HDBSCAN và kiểm tra DBCV | Nhãn noise, bảng C.2 và modularity non-noise |
| 13 | Trực quan hóa, isochrone, attention và phân tích kết quả | Bộ hình/bản đồ khoa học |
| 14 | Viết báo cáo, hoàn thiện phụ lục và chuẩn bị bảo vệ | Báo cáo cuối, slide và repository bàn giao |

---

## 6. Kết quả đạt được và sản phẩm bàn giao

### 6.1. Kết quả khoa học dự kiến / đã đạt

1. Một quy trình hoàn chỉnh từ dữ liệu đô thị mở đến biểu diễn đồ thị dị thể.
2. Tám mô hình GNN gồm bốn mô hình có thuộc tính và bốn mô hình structure-only, cùng Baseline PCA.
3. Tám ma trận embedding kích thước `985 x 32` và các trọng số semantic attention cho HAN-GAE.
4. Các phân vùng chức năng bằng K-Means và HDBSCAN.
5. Đánh giá định lượng trên ba mạng quan hệ và phân tích khả năng tiếp cận 5-30 phút.
6. Bộ bản đồ POI, phân cụm, attention, t-SNE, isochrone và đường cong suy giảm tương đồng.

### 6.2. Quy ước báo cáo các bảng thống kê

- **Giá trị tại `k*`:** `k*` được chọn là giá trị `k` có Silhouette cao nhất trong quét K-Means từ 2 đến 50. Đây là giá trị đại diện cho nghiệm phân cụm được lưu và dùng để lập bản đồ.
- **Mean ± SD trên dải `k=2..50`:** dùng để mô tả độ biến thiên của chỉ số trong toàn bộ quá trình quét, không phải độ lệch chuẩn giữa các lần huấn luyện mô hình. Khi trình bày, phải ghi rõ đây là “mean ± SD trên dải k”.
- **Độ ổn định theo seed tại `k*`:** nếu báo cáo nhiều lần chạy K-Means, Mean ± SD được tính trên các seed chạy lại tại cùng `k*`; đại lượng này khác với thống kê trên dải `k`.
- **HDBSCAN:** `mcs` được quét từ 5 đến 30 và cấu hình có DBCV cao nhất được chọn. Các ô nhãn `-1` được xem là noise.
- **Modularity của HDBSCAN:** tính trên đồ thị con gồm các ô **non-noise** và các cụm hợp lệ; do đó không so sánh trực tiếp với Modularity K-Means nếu không nêu rõ khác biệt về tập nút đánh giá.

### 6.3. Sản phẩm bàn giao

- Báo cáo đồ án và phụ lục phương pháp.
- Mã nguồn trong `src/`, `notebooks/` và `scripts/`.
- `config/experiment_config.yaml` phản ánh đúng cấu hình thực nghiệm đã huấn luyện: `lr=0.005`, `epochs=200`, `heads=4`, `out_dim=32`, `n_init=10`.
- Dữ liệu đã xử lý và đồ thị PyG.
- Checkpoint, embedding, nhãn phân cụm và bảng thống kê.
- Bộ hình ảnh/bản đồ khoa học và hướng dẫn tái lập trong README.

---

## 7. Hạn chế của đề tài và hướng phát triển tương lai

### 7.1. Hạn chế và đặc thù nghiên cứu

- Không có nhãn ground truth cho “vùng chức năng”, vì vậy kết quả được đánh giá bằng chỉ số nội tại, cấu trúc mạng và khả năng diễn giải đô thị.
- Không sử dụng train/validation/test split theo kiểu supervised; mô hình phục vụ self-supervised representation learning.
- Việc chọn `k` theo Silhouette và chọn nghiệm tốt nhất trong các seed có thể tạo thiên lệch tối ưu hóa nội bộ.
- Các mô hình 1S-4S chỉ dùng cấu trúc đồ thị nên có thể suy giảm biểu diễn hoặc tạo tỷ lệ noise cao; đây là kết quả ablation nhằm chứng minh vai trò của thuộc tính POI.
- HDBSCAN có thể gán nhiều ô là noise, đặc biệt ở các embedding structure-only; các ô này không nên bị diễn giải như khu vực không có chức năng đô thị.
- Chuyển mạng có hướng thành đồ thị vô hướng khi tính Newman Modularity giúp thống nhất phép đo cộng đồng nhưng làm mất thông tin chiều đi.
- Overture POI và GTFS là dữ liệu mở, có thể có thiếu hụt, sai phân loại và sai khác theo thời điểm cập nhật.
- GTFS được dùng ở dạng tĩnh, chưa phản ánh thời gian chờ thực tế, tắc nghẽn, ngày thường/cuối tuần hoặc biến động sáng-trưa-chiều-tối.
- Trọng số attention thể hiện mức đóng góp tương đối trong mô hình, không phải bằng chứng quan hệ nhân quả.
- t-SNE chỉ có giá trị minh họa trực quan, không được dùng làm thước đo định lượng chính.

### 7.2. Hướng phát triển tương lai

1. **Mô hình hóa GTFS đa khung giờ và đa ngày:** xây dựng các đồ thị sáng, trưa, chiều, tối; phân biệt ngày thường và cuối tuần; mở rộng sang temporal HAN-GAE hoặc dynamic GNN để theo dõi sự biến đổi của vùng chức năng.
2. **Tích hợp dữ liệu OD thực tế:** kết hợp GPS xe buýt, dữ liệu thẻ vé, mobile signaling hoặc ma trận Origin-Destination để kiểm định trực tiếp khả năng tiếp cận suy ra từ mô hình.
3. **Phân tích đa độ phân giải H3:** thử nghiệm Resolution 8, 9 và 10 nhằm đánh giá độ ổn định của vùng chức năng khi thay đổi quy mô không gian và giảm phụ thuộc vào một đơn vị phân tích duy nhất.
4. **Xây dựng không gian trạng thái cho RL/ITS:** sử dụng embedding của các ô H3 làm trạng thái; hành động có thể gồm điều chỉnh tuyến, tần suất xe buýt, vị trí điểm dừng hoặc phân bổ hạ tầng giao thông; hàm phần thưởng có thể kết hợp thời gian đi lại, độ phủ và công bằng tiếp cận.
5. **Đánh giá chuyển miền và bất định:** huấn luyện trên Hà Nội, kiểm thử trên đô thị khác; bổ sung bootstrap, khoảng tin cậy, nhiều random seed và kiểm định độ ổn định của cụm.

Các hướng trên được xem là mở rộng nghiên cứu sau đồ án, không phải điều kiện bắt buộc để hoàn thành mục tiêu hiện tại.

---

## Kết luận đề cương

Đề tài có phạm vi rõ ràng, dữ liệu đủ quy mô, phương pháp phù hợp với chuyên ngành Trí tuệ nhân tạo/Khoa học dữ liệu và sản phẩm đầu ra có thể kiểm tra định lượng. Đóng góp chính của đồ án là xây dựng một quy trình GeoAI có khả năng biểu diễn đồng thời thuộc tính POI, cấu trúc không gian và khả năng tiếp cận giao thông của khu vực nội đô Hà Nội trên nền tảng đồ thị dị thể.
