# KỊCH BẢN NỘI DUNG CHI TIẾT BỘ SLIDE THUYẾT TRÌNH BẢO VỆ CHUYÊN ĐỀ / TỐT NGHIỆP
## KHAI PHÁ HÌNH THÁI VÙNG CHỨC NĂNG ĐÔ THỊ 15 PHÚT TẠI HÀ NỘI BẰNG MẠNG NƠ-RON ĐỒ THỊ DỊ THỂ (HETEROGENEOUS GNNs)
> **Tác giả:** Lê Đức Tùng - 20227161  
> **Địa điểm báo cáo dự kiến:** Viện CNTT&TT / BKFintech (Đại học Bách Khoa Hà Nội) hoặc VinUni  
> **Khung cấu trúc:** 6 Phần - 28 Slide (Bao gồm 2 Slide Danh mục Thuật ngữ & Từ viết tắt chuẩn hóa; Quy tắc cặp đôi: 1 Slide luận điểm nội dung $\rightarrow$ 1 Slide ảnh trực quan / bảng số liệu thực nghiệm)

---

## TỔNG QUAN PHÂN BỔ 28 SLIDES (ĐÃ BỔ SUNG THUẬT NGỮ & TỪ VIẾT TẮT)

* **Khối Mở Đầu & Thuật Ngữ Nền Tảng (Slide 1 – 4):**
  * Slide 1: Bìa Báo Cáo (Title Slide)
  * Slide 2: Nội Dung Thuyết Trình (Presentation Agenda - 6 Phần cốt lõi)
  * Slide 3 [Danh mục Thuật ngữ 1]: Bảng Danh mục Từ viết tắt & Thuật ngữ Đô thị Không gian (Spatial & Urban Abbreviations)
  * Slide 4 [Danh mục Thuật ngữ 2]: Bảng Danh mục Từ viết tắt & Thuật ngữ AI / Đồ thị / Đánh giá (GNNs & Evaluation Metrics)
* **Phần 1: Đặt Vấn Đề, Bối Cảnh Đô Thị Nén & Khái Niệm 15 Phút (Slide 5 – 6):**
  * Slide 5 [Nội dung]: Bối cảnh Đô thị nén Hà Nội & Khái niệm Đô thị 15 phút
  * Slide 6 [Nội dung]: Thách thức Sai số MAUP & Giải pháp Chuẩn hóa Lưới Uber H3 Res 9
* **Phần 2: Xử Lý Dữ Liệu Không Gian & Ma Trận 16 Nhóm POIs (File 01) (Slide 7 – 8):**
  * Slide 7 [Nội dung]: Khai thác 152.000 POIs tầng trệt & Kỹ thuật Chuẩn hóa Logarit
  * Slide 8 [Ảnh thực nghiệm]: Bản đồ tổng mật độ POIs (Fig 10a) + Ma trận 16 nhóm POIs (Fig 10b) + Histogram phân phối chuẩn hóa
* **Phần 3: Xây Dựng Đồ Thị Dị Thể Đa Phương Thức 15 Phút (File 02) (Slide 9 – 12):**
  * Slide 9 [Nội dung]: Mô hình hóa Mạng lưới Đường bộ OSM, Mạng buýt GTFS & Vùng đẳng thời
  * Slide 10 [Ảnh thực nghiệm]: Mạng đường OSM (31k cạnh) + Đồ thị buýt GTFS (4.8k cạnh) + Vùng đẳng thời 15 phút Isochrones
  * Slide 11 [Nội dung]: Cấu trúc Đồ thị Dị thể PyG: 3 Siêu quan hệ (Metapaths) & 8 Quan hệ Vi mô
  * Slide 12 [Ảnh thực nghiệm]: Hình 1A (985 ô H3) + Hình 1B (Đồ thị tiếp giáp Contiguity) + Hình 2 (Ma trận 8 quan hệ vi mô)
* **Phần 4: Huấn Luyện Mạng GNN & Động Học Hội Tụ Loss (File 03) (Slide 13 – 16):**
  * Slide 13 [Nội dung]: Kiến trúc Mạng dị thể HAN-GAE, Bộ giải mã DistMult & Hàm mục tiêu kép Joint Loss
  * Slide 14 [Ảnh thực nghiệm]: Sơ đồ kiến trúc tổng thể HAN-GAE 2 cấp + Sơ đồ lấy mẫu âm Negative Sampling
  * Slide 15 [Nội dung]: Thiết kế Thực nghiệm Bóc tách 9 Mô hình & Quy luật Hội tụ Loss qua 200 Epochs
  * Slide 16 [Ảnh thực nghiệm]: Đồ thị hội tụ Loss mượt mà Model 4 đề xuất + Đối sánh Loss Model 1, 2, 3, 1S
* **Phần 5: Đánh Giá Phân Cụm Chức Năng & Giải Mã Cơ Chế Chú Ý (File 04) (Slide 17 – 20):**
  * Slide 17 [Nội dung]: Tối ưu hóa Số cụm $k^*=9$, Kiểm định Tính mô-đun Newman Modularity & Bảng 3 Đối chuẩn
  * Slide 18 [Ảnh thực nghiệm]: Đồ thị quét Silhouette đạt đỉnh + Davies-Bouldin + Bảng 3 Benchmark đối chuẩn 9 mô hình
  * Slide 19 [Nội dung]: Giải mã Cơ chế Chú ý Ngữ nghĩa 2 tầng ($\beta$) & Kiểm chứng Định luật 1 Tobler
  * Slide 20 [Ảnh thực nghiệm]: Phân bố trọng số Attention Tầng 1 ($\beta_1$) + Tầng 2 ($\beta_2$) + Đồ thị suy giảm Tobler
* **Phần 6: Khám Phá Không Gian, Kiểm Định HDBSCAN & Ứng Dụng Thực Tiễn (File 05 + HDBSCAN + FinTech) (Slide 21 – 26):**
  * Slide 21 [Nội dung]: Nhận diện Không gian 9 Vùng chức năng Hà Nội & Đóng góp mới: Giải mã bằng Radar Chart (Cell 10)
  * Slide 22 [Ảnh thực nghiệm]: Bản đồ 9 vùng chức năng (Figure 8 / Model 4) + Biểu đồ mạng nhện Radar Chart 16 POIs theo $Z$-score
  * Slide 23 [Nội dung]: Kiểm định Độ bền vững Phi tham số với HDBSCAN & Phân tích Không gian Tiềm ẩn t-SNE 2D
  * Slide 24 [Ảnh thực nghiệm]: Ma trận 3x3 Bản đồ HDBSCAN (Hình C.2) + Không gian t-SNE 2D (Hình C.3b) + Bảng C.2 Thống kê HDBSCAN
  * Slide 25 [Nội dung]: Định hướng Ứng dụng Thực tiễn: FinTech (Site Selection/Bất động sản) & Không gian trạng thái ITS/RL
  * Slide 26 [Ảnh thực nghiệm]: Sơ đồ khung luồng ứng dụng FinTech + Khung tích hợp Reinforcement Learning điều khiển ITS
* **Khối Tổng Kết (Slide 27 – 28):**
  * Slide 27 [Nội dung]: Tổng kết Toàn diện & 3 Đóng góp Kỹ thuật Cốt lõi
  * Slide 28 [Kết thúc]: Phiên Hỏi Đáp (Q&A) & Trao Đổi Khoa Học


---

# NỘI DUNG CHI TIẾT TỪNG SLIDE (SLIDE 1 ĐẾN 28)

---

### SLIDE 1: BÌA BÁO CÁO (TITLE SLIDE)
* **Tiêu đề lớn:** KHAI PHÁ HÌNH THÁI VÙNG CHỨC NĂNG ĐÔ THỊ 15 PHÚT TẠI HÀ NỘI BẰNG MẠNG NƠ-RON ĐỒ THỊ DỊ THỂ (HETEROGENEOUS GNNs)
* **Tiêu đề phụ:** Tái lập, chuyển giao và bản địa hóa khung phương pháp luận Sato et al. (CEUS 2026) cho vùng nội đô lịch sử Thủ đô Hà Nội
* **Người thực hiện:** Lê Đức Tùng -- MSSV: 20227161
* **Giáo viên hướng dẫn:** [Điền tên Thầy hướng dẫn]
* **Đơn vị:** Viện Công nghệ Thông tin & Truyền thông / BKFintech -- Trường Đại học Bách khoa Hà Nội (hoặc VinUni)

---

### SLIDE 2: NỘI DUNG THUYẾT TRÌNH (PRESENTATION AGENDA)
* **Tiêu đề slide:** Nội Dung Thuyết Trình (Khung Tiến Trình Nghiên Cứu 6 Giai Đoạn)
* **Nội dung hiển thị:**
  1. **Đặt Vấn Đề, Bối Cảnh Đô Thị Nén & Khái Niệm 15 Phút** (Thách thức mô hình nhà ống ngõ ngách & Sai số MAUP)
  2. **Xử Lý Dữ Liệu Không Gian & Ma Trận 16 Nhóm POIs** *(File 01: Lưới lục giác H3 & Chuẩn hóa Log-transform)*
  3. **Xây Dựng Đồ Thị Dị Thể Đa Phương Thức 15 Phút** *(File 02: 3 Siêu quan hệ Tiếp giáp, Đi bộ OSM & Buýt GTFS)*
  4. **Kiến Trúc Mạng HAN-GAE & Động Học Hội Tụ Loss** *(File 03: Cơ chế chú ý 2 cấp, DistMult & Hàm Joint Loss)*
  5. **Đánh Giá Phân Cụm Chức Năng & Cơ Chế Chú Ý Attention** *(File 04: Tối ưu $k^*=9$, Newman Modularity & Trọng số $\beta$)*
  6. **Bản Đồ Phân Vùng, Kiểm Định Bền Vững & Ứng Dụng Thực Tiễn** *(File 05 + HDBSCAN + Radar Chart + FinTech/ITS)*
* **Gợi ý thuyết minh:** *"Kính thưa Thầy Cô và Hội đồng, nghiên cứu của em được triển khai theo một tiến trình khoa học khép kín gồm 6 giai đoạn, đi từ cơ sở lý luận đô thị, xử lý dữ liệu lớn không gian, mô hình hóa đồ thị dị thể, huấn luyện mạng học sâu GNN, đến đánh giá phân cụm và kiểm định độ bền vững mô hình."*

---

### KHỐI DANH MỤC THUẬT NGỮ & TỪ VIẾT TẮT (SLIDE 3 – 4)

#### SLIDE 3 [THUẬT NGỮ 1]: DANH MỤC TỪ VIẾT TẮT & THUẬT NGỮ ĐÔ THỊ KHÔNG GIAN
* **Tiêu đề slide:** Danh Mục Từ Viết Tắt & Khái Niệm Đô Thị Không Gian (Spatial & Urban Analytics)
* **Bảng thuật ngữ chuẩn hóa đối chiếu:**
  | Viết tắt | Tên tiếng Anh đầy đủ | Giải thích & Vai trò trong Đề tài |
  | :--- | :--- | :--- |
  | **15-min City** | 15-Minute City | **Đô thị 15 phút:** Mô hình quy hoạch hướng tới việc cư dân tiếp cận toàn bộ nhu cầu thiết yếu trong 15 phút đi bộ hoặc xe buýt. |
  | **POI / POIs** | Point of Interest(s) | **Điểm tiện ích đô thị tầng trệt:** Dữ liệu vị trí điểm của các dịch vụ (quán ăn, cửa hàng, trường học, bệnh viện, v.v.). |
  | **H3 Res 9** | Uber H3 Hexagonal Hierarchical Spatial Index (Resolution 9) | **Lưới lục giác đều H3 phân giải cấp 9:** Chuẩn phân hoạch không gian rời rạc của Uber, mỗi ô có diện tích $\approx 0{,}1\text{ km}^2$, cự ly tâm $\approx 300\text{ m}$. |
  | **MAUP** | Modifiable Areal Unit Problem | **Sai số đơn vị diện tích khả biến:** Lệch lạc thống kê địa lý kinh điển (Openshaw, 1984) do chia ranh giới hành chính tùy tiện, triệt tiêu bằng lưới H3. |
  | **OSM** | OpenStreetMap | **Bản đồ dữ liệu mở:** Nền tảng cung cấp đồ thị mạng lưới đường bộ, ngõ ngách và lối đi bộ vi mô. |
  | **GTFS** | General Transit Feed Specification | **Chuẩn dữ liệu lịch trình giao thông công cộng:** Cung cấp vị trí trạm dừng, tuyến chạy và thời gian chuyến của mạng xe buýt Hà Nội. |
  | **Isochrone** | Isochronic Travel Zone | **Vùng đẳng thời:** Đường bao/vùng không gian có thể tiếp cận được từ một điểm xuất phát trong cùng một khoảng thời gian cố định (ví dụ: $\le 15$ phút). |
  | **OA** | Output Area (Hexagonal Cell) | **Đơn vị phân tích không gian:** Ký hiệu nút ô lục giác H3 trong cấu trúc đồ thị ($N = 985$ ô). |
* **Gợi ý thuyết minh:** *"Trước khi đi vào các phương pháp luận chuyên sâu, em xin phép tóm lược các thuật ngữ nền tảng về địa tin học đô thị. Trong đó, việc chuẩn hóa không gian bằng lưới lục giác Uber H3 cấp 9 đóng vai trò triệt tiêu triệt để sai số đơn vị diện tích khả biến MAUP thường gặp trong thống kê hành chính."*

#### SLIDE 4 [THUẬT NGỮ 2]: DANH MỤC TỪ VIẾT TẮT AI, ĐỒ THỊ DỊ THỂ & ĐÁNG GIÁ KHOA HỌC
* **Tiêu đề slide:** Danh Mục Thuật Ngữ Mạng Nơ-ron Đồ Thị (GNNs) & Chỉ Số Đối Chuẩn Định Lượng
* **Bảng thuật ngữ chuẩn hóa đối chiếu:**
  | Viết tắt | Tên tiếng Anh đầy đủ | Giải thích & Ý nghĩa thuật toán / chỉ số |
  | :--- | :--- | :--- |
  | **GNN** | Graph Neural Network | **Mạng nơ-ron đồ thị:** Họ kiến trúc học sâu học trực tiếp trên dữ liệu có cấu trúc đồ thị phi Euclid (Non-Euclidean). |
  | **HAN-GAE** | Heterogeneous Graph Attention Network - Graph Autoencoder | **Bộ mã hóa tự động đồ thị dị thể có cơ chế chú ý:** Kiến trúc AI cốt lõi dùng cơ chế chú ý 2 tầng (Node-level và Semantic-level) để tạo vector nhúng. |
  | **DistMult** | Distance Multiplicative Decoder | **Bộ giải mã đa quan hệ tích khoảng cách:** Bộ giải mã ma trận đường chéo $R_m$ phục vụ bài toán dự đoán liên kết (Link Prediction). |
  | **Metapath** | Meta-path (Siêu quan hệ) | Chuỗi liên kết ngữ nghĩa giữa các nút qua các loại cạnh quan hệ khác nhau (Tiếp giáp hình học, Đi bộ 15 phút, Buýt 15 phút). |
  | **$Q_{\text{contig}}$ / $Q_{\text{walk}}$** | Newman's Graph Modularity | **Tính mô-đun Newman:** Thước đo định lượng cấu trúc phân cụm đồ thị; điểm $Q > 0{,}4$ chứng minh cụm phân vùng tạo thành mảng liền khoảnh tự nhiên. |
  | **HDBSCAN** | Hierarchical Density-Based Spatial Clustering of Applications with Noise | **Thuật toán phân cụm phân cấp theo mật độ:** Phương pháp phân cụm phi tham số dùng để kiểm định độ bền vững, tự bóc tách điểm nhiễu ngoại vi (`-1`). |
  | **DBCV** | Density-Based Clustering Validation | **Độ đo hợp lệ phân cụm mật độ:** Thước đo đánh giá chất lượng phân cụm chuyên dụng cho HDBSCAN, dao động từ $[-1, 1]$. |
  | **t-SNE** | t-Distributed Stochastic Neighbor Embedding | **Phương pháp giảm chiều phi tuyến:** Kỹ thuật chiếu vector nhúng 32 chiều xuống không gian 2D để trực quan hóa cấu trúc các đảo cụm. |
* **Gợi ý thuyết minh:** *"Về phương diện trí tuệ nhân tạo và khoa học dữ liệu, mô hình HAN-GAE đóng vai trò bộ mã hóa tự động dị thể 2 tầng, kết hợp bộ giải mã DistMult và kiểm định tính liền khối không gian thông qua chỉ số Newman Modularity cùng thuật toán mật độ HDBSCAN."*

---

### MỤC 1: ĐẶT VẤN ĐỀ, BỐI CẢNH ĐÔ THỊ NÉN & KHÁI NIỆM 15 PHÚT

#### SLIDE 5 [NỘI DUNG]: BỐI CẢNH ĐÔ THỊ NÉN HÀ NỘI & KHÁI NIỆM ĐÔ THỊ 15 PHÚT
* **Khái niệm Đô thị 15 phút (Carlos Moreno et al., 2021):**
  * **Mục tiêu nhân bản:** Cư dân tiếp cận toàn bộ 6 nhu cầu thiết yếu (ở, làm việc, mua sắm, y tế, giáo dục, giải trí) trong vòng $\le 15$ phút di chuyển.
  * **Đa phương thức:** Tích hợp năng lực đi bộ vi mô và mạng lưới giao thông công cộng liên vùng.
  * **Xu thế hiện đại:** Chuyển dịch từ quy hoạch phụ thuộc ô tô sang mô hình lấy con người và khả năng tiếp cận làm trung tâm.
* **Nghịch lý hình thái đô thị nén Hà Nội:**
  * **Mô hình nhà ống hỗn hợp:** Tầng trệt kinh doanh thương mại/dịch vụ, các tầng trên sinh hoạt gia đình $\rightarrow$ Ranh giới chức năng đan cài phức tạp, không phân vùng đơn lẻ như phương Tây.
  * **Mạng lưới ngõ ngách chi phối:** Ngõ nhỏ, ngách sâu chiếm hơn $60\%$ lưu thông nội đô; cự ly đường chim bay hoàn toàn không phản ánh đúng thời gian tiếp cận thực tế.
  * **Bất cập quy hoạch tĩnh:** Bản đồ địa chính phân khu chức năng (Zoning) cứng nhắc không theo kịp nhịp sống kinh tế tầng trệt năng động.
* **Câu hỏi nghiên cứu cốt lõi:** Làm thế nào để định lượng hóa và giải mã chính xác hình thái chức năng đô thị nén từ dữ liệu lớn tiện ích POIs và mạng lưới tiếp cận đa phương thức thực tế?
* **Gợi ý thuyết minh:** *"Quy hoạch truyền thống chia vùng cứng nhắc đã bộc lộ nhiều bất cập trước một đô thị nén sống động như Hà Nội, nơi hàng quán ngõ ngách đan cài với nhà ở. Đề tài đặt mục tiêu giải mã ranh giới chức năng thực tế dựa trên khả năng tiếp cận 15 phút."*

#### SLIDE 6 [NỘI DUNG]: THÁCH THỨC SAI SỐ MAUP & GIẢI PHÁP CHUẨN HÓA LƯỚI UBER H3 RES 9
* **Thách thức Sai số Đơn vị Diện tích Khả biến (MAUP - Openshaw, 1984):**
  * **Hiệu ứng quy mô (Scale Effect):** Diện tích giữa 27 phường nội đô chênh lệch từ 5 đến 10 lần, gây méo mó nghiêm trọng mật độ dịch vụ thống kê.
  * **Hiệu ứng phân vùng (Zoning Effect):** Đường phân giới hành chính cắt ngang qua các lưu vực đi bộ tự nhiên của người dân.
  * **Lệch trọng tâm (Centroid Shift):** Trọng tâm hình học của phường thường rơi vào khoảng trống, xa rời các cụm tiện ích tầng trệt.
* **Giải pháp chuẩn hóa Lưới lục giác Uber H3 Resolution 9:**
  * **Phân hoạch đồng nhất:** Toàn bộ vùng lõi $113{,}22\text{ km}^2$ được chia đều thành **985 ô lục giác đều H3**, triệt tiêu hoàn toàn sai số MAUP.
  * **Tính đẳng hướng không gian (Isotropy):** Mỗi ô H3 có bán kính $\approx 174\text{ m}$, khoảng cách giữa các tâm $\approx 300\text{ m}$; 6 ô láng giềng kề sát có cự ly hoàn hảo bằng nhau.
  * **Khung đối chuẩn quốc tế 1:1:** Khớp tương đương quy mô nghiên cứu ca Liverpool của Sato et al. (CEUS 2026: $111{,}84\text{ km}^2$), tạo tiền đề kiểm chứng tính tổng quát hóa xuyên quốc gia của mô hình GNN.
* **Gợi ý thuyết minh:** *"Để xóa bỏ triệt để sai số MAUP của ranh giới phường xã, chúng em chuẩn hóa không gian nghiên cứu thành 985 ô lục giác H3 đồng nhất, tạo hạ tầng không gian chuẩn cho đồ thị dị thể."*

---

### MỤC 2: XỬ LÝ DỮ LIỆU KHÔNG GIAN & MA TRẬN 16 POIs (FILE 01)

#### SLIDE 7 [NỘI DUNG]: KHAI THÁC 152.000 POIs TẦNG TRỆT & KỸ THUẬT CHUẨN HÓA LOGARIT
* **Khai thác và phân loại tiện ích tầng trệt (POIs):**
  * Thu thập 152.120 điểm tiện ích từ nền tảng dữ liệu mở Overture Maps và OpenStreetMap.
  * Phân loại chuẩn hóa vào **16 nhóm chức năng cố định** theo Bảng A.1 trong bài báo gốc CEUS 2026:
    * *Thương mại & Ẩm thực:* F&B (`food_beverage`), Bán lẻ (`retail`), Giải trí (`entertainment`).
    * *Dịch vụ công & Thiết yếu:* Y tế (`healthcare`), Giáo dục (`education`), Dịch vụ công (`public_service`).
    * *Văn phòng & Công nghiệp:* Dịch vụ doanh nghiệp (`corporate_service`), Dịch vụ tiêu dùng (`consumer_service`), Công nghiệp nhẹ (`industrial_service`).
    * *Hạ tầng & Không gian xanh:* Khách sạn lưu trú (`hotel_lodging`), Công viên (`park`), Cơ sở thể thao (`sports_fitness`), Cơ sở giao thông (`transportation_facility`)...
* **Kỹ thuật xử lý phân phối lệch đuôi dài (Heavy-tailed Distribution):**
  * Mật độ POIs tập trung cực đoan tại vùng lõi Hoàn Kiếm $\rightarrow$ Gây hiện tượng bão hòa độ dốc (gradient vanishing/exploding) trong mạng nơ-ron.
  * **Giải pháp:** Áp dụng phép biến đổi logarit:
    $$X_{\text{norm}} = \log(1 + X_{\text{raw}})$$
  * Đưa ma trận đặc trưng $X \in \mathbb{R}^{985 \times 16}$ về dạng phân phối gần chuẩn Gaussian, giúp mạng GNN học biểu diễn ổn định.
* **Gợi ý thuyết minh:** *"Tại Bước 01, dữ liệu 152.000 POIs được làm sạch và gom vào đúng 16 nhóm chức năng. Phép biến đổi Log-transform đóng vai trò quyết định giúp triệt tiêu hiện tượng lệch cực trị."*

#### SLIDE 8 [ẢNH THỰC NGHIỆM]: BẢN ĐỒ TIỆN ÍCH TẦNG TRỆT & PHÂN PHỐI ĐẶC TRƯNG CHUẨN HÓA
* **Bố cục 3 ảnh trực quan (Từ `data/outputs/hanoi/figures`):**
  * **Ảnh 1 (Trái):** `hanoi_total_poi_density_fig10a.jpg` (Bản đồ phân bố tổng mật độ POIs trên 985 ô H3 nội đô Hà Nội).
  * **Ảnh 2 (Giữa):** `16_poi_categories.png` (Ma trận 16 bản đồ nhiệt phân bố không gian của từng loại hình tiện ích vi mô).
  * **Ảnh 3 (Phải):** `hanoi_raw_features_histogram.png` ghép cạnh `hanoi_scaled_features_histogram.png` (Biểu đồ phân phối thô lệch cực trị vs Phân phối mượt mà sau chuẩn hóa Log).
* **Điểm cần chỉ tay trên ảnh:**
  * *Ảnh 1:* Vùng màu vàng/đỏ rực tập trung quanh hồ Hoàn Kiếm và trục Phố Cổ (mật độ $> 800\text{ POIs/ô H3}$).
  * *Ảnh 2:* Các tiện ích chuyên biệt như `industrial_service` dạt ra ngoại vi phía Nam (Hoàng Mai), trong khi `hotel_lodging` và `food_beverage` ôm trọn quận Hoàn Kiếm.
  * *Ảnh 3:* Đồ thị sau biến đổi Log chuyển từ dạng chữ L lệch đuôi dài sang dạng phân phối hình chuông cân đối.
* **Thông điệp chốt hạ (Key Takeaway):** *"Dữ liệu tầng trệt phản ánh trung thực nhịp sống kinh tế vi mô; chuẩn hóa Log đưa miền giá trị về trạng thái tối ưu cho huấn luyện mạng GNN."*

---

### MỤC 3: XÂY DỰNG ĐỒ THỊ DỊ THỂ ĐA PHƯƠNG THỨC 15 PHÚT (FILE 02)

#### SLIDE 9 [NỘI DUNG]: MÔ HÌNH HÓA MẠNG LƯỚI GIAO THÔNG VI MÔ & VÙNG ĐẲNG THỜI
* **Định nghĩa không gian 15 phút đa phương thức:**
  * Ranh giới tiếp cận đô thị được định hình bởi **thời gian di chuyển đẳng thời (Isochronic Travel Time)** chứ không phải khoảng cách Euclidean.
* **Mạng đường đi bộ ngõ ngách OSM (15-min walk):**
  * Khai thác toàn bộ mạng lưới đường bộ, vỉa hè và ngõ nhỏ Hà Nội từ OpenStreetMap (OSM).
  * Thiết lập ngưỡng tiếp cận đi bộ vi mô $T_{\text{walk}} \le 900\text{ s}$ (15 phút) với vận tốc đi bộ thực tế trung bình $v = 4{,}5\text{ km/h}$.
* **Mạng xe buýt công cộng GTFS (15-min multi):**
  * Khai thác lộ trình và lịch trình từ dữ liệu chuẩn GTFS xe buýt Hà Nội.
  * Mô hình hóa hành trình đa phương thức: Đi bộ ra trạm đón ($T \le 300\text{ s}$) + Thời gian chờ và đi trên xe buýt ($T \le 600\text{ s}$).
* **Năng lực khuếch đại tiếp cận của giao thông công cộng:**
  * Xe buýt mở rộng bán kính tiếp cận từ $\approx 1{,}2\text{ km}$ (đi bộ) lên tới $\ge 3{,}5\text{ km}$, làm tăng diện tích lưu vực tiếp cận gấp 6.2 lần.
* **Gợi ý thuyết minh:** *"Khả năng tiếp cận 15 phút tại Hà Nội được kết hợp bởi 2 cấu trúc: mạng ngõ nhỏ phục vụ đi bộ hàng ngày và mạng buýt công cộng kết nối xuyên vùng."*

#### SLIDE 10 [ẢNH THỰC NGHIỆM]: BẢN ĐỒ ĐƯỜNG BỘ OSM, BUÝT GTFS & VÙNG ĐẲNG THỜI
* **Bố cục 3 ảnh trực quan:**
  * **Ảnh 1 (Trái):** `hanoi_street_network.png` (Mạng lưới đường bộ OSM vi mô len lỏi qua từng ngõ ngách của 27 phường).
  * **Ảnh 2 (Giữa):** `hanoi_gtfs_travel_graph.png` (Đồ thị các tuyến xe buýt và điểm trung chuyển GTFS Hà Nội).
  * **Ảnh 3 (Phải):** `isochrones.jpg` (Bản đồ đối sánh vùng đẳng thời 15 phút: Thuần đi bộ vs Xe buýt mở rộng diện tích gấp 6.2 lần).
* **Điểm cần chỉ tay trên ảnh:**
  * *Ảnh 1:* Cấu trúc mạng đường dạng mạng nhện đậm đặc ở trung tâm Hoàn Kiếm, Đống Đa và thưa dần ra ven sông Hồng.
  * *Ảnh 2:* Các trục xương sống giao thông buýt chạy dọc theo đường trục Giải Phóng, Nguyễn Trãi, Giảng Võ.
  * *Ảnh 3:* Đốm xanh nhỏ là phạm vi đi bộ 15 phút; vùng đệm màu đỏ/cam rộng lớn là phạm vi tiếp cận nhờ xe buýt từ cùng một tâm điểm.
* **Thông điệp chốt hạ (Key Takeaway):** *"Xe buýt đóng vai trò phá vỡ rào cản cự ly, kết nối các vùng dân sinh ngoại vi vào lõi tiện ích trung tâm trong ngưỡng 15 phút."*

#### SLIDE 11 [NỘI DUNG]: CẤU TRÚC ĐỒ THỊ DỊ THỂ PyG: 3 SIÊU QUAN HỆ & 8 QUAN HỆ VI MÔ
* **Tích hợp 3 siêu quan hệ cạnh không gian (Metapaths):**
  1. **Quan hệ kề cận (Contiguity):** 5.550 cạnh có hướng kết nối các cặp ô H3 tiếp giáp trực tiếp đường biên hình học.
  2. **Quan hệ đi bộ vi mô 15 phút (15-min walk):** 31.040 cạnh kết nối các ô H3 đến được nhau trong 15 phút đi bộ.
  3. **Quan hệ buýt đa phương thức 15 phút (15-min multi):** 4.898 cạnh kết nối các ô H3 thông qua mạng lưới xe buýt công cộng.
* **Đóng gói đối tượng HeteroData trong PyTorch Geometric (`hetero_multi.pt`):**
  * Tích hợp đầy đủ **8 quan hệ vi mô** liên kết giữa ô đất (`oa`), nút giao thông (`street_connector`) và trạm buýt (`transit_stop`).
  * Thực hiện thuật toán lọc bộ ba đồng bộ *(node, edge, attribute)* đảm bảo đồ thị liên thông 100%, triệt tiêu nút cô lập.
* **Gợi ý thuyết minh:** *"Đồ thị `hetero_multi.pt` của chúng em có hơn 41.000 cạnh liên kết, phản ánh đầy đủ cả cấu trúc hình học biên giới lẫn động thái di chuyển thực tế của người dân."*

#### SLIDE 12 [ẢNH THỰC NGHIỆM]: MA TRẬN 8 SIÊU QUAN HỆ KHÔNG GIAN DỊ THỂ
* **Bố cục 3 ảnh trực quan:**
  * **Ảnh 1 (Trái):** `hanoi_contig_oa.jpg` (Hình 1A: 985 đa giác H3 cùng tọa độ các tâm điểm centroid).
  * **Ảnh 2 (Giữa):** `hanoi_contig_edges.jpg` (Hình 1B: Đồ thị tiếp giáp kề cận với thanh màu thời gian di chuyển).
  * **Ảnh 3 (Phải):** `hanoi_fig2_hetero_8relations_matrix.png` (Hình 2 trong bài báo: Ma trận trực quan hóa cấu trúc 8 quan hệ vi mô trong PyG).
* **Điểm cần chỉ tay trên ảnh:**
  * *Ảnh 1:* Các ô lục giác H3 bao phủ khép kín 27 phường nội đô.
  * *Ảnh 2:* Các cạnh kết nối tiếp giáp giữa các ô có thời gian di chuyển đồng đều $\approx 4\text{ phút}$.
  * *Ảnh 3:* Ma trận thể hiện cấu trúc đồ thị dị thể liên kết giữa 3 loại nút: ô H3, nút đường bộ và điểm dừng xe buýt.
* **Thông điệp chốt hạ (Key Takeaway):** *"Khung đồ thị dị thể đa quan hệ nắm bắt trọn vẹn sự tương tác không gian đa tầng của vùng đô thị nén Hà Nội."*

---

### MỤC 4: HUẤN LUYỆN MẠNG GNN & HỘI TỤ LOSS (FILE 03)

#### SLIDE 13 [NỘI DUNG]: KIẾN TRÚC MẠNG HAN-GAE, BỘ GIẢI MÃ DISTMULT & JOINT LOSS
* **Kiến trúc Heterogeneous Graph Attention Network (HAN-GAE):**
  * **Cấp 1 - Node-level Attention (Edge-Aware GAT):** Tính trọng số chú ý cục bộ $\alpha_{ij}^{(m)}$ giữa các ô láng giềng trong từng loại quan hệ $m$.
  * **Cấp 2 - Semantic-level Attention ($\beta_m$):** Tự động học tỷ trọng đóng góp của từng lớp mạng (Kề cận vs Đi bộ vs Xe buýt).
* **Bộ giải mã DistMult & Hàm mục tiêu liên hợp (Joint Loss):**
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{link}} + \lambda_{\text{struct}} \mathcal{L}_{\text{struct}}$$
  * $\mathcal{L}_{\text{link}}$: Cross-entropy dự đoán liên kết đa quan hệ (Link Prediction) với negative sampling ratio = 1.0.
  * $\mathcal{L}_{\text{struct}}$: Tái cấu trúc ma trận bậc nút (Degree Reconstruction) với $\lambda_{\text{struct}} = 1.0$.
* **Không gian vector tiềm ẩn $Z \in \mathbb{R}^{985 \times 32}$:** Nén toàn bộ 16 thuộc tính POIs và 3 lớp đồ thị vào vector nhúng 32 chiều được chuẩn hóa mặt cầu L2 đơn vị ($||z||_2 = 1$).
* **Gợi ý thuyết minh:** *"Mạng HAN-GAE sử dụng cơ chế chú ý 2 tầng để nén dữ liệu đa tầng thành vector nhúng 32 chiều. Hàm Joint Loss đảm bảo vector vừa lưu giữ ngữ nghĩa POIs vừa bảo toàn cấu trúc mạng lưới."*

#### SLIDE 14 [ẢNH THỰC NGHIỆM]: SƠ ĐỒ KIẾN TRÚC TỔNG THỂ MẠNG DỊ THỂ HAN-GAE
* **Bố cục 2 sơ đồ trực quan:**
  * **Sơ đồ 1 (Trái):** Luồng chuyển đổi dữ liệu kiến trúc HAN-GAE: *Đầu vào 16 POIs + 3 Metapaths $\rightarrow$ Edge-Aware GAT (Layer 1 & 2) $\rightarrow$ Semantic Attention Pooling $\rightarrow$ Vector nhúng $Z \in \mathbb{R}^{32}$.*
  * **Sơ đồ 2 (Phải):** Sơ đồ bộ giải mã DistMult và cơ chế lấy mẫu cạnh âm (Negative Sampling) để tính toán Joint Loss.
* **Điểm cần chỉ tay trên ảnh:**
  * *Sơ đồ 1:* Hai tầng attention: Tầng 1 học trong từng đồ thị con; Tầng 2 gộp trọng số $\beta$ qua phép pooling ngữ nghĩa.
  * *Sơ đồ 2:* Ma trận đường chéo $R_m$ của DistMult cho phép học mối liên kết riêng biệt cho từng loại quan hệ.
* **Thông điệp chốt hạ (Key Takeaway):** *"Cơ chế chú ý hai cấp cho phép mô hình học được cả tương tác vi mô ngõ ngách lẫn cấu trúc giao thông vĩ mô toàn đô thị."*

#### SLIDE 15 [NỘI DUNG]: THIẾT KẾ THỰC NGHIỆM BÓC TÁCH 9 MÔ HÌNH & QUY LUẬT HỘI TỤ LOSS
* **Ma trận thiết kế thực nghiệm bóc tách (Ablation Study Matrix):**
  * **Baseline (PCA):** Giảm chiều tuyến tính thuần túy từ 16 POIs thô, không dùng đồ thị.
  * **Model 1 (GAT-GAE):** Đồ thị đơn thể thuần kề cận (Contiguity).
  * **Model 2 (HAN-GAE: Walk):** Đồ thị dị thể kết hợp Kề cận + Đi bộ 15 phút.
  * **Model 3 (HAN-GAE: Multi):** Đồ thị dị thể kết hợp Kề cận + Xe buýt 15 phút.
  * **Model 4 (HAN-GAE: All - Đề xuất):** Tích hợp trọn vẹn cả 3 lớp quan hệ không gian.
  * **Các mô hình Struct-Only (1S, 2S, 3S, 4S):** Triệt tiêu thuộc tính 16 POIs để kiểm chứng vai trò của dữ liệu tầng trệt.
* **Quy luật hội tụ nghiêm ngặt qua 200 Epochs:**
  * Sử dụng Adam optimizer ($lr = 0.005$), Early stopping patience = 100.
  * Thứ bậc hội tụ: $\mathcal{L}_{\text{Model 4}} (0{,}6517) < \mathcal{L}_{\text{Model 2}} (0{,}7042) < \mathcal{L}_{\text{Model 3}} (0{,}8124) < \mathcal{L}_{\text{Model 1}} (0{,}9594)$.
* **Gợi ý thuyết minh:** *"Chúng em thiết kế 9 mô hình đối chuẩn nghiêm ngặt. Kết quả thực nghiệm cho thấy Model 4 tích hợp đủ 3 quan hệ đạt hàm mất mát thấp nhất và hội tụ ổn định nhất."*

#### SLIDE 16 [ẢNH THỰC NGHIỆM]: ĐỒ THỊ ĐỐI CHUẨN ĐỘNG HỌC HỘI TỤ LOSS QUA 200 EPOCHS
* **Bố cục lưới 4 đồ thị Loss (Từ `data/outputs/hanoi/figures`):**
  * **Góc trên trái:** `hanoi_loss_progress_model4_han_gae_all.png` (Model 4 đề xuất: Hội tụ mượt mà, Loss thấp nhất 0.6517).
  * **Góc trên phải:** `hanoi_loss_progress_model2_han_gae_walk.png` (Model 2: Hội tụ tốt trên mạng đi bộ, Loss 0.7042).
  * **Góc dưới trái:** `hanoi_loss_progress_model1_gat_gae.png` (Model 1 đơn thể: Loss cao 0.9594).
  * **Góc dưới phải:** `hanoi_loss_progress_model1s_gat_gae_struct.png` (Model 1S triệt tiêu POI: Dao động mạnh, mất ổn định).
* **Điểm cần chỉ tay trên ảnh:**
  * Đường cong Model 4 dốc xuống nhanh trong 50 epoch đầu và đi ngang ổn định đến epoch 200.
  * Đồ thị Model 1S bị rung lắc mạnh, chứng minh rằng nếu không có thuộc tính POIs tầng trệt, mạng GNN rất khó nắm bắt cấu trúc đô thị.
* **Thông điệp chốt hạ (Key Takeaway):** *"Mô hình đa phương thức Model 4 có tốc độ hội tụ nhanh nhất và đạt hàm mất mát cực tiểu; dữ liệu 16 POIs là điều kiện tiên quyết giúp gradient ổn định."*

---

### MỤC 5: ĐÁNH GIÁ PHÂN CỤM & CƠ CHẾ CHÚ Ý (FILE 04)

#### SLIDE 17 [NỘI DUNG]: TỐI ƯU HÓA $k^* = 9$, KIỂM ĐỊNH MODULARITY & BẢNG 3 ĐỐI CHUẨN
* **Tối ưu hóa số cụm phân vùng K-Means ($k^* = 9$):**
  * Quét dải siêu tham số $k \in [2, 50]$, chạy lặp lại qua 10 hạt giống ngẫu nhiên (random seeds = 42).
  * Điểm Silhouette đạt đỉnh cực đại duy nhất tại **$k^* = 9$ ($0{,}36 \pm 0{,}01$)**, Calinski-Harabasz đạt đỉnh nhọn và Davies-Bouldin đạt giá trị nhỏ nhất ($0{,}87$).
* **Bảng 3 Đối chuẩn định lượng (Table 3 Benchmark):**
  * Model 4 áp đảo toàn diện: Silhouette $= 0{,}36$, Davies-Bouldin $= 0{,}87$, Modularity $Q_{\text{contig}} = 0{,}63$, $Q_{\text{walk}} = 0{,}58$.
  * Trong khi đó, Baseline PCA chỉ đạt $Q_{\text{contig}} = 0{,}38$; GAT đơn thể chỉ đạt $Q_{\text{contig}} = 0{,}54$.
* **Ý nghĩa tính mô-đun không gian (Newman Modularity):**
  * Điểm $Q$ vượt trội chứng minh các ô trong cùng cụm gắn kết chặt chẽ và tạo thành các mảng liền khoảnh tự nhiên, không bị xé vụn hình học.
* **Gợi ý thuyết minh:** *"Toàn bộ các chỉ số toán học độc lập đều đồng thuận xác nhận cấu trúc nội đô Hà Nội được phân định tối ưu nhất ở 9 phân vùng chức năng."*

#### SLIDE 18 [ẢNH THỰC NGHIỆM]: BIỂU ĐỒ TỐI ƯU HÓA $k$ & BẢNG 3 ĐỐI CHUẨN KHOA HỌC
* **Bố cục 3 ảnh/bảng trực quan:**
  * **Ảnh 1 (Trái):** `silhouette.jpg` (Đồ thị điểm Silhouette đạt đỉnh sắc nét tại $k=9$).
  * **Ảnh 2 (Giữa):** `davies_bouldin.jpg` ghép cạnh `calinski_harabasz.jpg` (Các chỉ số bổ trợ đồng thuận xác nhận $k=9$).
  * **Bảng 3 (Phải):** Bảng số liệu đối chuẩn 9 mô hình (Silhouette, DB, Modularity $Q$ trên 3 mạng lưới).
* **Điểm cần chỉ tay trên ảnh:**
  * *Ảnh 1:* Điểm Silhouette từ $k=2$ tăng dần và đạt đỉnh nhọn tại $k=9$, sau đó suy giảm khi $k > 15$.
  * *Ảnh 2:* Điểm Davies-Bouldin rơi xuống đáy thấp nhất tại $k=9$ (cụm càng tách biệt thì DB càng nhỏ).
  * *Bảng 3:* Cột $Q_{\text{contig}}$ của Model 4 đạt $0{,}63$ -- mức tính mô-đun rất cao trong lý thuyết đồ thị không gian.
* **Thông điệp chốt hạ (Key Takeaway):** *"Số cụm tối ưu $k^*=9$ là kết quả khách quan rút ra từ phân tích toán học chứ không phải do con người áp đặt chủ quan."*

#### SLIDE 19 [NỘI DUNG]: GIẢI MÃ CHÚ Ý ATTENTION $\beta$ & KIỂM CHỨNG ĐỊNH LUẬT 1 TOBLER
* **Giải mã động học cơ chế chú ý ngữ nghĩa ($\beta$):**
  * **Tầng 1 (Cục bộ vi mô):** Mạng đi bộ chiếm tỷ trọng áp đảo **$65{,}13\% \pm 25{,}33\%$**, mạng tiếp giáp chiếm $24{,}75\%$, xe buýt chiếm $10{,}12\%$.
  * $\rightarrow$ *Phát hiện khoa học:* Khả năng tiếp cận đi bộ ngõ ngách là yếu tố chi phối bản chất phân vùng chức năng tại Hà Nội.
  * **Tầng 2 (Làm mịn không gian):** Trọng số được phân bổ cân bằng hơn giữa đi bộ và xe buýt nhằm kết nối các trung tâm lớn.
* **Kiểm chứng Định luật 1 Địa lý học của Tobler:**
  * Độ tương đồng cosine giữa các vector nhúng $Z_i, Z_j$ suy giảm đơn điệu theo thời gian di chuyển (Distance Decay Effect).
* **Gợi ý thuyết minh:** *"Cơ chế chú ý của HAN-GAE mang tính diễn giải rất cao: nó chứng minh mạng đi bộ ngõ ngách chi phối tới hơn 65% nhịp sống chức năng của Thủ đô."*

#### SLIDE 20 [ẢNH THỰC NGHIỆM]: BIỂU ĐỒ ATTENTION $\beta$ & ĐỒ THỊ SUY GIẢM TOBLER
* **Bố cục 3 ảnh trực quan:**
  * **Ảnh 1 (Trái):** `attention_weight_layer_1.jpg` (Biểu đồ phân phối trọng số chú ý Tầng 1: Mạng đi bộ chiếm ưu thế tuyệt đối).
  * **Ảnh 2 (Giữa):** `attention_weight_layer_2.jpg` (Biểu đồ phân phối trọng số chú ý Tầng 2: Cân bằng ngữ nghĩa vĩ mô).
  * **Ảnh 3 (Phải):** `similarity_decay_by_time_bin.jpg` (Đồ thị kiểm chứng quy luật suy giảm tương đồng Tobler theo các lát cắt thời gian di chuyển).
* **Điểm cần chỉ tay trên ảnh:**
  * *Ảnh 1:* Cột `15_min_walk` vọt cao vượt trội so với `contig` và `15_min_multi`.
  * *Ảnh 3:* Đường biểu diễn độ tương đồng dốc xuống tuyến tính từ mốc 0 phút đến 30 phút, kiểm chứng hoàn hảo định luật Tobler.
* **Thông điệp chốt hạ (Key Takeaway):** *"Mạng HAN-GAE tái hiện chính xác quy luật địa lý khách quan: Các ô càng gần nhau về thời gian di chuyển thì càng có độ tương đồng chức năng cao."*

---

### MỤC 6: PHÂN VÙNG CHỨC NĂNG, KIỂM ĐỊNH HDBSCAN & ỨNG DỤNG (FILE 05 + HDBSCAN + FINTECH)

#### SLIDE 21 [NỘI DUNG]: HÌNH THÁI 9 VÙNG CHỨC NĂNG & GIẢI MÃ BẰNG RADAR CHART
* **Nhận diện không gian 9 vùng chức năng đô thị Hà Nội (Model 4):**
  * **Cụm 0 (Lõi lịch sử Hoàn Kiếm CBD):** Tách biệt hoàn toàn, chi phối bởi dịch vụ ẩm thực, du lịch và bán lẻ cao cấp.
  * **Cụm 1 & 2 (Ba Đình - Đống Đa):** Trục hành chính công quyền, cơ quan đầu não, ngoại giao và giáo dục đại học.
  * **Cụm 6 & 7 (Hoàng Mai - Hai Bà Trưng mở rộng):** Vùng dân sinh mật độ cao, thương mại truyền thống và công nghiệp nhẹ.
* **Đóng góp mới: Giải mã định lượng bằng Radar Chart (Cell 10 mới):**
  * Chuẩn hóa $Z$-score 16 nhóm POIs theo từng cụm so với mức trung bình toàn đô thị ($Z=0$).
  * Nhận diện chính xác cụm đơn chức năng chuyên biệt (Spike nhọn vọt cao) vs Cụm đa chức năng hỗn hợp (Đa giác phân bố đều).
* **Gợi ý thuyết minh:** *"Bước 05 kết xuất bản đồ 9 vùng chức năng. Đặc biệt, thông qua biểu đồ Radar Chart mới phát triển, chúng em giải mã định lượng được bản chất kinh tế xã hội của từng cụm dựa trên độ lệch chuẩn Z-score."*

#### SLIDE 22 [ẢNH THỰC NGHIỆM]: BẢN ĐỒ 9 VÙNG CHỨC NĂNG & BIỂU ĐỒ RADAR 16 POIS
* **Bố cục 2 ảnh lớn chất lượng cao:**
  * **Ảnh 1 (Trái):** `cluster_maps.jpg` (Bản đồ phân vùng chức năng 9 mô hình, làm nổi bật Model 4 liền khoảnh mạch lạc).
  * **Ảnh 2 (Phải):** `hanoi_cluster_poi_radar_chart.jpg` (Biểu đồ mạng nhện Radar Chart phân tích hồ sơ 16 nhóm POIs của 9 cụm).
* **Điểm cần chỉ tay trên ảnh:**
  * *Ảnh 1:* Bản đồ Model 4 có ranh giới các cụm liền khối tự nhiên, không có hiện tượng "da báo" lốm đốm như Baseline PCA.
  * *Ảnh 2:* Cụm 0 có đỉnh nhọn vọt ra ngoài biên ở các trục `Retail`, `Food & Beverage`, `Hotel Lodging`; Cụm 1 vọt nhọn ở `Public Service` và `Education`.
* **Thông điệp chốt hạ (Key Takeaway):** *"Bản đồ Model 4 tạo thành các phân vùng liền khoảnh tự nhiên; biểu đồ Radar Chart giải mã chính xác bản chất kinh tế - xã hội của từng cụm."*

#### SLIDE 23 [NỘI DUNG]: KIỂM ĐỊNH BỀN VỮNG PHI THAM SỐ VỚI HDBSCAN & t-SNE 2D
* **Kiểm định độ bền vững với HDBSCAN (Appendix C):**
  * Khắc phục giả định cụm hình cầu của K-Means; thuật toán mật độ **HDBSCAN** nhận diện cụm có hình thái bất kỳ và bóc tách các điểm nhiễu ngoại vi (`label = -1`).
  * Quét dải siêu tham số $MCS \in [5, 30]$, tối ưu hóa theo chỉ số hợp lệ mật độ DBCV.
* **Độ bền vững của mô hình đề xuất (Bảng Table C.2):**
  * Model 4 giữ vững cấu trúc phân vùng ổn định với tỷ lệ điểm nhiễu thấp ($18{,}2\%$) và điểm DBCV cao ($0{,}17 \pm 0{,}08$).
* **Không gian tiềm ẩn t-SNE 2D:**
  * Các cụm hình thành những "hòn đảo" tách biệt rõ rệt, triệt tiêu hoàn toàn hiện tượng sụp đổ biểu diễn (Representation Collapse).
* **Gợi ý thuyết minh:** *"Kiểm định HDBSCAN khẳng định 9 vùng chức năng là các thực thể mật độ khách quan chứ không phải sản phẩm nhân tạo của K-Means; các điểm nhiễu xám nằm ở dải đệm ven sông Hồng."*

#### SLIDE 24 [ẢNH THỰC NGHIỆM]: MA TRẬN BẢN ĐỒ HDBSCAN & KHÔNG GIAN NHÚNG t-SNE 2D
* **Bố cục 3 ảnh/bảng trực quan:**
  * **Ảnh 1 (Trái):** `cluster_maps_hdbscan.jpg` (Hình C.2: Ma trận $3 \times 3$ bản đồ HDBSCAN với điểm nhiễu màu xám `#CCCCCC`).
  * **Ảnh 2 (Giữa):** `tsne_embeddings_hdbscan.jpg` (Hình C.3b: Chiếu t-SNE 2D đồng bộ màu sắc với bản đồ địa lý).
  * **Ảnh 3 (Phải):** Bảng C.2 Thống kê phân cụm HDBSCAN (`cluster_stats_hdbscan.csv`: $Mean \pm SD$ qua dải $MCS$).
* **Điểm cần chỉ tay trên ảnh:**
  * *Ảnh 1:* Các điểm màu xám `#CCCCCC` (nhiễu) bị đẩy ra vùng ven sông Hồng hoặc dải đất chuyển tiếp, giữ lại lõi trung tâm thuần nhất.
  * *Ảnh 2:* Các cụm màu trên không gian t-SNE 2D gom thành từng đám cô đọng, tách rời nhau rõ rệt.
* **Thông điệp chốt hạ (Key Takeaway):** *"Không gian t-SNE tách đảo sắc nét; HDBSCAN lọc bỏ các ô đệm không thuần nhất, giữ lại lõi chức năng cô đọng nhất."*

#### SLIDE 25 [NỘI DUNG]: ĐỊNH HƯỚNG ỨNG DỤNG THỰC TIỄN: FINTECH, LOGISTICS & ITS/RL
* **Ứng dụng FinTech & Định vị Chi nhánh (Site Selection):**
  * Vector nhúng $Z \in \mathbb{R}^{32}$ lượng hóa toàn diện giá trị kinh tế và mức độ tiếp cận dịch vụ của từng ô đất $\rightarrow$ Làm đầu vào cho mô hình AI định vị chi nhánh ngân hàng, cây ATM, thẩm định rủi ro cho vay thế chấp bất động sản.
* **Tối ưu hóa Logistics chặng cuối (Last-mile Delivery):**
  * Nhận diện các "vùng trũng dịch vụ" và nút thắt cổ chai ngõ ngách để quy hoạch trạm gom hàng và tủ khóa thông minh (Smart Lockers).
* **Tiềm năng kết nối Hệ thống Giao thông Thông minh (ITS) & Học tăng cường (RL):**
  * Vector nhúng $Z_i$ đóng vai trò là **Không gian Trạng thái Không gian (Spatial State Space)** tĩnh và bán động, làm đầu vào cho các Agent RL điều khiển chu kỳ đèn giao thông thích ứng theo vùng chức năng ở nghiên cứu tiếp theo.
* **Gợi ý thuyết minh:** *"Vector nhúng $Z$ đóng vai trò hạ tầng dữ liệu biểu diễn không gian, mở ra hướng ứng dụng trực tiếp cho chấm điểm tín dụng bất động sản FinTech và điều khiển giao thông thông minh."*

#### SLIDE 26 [ẢNH THỰC NGHIỆM]: SƠ ĐỒ KHUNG ỨNG DỤNG FINTECH & KHÔNG GIAN TRẠNG THÁI RL
* **Bố cục 2 sơ đồ trực quan:**
  * **Sơ đồ 1 (Trái):** Luồng ứng dụng FinTech: *Vector nhúng H3 ($Z$) $\rightarrow$ Feature Vector $\rightarrow$ Mô hình dự báo giá trị thương mại & Tối ưu vị trí chi nhánh ngân hàng.*
  * **Sơ đồ 2 (Phải):** Khung tích hợp Reinforcement Learning: *Môi trường đô thị $\rightarrow$ State $s_t = [Z_i, \text{lưu lượng xe}] \rightarrow$ RL Agent $\rightarrow$ Action $a_t$ (Chu kỳ đèn) $\rightarrow$ Reward (Giảm ùn tắc).*
* **Điểm cần chỉ tay trên ảnh:**
  * *Sơ đồ 1:* Vector $Z$ thay thế cho hàng chục biến khảo sát thủ công tốn kém khi đánh giá địa điểm mở chi nhánh.
  * *Sơ đồ 2:* Vector $Z$ giúp Agent RL 'hiểu' được đặc tính khu vực (khu thương mại hay khu dân sinh) để ra quyết định điều tiết tín hiệu đèn tối ưu.
* **Thông điệp chốt hạ (Key Takeaway):** *"Học biểu diễn đồ thị không gian đóng vai trò nền tảng hạ tầng dữ liệu (Data Foundation) cho các thuật toán AI quyết định bậc cao trong Đô thị thông minh."*

---

### KHỐI TỔNG KẾT & KẾT THÚC (SLIDE 25 – 26)

#### SLIDE 27 [NỘI DUNG]: TỔNG KẾT TOÀN DIỆN & 3 ĐÓNG GÓP KỸ THUẬT CỐT LÕI
* **1. Chuyển giao phương pháp luận thành công 100%:**
  * Tái lập và bản địa hóa trọn vẹn khung Sato et al. (CEUS 2026) cho đô thị nén Hà Nội; chứng minh tính tổng quát của kiến trúc HAN-GAE.
* **2. Khám phá khoa học thực nghiệm:**
  * Xác lập bản đồ 9 vùng chức năng có cơ sở dữ liệu định lượng; chứng minh vai trò chi phối của mạng đi bộ ngõ ngách ($65{,}13\%$).
* **3. Bộ công cụ tái lập nguồn mở hoàn chỉnh:**
  * Đồng bộ hóa tuyệt đối chuỗi mã nguồn từ tiền xử lý, huấn luyện, phân cụm đến trực quan hóa và kiểm định bền vững, sẵn sàng mở rộng cho toàn bộ vùng Thủ đô.
* **Gợi ý thuyết minh:** *"Tóm lại, đề tài đã giải quyết trọn vẹn bài toán từ phương pháp luận lý thuyết, hiện thực hóa thuật toán GNN, đến kiểm định thực nghiệm và mở ra hướng ứng dụng thực tiễn."*

#### SLIDE 28: PHIÊN HỎI ĐÁP (Q&A) & TRAO ĐỔI KHOA HỌC
* **Tiêu đề:** Trân trọng cảm ơn Thầy Cô và Hội đồng!
* **Thẻ tóm tắt thông số cốt lõi (Quick Fact Card):**
  * **Địa bàn:** 985 ô H3 Res 9 (27 phường nội đô lịch sử Hà Nội).
  * **Dữ liệu:** 152.120 POIs (16 nhóm), mạng đường OSM, mạng buýt GTFS.
  * **Mô hình:** HAN-GAE (200 Epochs, Joint Loss, DistMult).
  * **Kết quả:** $k^* = 9$ cụm, $Q_{\text{contig}} = 0{,}63$, Silhouette $= 0{,}36$.
* **Mời câu hỏi & Góp ý từ Hội đồng BKFintech / VinUni.**
