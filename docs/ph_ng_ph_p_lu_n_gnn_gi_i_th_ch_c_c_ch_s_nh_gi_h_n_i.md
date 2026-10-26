# PHẦN I: PHƯƠNG PHÁP LUẬN MẠNG NƠ-RON ĐỒ THỊ TỰ MÃ HÓA (GNN METHODOLOGY)

*(Phần này được biên soạn theo chuẩn học thuật để đưa trực tiếp vào Chương 3 của Luận văn)*

## 3.x. Mô hình Học biểu diễn Không gian Không giám sát với Graph Autoencoder (GAE)

Để trích xuất cấu trúc chức năng đô thị từ dữ liệu tiện ích tầng trệt và mạng lưới giao thông đa phương thức tại Hà Nội mà không cần nhãn phân loại sẵn, nghiên cứu áp dụng kiến trúc **Bộ tự mã hóa Đồ thị (Graph Autoencoder - GAE)** kết hợp cơ chế **Tự chú ý hai tầng (Dual-level Attention)** kế thừa từ khung nghiên cứu của Sato et al. (2026).

Mục tiêu của mô hình là học một hàm ánh xạ phi tuyến $f_\theta$ để nén mỗi ô lục giác H3 ($N = 985$) từ không gian đặc trưng đầu vào thành một vector nhúng (embedding) chiều thấp $\mathbf{z}_i \in \mathbb{R}^d$ ($d = 32$) sao cho:

1. Các ô H3 có cơ cấu tiện ích chức năng tương đồng sẽ nằm gần nhau trong không gian vector tiềm ẩn.

2. Các ô H3 có kết nối giao thông chặt chẽ (kề sát địa lý, đi bộ 15 phút, hoặc xe buýt 15 phút) có thể tái tạo lại được liên kết từ chính vector nhúng của chúng.

### 1. Chuẩn hóa Đặc trưng Nút và Trọng số Cạnh Thời gian Di chuyển

* **Ma trận đặc trưng nút (**$\tilde{\mathbf{X}} \in \mathbb{R}^{N \times F}$**):** Với $N = 985$ ô lục giác H3 Resolution 9 và $F = 16$ danh mục tiện ích tầng trệt (POIs), do phân bố tần suất tiện ích đô thị có độ lệch dương lớn (skewed distribution), số lượng tiện ích thô $x_{i,f}$ tại ô $i$ thuộc nhóm $f$ được chuẩn hóa logarit kết hợp Z-score:

  $$
  \tilde{x}_{i,f} = \frac{\log(1 + x_{i,f}) - \mu_f}{\sigma_f} \quad (1)
  $$

  trong đó $\mu_f$ và $\sigma_f$ là giá trị trung bình và độ lệch chuẩn của $\log(1 + x_{i,f})$ trên toàn bộ $985$ ô H3.

* **Trọng số cạnh thời gian di chuyển (**$\tilde{\tau}_{ij}^{(m)}$**):** Cho tập hợp các quan hệ không gian $\mathcal{M} = \{\text{contig}, \text{15\_min\_walk}, \text{15\_min\_multi}\}$, mỗi cạnh $(i, j) \in \mathcal{E}^{(m)}$ mang thuộc tính thời gian di chuyển $\tau_{ij}^{(m)}$ (tính bằng giây, $\tau_{ij}^{(m)} \le 900\text{s}$). Trọng số này cũng được chuẩn hóa tương ứng:

  $$
  \tilde{\tau}_{ij}^{(m)} = \frac{\log(1 + \tau_{ij}^{(m)}) - \mu_\tau^{(m)}}{\sigma_\tau^{(m)}} \quad (2)
  $$

### 2. Kiến trúc Bộ Mã hóa Nhận biết Cạnh (Edge-Aware GAT & HAN Encoders)

Nghiên cứu thiết lập và đối sánh 4 cấu hình mô hình chính dựa trên tập quan hệ $\mathcal{M}$:

* **Model 1 (GAT-GAE — Đồ thị đồng nhất):** Chỉ sử dụng quan hệ tiếp giáp địa lý $\mathcal{M} = \{\text{contig}\}$.

* **Model 2 (HAN-GAE: Walk — Đồ thị dị thể đi bộ):** Kết hợp $\mathcal{M} = \{\text{contig}, \text{15\_min\_walk}\}$.

* **Model 3 (HAN-GAE: Multi — Đồ thị dị thể xe buýt):** Kết hợp $\mathcal{M} = \{\text{contig}, \text{15\_min\_multi}\}$.

* **Model 4 (HAN-GAE: All — Đồ thị dị thể toàn diện):** Tích hợp cả 3 quan hệ $\mathcal{M} = \{\text{contig}, \text{15\_min\_walk}, \text{15\_min\_multi}\}$.

Bộ mã hóa gồm $L = 2$ lớp truyền thông điệp (message-passing layers) nhằm thu thập thông tin lân cận trong phạm vi 2 bước nhảy (2-hops) mà không gây ra hiện tượng làm mịn quá mức (*oversmoothing*).

#### a) Tầng 1: Cơ chế Chú ý cấp độ Nút nhận biết thời gian di chuyển (Node-level Edge-Aware Attention)

Trong từng quan hệ $m \in \mathcal{M}$, mức độ đóng góp thông tin từ ô lân cận $j \in \mathcal{N}_m(i)$ tới ô trung tâm $i$ không đồng đều mà phụ thuộc vào cả đặc trưng tiện ích lẫn thời gian di chuyển $\tilde{\tau}_{ij}^{(m)}$. Tại lớp $l$ và đầu chú ý (attention head) thứ $h \in \{1, \dots, H\}$ (với $H = 4$), hệ số chú ý $\alpha_{ij}^{(l,m,h)}$ được tính như sau:

$$
e_{ij}^{(l,m,h)} = \text{LeakyReLU}\left( {\mathbf{a}^{(l,m,h)}}^\top \left[ \mathbf{W}^{(l,m,h)}\mathbf{h}_i^{(l)} \,\Vert{}\, \mathbf{W}^{(l,m,h)}\mathbf{h}_j^{(l)} \,\Vert{}\, \mathbf{U}^{(l,m,h)}\tilde{\tau}_{ij}^{(m)} \right] \right) \quad (3)
$$

$$
\alpha_{ij}^{(l,m,h)} = \frac{\exp\left(e_{ij}^{(l,m,h)}\right)}{\sum_{k \in \mathcal{N}_m(i) \cup \{i\}} \exp\left(e_{ik}^{(l,m,h)}\right)} \quad (4)
$$

trong đó $\Vert{}$ là phép nối vector (concatenation), $\mathbf{W}^{(l,m,h)}$, $\mathbf{U}^{(l,m,h)}$ và $\mathbf{a}^{(l,m,h)}$ là các tham số huấn luyện. Vector biểu diễn của ô $i$ theo quan hệ $m$ được tổng hợp qua $H = 4$ đầu chú ý (mỗi đầu có kích thước ẩn bằng $8$, tạo ra vector $32$ chiều):

$$
\mathbf{h}_i^{(l+1, m)} = \text{ELU}\left( \underset{h=1}{\overset{H}{\Big\Vert{}}} \sum_{j \in \mathcal{N}_m(i) \cup \{i\}} \alpha_{ij}^{(l,m,h)} \mathbf{W}^{(l,m,h)} \mathbf{h}_j^{(l)} \right) \in \mathbb{R}^{32} \quad (5)
$$

#### b) Tầng 2: Cơ chế Chú ý cấp độ Ngữ nghĩa giữa các phương thức (Semantic-level Attention)

Đối với các mô hình đồ thị dị thể (Models 2, 3, 4), mỗi ô H3 $i$ tại lớp $l+1$ sẽ có $\vert{}\mathcal{M}\vert{}$ vector biểu diễn riêng biệt $\mathbf{h}_i^{(l+1, m)}$ tương ứng với từng loại mạng lưới giao thông. Để hợp nhất chúng một cách thích ứng, mô hình sử dụng cơ chế chú ý ngữ nghĩa (HAN) nhằm tự động học trọng số đóng góp $\beta_{i,m}^{(l+1)}$ của từng quan hệ $m$ tại từng ô $i$:

$$
w_{i,m}^{(l+1)} = {\mathbf{q}^{(l+1)}}^\top \tanh\left( \mathbf{W}_{\text{sem}}^{(l+1)} \mathbf{h}_i^{(l+1, m)} + \mathbf{b}^{(l+1)} \right) \quad (6)
$$

$$
\beta_{i,m}^{(l+1)} = \frac{\exp\left(w_{i,m}^{(l+1)}\right)}{\sum_{m' \in \mathcal{M}} \exp\left(w_{i,m'}^{(l+1)}\right)} \quad \text{với } \sum_{m \in \mathcal{M}} \beta_{i,m}^{(l+1)} = 1 \quad (7)
$$

Vector nhúng hợp nhất cuối cùng tại lớp $l+1$ của ô $i$ là tổng có trọng số của các phương thức:

$$
\bar{\mathbf{h}}_i^{(l+1)} = \sum_{m \in \mathcal{M}} \beta_{i,m}^{(l+1)} \mathbf{h}_i^{(l+1, m)} \in \mathbb{R}^{32} \quad (8)
$$

Kết thúc $L = 2$ lớp, ta thu được ma trận nhúng cuối cùng $\mathbf{Z} = \bar{\mathbf{H}}^{(2)} \in \mathbb{R}^{985 \times 32}$.

### 3. Bộ Giải mã Kép (Dual Decoders) và Hàm Mất mát Huấn luyện

Để huấn luyện không giám sát, GAE sử dụng hai bộ giải mã song song nhằm tái tạo lại cả thuộc tính tiện ích lẫn cấu trúc liên kết mạng lưới từ ma trận nhúng $\mathbf{Z}$:

1. **Giải mã thuộc tính nút (Feature Reconstruction):** Sử dụng một mạng truyền thẳng (MLP) $g_\psi(\mathbf{Z})$ để khôi phục ma trận tiện ích ban đầu $\hat{\tilde{\mathbf{X}}} = g_\psi(\mathbf{Z}) \in \mathbb{R}^{985 \times 16}$, tối thiểu hóa sai số bình phương trung bình (MSE):

   $$
   \mathcal{L}_{\text{feat}} = \frac{1}{N \cdot F} \Vert{}\tilde{\mathbf{X}} - \hat{\tilde{\mathbf{X}}}\Vert{}_F^2 \quad (9)
   $$

2. **Giải mã cấu trúc đa quan hệ (Structure Reconstruction):** Sử dụng phép nhân vô hướng có trọng số quan hệ (DistMult decoder) để dự đoán xác suất tồn tại cạnh giữa hai ô $u$ và $v$ trên từng mạng lưới $m$:

   $$
   p_m(u, v) = \sigma\left( \mathbf{z}_u^\top \text{diag}(\mathbf{r}_m) \mathbf{z}_v \right) \in [0, 1] \quad (10)
   $$

   với $\mathbf{r}_m \in \mathbb{R}^{32}$ là vector tham số đặc trưng cho quan hệ $m$ và $\sigma(\cdot)$ là hàm Sigmoid. Hàm mất mát cấu trúc sử dụng Binary Cross-Entropy có lấy mẫu âm (Negative Sampling $\mathcal{E}^{(m)-}$):

   $$
   \mathcal{L}_{\text{struct}}^{(m)} = -\frac{1}{\vert{}\mathcal{E}^{(m)+}\vert{}} \sum_{(u,v) \in \mathcal{E}^{(m)+}} \log p_m(u,v) - \frac{1}{\vert{}\mathcal{E}^{(m)-}\vert{}} \sum_{(u,v) \in \mathcal{E}^{(m)-}} \log\left(1 - p_m(u,v)\right) \quad (11)
   $$

3. **Hàm mục tiêu tổng hợp:**

   $$
   \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{feat}} + \frac{1}{\vert{}\mathcal{M}\vert{}} \sum_{m \in \mathcal{M}} \mathcal{L}_{\text{struct}}^{(m)} \quad (12)
   $$

Bên cạnh 4 mô hình chính, nghiên cứu còn thiết lập 4 biến thể khuyết thuộc tính (**Structure-only Models 1S–4S**) bằng cách thay ma trận $\tilde{\mathbf{X}}$ bằng vector toàn số $1$ ($\mathbf{1}$) và đặt $\mathcal{L}_{\text{feat}} = 0$ nhằm bóc tách độc lập vai trò của hình học mạng lưới giao thông (Ablation Study). Toàn bộ 8 mô hình được tối ưu bằng thuật toán Adam ($\text{lr} = 0.005$, $\text{dropout} = 0.6$, $\text{Early Stopping patience} = 100$ epochs).

# PHẦN II: GIẢI THÍCH CHI TIẾT CÁC METRIC SO SÁNH Ở PHẦN 4 & PHẦN 5 (KẾT QUẢ HÀ NỘI)

*(Phần* này giúp bạn hiểu rõ bản chất toán học, ý nghĩa quy hoạch và cách đọc các con số của Hà Nội *để giải thích cho Thầy)*

Trước khi đưa vào phân cụm K-Means (quét $k \in [2, 50]$), tất cả các vector nhúng $\mathbf{z}_i$ đều được **chuẩn hóa** $L_2$ **(**$\Vert{}\mathbf{z}_i\Vert{}_2 = 1$**)** lên mặt cầu đơn vị. Nhờ đó, khoảng cách Euclid trong K-Means tương đương trực tiếp với **độ tương đồng góc Cosine (Cosine Similarity)**.

## 1. Nhóm Metric ở Phần 4 (`04_evaluation_hanoi.py` & Phụ lục HDBSCAN)

Phần 4 đánh giá chất lượng phân vùng đô thị trên **2 phương diện độc lập**: (A) Chất lượng phân tách trong không gian đặc trưng (Internal