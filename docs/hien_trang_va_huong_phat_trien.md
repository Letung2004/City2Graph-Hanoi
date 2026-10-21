# Đánh giá hiện trạng và hướng phát triển City2Graph-Hanoi

## Kết luận tổng quát

Không phát hiện dấu hiệu dữ liệu bị hỏng toàn bộ hoặc sai toán học nghiêm trọng trong các đồ thị, embedding và nhãn cụm đã xuất. Pipeline đủ cơ sở để bảo vệ, nhưng cần điều chỉnh cách trình bày phương pháp và trung thực về cấu hình thực tế.

## Mức 1: Vấn đề nghiêm trọng / Blocker

| Vấn đề | Bằng chứng | Mức rủi ro | Cách xử lý ngắn nhất |
|---|---|---:|---|
| **Cấu hình công bố không phải cấu hình mà notebook huấn luyện thực sự đọc** | Notebook tìm `configs/experiment_config.yaml` tại `notebooks/03_model_training_hanoi.py:145`, nhưng repository chỉ có `config/experiment_config.yaml`. Vì vậy nó rơi vào fallback tại `notebooks/03_model_training_hanoi.py:120`, dùng `lr=0.005`, trong khi file cấu hình công bố ghi `lr=0.001` tại `config/experiment_config.yaml:113`. Ngoài ra notebook dùng `model.out_dim`, còn YAML dùng `embedding_dim`. | **Cao nhất** | Trong luận văn phải ghi rõ các checkpoint hiện tại được huấn luyện bằng cấu hình fallback, đặc biệt learning rate thực tế là `0.005` nếu đúng notebook này đã tạo output. Không được tuyên bố kết quả là của `lr=0.001` nếu chưa có bằng chứng log tương ứng. |

Đây là điểm duy nhất được xếp vào nhóm có thể bị Hội đồng phản bác trực diện về tính tái lập. Không nhất thiết phải chạy lại toàn bộ nếu xác nhận và báo cáo đúng cấu hình thực tế của các checkpoint hiện tại.

## Mức 2: Vấn đề nhẹ / Đặc thù nghiên cứu

| Hiện tượng | Đánh giá và cách bảo vệ |
|---|---|
| **Không có train/validation/test split** | Đây là học tự giám sát trên toàn đồ thị, phù hợp với mục tiêu tạo embedding và khám phá không gian đô thị. Tuy nhiên Silhouette, Q, CH, DB không phải là độ chính xác dự báo. Luận điểm bảo vệ: “Mục tiêu là representation learning và exploratory spatial analysis, không phải supervised prediction.” |
| **Chọn `k` bằng Silhouette rồi chọn seed tốt nhất theo Silhouette** | Có thiên lệch tối ưu hóa nội bộ. Code chọn `k*` tại `notebooks/04_evaluation_hanoi.py:439` và tiếp tục chọn lần chạy tốt nhất. Hãy gọi đây là “internal clustering validation”, không gọi là bằng chứng khách quan tuyệt đối về số vùng chức năng. |
| **Bảng 3 dễ gây hiểu nhầm** | Code tính Mean ± SD trên toàn dải `k=2..50` tại `notebooks/04_evaluation_hanoi.py:540`, trong khi phần diễn giải lại đặt cạnh `k*`. Cần nói rõ mỗi con số là thống kê trên dải quét hay giá trị tại `k*`. Đây là lỗi báo cáo, không phải lỗi mô hình. |
| **`n_init` trong YAML là 20 nhưng code hard-code 10** | Thấy tại `notebooks/04_evaluation_hanoi.py:414`. Không làm hỏng kết quả, nhưng phải ghi đúng trong phần thực nghiệm. |
| **HDBSCAN loại noise trước khi tính modularity** | Code dùng đồ thị con tại `notebooks/appendix_evaluation_hdbscan_hanoi.py:340`. Vì vậy Q của HDBSCAN không hoàn toàn so sánh trực tiếp với Q của K-Means trên đủ 985 ô. Hãy chú thích rõ “Q trên tập non-noise”. |
| **Model 3S/4S suy giảm hoặc tạo rất nhiều noise** | Đây là hệ quả hợp lý khi bỏ toàn bộ thuộc tính POI và chỉ giữ cấu trúc. Nó chứng minh topology đơn thuần chưa đủ để khôi phục vùng chức năng. Đây là một ablation có giá trị, không phải lỗi cần sửa. |
| **HDBSCAN Model 3S có 100% noise** | Kết quả hiện tại có `985/985` noise. Không nên diễn giải là “mô hình thất bại”; nên trình bày là density structure không đủ ổn định dưới không gian embedding structure-only. |
| **Đánh giá biến mạng có hướng thành đồ thị vô hướng** | Phần NetworkX dùng `nx.Graph`. Điều này phù hợp với Newman modularity chuẩn, nhưng làm mất hướng đi bộ/xe buýt. Cần ghi rõ đây là phép chiếu phục vụ đánh giá cộng đồng, không phải tuyên bố mạng giao thông vốn vô hướng. |
| **Dữ liệu POI Overture có thể lệch theo thời điểm và phân loại** | 152 nghìn POI là quy mô tốt, nhưng không đồng nghĩa với độ đầy đủ tuyệt đối. Các nhóm ít mẫu và sai lệch phân loại nên đưa vào hạn chế dữ liệu nguồn. |
| **GTFS hiện là lát cắt tĩnh** | Không có biến động sáng/chiều/tối, ngày thường/cuối tuần, tắc nghẽn hoặc thời gian chờ thực đo. Đây là hạn chế nghiên cứu, không phải lỗi pipeline. |
| **Attention không phải quan hệ nhân quả** | `beta_walk` cao chỉ cho thấy mô hình dựa nhiều vào quan hệ đó trong nghiệm hiện tại; không thể kết luận đi bộ “gây ra” cấu trúc đô thị. |
| **t-SNE chỉ để minh họa** | Không dùng t-SNE làm bằng chứng định lượng về phân tách cụm; chỉ dùng để trực quan hóa. |

## Các luận điểm bảo vệ nên nhấn mạnh

- Model 4 không nhất thiết phải có Silhouette cao nhất; đóng góp của nó nằm ở việc tăng tính liên kết theo mạng, thể hiện qua modularity.
- Structure-only là ablation để chứng minh vai trò của thuộc tính POI, không phải mô hình triển khai chính.
- HDBSCAN bổ sung kiểm tra độ bền theo mật độ, nhưng DBCV thấp hoặc noise cao là thông tin về cấu trúc dữ liệu.
- Kết quả hiện tại là phân tích không gian tĩnh của Hà Nội, chưa phải mô phỏng giao thông thời gian thực.
- Không nên dùng các cụm thu được để khẳng định trực tiếp ranh giới hành chính hay quy hoạch pháp lý.

## Future Work

1. **GTFS đa khung giờ:** xây dựng đồ thị sáng, trưa, chiều, tối và ngày thường/cuối tuần; dùng temporal HAN-GAE hoặc dynamic GNN.
2. **Tích hợp OD thực tế:** bổ sung dữ liệu GPS, thẻ vé, mobile signaling hoặc ma trận OD để kiểm định khả năng tiếp cận suy ra từ GTFS.
3. **H3 đa độ phân giải:** đánh giá Res 8/9/10 để kiểm tra độ ổn định của vùng chức năng theo quy mô không gian.
4. **Không gian trạng thái cho RL/ITS:** dùng embedding của các ô H3 làm state; hành động có thể là điều chỉnh tuyến buýt, tần suất hoặc phân bổ hạ tầng.
5. **Đánh giá chuyển miền và bất định:** huấn luyện trên Hà Nội, kiểm thử tại một đô thị khác; bổ sung bootstrap, khoảng tin cậy và kiểm định độ ổn định theo seed.

## Tóm tắt khuyến nghị

Dự án đủ nền tảng để bảo vệ. Việc cần ưu tiên là **đính chính cấu hình huấn luyện thực tế và làm rõ quy ước tính các chỉ số**, không phải viết lại hay chạy lại toàn bộ pipeline.
