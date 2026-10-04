# Báo cáo Lab Day 2 — DeepWeeds

## Tóm tắt
Đã so sánh 5 backbone, 10 công thức huấn luyện và 6 phương pháp suy luận ngoài mốc.
Cấu hình F01: **convnext_tiny**, công thức **T05**, suy luận **fivecrop**, kèm temperature scaling khớp riêng trên val cho mỗi seed.
Macro-F1 test: **0.9746 ± 0.0039**; top-1 test: **0.9797 ± 0.0041**.
Mức cải thiện macro-F1 so với T00: +0.0025. Chênh lệch chưa vượt nhiễu giữa các seed; chưa đủ bằng chứng khẳng định tốt hơn.

## Dữ liệu và thiết lập
Dùng nguyên CSV fold 0 của tác giả; train/val/test không giao nhau và đủ 17.509 ảnh. Số đếm thật nằm trong `split_checks.json`. Chỉ dùng train cho trọng số; val cho mọi lựa chọn và nhiệt độ; test chỉ ở vòng cuối sau `selection_locked.json`.
Đối chiếu catalog `labels.csv` và fold CSV: có 1 nhãn khác nhau, chi tiết `[{'Filename': '20170714-110407-3.jpg', 'Label_split': 0, 'Label_catalog': 1}]`. Giữ nguyên nhãn CSV fold 0 theo đề, không sửa hay lọc ảnh; vì vậy số lớp trong fold có thể lệch Table 1 tương ứng. Checksum CSV trong `data_checksums.json`.
![Phân bố lớp](class_distribution.png)
![Ảnh mẫu train](sample_images.png)
Kiểm tra pipeline với mạng tuyến tính nhỏ, độc lập với kết quả backbone, nằm trong `pipeline_checks.json`; ảnh sau augmentation ở `augmented_images.png`.
Nền: ImageNet finetune, ảnh 224, crop/lật ngang, AdamW, LR backbone 1e-4/head 1e-3, decay 0.05 (trừ norm/bias), warmup một epoch rồi cosine theo bước, CE, 10 epoch, batch 32, AMP. Tiền xử lý dùng mean/std đúng trọng số; val/test resize 256 rồi crop 224. GMAC từ fvcore, phép tính không được hỗ trợ được ghi trong log.
GPU: Tesla T4; torch: 2.11.0+cu128. Phiên bản đầy đủ trong `environment.json`. Seed vòng cuối: [0, 1, 2]; std mẫu ddof=1. Vòng sàng và ablation chỉ một seed.

## So sánh backbone
| exp_id   | backbone               | pretrained_tag   |   params_m |   gmac |   macro_f1_val |   top1_val |   train_seconds_per_epoch |   latency_ms |
|:---------|:-----------------------|:-----------------|-----------:|-------:|---------------:|-----------:|--------------------------:|-------------:|
| B01      | resnet50               | a1_in1k          |    23.5265 | 4.1095 |         0.8427 |     0.8855 |                   54.4334 |       6.0189 |
| B02      | resnext50_32x4d        | a1h_in1k         |    22.9983 | 4.2574 |         0.8783 |     0.9055 |                   74.0032 |       7.7864 |
| B03      | convnext_tiny          | in12k_ft_in1k    |    27.8270 | 4.4697 |         0.9694 |     0.9766 |                   68.2169 |       5.6051 |
| B04      | deit_small_patch16_224 | fb_in1k          |    21.6691 | 4.6080 |         0.9563 |     0.9686 |                   46.5924 |       4.6599 |
| B05      | mobilenetv3_large_100  | ra_in1k          |     4.2136 | 0.2242 |         0.8565 |     0.8926 |                   29.6413 |       5.9792 |
Chọn convnext_tiny vì macro-F1 val cao nhất trong vòng sàng. Bảng vẫn ghi tốc độ và kích thước để thể hiện chi phí; lựa chọn này ưu tiên chất lượng.

## Công thức huấn luyện
| exp_id   | axis             | changes                                                 |   macro_f1_val |   top1_val |   delta_macro_f1 |
|:---------|:-----------------|:--------------------------------------------------------|---------------:|-----------:|-----------------:|
| T00      | baseline         | {}                                                      |         0.9694 |     0.9766 |           0.0000 |
| T01      | A initialization | {"init": "scratch"}                                     |         0.3240 |     0.5821 |          -0.6453 |
| T02      | A initialization | {"init": "frozen"}                                      |         0.8527 |     0.8826 |          -0.1166 |
| T03      | B augmentation   | {"aug": "color"}                                        |         0.9701 |     0.9760 |           0.0007 |
| T04      | B augmentation   | {"mix": "cutmix"}                                       |         0.9710 |     0.9783 |           0.0017 |
| T05      | C loss           | {"loss": "ls", "label_smoothing": 0.1}                  |         0.9721 |     0.9786 |           0.0027 |
| T06      | C loss           | {"loss": "focal"}                                       |         0.9671 |     0.9749 |          -0.0023 |
| T07      | C loss           | {"loss": "ce_weighted"}                                 |         0.9681 |     0.9749 |          -0.0013 |
| T08      | F regularization | {"ema_decay": 0.99}                                     |         0.9687 |     0.9760 |          -0.0006 |
| T09      | combined         | {"mix": "cutmix", "loss": "ls", "label_smoothing": 0.1} |         0.9715 |     0.9783 |           0.0022 |
Mỗi T01–T08 chỉ đổi một trục so với T00. T09 là kết hợp thử nghiệm; lựa chọn cuối dựa trên macro-F1 val, không mặc định kết hợp sẽ tốt hơn.
Các delta một seed là dấu hiệu sàng lọc, chưa phải bằng chứng thống kê. Vòng cuối mới so sánh với nhiễu đa seed.
EMA trung bình tham số và sao chép buffer BN từ mạng đang huấn luyện; checkpoint được đánh giá bằng bản EMA. Seed đầu của vòng cuối tái sử dụng checkpoint của đúng công thức đã chạy, nguồn được ghi trong `Final`.

## Suy luận, hiệu chuẩn và độ trễ
| exp_id   | method        |   K |   macro_f1 |   top1 |    ece |     p50 |     p95 |     p99 |   relative_cost |
|:---------|:--------------|----:|-----------:|-------:|-------:|--------:|--------:|--------:|----------------:|
| I00      | single        |   1 |     0.9721 | 0.9786 | 0.0829 |  5.7282 |  9.4159 | 10.7119 |          1.0000 |
| I01      | hflip_prob    |   2 |     0.9721 | 0.9783 | 0.0845 | 10.8612 | 11.6049 | 12.0630 |          1.8961 |
| I02      | hflip_logit   |   2 |     0.9721 | 0.9783 | 0.0844 | 11.1792 | 11.8246 | 12.0383 |          1.9516 |
| I03      | fivecrop      |   5 |     0.9728 | 0.9789 | 0.0837 | 27.8048 | 28.7466 | 30.1875 |          4.8541 |
| I04      | resolution256 |   1 |     0.9722 | 0.9783 | 0.0941 |  6.6624 |  8.6412 |  9.5629 |          1.1631 |
| I05      | temperature   |   1 |     0.9721 | 0.9786 | 0.0067 |  6.0329 |  6.4831 |  7.0172 |          1.0532 |
| I06      | amp           |   1 |     0.9721 | 0.9786 | 0.0828 |  8.0966 |  8.6395 |  8.8096 |          1.4135 |
![Đánh đổi độ chính xác–độ trễ](accuracy_latency.png)
Độ trễ đo với 10 lượt warmup, 50 lượt có đồng bộ GPU, batch 1 và 32, gồm forward/softmax/gộp view, không gồm đọc ảnh và tiền xử lý. TTA và AMP được đo thực tế.
ECE test trước hiệu chuẩn: 0.0863; sau: 0.0058. Tối ưu temperature theo NLL trên val không đảm bảo giảm ECE trên test; số liệu trên là kết quả thực tế.

## Chung kết và phân tích lỗi
| exp_id   |   n_seeds |   macro_f1_val |   macro_f1_test |   macro_f1_test_std |   top1_test |   top1_test_std |   ece_test |
|:---------|----------:|---------------:|----------------:|--------------------:|------------:|----------------:|-----------:|
| F01      |         3 |         0.9710 |          0.9746 |              0.0039 |      0.9797 |          0.0041 |     0.0058 |
| T00      |         3 |         0.9682 |          0.9721 |              0.0035 |      0.9785 |          0.0029 |     0.0128 |
| R01      |         3 |         0.9710 |          0.9746 |              0.0039 |      0.9797 |          0.0041 |     0.0058 |
![Ma trận nhầm lẫn](confusion_matrix.png)
| configuration   | class          |   support |   precision |   recall |     f1 |
|:----------------|:---------------|----------:|------------:|---------:|-------:|
| F01             | Chinee Apple   |  226.0000 |      0.9742 |   0.9454 | 0.9595 |
| F01             | Lantana        |  213.0000 |      0.9661 |   0.9765 | 0.9713 |
| F01             | Parkinsonia    |  207.0000 |      0.9809 |   0.9887 | 0.9848 |
| F01             | Parthenium     |  205.0000 |      0.9951 |   0.9789 | 0.9869 |
| F01             | Prickly Acacia |  213.0000 |      0.9483 |   0.9750 | 0.9614 |
| F01             | Rubber Vine    |  202.0000 |      0.9851 |   0.9769 | 0.9810 |
| F01             | Siam Weed      |  215.0000 |      0.9845 |   0.9829 | 0.9837 |
| F01             | Snake Weed     |  204.0000 |      0.9545 |   0.9592 | 0.9568 |
| F01             | Negatives      | 1822.0000 |      0.9856 |   0.9861 | 0.9858 |
| T00             | Chinee Apple   |  226.0000 |      0.9727 |   0.9381 | 0.9549 |
| T00             | Lantana        |  213.0000 |      0.9767 |   0.9750 | 0.9758 |
| T00             | Parkinsonia    |  207.0000 |      0.9824 |   0.9887 | 0.9855 |
| T00             | Parthenium     |  205.0000 |      0.9933 |   0.9691 | 0.9811 |
| T00             | Prickly Acacia |  213.0000 |      0.9383 |   0.9734 | 0.9555 |
| T00             | Rubber Vine    |  202.0000 |      0.9784 |   0.9719 | 0.9751 |
| T00             | Siam Weed      |  215.0000 |      0.9770 |   0.9845 | 0.9807 |
| T00             | Snake Weed     |  204.0000 |      0.9454 |   0.9624 | 0.9538 |
| T00             | Negatives      | 1822.0000 |      0.9863 |   0.9863 | 0.9863 |
| R01             | Chinee Apple   |  226.0000 |      0.9742 |   0.9454 | 0.9595 |
| R01             | Lantana        |  213.0000 |      0.9661 |   0.9765 | 0.9713 |
| R01             | Parkinsonia    |  207.0000 |      0.9809 |   0.9887 | 0.9848 |
| R01             | Parthenium     |  205.0000 |      0.9951 |   0.9789 | 0.9869 |
| R01             | Prickly Acacia |  213.0000 |      0.9483 |   0.9750 | 0.9614 |
| R01             | Rubber Vine    |  202.0000 |      0.9851 |   0.9769 | 0.9810 |
| R01             | Siam Weed      |  215.0000 |      0.9845 |   0.9829 | 0.9837 |
| R01             | Snake Weed     |  204.0000 |      0.9545 |   0.9592 | 0.9568 |
| R01             | Negatives      | 1822.0000 |      0.9856 |   0.9861 | 0.9858 |
Cặp nhầm có số ảnh trung bình lớn nhất: Negatives → Prickly Acacia (8.7 ảnh). Cần xem ảnh lỗi trước khi kết luận nguyên nhân; tương đồng hình dạng và bối cảnh là giả thuyết, chưa được kiểm chứng nhân quả.
![Ảnh dự đoán sai](misclassified_images.png)
Các ví dụ ưu tiên cặp Chinee Apple ↔ Snake Weed. Tên ảnh và nhãn nằm trong `error_examples.json`.
Chênh lệch macro-F1 val/test tuyệt đối từng seed: [0.004389214658120366, 0.0027617594971395265, 0.0034986980921453137]. Có thể khác biệt do độ khó mẫu và chọn checkpoint trên val; không điều chỉnh lại cấu hình sau khi đọc test.

## Kết luận và khuyến nghị
Chênh lệch chưa vượt nhiễu giữa các seed; chưa đủ bằng chứng khẳng định tốt hơn.
Khoảng chênh trên val: backbone 0.1267; huấn luyện 0.0027; suy luận 0.0007. Trong các phép so sánh đã chạy, **chọn backbone (khoảng tốt nhất–kém nhất)** có khoảng lớn nhất. Các khoảng này có mốc khác nhau và chỉ một seed, nên không coi là phân rã nhân quả hay bằng chứng phổ quát.
Cấu hình R01 dùng fivecrop; p95 trên val = 35.61 ms (trung bình đo qua seed), macro-F1 test = 0.9746 ± 0.0039.
TTA nhiều view tốn thêm forward; dùng bảng độ trễ thực tế để quyết định chạy ngoại tuyến hay trên robot.

## Hạn chế và việc tiếp theo
Chỉ dùng một fold, các vòng sàng một seed và ngân sách 10 epoch. Split ngẫu nhiên không theo địa điểm có thể cho test lạc quan khi triển khai nơi mới. Seed cố định và cuDNN deterministic không bảo đảm giống tuyệt đối giữa phần cứng; khi resume, trạng thái worker augmentation có thể khác phiên liên tục.
Các thất bại nếu có được lưu trong `inference_failures.json`. Không dùng số tham khảo bài báo làm kết quả của mình. Bài báo gốc huấn luyện lâu hơn; chưa kiểm chứng các miền mùa/ánh sáng khác, ONNX hay robot thật.

## Phụ lục
Cấu hình mọi lần chạy: `runs/<exp_id>/seed<k>/config.json`; log: `history.csv`; trọng số tốt nhất: `best.pt`; biểu đồ: `curves/`; dự đoán: `predictions/`. Nguồn trọng số và tag: `pretrained.json`. Đánh giá đối chiếu: `eval_score_*.txt`, `eval_grade.txt`, `eval_out/`.
Notebook: https://www.kaggle.com/code/thngonquang/deepweeds-day2
