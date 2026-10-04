# Lab Day 2 — DeepWeeds | Doan Quang Thang — 2A202602395

**Bản hoàn thiện trước khi nộp:** notebook công khai, dùng code trực tiếp từ Git; mặc định xem kết quả đã chạy. Chạy mới không cần quyền truy cập dataset checkpoint riêng tư. Các lần chạy mới tái sử dụng đánh giá F01 cho R01 khi cùng cấu hình, không forward test lần nữa. Báo cáo ghi rõ lần chạy phiên bản 5 trước sửa đã forward riêng theo hai tên; không thay đổi lịch sử hay số liệu cũ.

Notebook: https://www.kaggle.com/code/thngonquang/deepweeds-day2

Phục hồi là tuỳ chọn cho chủ tài khoản: [checkpoint phiên bản 3](https://www.kaggle.com/datasets/thngonquang/deepweeds-day2-checkpoints-v3). Dataset này vẫn riêng tư và không gắn vào notebook công khai. Nếu chọn `RESUME_FROM_INPUT = True`, cần chủ động gắn nguồn phục hồi đúng; thiếu dữ liệu sẽ dừng. Chạy mới với `RESUME_FROM_INPUT = False` dùng nguồn DeepWeeds công khai.

## Cách chạy

Notebook `code/lab_day2.ipynb` dùng `git clone`, checkout commit cố định ghi trong `CODE_REVISION` và dùng trực tiếp code Python cùng `eval.py` gốc. Bật Internet. Mặc định `RUN_EXPERIMENTS = False` mở kết quả đã chạy xong ở phiên bản 5; đổi thành `True` và bật GPU để chạy mới toàn bộ thí nghiệm, không cần gắn Input. `RUN_CHECKS = True` chạy kiểm tra chống test trùng ngay cả ở chế độ xem. Notebook không chứa PAYLOAD. Các ô kết quả tách riêng bảy bảng Excel, biểu đồ phân bố lớp, đánh đổi độ trễ, ma trận nhầm lẫn và ảnh lỗi để xem trực tiếp trên Kaggle. Bản đưa lên Kaggle nằm ở `kaggle/deepweeds-day2/` từ thư mục gốc repo.

- Dùng nguyên DeepWeeds fold 0: tải ảnh Zenodo và kiểm MD5; tải CSV từ GitHub tác giả. File split gốc chỉ có `Filename,Label`; code không yêu cầu cột Species. Nguồn hiện có một nhãn khác giữa catalog và fold (`20170714-110407-3.jpg`: fold=0, catalog=1); giữ nguyên nhãn fold theo đề và ghi `catalog_label_discrepancies.json`/checksum CSV vào báo cáo.
- Nền chung: 10 epoch, batch 32, ảnh 224, ImageNet finetune, AdamW, warmup một epoch + cosine, CE, AMP; no weight decay cho norm/bias. Chuẩn hoá theo trọng số thực sự tải.
- B01–B05: ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small (độ phân giải động), MobileNetV3-Large.
- T00–T09: nền; scratch/frozen/finetune; augmentation basic/color/CutMix; CE/label smoothing/focal/weighted CE; EMA; một kết hợp. So sánh từng thay đổi với T00 trên backbone thắng vòng sàng.
- I00–I07: mốc, TTA lật (gộp xác suất/logit), 5 crop, ảnh 256, hiệu chuẩn, AMP, gộp BN. Đo thật 10 warmup +50 lượt, đồng bộ GPU, batch 1 và 32; không tính tiền xử lý/đọc ảnh.
- Chốt mọi lựa chọn trên val trong `selection_locked.json`. F01 và T00 dùng seed **0,1,2**, std mẫu `ddof=1`. R01 bổ sung cấu hình có p95 ≤100 ms nếu tìm được trên val; cũng chốt trước test.
- Seed 0 tái sử dụng đúng checkpoint đã chạy ở vòng sàng công thức; hai seed còn lại được huấn luyện với cùng công thức. Không lặp train seed 0 chỉ để lấy kết quả khác.
- Code hiện tại: test toàn bộ, một lần cho mỗi cấu hình/seed sau khi chốt; R01 dùng chung kết quả với F01 khi cùng cấu hình. Lịch sử phiên bản 5 có ngoại lệ F01/R01 được nêu trong report.md. Logits test được lưu để tính trước/sau hiệu chuẩn mà không forward lại. Không thay đổi cấu hình theo điểm test.

## Kết quả sau khi chạy

Kaggle lưu trong `/kaggle/working/lab_output/`:

| Sản phẩm | Nội dung |
|---|---|
| `results.xlsx` | Backbones, Training, Inference, Final, PerClass, Latency, Summary |
| `report.md` | Báo cáo sinh từ số liệu thật, so sánh, hiệu chuẩn, phân tích lỗi và hạn chế |
| `curves/*.png` | Loss train/val, macro-F1/top-1 val, LR từng thí nghiệm/seed |
| `predictions/*.csv` | Val/test của cấu hình cuối và nền; test chưa hiệu chuẩn; R01 nếu có |
| `runs/` | config.json, pretrained.json, history.csv, best.pt, last.pt, val logits |
| `eval_out/`, `eval_score_*.txt`, `eval_grade.txt` | Chỉ số tính lại và điểm phần I từ eval.py gốc |
| `environment.json` | Phiên bản thực của Python, torch, timm và thư viện, tên GPU |
| `execution_status.json`, `execution_error.json` (nếu lỗi) | Tiến độ chạy và bằng chứng lỗi |

**Đã chạy xong trên Kaggle phiên bản 5 (`COMPLETE`) và đã lấy kết quả thật về Git.** Cấu hình cuối: ConvNeXt-Tiny, label smoothing 0.1, suy luận 5 crop, temperature khớp trên validation. Test fold 0 gồm 3.507 ảnh, seed 0/1/2: macro-F1 **0.9746 ± 0.0039**, top-1 **97.97% ± 0.41 điểm phần trăm**, ECE **0.0058 ± 0.0019**. Công cụ `eval.py` gốc tự chấm **mục I: 17/20**; đây không phải điểm toàn bài hay điểm giảng viên xác nhận. Chênh macro-F1 so với nền khoảng +0.0025, nhỏ hơn nhiễu giữa các seed; báo cáo không kết luận chắc chắn tốt hơn.

Trong bản Git, `report.md`, `results.xlsx`, `curves/` và `predictions/` nằm ngay ở thư mục này. Các JSON môi trường/cấu hình chốt/kết quả và đánh giá nằm trong `logs/`; cấu hình, nguồn trọng số và history ở `logs/training/<exp_id>/seed<k>/`; CSV fold gốc ở `logs/labels/`. Tên đường dẫn trong báo cáo gốc tương ứng với Output Kaggle. Checkpoint và toàn bộ dữ liệu ảnh vẫn nằm ngoài Git.

Đã đối chiếu CSV bằng `eval.py` gốc và kiểm tra số trong báo cáo/Excel: 35 file dự đoán, 19 history đủ 10 epoch, 23 biểu đồ, 7 sheet Excel. Báo cáo/Excel gốc được giữ tại `logs/prior_version5_submission/`; bản hiện tại bổ sung giải thích lịch sử và Summary, không đổi số liệu. Bằng chứng tại `logs/local_verification.json`; nguồn và SHA-256 từng sản phẩm trong `logs/artifact_manifest.json`. Notebook phiên bản 5 lưu trên Kaggle đã được đối chiếu cell source và code đóng gói với bản Git tại commit `902702c`; 18 kiểm tra code đã đạt trên GPU Kaggle trước thí nghiệm.

## Code và kiểm tra

Package ở `code/deepweeds_lab/`: `dataset.py` → `model.py`/`losses.py` → `train.py` → `inference.py`/`benchmark.py` → `experiments.py` → `reporting.py`. `prepare.py` tải dữ liệu, EDA và kiểm tra pipeline. Một hàm `train.run(Config(...))` dùng chung.

Môi trường Python >=3.10, torch >=2.3 và torchvision tương thích; giữ cặp torch/torchvision GPU của Kaggle. Các thư viện còn lại trong `code/requirements.txt`; notebook ghi lại phiên bản thực trước chạy. CPU tests chỉ dùng dữ liệu tổng hợp, không chứng minh chất lượng trên DeepWeeds.

```bash
# Từ thư mục gốc repo, sau khi cài torch/torchvision phù hợp và code/requirements.txt:
python -m unittest discover -s tests -v
python -m unittest discover -s submissions/2A202602395_DoanQuangThang/code/tests -v

# Sau khi commit và push code, tạo notebook clone đúng commit đó:
python submissions/2A202602395_DoanQuangThang/code/build_notebook.py

# Chạy trên máy GPU (chỉ giữ test ở vòng chung kết):
PYTHONPATH=submissions/2A202602395_DoanQuangThang/code:. python -m deepweeds_lab.experiments --output lab_output --epochs 10 --batch-size 32

# Đẩy notebook lên; chế độ mặc định chỉ mở kết quả đã hoàn tất:
kaggle kernels push -p kaggle/deepweeds-day2 -t 43200
kaggle kernels status thngonquang/deepweeds-day2
```

## Tiếp tục khi bị ngắt

`last.pt` lưu sau mỗi epoch: model, optimizer, scheduler, scaler, EMA, lịch sử, trạng thái ngẫu nhiên và DataLoader generator. `best.pt` chỉ đổi khi macro-F1 val cao hơn; hoà giữ epoch sớm. `run()` từ chối dùng chung đường dẫn với config khác, và tái sử dụng kết quả đã hoàn tất. Không bảo đảm augmentation worker giống tuyệt đối khi resume.

Muốn tiếp tục trên phiên Kaggle mới: giữ nguyên notebook/config, gắn output phiên trước bằng Add Input, **sao chép nguyên `lab_output` từ input vào `/kaggle/working/lab_output` trước ô chạy thí nghiệm**. Bao gồm checkpoint, prediction, `final_cache`, `selection_locked.json` và JSON kết quả. Không chỉ mang theo trọng số. File test logits đã có sẽ được tái sử dụng; không dùng kết quả test để đổi lựa chọn. Nếu không có đủ output trước, phải báo rõ việc thiếu bằng chứng, không tuyên bố là resume hợp lệ.

Chạy đầy đủ có thể kéo dài nhiều giờ; nếu đạt giới hạn phiên, trạng thái còn dở phải được ghi nhận. Không coi push thành công hoặc GPU đang RUNNING là đã hoàn thành thực nghiệm.

## Đối chiếu trước khi nộp

- Summary có top 10 và mốc T00, ghi đơn vị, loại percentile và nguồn độ trễ; T04/T09 ghi “Chưa đo riêng” cùng chi phí huấn luyện thật.
- Báo cáo có số đếm split, loss kiểm tra pipeline và giải thích F01/R01 trong lịch sử phiên bản 5. Không tuyên bố việc sửa code làm lần chạy cũ đạt yêu cầu một forward test mỗi cấu hình/seed.
- Notebook công khai chỉ tải Git và nguồn công khai; không chứa URL tải riêng tư, không phụ thuộc input của chủ tài khoản khi chạy mới.
- Điểm phần I là 17/20 theo công cụ gốc; điểm toàn bài do giảng viên chấm.
