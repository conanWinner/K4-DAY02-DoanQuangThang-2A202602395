# Lab Day 2 — DeepWeeds | Doan Quang Thang — 2A202602395

**Bản sửa lỗi suy luận:** khắc phục trường `method` bị truyền trùng khi ghép thông tin đo độ trễ. Notebook hiện gắn dataset checkpoint riêng tư lấy từ output phiên bản 3 và tự kiểm tra/khôi phục đầy đủ 5 backbone + 10 cấu hình huấn luyện trước khi tiếp tục; thiếu checkpoint sẽ dừng, không tự train lại. Lỗi cũ được giữ trong `prior_errors/`, môi trường cũ ở `prior_environment/`.

Notebook riêng tư: https://www.kaggle.com/code/thngonquang/deepweeds-day2

Nguồn phục hồi riêng tư: [checkpoint phiên bản 3](https://www.kaggle.com/datasets/thngonquang/deepweeds-day2-checkpoints-v3) lưu 132 tệp, gồm đủ checkpoint/cấu hình/log của 15 lần huấn luyện đã hoàn tất. [Nguồn tải trực tiếp](https://www.kaggle.com/datasets/thngonquang/deepweeds-day2-recovery-links-v3) được dùng ở notebook phiên bản 5; phiên bản dataset mới cũng có ảnh gốc đã xác thực MD5. Code xác thực SHA-256 từng tệp trước khi khôi phục; các URL tải riêng tư nằm trong dataset, không đưa lên Git hay in vào log.

## Cách chạy

Notebook `code/lab_day2.ipynb` đã đóng gói toàn bộ code và bản `eval.py` gốc (kiểm tra SHA-256). Bật GPU và Internet, chạy toàn bộ. Không cần clone repo hoặc sửa đường dẫn. Bản đưa lên Kaggle nằm ở `kaggle/deepweeds-day2/` từ thư mục gốc repo.

- Dùng nguyên DeepWeeds fold 0: tải ảnh Zenodo và kiểm MD5; tải CSV từ GitHub tác giả. File split gốc chỉ có `Filename,Label`; code không yêu cầu cột Species. Nguồn hiện có một nhãn khác giữa catalog và fold (`20170714-110407-3.jpg`: fold=0, catalog=1); giữ nguyên nhãn fold theo đề và ghi `catalog_label_discrepancies.json`/checksum CSV vào báo cáo.
- Nền chung: 10 epoch, batch 32, ảnh 224, ImageNet finetune, AdamW, warmup một epoch + cosine, CE, AMP; no weight decay cho norm/bias. Chuẩn hoá theo trọng số thực sự tải.
- B01–B05: ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small (độ phân giải động), MobileNetV3-Large.
- T00–T09: nền; scratch/frozen/finetune; augmentation basic/color/CutMix; CE/label smoothing/focal/weighted CE; EMA; một kết hợp. So sánh từng thay đổi với T00 trên backbone thắng vòng sàng.
- I00–I07: mốc, TTA lật (gộp xác suất/logit), 5 crop, ảnh 256, hiệu chuẩn, AMP, gộp BN. Đo thật 10 warmup +50 lượt, đồng bộ GPU, batch 1 và 32; không tính tiền xử lý/đọc ảnh.
- Chốt mọi lựa chọn trên val trong `selection_locked.json`. F01 và T00 dùng seed **0,1,2**, std mẫu `ddof=1`. R01 bổ sung cấu hình có p95 ≤100 ms nếu tìm được trên val; cũng chốt trước test.
- Seed 0 tái sử dụng đúng checkpoint đã chạy ở vòng sàng công thức; hai seed còn lại được huấn luyện với cùng công thức. Không lặp train seed 0 chỉ để lấy kết quả khác.
- Test toàn bộ, một lần cho mỗi cấu hình/seed sau khi chốt. Logits test được lưu để tính trước/sau hiệu chuẩn mà không forward lại. Không thay đổi cấu hình theo điểm test.

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

Đã đối chiếu CSV bằng `eval.py` gốc và kiểm tra số trong báo cáo/Excel: 35 file dự đoán, 19 history đủ 10 epoch, 23 biểu đồ, 7 sheet Excel. Bằng chứng tại `logs/local_verification.json`; nguồn và SHA-256 từng sản phẩm trong `logs/artifact_manifest.json`. Notebook lưu trên Kaggle có toàn bộ cell source và mã code đóng gói khớp với notebook Git; 18 kiểm tra code đã đạt trên GPU Kaggle trước thí nghiệm.

## Code và kiểm tra

Package ở `code/deepweeds_lab/`: `dataset.py` → `model.py`/`losses.py` → `train.py` → `inference.py`/`benchmark.py` → `experiments.py` → `reporting.py`. `prepare.py` tải dữ liệu, EDA và kiểm tra pipeline. Một hàm `train.run(Config(...))` dùng chung.

Môi trường Python >=3.10, torch >=2.3 và torchvision tương thích; giữ cặp torch/torchvision GPU của Kaggle. Các thư viện còn lại trong `code/requirements.txt`; notebook ghi lại phiên bản thực trước chạy. CPU tests chỉ dùng dữ liệu tổng hợp, không chứng minh chất lượng trên DeepWeeds.

```bash
# Từ thư mục gốc repo, sau khi cài torch/torchvision phù hợp và code/requirements.txt:
python -m unittest discover -s tests -v
python -m unittest discover -s submissions/2A202602395_DoanQuangThang/code/tests -v

# Cập nhật notebook tự chứa sau khi sửa code:
python submissions/2A202602395_DoanQuangThang/code/build_notebook.py

# Chạy trên máy GPU (chỉ giữ test ở vòng chung kết):
PYTHONPATH=submissions/2A202602395_DoanQuangThang/code:. python -m deepweeds_lab.experiments --output lab_output --epochs 10 --batch-size 32

# Đẩy notebook lên và chạy toàn bộ:
kaggle kernels push -p kaggle/deepweeds-day2 -t 43200
kaggle kernels status thngonquang/deepweeds-day2
```

## Tiếp tục khi bị ngắt

`last.pt` lưu sau mỗi epoch: model, optimizer, scheduler, scaler, EMA, lịch sử, trạng thái ngẫu nhiên và DataLoader generator. `best.pt` chỉ đổi khi macro-F1 val cao hơn; hoà giữ epoch sớm. `run()` từ chối dùng chung đường dẫn với config khác, và tái sử dụng kết quả đã hoàn tất. Không bảo đảm augmentation worker giống tuyệt đối khi resume.

Muốn tiếp tục trên phiên Kaggle mới: giữ nguyên notebook/config, gắn output phiên trước bằng Add Input, **sao chép nguyên `lab_output` từ input vào `/kaggle/working/lab_output` trước ô chạy thí nghiệm**. Bao gồm checkpoint, prediction, `final_cache`, `selection_locked.json` và JSON kết quả. Không chỉ mang theo trọng số. File test logits đã có sẽ được tái sử dụng; không dùng kết quả test để đổi lựa chọn. Nếu không có đủ output trước, phải báo rõ việc thiếu bằng chứng, không tuyên bố là resume hợp lệ.

Chạy đầy đủ có thể kéo dài nhiều giờ; nếu đạt giới hạn phiên, trạng thái còn dở phải được ghi nhận. Không coi push thành công hoặc GPU đang RUNNING là đã hoàn thành thực nghiệm.
