# DeepWeeds Day2 — kết quả chạy thật
Notebook: https://www.kaggle.com/code/thngonquang/deepweeds-day2

Đọc report.md và results.xlsx; cấu hình đã chốt nằm trong selection_locked.json. Lệnh chạy lại: `python -m deepweeds_lab.experiments --output lab_output --epochs 10 --batch-size 32`, từ môi trường có package và eval.py trong PYTHONPATH. Phiên bản ở environment.json; seed 0,1,2. Giữ nguyên config khi resume, không đổi split hoặc dùng test để chọn lại model. Checkpoint và dataset chỉ lưu ngoài Git.
