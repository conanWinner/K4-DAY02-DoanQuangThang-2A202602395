"""Build a readable Kaggle notebook that clones an immutable Git revision."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = Path(__file__).resolve().parent
TARGET = ROOT / 'kaggle' / 'deepweeds-day2'


def cell(kind, text):
    result = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build():
    revision = '902702cea102fb1605094b2813d32266a4914c3c'
    paths = [ROOT / 'eval.py', CODE / 'requirements.txt',
             *sorted((CODE / 'deepweeds_lab').glob('*.py')),
             *sorted((CODE / 'tests').glob('*.py'))]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in paths}
    setup = f'''import hashlib, json, sys, subprocess
from pathlib import Path

# False: open verified results from version 5. True: install dependencies and run experiments.
RUN_EXPERIMENTS = False
REPO_URL = 'https://github.com/conanWinner/K4-DAY02-DoanQuangThang-2A202602395.git'
CODE_REVISION = '{revision}'
REPO_DIR = Path('/kaggle/working/deepweeds_repo_' + CODE_REVISION[:12])
if not REPO_DIR.exists():
    subprocess.run(['git', 'clone', '--no-checkout', REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(['git', '-C', str(REPO_DIR), 'checkout', '--detach', CODE_REVISION], check=True)
actual_revision = subprocess.check_output(['git', '-C', str(REPO_DIR), 'rev-parse', 'HEAD'], text=True).strip()
if actual_revision != CODE_REVISION:
    raise RuntimeError('Existing repository has another revision; preserve it and use a new directory')
if subprocess.check_output(['git', '-C', str(REPO_DIR), 'status', '--porcelain'], text=True).strip():
    raise RuntimeError('Existing repository has local changes; preserve them before running')
SUBMISSION_DIR = REPO_DIR / 'submissions/2A202602395_DoanQuangThang'
SOURCE_DIR = SUBMISSION_DIR / 'code'
sys.path[:0] = [str(SOURCE_DIR), str(REPO_DIR)]
if RUN_EXPERIMENTS:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(SOURCE_DIR / 'requirements.txt')], check=True)
print('Git revision:', actual_revision)
print('Code directory:', SOURCE_DIR)
print('Run experiments:', RUN_EXPERIMENTS)
'''
    verify = '''import importlib.metadata as metadata, platform
import torch
from deepweeds_lab.train import write_json
OUTPUT_DIR = Path('/kaggle/working/lab_output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
from deepweeds_lab.recovery import restore_kaggle_input
# Version 3 completed all training before the inference-row bug. Resume its immutable output.
restore_kaggle_input(OUTPUT_DIR, required=True)
versions = {name: metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy', 'pandas', 'scipy', 'matplotlib', 'openpyxl', 'fvcore')}
versions.update(python=platform.python_version(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
write_json(OUTPUT_DIR / 'environment.json', versions)
print(versions, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('Full lab requires GPU; enable an accelerator before running')
# CPU arithmetic checks and a tiny synthetic train only; not DeepWeeds results.
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(SOURCE_DIR / 'tests'), '-v'], check=True, cwd=SOURCE_DIR)
'''
    verify = "if RUN_EXPERIMENTS:\n" + ''.join('    ' + line + '\n' for line in verify.splitlines()) + "else:\n    print('Verified version 5 results: no new training requested')\n"
    execute = '''from deepweeds_lab.experiments import execute
# All required stages; test is locked until backbone/training/inference selection completes.
# Kaggle output retains per-epoch last.pt/best.pt/history.csv. Reuse the same configuration
# and restored lab_output to resume after interruption; never reselect from test results.
try:
    selection = execute(str(OUTPUT_DIR), epochs=10, batch_size=32, seeds=(0, 1, 2))
except Exception as exc:
    import traceback
    failure = {'status': 'error', 'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
    write_json(OUTPUT_DIR / 'execution_error.json', failure)
    raise
'''
    execute = "if RUN_EXPERIMENTS:\n" + ''.join('    ' + line + '\n' for line in execute.splitlines()) + "else:\n    print('Training skipped; displaying completed version 5 report')\n"
    cells = [cell('markdown', '''# Lab Day 2 — DeepWeeds | Doan Quang Thang — 2A202602395

Notebook chạy đầy đủ bài lab, không chứa số liệu giả. Mọi lựa chọn dựa trên val; chỉ mở test sau khi chốt cấu hình.

**Code tải trực tiếp bằng git clone**, cố định commit đã tạo kết quả phiên bản 5. Mặc định `RUN_EXPERIMENTS = False` để xem kết quả đã hoàn tất. Đổi thành `True` khi muốn chạy thí nghiệm; khi đó khôi phục 15 lần huấn luyện từ dataset checkpoint riêng tư trước khi tiếp tục.

**Luồng:** kiểm tra dữ liệu → 5 backbone → 3 trục huấn luyện và EMA/kết hợp → 6 phương pháp suy luận ngoài mốc (gộp BN không áp dụng cho ConvNeXt) → chung kết và mốc với seed 0/1/2 → Excel, biểu đồ, báo cáo và đánh giá bằng eval.py gốc.

Bật **GPU** và **Internet**. Cấu hình nền 10 epoch, batch 32 cho mọi backbone. Chạy toàn bộ có thể cần nhiều giờ; thời gian thực được lưu theo từng epoch. Không thay đổi cấu hình sau khi đã xem test.

Nguồn: https://github.com/conanWinner/K4-DAY02-DoanQuangThang-2A202602395
'''), cell('markdown', '## 0. Clone Git và chọn chế độ chạy\nCode Python nằm trực tiếp trong repo, không có PAYLOAD. Chỉ cài thư viện khi RUN_EXPERIMENTS = True.'), cell('code', setup),
        cell('markdown', '## 1. Kiểm tra môi trường và tính đúng của code\nCác fixture tổng hợp chỉ dùng cho kiểm tra, không đưa vào báo cáo DeepWeeds.'), cell('code', verify),
        cell('markdown', '''## 2–5. Chạy các vòng thí nghiệm

- B01–B05: ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small, MobileNetV3-Large; cùng nền và seed.
- T00–T09: khởi tạo, augmentation, loss; thêm EMA và một kết hợp. Mỗi thí nghiệm riêng chỉ đổi một trục.
- I00–I07: một ảnh, lật/gộp xác suất, lật/gộp logit, 5 crop, độ phân giải 256, temperature scaling, AMP và gộp BN.
- F01/T00: cấu hình cuối và nền với ≥3 seed; R01 nếu có phương pháp p95 ≤100 ms. Nhiệt độ khớp trên val của từng seed trước test. Mỗi cấu hình/seed dùng một lần forward trên test (TTA gồm các view đã khai báo); tái sử dụng logits cho so sánh hiệu chuẩn.
- Báo cáo sinh từ log, prediction CSV; eval.py gốc tính lại toàn bộ test.
'''), cell('code', execute), cell('markdown', '## 6. Xem kết quả thật\nChế độ xem dùng kết quả phiên bản 5 đã lưu trên Git; chế độ chạy dùng lab_output mới. Đường dẫn file được in bên dưới.'), cell('code', '''from IPython.display import display, Markdown
RESULT_DIR = OUTPUT_DIR if RUN_EXPERIMENTS else SUBMISSION_DIR
STATUS_FILE = RESULT_DIR / 'execution_status.json' if RUN_EXPERIMENTS else RESULT_DIR / 'logs/execution_status.json'
print('Result origin:', 'current execution' if RUN_EXPERIMENTS else 'completed Kaggle version 5, archived in Git')
print(STATUS_FILE.read_text())
print('Excel:', RESULT_DIR / 'results.xlsx')
print('Prediction files:', len(list((RESULT_DIR / 'predictions').glob('*.csv'))))
''')]
    # Each table is read from the verified workbook, never reconstructed from constants.
    sections = [
        ('6.1. So sánh backbone', 'Backbones', '5 backbone, cùng công thức nền; chọn theo validation.'),
        ('6.2. Công thức huấn luyện', 'Training', 'Các trục khởi tạo, augmentation, loss và các thử nghiệm bổ sung.'),
        ('6.3. Phương pháp suy luận', 'Inference', 'Chất lượng trên validation, hiệu chuẩn và chi phí suy luận.'),
        ('6.4. Kết quả cuối qua ba seed', 'Final', 'Kết quả test sau khi chốt cấu hình; có dòng tổng hợp mean và std.'),
        ('6.5. Chỉ số từng lớp', 'PerClass', 'Precision, recall và F1 của cấu hình cuối và mốc.'),
        ('6.6. Độ trễ', 'Latency', 'p50/p95/p99 và thông lượng theo batch, GPU và dtype.'),
        ('6.7. Tổng hợp cấu hình', 'Summary', 'Bảng tổng hợp từ workbook đã xuất sau thí nghiệm.'),
    ]
    cells.append(cell('code', "import pandas as pd\npd.set_option('display.max_columns', None)\npd.set_option('display.precision', 4)\n"))
    for title, sheet, explanation in sections:
        cells.extend([cell('markdown', f'## {title}\n{explanation}'),
                      cell('code', f"display(pd.read_excel(RESULT_DIR / 'results.xlsx', sheet_name={sheet!r}))\n")])
    cells.extend([
        cell('markdown', '## 6.8. Biểu đồ và ảnh phân tích\nẢnh lấy từ kết quả thật của phiên chạy tương ứng.'),
        cell('code', "from IPython.display import Image\nfor filename in ('class_distribution.png', 'accuracy_latency.png', 'confusion_matrix.png', 'misclassified_images.png'):\n    print(filename)\n    display(Image(filename=str(RESULT_DIR / filename)))\n"),
        cell('markdown', '## 7. Báo cáo đầy đủ\nNội dung gốc sinh từ log chạy thật, gồm hạn chế và phân tích kết quả.'),
        cell('code', "display(Markdown((RESULT_DIR / 'report.md').read_text()))\n"),
    ])
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'deepweeds_source_revision': revision, 'deepweeds_source_sha256': hashes}, 'cells': cells}
    for index, c in enumerate(cells):
        c['id'] = f'lab2-{index:02d}'
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / 'lab_day2.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    (CODE / 'lab_day2.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata_file = TARGET / 'kernel-metadata.json'
    meta = json.loads(metadata_file.read_text())
    meta.update(enable_gpu=True, enable_internet=True, is_private=True)
    metadata_file.write_text(json.dumps(meta, indent=2))
    print('Built Git-clone notebook:', TARGET / 'lab_day2.ipynb')


if __name__ == '__main__':
    build()
