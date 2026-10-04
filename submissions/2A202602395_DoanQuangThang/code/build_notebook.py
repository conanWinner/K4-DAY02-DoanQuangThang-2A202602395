"""Build a self-contained Kaggle notebook from versioned code; no Git/network bootstrapping."""
import base64
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
    payload = {'eval.py': base64.b64encode((ROOT / 'eval.py').read_bytes()).decode(),
               'requirements.txt': base64.b64encode((CODE / 'requirements.txt').read_bytes()).decode()}
    for path in sorted((CODE / 'deepweeds_lab').glob('*.py')):
        payload['deepweeds_lab/' + path.name] = base64.b64encode(path.read_bytes()).decode()
    for path in sorted((CODE / 'tests').glob('*.py')):
        payload['tests/' + path.name] = base64.b64encode(path.read_bytes()).decode()
    digest = hashlib.sha256((ROOT / 'eval.py').read_bytes()).hexdigest()
    setup = '''import base64, hashlib, json, os, sys, subprocess
from pathlib import Path
SOURCE_DIR = Path('/kaggle/working/deepweeds_source')
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
PAYLOAD = ''' + repr(payload) + '''
for name, encoded in PAYLOAD.items():
    path = SOURCE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = base64.b64decode(encoded)
    if path.exists() and path.read_bytes() != contents:
        raise RuntimeError(f'Existing source differs: {path}; preserve it and use a new output directory')
    if not path.exists():
        path.write_bytes(contents)
assert hashlib.sha256((SOURCE_DIR / 'eval.py').read_bytes()).hexdigest() == ''' + repr(digest) + '''
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(SOURCE_DIR / 'requirements.txt')], check=True)
sys.path.insert(0, str(SOURCE_DIR))
print('Source ready:', SOURCE_DIR, flush=True)
'''
    verify = '''import importlib.metadata as metadata, platform
import torch
from deepweeds_lab.train import write_json
OUTPUT_DIR = Path('/kaggle/working/lab_output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
versions = {name: metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy', 'pandas', 'scipy', 'matplotlib', 'openpyxl', 'fvcore')}
versions.update(python=platform.python_version(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
write_json(OUTPUT_DIR / 'environment.json', versions)
print(versions, flush=True)
if not torch.cuda.is_available():
    raise RuntimeError('Full lab requires GPU; enable an accelerator before running')
# CPU arithmetic checks and a tiny synthetic train only; not DeepWeeds results.
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(SOURCE_DIR / 'tests'), '-v'], check=True, cwd=SOURCE_DIR)
'''
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
    cells = [cell('markdown', '''# Lab Day 2 — DeepWeeds | Doan Quang Thang — 2A202602395

Notebook chạy đầy đủ bài lab, không chứa số liệu giả. Mọi lựa chọn dựa trên val; chỉ mở test sau khi chốt cấu hình.

**Luồng:** kiểm tra dữ liệu → 5 backbone → 3 trục huấn luyện và EMA/kết hợp → 7 phương pháp suy luận ngoài mốc → chung kết và mốc với seed 0/1/2 → Excel, biểu đồ, báo cáo và đánh giá bằng eval.py gốc.

Bật **GPU** và **Internet**. Cấu hình nền 10 epoch, batch 32 cho mọi backbone. Chạy toàn bộ có thể cần nhiều giờ; thời gian thực được lưu theo từng epoch. Không thay đổi cấu hình sau khi đã xem test.

Nguồn: https://github.com/conanWinner/K4-DAY02-DoanQuangThang-2A202602395
'''), cell('markdown', '## 0. Nạp code và cài thư viện\nCode được đóng gói từ repo; không phụ thuộc đường dẫn clone.'), cell('code', setup),
        cell('markdown', '## 1. Kiểm tra môi trường và tính đúng của code\nCác fixture tổng hợp chỉ dùng cho kiểm tra, không đưa vào báo cáo DeepWeeds.'), cell('code', verify),
        cell('markdown', '''## 2–5. Chạy các vòng thí nghiệm

- B01–B05: ResNet-50, ResNeXt-50, ConvNeXt-Tiny, DeiT-Small, MobileNetV3-Large; cùng nền và seed.
- T00–T09: khởi tạo, augmentation, loss; thêm EMA và một kết hợp. Mỗi thí nghiệm riêng chỉ đổi một trục.
- I00–I07: một ảnh, lật/gộp xác suất, lật/gộp logit, 5 crop, độ phân giải 256, temperature scaling, AMP và gộp BN.
- F01/T00: cấu hình cuối và nền với ≥3 seed; R01 nếu có phương pháp p95 ≤100 ms. Nhiệt độ khớp trên val của từng seed trước test. Mỗi cấu hình/seed dùng một lần forward trên test (TTA gồm các view đã khai báo); tái sử dụng logits cho so sánh hiệu chuẩn.
- Báo cáo sinh từ log, prediction CSV; eval.py gốc tính lại toàn bộ test.
'''), cell('code', execute), cell('markdown', '## 6. Xem kết quả thật\nTải `lab_output` trong Output. Dataset và checkpoint không đưa lên Git.'), cell('code', '''from IPython.display import display, Markdown
print((OUTPUT_DIR / 'execution_status.json').read_text())
display(Markdown((OUTPUT_DIR / 'report.md').read_text()))
print('Excel:', OUTPUT_DIR / 'results.xlsx')
print('Prediction files:', len(list((OUTPUT_DIR / 'predictions').glob('*.csv'))))
''')]
    nb = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'deepweeds_source_sha256': {k: hashlib.sha256(base64.b64decode(v)).hexdigest() for k, v in payload.items()}}, 'cells': cells}
    for index, c in enumerate(cells):
        c['id'] = f'lab2-{index:02d}'
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / 'lab_day2.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    (CODE / 'lab_day2.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
    metadata_file = TARGET / 'kernel-metadata.json'
    meta = json.loads(metadata_file.read_text())
    meta.update(enable_gpu=True, enable_internet=True, is_private=True)
    metadata_file.write_text(json.dumps(meta, indent=2))
    print('Built self-contained notebook:', TARGET / 'lab_day2.ipynb')


if __name__ == '__main__':
    build()
