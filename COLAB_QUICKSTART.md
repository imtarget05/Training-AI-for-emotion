# COLAB QUICKSTART — Copy-paste cells (branch `ml-optimization-v2`)

> Sinh tự động từ `training.ipynb` — KHÔNG sửa tay file này; sửa notebook rồi sinh lại.

> Protocol: PREFLIGHT → DATASET → **E0 → HARD GATE** → E1 → E2 → E3 → COMPARE.

> Gate FAIL thì DỪNG — không chạy E1/E2/E3. Kỳ vọng cải thiện chính là Macro-F1 + Fear recall,

> không chỉ accuracy.


## CELL 1 — Preflight GPU (phải thấy CUDA: True, Tesla T4)

Trước khi chạy: Menu **Runtime → Change runtime type → T4 GPU**.

```python
import torch, platform
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("GPU REQUIRED: CUDA unavailable. Do NOT fall back to CPU.")
print("GPU:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print("VRAM GB:", round(props.total_memory/1024**3, 2))
print("CUDA runtime:", torch.version.cuda)
```


## CELL 2 — Clone repo + xác minh nhánh/commit

Phải thấy commit có `E1-E3 full data` (HEAD mới nhất của nhánh). Nếu thiếu => DỪNG, báo lại.

```python
import os, subprocess
REPO = os.environ.get("REPO_DIR", "/content/Training-AI-for-emotion")
if not os.path.isdir(REPO):
    os.system(f"git clone https://github.com/imtarget05/Training-AI-for-emotion.git {REPO}")
os.chdir(REPO)
os.system("git fetch origin && git checkout ml-optimization-v2 && git pull origin ml-optimization-v2")
print(subprocess.run(["git","log","--oneline","-2"],capture_output=True,text=True).stdout)
os.system("python -m py_compile train.py && echo TRAIN_PY_OK")
```


## CELL 3 — Cài dependencies

~1 phút.

```python
import sys
os.system(f"{sys.executable} -m pip install -q mlflow tqdm scikit-learn pandas pyyaml kagglehub")
import sklearn, pandas, mlflow, tqdm, yaml
print("imports OK")
```


## CELL 4 — Tải FER2013 + assert test = 7178

Số ảnh từng class phải khớp audit (angry 3995 · disgust 436 · fear 4097 · happy 7215 · neutral 4965 · sad 4830 · surprise 3171).

```python
from pathlib import Path
import kagglehub
path = Path(kagglehub.dataset_download("msambare/fer2013"))
# Tim dong chua train/ + test/ (kagglehub co the doi cau truc version)
root = None
for p in [path, *path.rglob("train")]:
    if (p/"test").exists() if p.name=="train" else ((p/"train").exists() and (p/"test").exists()):
        root = p.parent if p.name=="train" else p
        break
assert root, f"Cannot locate FER2013 train/test under {path}"
DATA_DIR = str(root/"train")   # train.py tu stratified-split train/val
TEST_DIR = str(root/"test")
total = sum(len(f) for _,_,f in os.walk(DATA_DIR))
test_total = sum(len(f) for _,_,f in os.walk(TEST_DIR))
print("root:", root)
print("train images:", total)
print("test images:", test_total, "(expect 7178)")
assert test_total == 7178, f"TEST SET MISMATCH: got {test_total}"
for d in sorted(os.listdir(DATA_DIR)):
    print(" ", d, len(os.listdir(os.path.join(DATA_DIR,d))))
```


## CELL 5 — LAUNCH E0 (retrain V1 baseline, FULL data)

Chạy nền ~2–4h. Cell trả về sau ~90s để bạn thấy log đầu.

```python
LOG_DIR = os.environ.get("LOG_DIR", "/content")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////content/mlflow.db")
E0_LOG = f"{LOG_DIR}/e0.log"

# Runtime config: dong bo path dataset da discover (kagglehub co the doi version dir)
import yaml
base = yaml.safe_load(open("configs/train_v1_baseline.yaml"))
assert base.get("max_train_images_per_class") in (None, "null"), "E0 phai FULL data"
base["train_split_dir"] = DATA_DIR
RUNTIME_CFG = "/content/e0_runtime.yaml"
yaml.safe_dump(base, open(RUNTIME_CFG, "w"), sort_keys=False)
print(f"Runtime config: {RUNTIME_CFG} | train_split_dir={DATA_DIR}")

cmd = (f"MLFLOW_TRACKING_URI={MLFLOW_URI} "
       f"python train.py --config {RUNTIME_CFG} "
       f"> {E0_LOG} 2>&1 &")
os.system(cmd)
print("E0 launched (train.py, full FER2013 train split). Monitor voi cell tiep theo:")
print(f"  tail -20 {E0_LOG}")
os.system(f"sleep 90; tail -8 {E0_LOG}")
```


## CELL 6 — Monitor E0 (chạy lại mỗi ~30 phút)

Xong khi log có `Training complete` + `Final checkpoint:`.

```python
os.system("tail -15 /content/e0.log")
os.system("ps aux | grep 'train.py' | grep -v grep | head -2 || echo PROCESS_ENDED")
os.system("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
```


## CELL 7 — HARD GATE: evaluate ĐÚNG checkpoint E0 trên test set

CHỈ chạy khi CELL 6 thấy `Training complete`. Paste TOÀN BỘ output về để chấm gate.

```python
# Chi chay khi e0.log co "Training complete"
import glob, os.path
ckpts = sorted(glob.glob("outputs/*_final.pth"), key=os.path.getmtime)
assert ckpts, "NO E0 CHECKPOINT - E0 chua hoan thanh"
E0_CKPT = ckpts[-1]
print("Evaluating E0 checkpoint:", E0_CKPT)
assert "final_model.pth" not in E0_CKPT, "DAU PHIEN: khong duoc danh gia frozen weights!"
os.system(f"MLFLOW_TRACKING_URI=sqlite:////content/mlflow.db "
          f"python evaluate.py eval --weights {E0_CKPT} --data-dir {TEST_DIR!r}")
# GATE so voi baseline frozen: acc~49.83 | macro_f1~0.4210 | fear_recall~0.046
# Luu y: E0 la RETRAIN tu scratch (seed=42) nen KHONG doi trung khop tuyet doi;
# phan loai REPRODUCED / APPROXIMATELY REPRODUCED / NOT REPRODUCED do nguoi xet duyet.
# Paste toan bo output tren ve de cham gate truoc khi chay E1/E2/E3.
```


## CELL 8 — E1/E2/E3 (CHỈ sau khi gate PASS)

Bỏ comment `launch_experiment("weighted")` chạy E1 (~2–4h) → xong tới E2 (`finetune`) → E3 (`combined`). Monitor bằng lệnh tail mà cell in ra.

```python
# CHI chay khi HARD GATE E0 da PASS - bo comment tung experiment
# Moi run ~2-4h tren T4 (FULL data, khong con subset)
import yaml

def launch_experiment(name):
    base = yaml.safe_load(open(f"configs/train_v2_{name}.yaml"))
    assert base.get("max_train_images_per_class") is None, f"{name}: config phai FULL data"
    base["train_split_dir"] = DATA_DIR
    rcfg = f"/content/{name}_runtime.yaml"
    yaml.safe_dump(base, open(rcfg, "w"), sort_keys=False)
    log = f"/content/{name}.log"
    os.system(f"MLFLOW_TRACKING_URI=sqlite:////content/mlflow.db "
              f"python train.py --config {rcfg} > {log} 2>&1 &")
    print(f"{name} launched -> monitor: tail -20 {log}")

# launch_experiment("weighted")   # E1
# launch_experiment("finetune")   # E2
# launch_experiment("combined")   # E3
```


## CELL 9 — So sánh tất cả runs từ MLflow

```python
mlflow.set_tracking_uri("sqlite:////content/mlflow.db")
df = mlflow.search_runs(order_by=["start_time DESC"])
cols = [c for c in df.columns if "metrics." in c or c in ("run_id","status")]
print(df[cols].head(10).to_string())
```


## CELL 10 — EXPORT về Drive (SAU MỖI run!)

Colab XÓA HẾT dữ liệu khi runtime reset — export checkpoint + mlflow.db ngay sau mỗi experiment.

```python
# Colab mat du lieu khi runtime reset - EXPORT NGAY sau moi run xong
from google.colab import drive
drive.mount("/content/drive")
DEST = "/content/drive/MyDrive/emotion_ckpts"
import os
os.makedirs(DEST, exist_ok=True)
os.system(f"cp outputs/*.pth {DEST}/ 2>/dev/null")
os.system(f"cp mlflow.db {DEST}/ 2>/dev/null")   # sqlite tracking store
os.system(f"[ -d mlruns ] && cp -r mlruns {DEST}/ 2>/dev/null")
os.system(f"ls -la {DEST}/")
print("Exported:", DEST)
```
