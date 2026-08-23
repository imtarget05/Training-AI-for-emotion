"""Run INSIDE the torch-enabled container image to verify MLflow logging."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlflow.db"
from evaluate import _git_commit, _log_mlflow

ok = _log_mlflow(
    {
        "experiment": "emotion-resnet50",
        "model_arch": "resnet50",
        "weights": "final_model.pth",
        "data_dir": "demo",
        "num_classes": 7,
        "images": 100,
        "git_commit": _git_commit(),
    },
    {"accuracy": 0.82, "macro_f1": 0.79, "eval_seconds": 3.2},
    "image/confusion_matrix.png",
)
print("MLFLOW_LOGGED =", ok)
assert ok
print("CONTAINER_MLFLOW_OK")