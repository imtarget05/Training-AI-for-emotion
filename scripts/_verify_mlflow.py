"""Ad-hoc verification of MLflow integration (Phase 4) — run with the venv Python."""
import os
import sys

import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# --- Stub heavy ML imports so evaluate.py can be imported without torch/cv2 ---
# A metaclass that makes every attribute access return a *class*
# (subclass-of-_StubBase), so code like ``class ResNetEncoder(nn.Module)``
# works even though torch is absent.

class _StubMeta(type):
    def __getattr__(cls, name):
        return _StubMeta(name, (_StubBase,), {})

    def __add__(cls, other):
        return str(other)

    def __radd__(cls, other):
        return str(other)


class _StubBase(metaclass=_StubMeta):
    """Universal stub: subclassable, callable, iterable, truthy-false."""
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _StubBase()
    def __iter__(self): return iter([])
    def __len__(self): return 0
    def __bool__(self): return False
    def __getitem__(self, key): return _StubBase()
    def __setitem__(self, key, value): pass
    def to(self, *a, **k): return self
    def eval(self): return self
    def state_dict(self): return {}


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        return _StubMeta(name, (_StubBase,), {})


_STUBBED = (
    "torch", "torch.nn", "torch.cuda", "torchvision", "torchvision.models",
    "torchvision.transforms", "cv2", "cv2.data", "cv2.CascadeClassifier",
)
for name in _STUBBED:
    if name not in sys.modules:
        sys.modules[name] = _StubModule(name)

# Explicit overrides — behave like real torch objects
sys.modules["torch"].nn = _StubMeta("nn", (_StubBase,), {})
sys.modules["torch"].cuda = _StubModule("torch.cuda")
sys.modules["torch"].cuda.is_available = lambda: False  # type: ignore[attr-defined]
sys.modules["torch"].device = lambda *a, **k: "cpu"  # type: ignore[attr-defined]
sys.modules["torch"].no_grad = lambda fn=None: fn or (lambda f: f)  # type: ignore[attr-defined]

from evaluate import _git_commit, _log_mlflow  # noqa: E402

os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

params = {
    "experiment": "emotion-resnet50",
    "model_arch": "resnet50",
    "weights": "final_model.pth",
    "data_dir": "demo",
    "num_classes": 7,
    "images": 100,
    "git_commit": _git_commit(),
}
metrics = {"accuracy": 0.82, "macro_f1": 0.79, "eval_seconds": 3.2}
logged = _log_mlflow(params, metrics, os.path.join(ROOT, "image", "confusion_matrix.png"))
print(f"MLflow logged: {logged}")
assert logged is True, "MLflow was expected to log a run"
print("git_commit:", _git_commit() or "n/a")
print("MLFLOW_OK")