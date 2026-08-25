"""
train.py — minimal, controlled training harness for FER2013 emotion classification.

Built ONLY for optimization experiments. The production application (main.py)
does NOT depend on this file. Inference stays in model.py unchanged.

Experiments:
  E0 baseline reproduction   (configs/train_v1_baseline.yaml)
  E1 class-weighted loss     (configs/train_v2_weighted.yaml)
  E2 layer4 finetune         (configs/train_v2_finetune.yaml)
  E3 combined (E1+E2)        (configs/train_v2_combined.yaml)

CLI:
  python train.py --config configs/train_v1_baseline.yaml
  python train.py --config configs/train_v2_weighted.yaml
  python train.py --config configs/train_v2_finetune.yaml

Outputs a checkpoint to outputs/<run_hash>.pth and logs to MLflow.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import CLASS_NAMES, ResNetEncoder, FinetuneClassifier, device as model_device
from torchvision import transforms

Built ONLY for optimization experiments. The production application (main.py)
does NOT depend on this file. Inference stays in model.py unchanged.

Experiments:
  E0 baseline reproduction   (configs/train_v1_baseline.yaml)
  E1 class-weighted loss     (configs/train_v2_weighted.yaml)
  E2 layer4 finetune         (configs/train_v2_finetune.yaml)
  E3 combined (E1+E2)        (configs/train_v2_combined.yaml)

CLI:
  python train.py --config configs/train_v1_baseline.yaml
  python train.py --config configs/train_v2_weighted.yaml
  python train.py --config configs/train_v2_finetune.yaml

Outputs a checkpoint to outputs/<run_hash>.pth and logs to MLflow.

Design rules:
- Test set (FER2013 test/) is NEVER used during training/tuning.
- Train/val split from train/ only (stratified, seed=42).
- Exact preprocessing must match model.py inference_transform.
- Class names/order MUST match model.CLASS_NAMES exactly.
"""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import CLASS_NAMES, ResNetEncoder, FinetuneClassifier, device as model_device
from torchvision import transforms


# ── Re-use the EXACT inference preprocessing from model.py ──────────────
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
        if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


FER_LABEL_MAP = {"angry": "Anger", "disgust": "Disgust", "fear": "Fear",
                  "happy": "Happiness", "neutral": "Neutral", "sad": "Sadness",
                  "surprise": "Surprise"}


class FERDataset(Dataset):
    """Directory dataset: <root>/<ClassName>/*.jpg"""

    def __init__(self, root: Path, transform=None, max_per_class: Optional[int] = None, seed: int = 42):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[tuple[str, str]] = []
        fer_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        rng = random.Random(seed)
        for fer in fer_names:
            d = self.root / fer
            if not d.exists():
                d = self.root / fer.capitalize()  # project CLASS_NAMES format
            if not d.exists():
                continue
            files = sorted([f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".png"}])
            if max_per_class:
                rng.shuffle(files)
                files = files[:max_per_class]
            label_name = FER_LABEL_MAP.get(fer, fer.capitalize())
            self.samples += [(str(f), label_name) for f in files]
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_name = self.samples[idx]
        label_idx = CLASS_NAMES.index(label_name)
        from PIL import Image
        img = Image.open(path).convert("L")
                if self.transform:
            img = self.transform(img)
        return img, label_idx


def stratified_split(root: Path, val_ratio: float, seed: int) -> tuple[Path, Path]:
    """Create train/val directories with stratified split (copies files)."""
    import shutil
    tmp_root = Path("/tmp/fer_split")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    (tmp_root / "train").mkdir(parents=True, exist_ok=True)
    (tmp_root / "val").mkdir(parents=True, exist_ok=True)

    mapping = FER_LABEL_MAP.copy()
    rng = random.Random(seed)
    for fer, proj in mapping.items():
        src = root / fer
        if not src.exists():
            src = root / proj
        if not src.exists():
            continue
        files = sorted([f for f in src.iterdir() if f.suffix.lower() in {".jpg", ".png"}])
        rng.shuffle(files)
        n_val = int(len(files) * val_ratio)
        val_files = files[:n_val]
        train_files = files[n_val:]
        for f in train_files:
            dst = tmp_root / "train" / proj
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst / f.name)
        for f in val_files:
            dst = tmp_root / "val" / proj
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy(f, dst / f.name)
    return tmp_root / "train", tmp_root / "val"


def compute_class_weights(train_root: Path) -> torch.Tensor:
    """Inverse-frequency weights from train split only."""
    counts = torch.zeros(len(CLASS_NAMES))
    for proj in CLASS_NAMES:
        d = train_root / proj
        if d.exists():
            idx = CLASS_NAMES.index(proj)
            counts[idx] = len([f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".png"}])
    counts = torch.clamp(counts, min=1)
    weights = counts.sum() / (len(CLASS_NAMES) * counts)
    return weights


def build_model(freeze_backbone: bool, trainable_layers: str) -> FinetuneClassifier:
    encoder = ResNetEncoder(architecture="resnet50", pretrained=False)
    model = FinetuneClassifier(encoder, num_classes=len(CLASS_NAMES))

    # Freeze all backbone by default
    for p in model.encoder.parameters():
        p.requires_grad = False

    if trainable_layers in ("layer4_plus_head", "layer4"):
        # Unfreeze ResNet layer4 (last residual block group, index 6 in children)
        layer4 = list(model.encoder.resnet.children())[6]
        for p in layer4.parameters():
            p.requires_grad = True

        return model.to(model_device)


def train(cfg: dict):
    set_seed(cfg["seed"])

    # 1. Stratified split from train/
    train_dir, val_dir = stratified_split(
        Path(cfg["train_split_dir"]), cfg["validation_split"], cfg["seed"])

    # 2. Datasets
    aug = cfg.get("training", {}).get("augmentation", "train")
    train_transform_used = train_transform if aug.lower() in ("train", "true", "yes") else inference_transform
    train_ds = FERDataset(train_dir, transform=train_transform_used,
                          max_per_class=cfg.get("max_train_images_per_class"),
                          seed=cfg["seed"])
    val_ds = FERDataset(val_dir, transform=inference_transform,
                        max_per_class=cfg.get("max_val_images_per_class"),
                        seed=cfg["seed"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # 3. Model
    model = build_model(cfg.get("freeze_backbone", True), cfg["trainable_layers"])

    # 4. Loss
    loss_cfg = cfg.get("loss", {})
    if loss_cfg.get("class_weights") == "train_inverse_frequency":
        weights = compute_class_weights(train_dir).to(model_device)
        print(f"Class weights (E1): {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 5. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg["learning_rate"],
                                  weight_decay=cfg.get("weight_decay", 1e-5))

    # 6. Scheduler
    scheduler = None
    sched_cfg = cfg.get("scheduler", {})
    if sched_cfg.get("type") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=sched_cfg.get("T_max", cfg["max_epochs"]))

    # 7. MLflow
    _init_mlflow(cfg)

    # 8. Training loop
    best_val_f1 = -1.0
    patience_counter = 0
        run_hash = hashlib.md5(json.dumps({k: str(v) for k, v in cfg.items()}, sort_keys=True).encode()).hexdigest()[:8]
    last_ckpt = None

    for epoch in range(cfg["max_epochs"]):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(model_device), labels.to(model_device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if scheduler:
            scheduler.step()

        # 9. Validation
        val_metrics = _evaluate(model, val_loader)
        avg_loss = total_loss / max(len(train_loader), 1)

        print(f"Epoch {epoch+1}/{cfg['max_epochs']} | loss={avg_loss:.4f} "
              f"| val_acc={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}")

        # 10. Early stopping on macro-F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            out_path = Path("outputs") / f"model_{run_hash}_epoch{epoch+1}.pth"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_path)
            last_ckpt = out_path
            print(f"   → checkpoint saved: {out_path}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.get("early_stopping_patience", 5):
                print(f"   → early stopping at epoch {epoch+1}")
                break

    # 11. Final checkpoint + metadata
    final_path = Path("outputs") / f"model_{run_hash}_final.pth"
    if last_ckpt and last_ckpt.exists():
        model.load_state_dict(torch.load(last_ckpt, map_location="cpu", weights_only=True))
    torch.save(model.state_dict(), final_path)

    _log_mlflow(cfg, val_metrics, final_path, run_hash)

    print(f"\nTraining complete.")
    print(f"Best val macro-F1: {best_val_f1:.4f}")
    print(f"Final checkpoint: {final_path}")
        return final_path


def _evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(model_device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for p, l in zip(all_preds, all_labels):
        cm[l, p] += 1

    from evaluate import _classification_report
    report, totals = _classification_report(cm, CLASS_NAMES)
    return totals


def _init_mlflow(cfg):
    try:
        import mlflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(cfg.get("mlflow_experiment", "emotion-resnet50"))
        return True
    except Exception:
        return False


def _log_mlflow(cfg, metrics, artifact_path, run_hash):
    try:
        import mlflow
        with mlflow.start_run():
            mlflow.log_params({k: str(v) for k, v in cfg.items()})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("train.py_run_id", run_hash)
            if artifact_path and Path(str(artifact_path)).exists():
                mlflow.log_artifact(str(artifact_path))
    except Exception:
        pass


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f)
    result = {}
    result.update(cfg)
    result["freeze_backbone"] = cfg.get("freeze_backbone", True)
    if cfg.get("trainable_layers") == "layer4_plus_head":
        result["freeze_backbone"] = False
    result["loss"] = cfg.get("loss", {"type": "cross_entropy"})
    result["training"] = cfg.get("training", {})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    train(cfg)
    print(f"Total wall time: {time.time()-t0:.1f}s")






