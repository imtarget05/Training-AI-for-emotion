"""
reproduce_v1.py — minimal V1 reproduction from train split.

Loads the existing final_model.pth checkpoint, evaluates it on FER2013 test,
and records metrics to MLflow under a dedicated "v1-reproduction" run.

CLI:
  python reproduce_v1.py [--weights final_model.pth] [--data-dir data/test]

Does NOT train. Does NOT modify production code or final_model.pth.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import CLASS_NAMES, load_model, predict_image
from evaluate import _classification_report, _log_mlflow

from PIL import Image
from torchvision import transforms
import numpy as np
import torch


inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def evaluate_folder(model, data_dir):
    """Run inference on <data_dir>/<CLASS_NAME>/ images for all 7 classes."""
    data_dir = Path(data_dir)
    all_preds, all_labels = [], []
    for i, cls in enumerate(CLASS_NAMES):
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                continue
            img = Image.open(f).convert("L")
            result = predict_image(model, img)
            pred = CLASS_NAMES.index(result["label"])
            all_preds.append(pred)
            all_labels.append(i)

    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for p, l in zip(all_preds, all_labels):
        cm[l, p] += 1

    report, totals = _classification_report(cm, CLASS_NAMES)
    return report, totals, cm.tolist()


def main():
    parser = argparse.ArgumentParser(description="Reproduce V1 baseline evaluation")
    parser.add_argument("--weights", default="final_model.pth")
    parser.add_argument("--data-dir", default="data/test")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.is_file():
        print(f"✋ Weights not found: {weights_path}")
        return 1

    print(f"Loading model: {weights_path}")
    model = load_model(str(weights_path))
    print("✅ Model loaded")

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"✋ Test data not found: {data_dir}")
        print("   Run: python scripts/prepare_fer2013.py first")
        return 1

    print("Running evaluation on test set...")
    report, totals, cm = evaluate_folder(model, data_dir)

    print(f"\n=== V1 Reproduction ===")
    print(f"Test samples: {totals['images']}")
    print(f"Test accuracy: {totals['accuracy']*100:.2f}%")
    print(f"Macro-F1: {totals['macro_f1']:.4f}")
    print("\nPer-class:")
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    for cls, m in report.items():
        print(f"{cls:<10} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>6.4f}")

    # Save artifact
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.array(cm), cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("V1 Baseline — Confusion Matrix")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                    color="white" if cm[i][j] > cm.max()/2 else "black")
    plt.colorbar(im)
    out_path = "image/v1_reproduction_cm.png"
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"\nConfusion matrix saved to {out_path}")

    # Log to MLflow
    params = {
        "experiment": "v1-reproduction",
        "model": "emotion-resnet50-v1",
        "arch": "resnet50",
        "weights": str(weights_path),
        "data_dir": str(data_dir),
        "images": totals["images"],
    }
    metrics = {
        "accuracy": totals["accuracy"],
        "macro_f1": totals["macro_f1"],
    }
    logged = _log_mlflow(params, metrics, out_path)
    print("🖥️  MLflow run logged." if logged else "ℹ️  MLflow skipped (not installed/configured).")

    # Summary
    fear = report.get("Fear", {})
    print(f"\n=== Summary ===")
    print(f"v1 baseline: accuracy={totals['accuracy']*100:.2f}% macro_f1={totals['macro_f1']:.4f} fear_recall={fear.get('recall',0):.3f}")
    print(f"Expected:    accuracy≈49.83%            macro_f1≈0.4210   fear_recall≈0.046")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())