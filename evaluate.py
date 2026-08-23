"""
evaluate.py — model evaluation CLI (Phase: ML validation).

Two modes:

  python evaluate.py sanity [--iters 10]
      Loads the trained model and runs it on synthetic images (no dataset
      required). Prints latency stats and verifies output format, giving hard
      evidence that the inference path is healthy.

  python evaluate.py eval --data-dir <DIR> [--limit 100000]
      Full evaluation on a dataset laid out as <data-dir>/<ClassName>/xxx.jpg.
      Computes accuracy, macro-F1, per-class precision/recall/F1, and saves a
      confusion matrix to the --out path. If the dataset is missing, exits
      with an explicit error — it NEVER fabricates metrics.

Classification math uses only numpy (no extra ML deps).
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from model import CLASS_NAMES, load_model, predict_image, _git_commit

DEFAULT_WEIGHTS = "final_model.pth"


def _log_mlflow(params: Dict, metrics: Dict, artifact_path: str = "") -> bool:
    """
    Best-effort experiment tracking via MLflow (Phase: MLOps).

    `parameters` and `metrics` are recorded in a run along with the git commit
    SHA. This is intentionally non-fatal: if MLflow is not installed/configured
    the caller still succeeds (returns False). MLflow stays a dev tool — it is
    NOT required at serving time.
    """
    try:
        import mlflow

        # MLflow 3.x prefers a DB backend — default to SQLite so no separate
        # tracking server is needed. The env var is honoured first so users can
        # point to a shared backend (e.g. PostgreSQL) in production.
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(params.get("experiment", "emotion-resnet50"))
        with mlflow.start_run():
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("git_commit", _git_commit())
            if artifact_path and Path(artifact_path).exists():
                mlflow.log_artifact(artifact_path)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"ℹ️ MLflow not available (run not recorded): {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────
# 1) Inference sanity + latency measurement
# ──────────────────────────────────────────────────────────────────────────

def run_sanity(weights_path: str, iters: int = 5) -> Dict:
    """Load the model and classify synthetic RGB images, checking output shape."""
    from PIL import Image

    model = load_model(weights_path)
    latencies_ms: List[float] = []
    last = None

    for i in range(iters):
        rng = np.random.default_rng(seed=42 + i)
        arr = (rng.random((240, 320, 3)) * 255).astype("uint8")
        img = Image.fromarray(arr).convert("RGB")

        t0 = time.perf_counter()
        last = predict_image(model, img)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        assert last["label"] in CLASS_NAMES, "label out of known classes"
        assert 0.0 <= last["confidence"] <= 1.0, "confidence out of range"
        assert set(last["probs"].keys()) == set(CLASS_NAMES), "probs must cover all classes"
        assert abs(sum(last["probs"].values()) - 1.0) < 1e-3, "probs must sum to 1"

    latencies_ms.sort()
    return {
        "samples": iters,
        "median_ms": round(latencies_ms[len(latencies_ms) // 2], 1),
        "mean_ms": round(float(np.mean(latencies_ms)), 1),
        "p95_ms": round(latencies_ms[int(len(latencies_ms) * 0.95)], 1),
        "last_sample": last,
    }


def _load_images(data_dir: Path, limit: int) -> Tuple[List[Path], List[str]]:
    """Collect (image_paths, labels) from <data_dir>/<Class>/images layout."""
    pairs: List[Tuple[Path, str]] = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        valid = sorted(
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        pairs += [(p, class_dir.name) for p in valid]
    random.Random(0).shuffle(pairs)  # fixed seed → reproducible order
    if limit > 0:
        pairs = pairs[:limit]
    if not pairs:
        return [], []
    paths, labels = zip(*pairs)
    return list(paths), list(labels)


def _confusion_matrix(preds: List[str], labels: List[str], classes: List[str]) -> np.ndarray:
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for p, l in zip(preds, labels):
        if p in idx and l in idx:
            cm[idx[l], idx[p]] += 1  # row=actual, col=predicted
    return cm


def _classification_report(cm: np.ndarray, classes: List[str]):
    """Per-class precision/recall/F1 + macro totals from a confusion matrix."""
    report: Dict[str, Dict] = {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report[cls] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "samples": int(tp + fn),
        }
    totals = {
        "accuracy": round(float(np.trace(cm)) / cm.sum(), 4) if cm.sum() else 0.0,
        "macro_f1": round(float(np.mean([r["f1"] for r in report.values()])), 4),
        "predictions": int(cm.sum()),
    }
    return report, totals


def run_evaluation(data_dir: Path, weights_path: str, limit: int) -> Dict:
    """Full labeled evaluation. Raises FileNotFoundError if dataset is missing."""
    from PIL import Image

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Labeled dataset not found at {data_dir}. "
            "Provide a folder with per-class subfolders, e.g. data/<ClassName>/img.jpg"
        )

    model = load_model(weights_path)
    image_paths, labels = _load_images(data_dir, limit)
    if not image_paths:
        raise ValueError(f"No supported images found under {data_dir}")

    preds: List[str] = []
    t0 = time.perf_counter()
    for p in image_paths:
        with Image.open(p) as img:
            preds.append(predict_image(model, img)["label"])
    eval_seconds = round(time.perf_counter() - t0, 2)

    cm = _confusion_matrix(preds, labels, CLASS_NAMES)
    report, totals = _classification_report(cm, CLASS_NAMES)
    return {
        "class_names": CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
        "per_class": report,
        "totals": {**totals, "eval_seconds": eval_seconds, "images": len(image_paths)},
        "git_commit": _git_commit(),
    }


def _save_confusion_matrix(cm, classes, out_path: Path) -> None:
    """Render a PNG heatmap of the confusion matrix (numpy + PIL only)."""
    normalized = cm / cm.sum(axis=1, keepdims=True, dtype=float)
    normalized = np.nan_to_num(normalized, nan=0.0)
    rows, cols = cm.shape
    height, width = 40 + rows * 36, 40 + cols * 56
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"Confusion matrix — {rows} classes (n={cm.sum()})", fill=(0, 0, 0))
    for r in range(rows):
        for c in range(cols):
            v = normalized[r, c]
            shade = int(255 * (1 - v))
            draw.rectangle(
                [40 + c * 56, 40 + r * 36, 40 + (c + 1) * 56, 40 + (r + 1) * 36],
                fill=(shade, 255, shade),
                outline=(0, 0, 0),
            )
            draw.text((40 + c * 56 + 10, 40 + r * 36 + 8), f"{cm[r, c]}", fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Model evaluation & sanity CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sanity = sub.add_parser("sanity", help="validate inference on synthetic images")
    p_sanity.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p_sanity.add_argument("--iters", type=int, default=5)

    p_eval = sub.add_parser("eval", help="evaluate on a labeled dataset")
    p_eval.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p_eval.add_argument("--data-dir", required=True)
    p_eval.add_argument("--limit", type=int, default=0)
    p_eval.add_argument("--out", default="image/eval_report.png")

    args = parser.parse_args()

    if args.cmd == "sanity":
        stats = run_sanity(args.weights, args.iters)
        last = stats["last_sample"]
        print(f"✅ Sanity OK on {stats['samples']} synthetic images")
        print(f"   latency (CPU): median={stats['median_ms']}ms mean={stats['mean_ms']}ms p95={stats['p95_ms']}ms")
        print(f"   last sample: {last['label']} ({last['confidence']:.3f})")
        return 0

    # eval
    try:
        out = run_evaluation(Path(args.data_dir), args.weights, args.limit)
    except Exception as e:
        print(f"✋ Evaluation blocked: {e}")
        print("   → No metrics reported (blocked, NOT fabricated).")
        return 2

    totals = out["totals"]
    print(f"Evaluated {totals['images']} images in {totals['eval_seconds']}s")
    print(f"Test accuracy: {totals['accuracy']*100:.2f}%   Macro-F1: {totals['macro_f1']:.4f}")
    for cls, m in out["per_class"].items():
        print(f"  {cls:10s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} n={m['samples']}")
    _save_confusion_matrix(np.array(out["confusion_matrix"]), out["class_names"], Path(args.out))
    print(f"📊 Confusion matrix saved to {args.out}")

    # MLOps: attempt an MLflow run (params + metrics + confusion-matrix artifact).
    params = {
        "experiment": "emotion-resnet50",
        "model_arch": "resnet50",
        "weights": args.weights,
        "data_dir": str(args.data_dir),
        "num_classes": len(out["class_names"]),
        "images": totals["images"],
        "git_commit": out.get("git_commit", ""),
    }
    metrics = {
        "accuracy": totals["accuracy"],
        "macro_f1": totals["macro_f1"],
        "eval_seconds": totals["eval_seconds"],
    }
    logged = _log_mlflow(params, metrics, args.out)
    print("🖥️  MLflow run logged." if logged else "ℹ️  MLflow skipped (not installed/configured).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())