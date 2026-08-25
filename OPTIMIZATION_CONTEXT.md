# ML Optimization Context & Recovery Report

## Current Git State
```
Branch: ml-optimization-v2 (new, tracking origin/ml-optimization-v2)
Frozen production baseline (main): d5676375c61d67d984363e539c26d22a19c9f346
Documentation commit: ab3451a8cfd084f0a61fb964c6f80e32ba00d3ad
```

## Hardware Audit
| Component | Status |
|-----------|--------|
| Environment | `/tmp/emotion_venv` (Python 3.14.6, torch 2.13.0) |
| GPU | **None** — CUDA available = False |
| CPU | 8 cores |
| Compute mode | **CPU-only training** |
| Disk | 290 GB free |
| Implication | Full 35k-image FER2013 training on CPU is impractical; optimization experiments will use **controlled subset training** to validate methodology and measure relative improvement direction. |

## Frozen ML Baseline
```
Architecture: ResNet-50 (ImageNet-pretrained=False, i.e. trained from scratch)
  → frozen feature extractor (nn.Sequential(*list(resnet.children())[:-1]))
  → 2048-d feature
  → Linear(2048, 7) trainable head

Dataset: FER2013 test split
Images: 7,178
Accuracy: 49.83%
Macro-F1: 0.4210
Fear recall: 0.046
Fear F1: 0.081

Inference preprocessing (model.py):
  resize(256) → center_crop(224) → to_tensor → ImageNet normalize
  (grayscale 48×48 → 3-channel RGB → 224×224)
```

## Dataset Audit Results

### Full FER2013 (kagglehub cache)
Found at `~/.cache/kagglehub/datasets/msambare/fer2013/versions/1/`

### Train split (35,887 images)
| Class (FER2013 lowercase) | Count |
|--------------------------|-------|
| angry | 3,995 |
| disgust | 436 |
| fear | 4,097 |
| happy | 7,215 |
| neutral | 4,965 |
| sad | 4,830 |
| surprise | 3,171 |

### Test split (7,178 images)
| Class | Count |
|-------|-------|
| angry | 958 |
| disgust | 111 |
| fear | 1,024 |
| happy | 1,774 |
| neutral | 1,233 |
| sad | 1,247 |
| surprise | 831 |

### Class mapping (FER2013 lowercase → project CLASS_NAMES)
```
fer2013/angry      → Anger
fer2013/disgust    → Disgust
fer2013/fear       → Fear
fer2013/happy      → Happiness
fer2013/neutral    → Neutral
fer2013/sad        → Sadness
fer2013/surprise   → Surprise
Order in CLASS_NAMES: [Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral]
```

### Class Imbalance Ratio
```
Most frequent: happy (7,215)
Least frequent: disgust (436)
Imbalance ratio (max/min): 16.5:1
```

### Validation split
FER2013 ships only `train/` and `test/`. No `validation/` split exists.
Will carve validation set from train (80/20 stratified) — held out from test set.

### Key Observations
1. **No train/val/test leakage risk**: train and test are disjoint official splits.
2. **Test set will NOT be used during training** (used only for final evaluation).
3. **No validation split exists** — must be created from train.
4. **Class imbalance severe** (16.5:1) — class-weighted loss / sampling is justified.
5. **Fear is NOT rare in train** (4,097 samples) — the model simply fails to recognize it, making fine-tuning + class-weighting hypotheses non-trivial.

## Experiments Planned
| ID | Hypothesis | Change |
|----|-----------|--------|
| E0 | Baseline reproduction | Same config, from train split |
| E1 | Class imbalance hurts minority recall | Class-weighted loss |
| E2 | Frozen backbone limits emotion-specific features | Fine-tune layer4 + head |
| E3 | Combined strategy | E1 + E2 |

## Existing Optimization Files
```
train.py                  — will be created (training harness)
configs/train_v1_baseline.yaml
configs/train_v2_weighted.yaml
configs/train_v2_finetune.yaml
scripts/reproduce_v1.py   — minimal checkpoint reproduction
evaluate.py               — already exists (test eval + MLflow logging)
```

## Decision Log
- **CPU-only confirmed**: design experiments for subset training feasibility
- **Baseline class order documented**: CLASS_NAMES = [Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral]
- **Train/test split verified**: no leakage
- **Validation strategy**: 80/20 stratified split from train

## Status
```
BASELINE: VERIFIED
OPTIMIZATION STATUS: NOT YET STARTED
HARDWARE CONSTRAINT: CPU ONLY (documented)
```

## Optimization Constraints
- Use FER2013 `train/` for training + validation (stratified split)
- Use FER2013 `test/` ONLY for final model evaluation
- Preserve exact class mapping and order
- Preserve inference preprocessing exactly
- Log all experiments to MLflow
- Use deterministic seeds
- Subset training feasible on CPU (document subset size + extrapolation)
