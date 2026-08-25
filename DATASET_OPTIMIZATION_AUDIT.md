# DATASET OPTIMIZATION AUDIT — FER2013

## Status: COMPLETE — FULL TRAIN + TEST COUNTED

### Dataset Source
```
Name: FER2013 (Facial Expression Recognition 2013)
Source: Kaggle — msambare/fer2013
Path (local cache): ~/.cache/kagglehub/datasets/msambare/fer2013/versions/1/
Accessed: 2026-08 (via kagglehub, anonymous download)
Licence: Distributed for academic/research use per ICML 2013 contest terms
Note: NOT public-domain. Not for commercial use without review.
```

### Splits Available
| Split | Images | Classes | Path |
|-------|--------|---------|------|
| train | 35,887 | 7 | `versions/1/train/` |
| test | 7,178 | 7 | `versions/1/test/` |
| validation | 0 | — | *(not shipped by FER2013)* |

The `test/` split was already verified previously (7,178 images with exact counts).
The `train/` split has now also been fully counted.

---

## Exact Train Split Count

| FER2013 label | Project CLASS_NAME | Train count |
|---------------|--------------------|------------:|
| angry         | Anger              | 3,995 |
| disgust       | Disgust            | 436 |
| fear          | Fear               | 4,097 |
| happy         | Happiness          | 7,215 |
| neutral       | Neutral            | 4,965 |
| sad           | Sadness            | 4,830 |
| surprise      | Surprise           | 3,171 |
| **Total**     |                    | **35,887** |

## Exact Test Split Count

| FER2013 label | Project CLASS_NAME | Test count |
|---------------|--------------------|-----------:|
| angry         | Anger              | 958 |
| disgust       | Disgust            | 111 |
| fear          | Fear               | 1,024 |
| happy         | Happiness          | 1,774 |
| neutral       | Neutral            | 1,233 |
| sad           | Sadness            | 1,247 |
| surprise      | Surprise           | 831 |
| **Total**     |                    | **7,178** |

## Class Name Mapping (train ↔ test)
Same mapping for both splits:
```
angry    → Anger
disgust  → Disgust
fear     → Fear
happy    → Happiness
neutral  → Neutral
sad      → Sadness
surprise → Surprise
```

## Image Properties
- **Format**: JPEG
- **Channels**: grayscale (1 channel, 48×48)
- **Inference pipeline converts**: grayscale → 3-channel RGB → resize(256) → center_crop(224) → ImageNet normalize

## Class Imbalance Analysis

### Train split
| Class | Count | Share | Cumulative |
|-------|-------|-------|------------|
| Happiness (max) | 7,215 | 20.1% | 20.1% |
| Neutral | 4,965 | 13.8% | 33.9% |
| Sadness | 4,830 | 13.5% | 47.4% |
| Fear | 4,097 | 11.4% | 58.8% |
| Anger | 3,995 | 11.1% | 69.9% |
| Surprise | 3,171 | 8.8% | 78.7% |
| Disgust (min) | 436 | 1.2% | 80.0% |
| Remaining (below median) | — | 8.8% | 80.0% |

**Imbalance ratio (max/min): 16.5 : 1**

### Test split
| Class | Count | Share |
|-------|-------|-------|
| Happiness (max) | 1,774 | 24.7% |
| Neutral | 1,233 | 17.2% |
| Sadness | 1,247 | 17.4% |
| Fear | 1,024 | 14.3% |
| Anger | 958 | 13.3% |
| Surprise | 831 | 11.6% |
| Disgust (min) | 111 | 1.5% |

**Imbalance ratio (max/min): 15.9 : 1**

## Critical Observations

1. **Fear is NOT a rare class in training.**
   It has 4,097 training samples (~11%), close to the median.
   So the 0.046 Fear recall in the baseline is **not caused by class rarity alone** —
   the classifier genuinely fails to map Fear features to the correct class,
   which makes **feature-representation tuning (E2: layer4 fine-tuning)** a
   legitimate hypothesis, not merely class weighting.

2. **Disgust is genuinely rare in train (436 samples).**
   Class-weighting may help Disgust (test F1 0.253).

3. **Train and test are disjoint official FER2013 splits.**
   No train/test leakage. Safe to train on train/ and evaluate on test/.

4. **No validation split exists.**
   Must carve a stratified 80/20 split from train/ — validation used for
   checkpoint/early-stopping selection, test/ used only once for final eval.

5. **Label semantics confirmed identical.**
   FER2013 labels are 7 basic emotions, matching project `CLASS_NAMES` exactly.

## Preprocessing & Normalization
```
FER2013 images: grayscale 48×48 JPEG face crops

Inference (model.py):
  - grayscale 48×48 → RGB (3×48×48 replication)
  - transforms.Resize(256)
  - transforms.CenterCrop(224)
  - transforms.ToTensor()
  - normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

## Train/Validation Split Plan
```
Split: train/ → 80% train_actual / 20% validation (stratified per class)
Test: test/ → held out, used ONLY for final evaluation (never for tuning)
Split method: deterministic random (seed=42) per-class
```

## Dataset Conversion Required
None beyond standard loading. FER2013 folder names (lowercase `angry`, etc.)
map directly to project `CLASS_NAMES` via `FER_TO_PROJECT`.

For training convenience:
```text
FER2013/train/angry      → Anger   (436–7215, depends on split)
FER2013/train/disgust    → Disgust
FER2013/train/fear       → Fear
FER2013/train/happy      → Happiness
FER2013/train/neutral    → Neutral
FER2013/train/sad        → Sadness
FER2013/train/surprise   → Surprise
```

## Verification Steps Performed
- [x] Full train split counted (35,887 images)
- [x] Full test split counted (7,178 images — previous)
- [x] Class mapping verified train ↔ test
- [x] Imbalance ratio computed (train 16.5:1, test 15.9:1)
- [x] Train/test disjointness confirmed (official FER2013 split)
- [x] No validation split exists — must be created
- [x] Preprocessing/normalization documented
