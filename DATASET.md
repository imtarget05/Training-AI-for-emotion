# DATASET.md — Evaluation dataset provenance & reproducibility

## Dataset used for evaluation

| Field | Value |
|---|---|
| Name | FER2013 (facial expression recognition) |
| Source | Kaggle — `msambare/fer2013` mirror of the original ICML 2013 challenge data |
| URL | https://www.kaggle.com/datasets/msambare/fer2013 |
| Original provenance | Goodfellow et al., *Challenges in Representation Learning: A report on three machine learning contests* (ICML/NIPS 2013) |
| Licence | Distributed for academic/research use. **NOT public-domain.** Verify the original competition terms before any commercial use. |
| Version | kagglehub dataset version 1 (accessed 2026-08) |
| Classes | 7 — matches this project's label space exactly after folder renaming |
| Test split size | 7,178 grayscale 48×48 JPEG face crops |
| Split used | `test/` only (the model was trained externally; using the official test split avoids train/eval overlap with our checkpoint's training run to the extent FER2013's split guarantees it) |

## Class name mapping

FER2013 folders are lowercase; this project's `CLASS_NAMES` (model.py) are capitalized:

```
angry     -> Anger
disgust   -> Disgust      (only 111 test images — rare class)
fear      -> Fear
happy     -> Happiness
neutral   -> Neutral
sad       -> Sadness
surprise  -> Surprise
```

## Reproduce

```bash
pip install kagglehub          # anonymous download works for this public dataset
python scripts/prepare_fer2013.py            # creates data/test/<ClassName> symlinks
python evaluate.py eval --data-dir data/test  # real metrics + confusion matrix + MLflow run
```

Manual alternative: download the zip from Kaggle, extract, and run
`python scripts/prepare_fer2013.py --source /path/to/fer2013`.

## Notes / caveats

- Images are 48×48 grayscale; the inference pipeline converts to RGB and resizes to 224×224.
- FER2013 labels contain known annotation noise (~10%); published human agreement tops out around 65±5%, so treat absolute accuracy in that light.
- The dataset is NOT committed to git (`data/` is gitignored). Only symlinks are created locally.
