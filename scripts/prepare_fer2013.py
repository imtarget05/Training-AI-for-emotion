"""
prepare_fer2013.py — build the evaluation dataset expected by evaluate.py.

FER2013 (msambare/fer2013 mirror on Kaggle) ships lowercase class folder names;
this project's model uses the exact CLASS_NAMES strings from model.py. This
script creates data/test/<ClassName> symlinks pointing at the downloaded
kagglehub copy, so no large data is duplicated or committed to git.

Usage:
    python scripts/prepare_fer2013.py [--source <kagglehub-or-manual-path>] [--out data/test]

If --source is omitted, kagglehub is used to fetch/locate the dataset
(anonymous download works for this public dataset).
"""

import argparse
import pathlib
import shutil
import sys

# FER2013 folder name -> project CLASS_NAMES string (model.py)
FER_TO_PROJECT = {
    "angry": "Anger",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happiness",
    "neutral": "Neutral",
    "sad": "Sadness",
    "surprise": "Surprise",
}


def locate_source(explicit: str | None) -> pathlib.Path:
    if explicit:
        p = pathlib.Path(explicit).expanduser().resolve()
        if not (p / "test").is_dir():
            sys.exit(f"✋ {p} does not contain a 'test/' directory")
        return p
    try:
        import kagglehub

        path = pathlib.Path(kagglehub.dataset_download("msambare/fer2013"))
    except Exception as e:  # noqa: BLE001
        sys.exit(
            f"✋ Could not locate FER2013 via kagglehub ({e}).\n"
            "   Download manually from https://www.kaggle.com/datasets/msambare/fer2013\n"
            "   and pass --source /path/to/fer2013 (must contain test/<class>/ dirs)."
        )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None, help="path containing train/ and test/")
    parser.add_argument("--out", default="data/test", help="output directory to create")
    args = parser.parse_args()

    src_root = locate_source(args.source)
    src_test = src_root / "test"
    out_root = pathlib.Path(args.out)

    missing = [f for f in FER_TO_PROJECT if not (src_test / f).is_dir()]
    if missing:
        sys.exit(f"✋ Source test split is missing classes: {missing}")

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    total = 0
    for fer_name, proj_name in sorted(FER_TO_PROJECT.items()):
        src = (src_test / fer_name).resolve()
        dst = out_root / proj_name
        # Copy (not symlink): the evaluation may run inside a Docker container
        # where absolute host paths from kagglehub's cache don't exist.
        shutil.copytree(src, dst)
        n = sum(1 for f in dst.rglob("*") if f.is_file())
        total += n
        print(f"  {fer_name:<10s} -> {proj_name:<10s} ({n} images)")

    print(f"✅ Prepared {total} evaluation images under {out_root}/")
    print("   Next: python evaluate.py eval --data-dir", str(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
