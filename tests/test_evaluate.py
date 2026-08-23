"""
Regression tests for the evaluate.py dataset loader (Phase: ML evaluation).

These run without torch/weights: the `model` module is stubbed by conftest,
so only the pure-Python loader/report logic is exercised.
"""

import importlib.util
import pathlib
import sys

import pytest
from PIL import Image


@pytest.fixture(scope="module")
def evaluate_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_under_test",
        pathlib.Path(__file__).resolve().parents[1] / "evaluate.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mini_dataset(tmp_path):
    """2 classes × 3 JPEGs each, in <root>/<Class>/ layout."""
    for cls in ["Sadness", "Anger"]:
        d = tmp_path / cls
        d.mkdir()
        for i in range(3):
            Image.new("RGB", (48, 48), (200, 0, 0)).save(d / f"img{i}.jpg")
    return tmp_path


def test_loader_returns_all_images(evaluate_module, mini_dataset):
    paths, labels = evaluate_module._load_images(mini_dataset, limit=0)
    assert len(paths) == 6 and len(labels) == 6  # regression: was None (dead-code bug)
    assert set(labels) == {"Sadness", "Anger"}


def test_loader_limit_and_determinism(evaluate_module, mini_dataset):
    p1, l1 = evaluate_module._load_images(mini_dataset, limit=4)
    p2, l2 = evaluate_module._load_images(mini_dataset, limit=4)
    assert len(p1) == 4
    assert p1 == p2 and l1 == l2  # fixed seed → reproducible order


def test_loader_empty_dir(evaluate_module, tmp_path):
    assert evaluate_module._load_images(tmp_path, limit=0) == ([], [])


def test_confusion_matrix_orientation(evaluate_module):
    classes = ["A", "B"]
    cm = evaluate_module._confusion_matrix(["A", "B", "A"], ["A", "B", "B"], classes)
    # row=actual, col=predicted: A→A correct, B→B correct, actual B predicted A
    assert cm.tolist() == [[1, 0], [1, 1]]


def test_report_macro_f1(evaluate_module):
    import numpy as np

    cm = np.array([[2, 0], [1, 1]])
    report, totals = evaluate_module._classification_report(cm, ["A", "B"])
    # class A: P=2/2, R=2/3, F1=0.8 ; class B: P=1/1, R=1/2, F1≈0.667
    assert abs(totals["macro_f1"] - (0.8 + 2 / 3) / 2) < 1e-3
    assert abs(totals["accuracy"] - 3 / 4) < 1e-3
