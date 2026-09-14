"""Stage 1 of the training pipeline: fetch, audit and split Banking77.

Run: `python -m training.prepare_data`

Why this is a separate stage: data preparation must be reproducible and
auditable on its own. If it lived inside the training script, the leakage audit
below would run only when someone trained, and its findings would be buried in
training logs instead of written to a report file.

The audit is the part that matters. Banking77 ships an official train/test
split, and it is tempting to trust it blindly. We check anyway:
  - exact duplicate texts *within* train (inflates CV optimism),
  - exact duplicate texts *across* train and test (direct test leakage),
  - label coverage (every intent must appear in both splits),
  - class balance (drives the choice of macro F1 over accuracy).
Findings are written to reports/data_audit.json and printed. We do not silently
drop the overlap: we record it, quantify its effect on the headline metric, and
report both numbers.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
SOURCE = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
)
VAL_SIZE = 0.15
RANDOM_STATE = 42


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_if_missing() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, remote in (("banking77_train.csv", "train.csv"), ("banking77_test.csv", "test.csv")):
        target = RAW_DIR / name
        if target.exists():
            continue
        import urllib.request

        url = f"{SOURCE}/{remote}"
        print(f"downloading {url} -> {target}")
        urllib.request.urlretrieve(url, target)  # noqa: S310 - fixed, trusted URL


def normalise(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def audit(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    train_norm, test_norm = normalise(train.text), normalise(test.text)
    overlap = sorted(set(train_norm) & set(test_norm))
    counts = train.category.value_counts()
    report = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_labels_train": int(train.category.nunique()),
        "n_labels_test": int(test.category.nunique()),
        "labels_only_in_train": sorted(set(train.category) - set(test.category)),
        "labels_only_in_test": sorted(set(test.category) - set(train.category)),
        "duplicate_texts_within_train": int(train_norm.duplicated().sum()),
        "duplicate_texts_within_test": int(test_norm.duplicated().sum()),
        "train_test_exact_overlap": len(overlap),
        "train_test_overlap_pct_of_test": round(100 * len(overlap) / len(test), 3),
        "overlap_examples": overlap[:5],
        "class_count_min": int(counts.min()),
        "class_count_max": int(counts.max()),
        "class_imbalance_ratio": round(float(counts.max() / counts.min()), 2),
        "sha256_train": sha256_file(RAW_DIR / "banking77_train.csv"),
        "sha256_test": sha256_file(RAW_DIR / "banking77_test.csv"),
    }
    return report


def main() -> int:
    download_if_missing()
    train = pd.read_csv(RAW_DIR / "banking77_train.csv")
    test = pd.read_csv(RAW_DIR / "banking77_test.csv")
    for frame, name in ((train, "train"), (test, "test")):
        missing = {"text", "category"} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns {missing}")
        if frame.text.isna().any() or frame.category.isna().any():
            raise ValueError(f"{name} contains nulls")

    report = audit(train, test)

    # Train/validation split. Stratified, because the rarest intent has ~35
    # examples and a random split can leave a class with none in validation.
    tr, va = train_test_split(
        train, test_size=VAL_SIZE, stratify=train.category, random_state=RANDOM_STATE
    )
    report["n_train_split"] = int(len(tr))
    report["n_val_split"] = int(len(va))
    report["val_size"] = VAL_SIZE
    report["random_state"] = RANDOM_STATE

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tr.to_csv(PROCESSED_DIR / "train.csv", index=False)
    va.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    (REPORTS_DIR / "data_audit.json").write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    if report["train_test_exact_overlap"]:
        print(
            f"\nNOTE: {report['train_test_exact_overlap']} test rows "
            f"({report['train_test_overlap_pct_of_test']}% of test) are verbatim duplicates of "
            "training rows. Reported alongside the headline metric; "
            "training/evaluate.py also reports macro F1 on the de-duplicated test set."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
