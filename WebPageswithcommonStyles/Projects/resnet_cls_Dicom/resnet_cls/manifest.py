"""Manifest (CSV) creation with PATIENT-LEVEL splits, and the leakage check.
Imports no torch.  Columns: path,label,patient_id,split"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
from sklearn.model_selection import GroupShuffleSplit

from .dicom_io import hash_patient_id, is_dicom

IMAGE_EXTS = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def assert_no_leakage(rows) -> None:
    seen = defaultdict(set)
    for r in rows:
        seen[r["patient_id"]].add(r["split"])
    bad = [p for p, s in seen.items() if len(s) > 1]
    if bad:
        raise ValueError(f"{len(bad)} patient(s) appear in more than one split, e.g. {bad[:3]}")


def build_manifest(root, out_csv, salt: str, val_frac=0.15, test_frac=0.15, seed=42,
                   max_attempts=50) -> dict:
    """root/<class>/<file>. Patient id comes from the DICOM header (hashed), or from the file
    name prefix before the first '_' for non-DICOM files. Splits are assigned PER PATIENT."""
    rows = []
    for cls_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for f in sorted(cls_dir.rglob("*")):
            if f.suffix.lower() not in IMAGE_EXTS and not (f.is_file() and not f.suffix and is_dicom(f)):
                continue
            if is_dicom(f):
                ds = pydicom.dcmread(str(f), stop_before_pixels=True)   # header only: fast
                pid = str(ds.get("PatientID", "")) or f.stem
            else:
                pid = f.stem.split("_")[0]
            rows.append({"path": str(f), "label": cls_dir.name,
                         "patient_id": hash_patient_id(pid, salt)})
    if not rows:
        raise ValueError(f"No images found under {root}")

    groups = np.array([r["patient_id"] for r in rows])
    labels = {r["label"] for r in rows}
    idx = np.arange(len(rows))

    # Patient-level splits can leave a rare class out of val/test by chance. Try successive
    # seeds until every class appears in every split; fail loudly if that is impossible.
    for attempt in range(max_attempts):
        rs = seed + attempt
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=rs)
        train_i, rest_i = next(gss.split(idx, groups=groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=test_frac / (val_frac + test_frac), random_state=rs)
        val_rel, test_rel = next(gss2.split(rest_i, groups=groups[rest_i]))
        parts = {"train": train_i, "val": rest_i[val_rel], "test": rest_i[test_rel]}
        if all({rows[i]["label"] for i in ix} == labels for ix in parts.values()):
            break
    else:
        raise ValueError(f"Could not find a patient-level split containing all {len(labels)} classes "
                         f"in train, val and test after {max_attempts} seeds: collect more patients "
                         f"for the rare classes or reduce val_frac/test_frac.")
    for name, ix in parts.items():
        for i in ix:
            rows[i]["split"] = name
    assert_no_leakage(rows)

    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "patient_id", "split"])
        w.writeheader()
        w.writerows(rows)

    summary = defaultdict(lambda: defaultdict(int))     # split -> class -> count
    for r in rows:
        summary[r["split"]][r["label"]] += 1
    return {s: dict(c) for s, c in summary.items()}     # per-split class counts, for your logs


def read_manifest(csv_path, split: str):
    with open(csv_path, newline="") as fh:
        all_rows = list(csv.DictReader(fh))
    assert_no_leakage(all_rows)
    classes = sorted({r["label"] for r in all_rows})    # one class mapping for all splits
    return classes, [r for r in all_rows if r["split"] == split]
