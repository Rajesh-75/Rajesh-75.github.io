import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from resnet_cls.dicom_io import InvalidInputError, load_image
from resnet_cls.manifest import assert_no_leakage, build_manifest


def make_dcm(path, arr, photometric="MONOCHROME2", modality="DX", view="PA", pid="P1",
             slope=1.0, intercept=0.0):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.is_little_endian, ds.is_implicit_VR = True, False
    ds.Modality, ds.ViewPosition, ds.PatientID = modality, view, pid
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, photometric
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    ds.PixelData = arr.astype(np.uint16).tobytes()
    ds.save_as(str(path))


GRADIENT = np.tile(np.linspace(0, 4095, 64), (64, 1))        # 12-bit horizontal ramp


def test_monochrome1_is_inverted(tmp_path):
    make_dcm(tmp_path / "a.dcm", GRADIENT, "MONOCHROME2")
    make_dcm(tmp_path / "b.dcm", GRADIENT, "MONOCHROME1")
    a, b = np.asarray(load_image(tmp_path / "a.dcm")), np.asarray(load_image(tmp_path / "b.dcm"))
    assert a.dtype == np.uint8 and a[0, 0] < a[0, -1]        # MONOCHROME2: dark -> bright
    assert np.abs(a.astype(int) + b.astype(int) - 255).max() <= 1   # MONOCHROME1: mirrored


def test_wrong_view_and_modality_rejected(tmp_path):
    make_dcm(tmp_path / "lat.dcm", GRADIENT, view="LL")
    make_dcm(tmp_path / "ct.dcm", GRADIENT, modality="CT")
    for name in ("lat.dcm", "ct.dcm"):
        with pytest.raises(InvalidInputError):
            load_image(tmp_path / name)


def test_manifest_has_no_patient_leakage(tmp_path):
    for cls in ("Normal", "Pneumonia"):
        (tmp_path / "raw" / cls).mkdir(parents=True)
        for p in range(12):                                   # 12 patients x 3 images per class
            for k in range(3):
                make_dcm(tmp_path / "raw" / cls / f"{p}_{k}.dcm", GRADIENT, pid=f"{cls}{p}")
    summary = build_manifest(tmp_path / "raw", tmp_path / "m.csv", salt="s")
    assert set(summary) == {"train", "val", "test"}
    assert all(set(c) == {"Normal", "Pneumonia"} for c in summary.values())   # every class in every split
    import csv
    rows = list(csv.DictReader(open(tmp_path / "m.csv")))
    assert_no_leakage(rows)
    rows[0]["split"] = "train"; rows[1] = dict(rows[0], split="val")   # force a leak
    with pytest.raises(ValueError):
        assert_no_leakage(rows)


def test_impossible_split_fails_loudly(tmp_path):
    # the rare class has only ONE patient, so it can never be in train, val and test at once
    for cls, n_patients in (("Normal", 12), ("Rare", 1)):
        (tmp_path / "raw" / cls).mkdir(parents=True)
        for p in range(n_patients):
            make_dcm(tmp_path / "raw" / cls / f"{p}_0.dcm", GRADIENT, pid=f"{cls}{p}")
    with pytest.raises(ValueError, match="all 2 classes"):
        build_manifest(tmp_path / "raw", tmp_path / "m.csv", salt="s", max_attempts=5)
