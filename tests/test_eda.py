"""Tests that only need scikit-learn / pandas (no TensorFlow)."""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def test_dx_metadata_consistency():
    from EDA import DX_FULL_NAME, N_CLASSES

    assert len(DX_FULL_NAME) == N_CLASSES
    assert set(DX_FULL_NAME) == {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}


def test_group_aware_split_no_lesion_leakage():
    import pandas as pd
    from EDA import group_aware_split

    rows = []
    # Enough lesions per class for stratified group folding
    for i in range(40):
        dx = ["nv", "mel", "bkl", "bcc"][i % 4]
        for j in range(2):
            rows.append(
                {
                    "lesion_id": f"L{i}",
                    "image_id": f"I{i}_{j}",
                    "dx": dx,
                    "image_path": f"/tmp/I{i}_{j}.jpg",
                }
            )
    df = pd.DataFrame(rows)
    train_df, val_df, test_df = group_aware_split(df, test_size=0.2, val_size=0.2)

    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    assert train_lesions.isdisjoint(val_lesions)
    assert train_lesions.isdisjoint(test_lesions)
    assert val_lesions.isdisjoint(test_lesions)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


def test_group_aware_split_preserves_all_classes():
    import pandas as pd
    from EDA import group_aware_split

    rows = []
    classes = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
    for i, dx in enumerate(classes * 8):
        rows.append(
            {
                "lesion_id": f"L{i}",
                "image_id": f"I{i}",
                "dx": dx,
                "image_path": f"/tmp/I{i}.jpg",
            }
        )
    df = pd.DataFrame(rows)
    train_df, val_df, test_df = group_aware_split(df, test_size=0.2, val_size=0.2)

    # Train should still contain every class with this balanced synthetic set
    assert set(train_df["dx"]) == set(classes)
