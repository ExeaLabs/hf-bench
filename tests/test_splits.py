"""Tests for temporal splitting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest

from hfbench.data.demo import make_demo_dataset
from hfbench.data.split import temporal_split


@pytest.fixture
def demo_df():
    return make_demo_dataset(n=300, seed=42)


def test_split_sizes(demo_df):
    train, val, test = temporal_split(demo_df)
    n = len(demo_df)
    assert len(train) + len(val) + len(test) == n
    assert len(train) > len(val)
    assert len(train) > len(test)


def test_split_chronological_order(demo_df):
    train, val, test = temporal_split(demo_df)
    # Last train admit must be <= first val admit
    assert train["admittime"].max() <= val["admittime"].min()
    # Last val admit must be <= first test admit
    assert val["admittime"].max() <= test["admittime"].min()


def test_no_hadm_id_overlap(demo_df):
    train, val, test = temporal_split(demo_df)
    train_ids = set(train["hadm_id"])
    val_ids = set(val["hadm_id"])
    test_ids = set(test["hadm_id"])
    assert train_ids.isdisjoint(val_ids), "Train/val overlap"
    assert train_ids.isdisjoint(test_ids), "Train/test overlap"
    assert val_ids.isdisjoint(test_ids), "Val/test overlap"


def test_label_preserved(demo_df):
    train, val, test = temporal_split(demo_df)
    for split in [train, val, test]:
        assert "readmit_30d" in split.columns
        assert split["readmit_30d"].isin([0, 1]).all()
