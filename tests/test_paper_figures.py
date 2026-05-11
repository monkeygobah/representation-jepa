from __future__ import annotations

import pandas as pd
import pytest

from paper_figures.data import assert_no_seginit, dataset_geometry, dataset_landmarks, rep_geometry, rep_landmarks
from paper_figures.metadata import parse_run_name


def test_parse_run_name_handles_baselines_and_variants() -> None:
    parts = parse_run_name("geometry-fixedcompute-1m-bhep-imagenet-50ksteps")
    assert parts.scale == "1m"
    assert parts.objective == "bhep"
    assert parts.init == "imagenet"
    assert parts.is_50k


def test_assert_no_seginit_rejects_main_output_rows() -> None:
    with pytest.raises(ValueError):
        assert_no_seginit(pd.DataFrame({"init": ["random", "seginit"]}), "unit")


def test_current_dataset_track_filters_have_expected_shape() -> None:
    geometry = dataset_geometry()
    landmarks, baselines, external_baselines = dataset_landmarks()
    assert len(geometry) == 54
    assert len(landmarks) == 36
    assert len(baselines) == 4
    assert len(external_baselines) == 4
    assert "seginit" not in set(geometry["init"].astype(str))
    assert "seg_init" not in set(landmarks["init_mode"].astype(str))


def test_current_rep_learning_filters_have_expected_shape() -> None:
    geometry = rep_geometry()
    landmarks = rep_landmarks()
    assert len(geometry) == 18
    assert len(landmarks) == 12
    assert set(geometry["objective"].astype(str)) == {"lejepa", "bhep", "eppartial"}
    assert set(landmarks["objective"].astype(str)) == {"lejepa", "bhep", "eppartial"}
