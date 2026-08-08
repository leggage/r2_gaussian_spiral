import numpy as np

from data_preprocess.norm_pipeline import (
    dataset_output_path,
    _set_real_volume_bounds,
    _split_indices,
    stitch_projections,
    validate_config,
)


def test_dataset_output_path_is_hierarchical():
    cfg = {"dataset_type": "real", "organ": "chest", "model": "r2gs", "n_train": 400}
    assert dataset_output_path("data", cfg, "spiral").as_posix() == (
        "data/real/chest/spiral/ntrain400/r2gs"
    )
    assert dataset_output_path("data", cfg, "stitch", 200).as_posix() == (
        "data/real/chest/stitch/ntrain200/r2gs"
    )


def test_split_is_disjoint_and_deterministic():
    train, test = _split_indices(20, 6, 5, 7)
    assert len(train) == 6 and len(test) == 5
    assert not set(train) & set(test)
    assert np.array_equal(test, _split_indices(20, 6, 5, 7)[1])


def test_stitch_places_rotations_in_angle_slots():
    projs = np.stack([np.full((2, 3), i + 1, np.float32) for i in range(4)])
    stitched, angles, info = stitch_projections(
        projs, np.arange(4, dtype=float), np.array([0, 0.5, 1, 1.5]), 2, 1.0
    )
    assert stitched.shape == (2, 4, 3)
    assert angles.tolist() == [0.0, 1.0]
    assert info["raw_views"] == 4


def test_real_requires_raw_projection_path():
    cfg = dict(dataset_type="real", organ="aorta", model="r2gs", raw_gt="x", output_root="o", n_train=1, n_test=1, scanner={})
    try:
        validate_config(cfg)
    except ValueError as exc:
        assert "raw_proj" in str(exc)
    else:
        raise AssertionError("expected validation failure")


def test_real_volume_bounds_follow_projection_extrema():
    scanner = {"sVoxel": [10, 11, 12], "offOrigin": [0, 0, 0]}
    result = _set_real_volume_bounds(scanner, np.array([-4.0, 2.0]), {})
    assert scanner["sVoxel"] == [6.0, 6.0, 6.0]
    assert scanner["offOrigin"][2] == -1.0
    assert result["z_shift_range"] == [-4.0, 2.0]
