import numpy as np
import pandas as pd
import pytest

from groupby_numpy import group_mean, group_median


def test_group_mean_and_median_small():
    keys = np.array([2, 1, 2, 1, 1])
    vals = np.array([10.0, 1.0, 20.0, 3.0, 2.0])
    u, m = group_mean(keys, vals)
    assert u.tolist() == [1, 2] and np.allclose(m, [2.0, 15.0])
    _, med = group_median(keys, vals)
    assert np.allclose(med, [2.0, 15.0])


def test_copy_on_write_subset_does_not_touch_parent():
    df = pd.DataFrame({"a": [1, 2, 3]})
    sub = df[df.a > 1]
    sub.loc[:, "a"] = 0
    assert df.a.tolist() == [1, 2, 3]


def test_merge_validate_catches_duplicate_keys():
    left = pd.DataFrame({"k": [1, 1], "v": [0, 1]})
    right = pd.DataFrame({"k": [1, 1], "w": [5, 6]})
    with pytest.raises(pd.errors.MergeError):
        left.merge(right, on="k", validate="many_to_one")
