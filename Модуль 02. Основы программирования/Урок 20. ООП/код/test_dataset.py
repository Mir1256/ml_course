import dataclasses

import numpy as np
import pytest

from dataset import BaseDataset, Box, Compose, HFlip, Sample, SyntheticDetection, Transform


def test_box_is_immutable_and_hashable():
    b = Box(0, 0, 10, 10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.x1 = 5
    assert len({b, Box(0, 0, 10, 10)}) == 1


def test_box_validation():
    with pytest.raises(ValueError):
        Box(5, 5, 5, 10)


def test_hflip_twice_is_identity():
    s = Sample(np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3), [Box(1, 0, 3, 2)])
    twice = Compose([HFlip(), HFlip()])(s)
    assert np.array_equal(twice.image, s.image) and twice.boxes == s.boxes


def test_dataset_protocols():
    ds = SyntheticDetection(4)
    assert len(ds) == 4 and len(list(ds)) == 4
    with pytest.raises(IndexError):
        ds[4]
    assert isinstance(HFlip(), Transform)
    with pytest.raises(TypeError):
        BaseDataset()


def test_default_factory_not_shared():
    a, b = Sample(np.zeros((1, 1, 3))), Sample(np.zeros((1, 1, 3)))
    a.boxes.append(Box(0, 0, 1, 1))
    assert b.boxes == []
