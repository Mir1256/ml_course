import numpy as np
import pytest

from discrete import box_iou, connected_components, mask_iou, nms


def test_iou_properties():
    b = (0, 0, 10, 10)
    assert box_iou(b, b) == 1.0
    assert box_iou(b, (20, 20, 30, 30)) == 0.0
    assert box_iou(b, (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert box_iou((0, 0, 4, 4), (1, 1, 3, 3)) == pytest.approx(4 / 16)     # вложенная рамка


def test_mask_iou_matches_box_iou():
    a = np.zeros((20, 20), bool); a[0:10, 0:10] = True
    b = np.zeros((20, 20), bool); b[0:10, 5:15] = True
    assert mask_iou(a, b) == pytest.approx(box_iou((0, 0, 10, 10), (5, 0, 15, 10)))


def test_connectivity_matters():
    diag = np.eye(4, dtype=bool)
    assert connected_components(diag, 4)[1] == 4
    assert connected_components(diag, 8)[1] == 1


def test_nms_keeps_best_and_far_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    keep = nms(boxes, np.array([0.6, 0.9, 0.5]), 0.5)
    assert keep == [1, 2]
