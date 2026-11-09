import numpy as np

from vis import PALETTE, draw_boxes_np, overlay_mask_np


def test_draw_does_not_modify_input_and_draws_edges():
    img = np.zeros((20, 20, 3), np.uint8)
    out = draw_boxes_np(img, np.array([[2, 3, 10, 12.0]]), np.array([1]), thickness=1)
    assert img.sum() == 0                                    # исходник не изменён
    assert (out[3, 5] == PALETTE[1]).all() and (out[12, 5] == PALETTE[1]).all()
    assert (out[7, 6] == 0).all()                            # внутри рамки пусто


def test_box_outside_frame_is_clipped():
    img = np.zeros((10, 10, 3), np.uint8)
    out = draw_boxes_np(img, np.array([[-5, -5, 50, 50.0]]), np.array([0]), thickness=1)
    assert out[0, 0].any() and out[9, 9].any()


def test_overlay_alpha():
    img = np.full((2, 2, 3), 100, np.uint8)
    mask = np.array([[True, False], [False, False]])
    out = overlay_mask_np(img, mask, (200, 200, 200), alpha=0.5)
    assert (out[0, 0] == 150).all() and (out[1, 1] == 100).all()
