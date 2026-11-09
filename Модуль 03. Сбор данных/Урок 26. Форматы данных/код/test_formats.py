import json

import pytest

from annotation_formats import (VOC_EXAMPLE, Ann, ImageAnns, from_coco, from_yolo, read_voc, to_coco, to_yolo,
                                write_voc)

CATS = ["person", "chair"]


def boxes(img):
    return [(a.label, a.x1, a.y1, a.x2, a.y2) for a in img.anns]


def test_voc_is_one_based_inclusive():
    img = read_voc(VOC_EXAMPLE)
    person = [a for a in img.anns if a.label == "person"][0]
    assert (person.x1, person.y1, person.x2, person.y2) == (0, 0, 500, 375)    # рамка на весь кадр 500x375


def test_coco_round_trip_exact():
    img = read_voc(VOC_EXAMPLE)
    assert boxes(from_coco(json.loads(json.dumps(to_coco([img], CATS))))[0]) == boxes(img)


def test_yolo_round_trip_subpixel():
    img = read_voc(VOC_EXAMPLE)
    back = from_yolo(to_yolo(img, CATS), img.file_name, img.width, img.height, CATS)
    for (l1, *a), (l2, *b) in zip(boxes(img), boxes(back)):
        assert l1 == l2 and max(abs(p - q) for p, q in zip(a, b)) < 1e-3


def test_voc_write_read_round_trip():
    img = ImageAnns("a.jpg", 100, 80, [Ann("chair", 10, 20, 30, 40)])
    assert boxes(read_voc(write_voc(img))) == boxes(img)


def test_yolo_values_normalized():
    for line in to_yolo(read_voc(VOC_EXAMPLE), CATS).splitlines():
        assert all(0 <= float(v) <= 1 for v in line.split()[1:])


def test_unknown_class_raises():
    img = ImageAnns("a.jpg", 10, 10, [Ann("dog", 0, 0, 5, 5)])
    with pytest.raises(ValueError):
        to_yolo(img, CATS)
