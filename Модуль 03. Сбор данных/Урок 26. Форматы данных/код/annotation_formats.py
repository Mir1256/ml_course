"""Конвертеры CV-разметки: Pascal VOC XML <-> внутреннее представление <-> COCO JSON <-> YOLO txt.

Внутреннее представление — пиксельные xyxy в непрерывных координатах (левый верхний угол кадра = (0, 0)).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass


@dataclass
class Ann:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ImageAnns:
    file_name: str
    width: int
    height: int
    anns: list[Ann]


# ---------- Pascal VOC: XML на картинку, пиксели 1-based и включительно ----------
def read_voc(xml_text: str) -> ImageAnns:
    root = ET.fromstring(xml_text)
    size = root.find("size")
    anns = []
    for obj in root.iter("object"):
        b = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        # VOC: пиксели нумеруются с 1, xmax включительно -> непрерывные координаты [xmin-1, xmax)
        anns.append(Ann(obj.find("name").text, xmin - 1, ymin - 1, xmax, ymax))
    return ImageAnns(root.find("filename").text, int(size.find("width").text), int(size.find("height").text), anns)


def write_voc(img: ImageAnns) -> str:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = img.file_name
    size = ET.SubElement(root, "size")
    for k, v in (("width", img.width), ("height", img.height), ("depth", 3)):
        ET.SubElement(size, k).text = str(v)
    for a in img.anns:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = a.label
        ET.SubElement(obj, "difficult").text = "0"
        b = ET.SubElement(obj, "bndbox")
        for k, v in (("xmin", a.x1 + 1), ("ymin", a.y1 + 1), ("xmax", a.x2), ("ymax", a.y2)):
            ET.SubElement(b, k).text = str(round(v))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


# ---------- COCO: один JSON на датасет, bbox = [x, y, w, h] в пикселях ----------
def to_coco(images: list[ImageAnns], categories: list[str]) -> dict:
    cat_id = {name: i + 1 for i, name in enumerate(categories)}
    out = {"images": [], "annotations": [], "categories": [{"id": i, "name": n} for n, i in cat_id.items()]}
    aid = 0
    for img_id, img in enumerate(images):
        out["images"].append({"id": img_id, "file_name": img.file_name, "width": img.width, "height": img.height})
        for a in img.anns:
            w, h = a.x2 - a.x1, a.y2 - a.y1
            out["annotations"].append({"id": aid, "image_id": img_id, "category_id": cat_id[a.label],
                                       "bbox": [a.x1, a.y1, w, h], "area": w * h, "iscrowd": 0})
            aid += 1
    return out


def from_coco(coco: dict) -> list[ImageAnns]:
    names = {c["id"]: c["name"] for c in coco["categories"]}
    by_image = {im["id"]: ImageAnns(im["file_name"], im["width"], im["height"], []) for im in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        by_image[a["image_id"]].anns.append(Ann(names[a["category_id"]], x, y, x + w, y + h))
    return list(by_image.values())


# ---------- YOLO: txt на картинку, «class cx cy w h», всё нормировано на [0, 1] ----------
def to_yolo(img: ImageAnns, categories: list[str]) -> str:
    lines = []
    for a in img.anns:
        cx, cy = (a.x1 + a.x2) / 2 / img.width, (a.y1 + a.y2) / 2 / img.height
        w, h = (a.x2 - a.x1) / img.width, (a.y2 - a.y1) / img.height
        lines.append(f"{categories.index(a.label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def from_yolo(text: str, file_name: str, width: int, height: int, categories: list[str]) -> ImageAnns:
    anns = []
    for line in text.strip().splitlines():
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * width, float(cy) * height, float(w) * width, float(h) * height
        anns.append(Ann(categories[int(c)], cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return ImageAnns(file_name, width, height, anns)


VOC_EXAMPLE = """<annotation>
  <filename>000005.jpg</filename>
  <size><width>500</width><height>375</height><depth>3</depth></size>
  <object><name>chair</name><difficult>0</difficult>
    <bndbox><xmin>263</xmin><ymin>211</ymin><xmax>324</xmax><ymax>339</ymax></bndbox></object>
  <object><name>person</name><difficult>0</difficult>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>500</xmax><ymax>375</ymax></bndbox></object>
</annotation>"""

if __name__ == "__main__":
    cats = ["person", "chair"]
    img = read_voc(VOC_EXAMPLE)
    print("VOC ->", img)
    coco = to_coco([img], cats)
    print("COCO bbox:", [a["bbox"] for a in coco["annotations"]])
    yolo = to_yolo(img, cats)
    print("YOLO:\n" + yolo)
    back = from_yolo(yolo, img.file_name, img.width, img.height, cats)
    voc_again = read_voc(write_voc(from_coco(json.loads(json.dumps(to_coco([back], cats))))[0]))
    max_err = max(abs(getattr(a, k) - getattr(b, k)) for a, b in zip(img.anns, voc_again.anns) for k in ("x1", "y1", "x2", "y2"))
    print(f"round-trip VOC -> COCO -> YOLO -> COCO -> VOC: макс. ошибка {max_err:.2e} px")
