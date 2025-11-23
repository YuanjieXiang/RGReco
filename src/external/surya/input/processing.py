from typing import List

import cv2
import numpy as np
from PIL import Image
from src.external.surya.settings import settings


def convert_if_not_rgb(images: List[Image.Image]) -> List[Image.Image]:
    new_images = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        new_images.append(image)
    return new_images


def slice_bboxes_from_image(image: Image.Image, bboxes):
    lines = []
    for bbox in bboxes:
        line = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        if line.size[0] == 0:
            print(f"Warning: found an empty line with bbox {bbox}")
        lines.append(line)
    return lines


def slice_polys_from_image(image: Image.Image, polys):
    image_array = np.array(image, dtype=np.uint8)
    lines = []
    for idx, poly in enumerate(polys):
        lines.append(slice_and_pad_poly(image_array, poly))
    return lines


def slice_and_pad_poly(image_array: np.array, coordinates):
    # Draw polygon onto mask
    coordinates = [(corner[0], corner[1]) for corner in coordinates]
    bbox = [min([x[0] for x in coordinates]), min([x[1] for x in coordinates]), max([x[0] for x in coordinates]), max([x[1] for x in coordinates])]

    # We mask out anything not in the polygon
    cropped_polygon = image_array[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
    coordinates = [(x - bbox[0], y - bbox[1]) for x, y in coordinates]

    # Pad the area outside the polygon with the pad value
    mask = np.zeros(cropped_polygon.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(coordinates)], 1)
    mask = np.stack([mask] * 3, axis=-1)

    cropped_polygon[mask == 0] = settings.RECOGNITION_PAD_VALUE
    rectangle_image = Image.fromarray(cropped_polygon)

    return rectangle_image
