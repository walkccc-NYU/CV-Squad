import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

from constants import (ANNOTATION_DIR, BBOXES_COORDS_FILE, GENERATED_DIR,
                       IMAGE_COORDS_FILE, ORIGINAL_DENSITIES_DIR,
                       ORIGINAL_IMAGES_DIR, RESIZED_DENSITIES_DIR,
                       RESIZED_IMAGES_DIR)

parser = argparse.ArgumentParser(description='Few Show Counting Resize')
parser.add_argument('-ms', '--max-size', type=int, default=192)
parser.add_argument('-p', '--use-pad',
                    action='store_true', default=False)
args = parser.parse_args()


def __shrink(src, H: int, W: int, max_size: int):
    shrink_ratio = max_size / max(H, W)
    H = int(H * shrink_ratio)
    W = int(W * shrink_ratio)
    assert H <= max_size and W <= max_size
    return cv2.resize(src, (W, H)), H, W, shrink_ratio


def __shrink_bboxes(bboxes, shrink_ratio: float, H: int):
    """ Shrink bboxes while shrinking the image """
    bboxes *= shrink_ratio
    bboxes[:, :2] = np.floor(bboxes[:, :2])
    bboxes[:, 2:] = np.ceil(bboxes[:, 2:])
    bboxes = np.clip(bboxes, a_min=0, a_max=H)


def __translate_bboxes(bboxes, pad_top: int, pad_left: int):
    """ Translate bboxes to abide new coordinates after padding the image """
    bboxes[:, 0] += pad_left
    bboxes[:, 1] += pad_top
    bboxes[:, 2] += pad_left
    bboxes[:, 3] += pad_top


def __get_pads(H: int, W: int, max_size: int) -> Tuple[int, int, int, int]:
    height_diff = max_size - H
    width_diff = max_size - W

    half_height_diff = height_diff // 2
    half_width_diff = width_diff // 2

    pad_top = pad_bottom = half_height_diff
    pad_left = pad_right = half_width_diff

    if height_diff & 1:
        pad_top += 1
    if width_diff & 1:
        pad_left += 1

    return pad_top, pad_bottom, pad_left, pad_right


if __name__ == '__main__':
    with open(ANNOTATION_DIR) as f:
        annotations = json.load(f)

    if not os.path.exists(GENERATED_DIR):
        os.mkdir(GENERATED_DIR)

    for _dir in [RESIZED_IMAGES_DIR, RESIZED_DENSITIES_DIR]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    bboxes_coords_dict: Dict[str, List[List[int]]] = {}
    image_coords_dict: Dict[str, List[int]] = {}

    for filename in os.listdir(ORIGINAL_IMAGES_DIR):
        print(f'Resizing image {filename}...')
        image_path = os.path.join(ORIGINAL_IMAGES_DIR, filename)
        image = cv2.imread(image_path)

        anno = annotations[filename]
        example_coordinates = anno['box_examples_coordinates']
        bboxes = []

        for coord in example_coordinates:
            x_min, y_min = coord[0]  # top-left
            x_max, y_max = coord[2]  # bottom-right
            bboxes.append([x_min, y_min, x_max, y_max])

        bboxes = np.array(bboxes).astype(float)

        H, W, C = image.shape
        if H > args.max_size or W > args.max_size:
            image, H, W, shrink_ratio = __shrink(
                image, H, W, args.max_size)
            __shrink_bboxes(bboxes, shrink_ratio, H)

        if args.use_pad:
            t, b, l, r = __get_pads(H, W, args.max_size)
            image = cv2.copyMakeBorder(
                image, t, b, l, r, cv2.BORDER_CONSTANT)
            assert image.shape == (args.max_size, args.max_size, 3)
            __translate_bboxes(bboxes, t, l)
            # (x_min, y_min, x_max, y_max)
            image_coords_dict[filename] = [l, t, l + W, t + H]
        else:
            # (x_min, y_min, x_max, y_max)
            image_coords_dict[filename] = [0, 0, W, H]

        resized_image_path = os.path.join(RESIZED_IMAGES_DIR, filename)
        cv2.imwrite(resized_image_path, image)
        bboxes_coords_dict[filename] = np.asarray(bboxes).astype(int).tolist()

    with open(IMAGE_COORDS_FILE, 'w') as f:
        f.write(json.dumps(image_coords_dict))

    with open(BBOXES_COORDS_FILE, 'w') as f:
        f.write(json.dumps(bboxes_coords_dict))

    for filename in os.listdir(ORIGINAL_DENSITIES_DIR):
        print(f'Resizing density {filename}...')
        density_path = os.path.join(ORIGINAL_DENSITIES_DIR, filename)
        density = np.load(density_path)

        H, W = density.shape
        if H > args.max_size or W > args.max_size:
            original_density_sum: float = np.sum(density)
            density, H, W, _ = __shrink(density, H, W, args.max_size)
            new_density_sum: float = np.sum(density)
            if new_density_sum > 0:
                # make sure sum(density) remain unchanged
                density *= (original_density_sum / new_density_sum)

        if args.use_pad:
            t, b, l, r = __get_pads(H, W, args.max_size)
            density = np.pad(density, ((t, b), (l, r)), 'constant')
            assert density.shape == (args.max_size, args.max_size)

        resized_density_path = os.path.join(
            RESIZED_DENSITIES_DIR, filename)
        np.save(resized_density_path, density)
