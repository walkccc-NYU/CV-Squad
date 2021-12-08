import argparse
import json
import os
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np

from constants import (ANNOTATION_DIR, BBOXES_COORDS_FILE, IMAGE_COORDS_FILE,
                       ORIGINAL_DENSITIES_DIR, ORIGINAL_IMAGES_DIR,
                       RESIZED_DENSITIES_DIR, RESIZED_IMAGES_DIR)

parser = argparse.ArgumentParser(description='Few Show Counting Visualizer')
parser.add_argument('-i', '--image-num', type=int, default=2)
args = parser.parse_args()


def get_original_bboxes(image_id: int) -> List[List[int]]:
    bboxes = []

    with open(ANNOTATION_DIR) as f:
        annotations = json.load(f)
    anno = annotations[image_id]
    example_coordinates = anno['box_examples_coordinates']

    for coord in example_coordinates:
        x_min, y_min = coord[0]  # top-left
        x_max, y_max = coord[2]  # botto
        bboxes.append([x_min, y_min, x_max, y_max])
    return bboxes


def mark_image_with_bboxes(image, bboxes) -> None:
    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image, start_point, end_point, (255, 0, 0), 1)


if __name__ == '__main__':
    with open(IMAGE_COORDS_FILE) as f:
        image_coords_dict = json.load(f)

    with open(BBOXES_COORDS_FILE) as f:
        bboxes_coords_dict = json.load(f)

    image_id = f'{args.image_num}.jpg'

    fig, axs = plt.subplots(2, 2)
    fig.suptitle(image_id)

    for i, (images_dir, densities_dir, bboxes) in enumerate(
        zip([ORIGINAL_IMAGES_DIR, RESIZED_IMAGES_DIR],
            [ORIGINAL_DENSITIES_DIR, RESIZED_DENSITIES_DIR],
            [get_original_bboxes(image_id), bboxes_coords_dict[image_id]])):
        image_path = '{}/{}'.format(images_dir, image_id)
        image = cv2.imread(image_path)
        mark_image_with_bboxes(image, bboxes)

        density_path = '{}/{}.npy'.format(densities_dir, args.image_num)
        density = np.load(density_path)

        if i == 1:
            x_min, y_min, x_max, y_max = image_coords_dict[image_id]
            image = cv2.rectangle(image, (x_min, y_min),
                                  (x_max, y_max), (0, 0, 255), 1)

        axs[i, 0].imshow(image)
        axs[i, 1].imshow(density)
        count = '{:.2f}'.format(np.sum(density))
        axs[i, 1].set_title(f'Count = {count}')

    axs[0, 0].set_title('Original image')
    axs[1, 0].set_title('Resized image')

    plt.show()
