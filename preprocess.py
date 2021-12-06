import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
from PIL import Image

from constants import (ANNOTATION_DIR, BBOXES_COORDS_FILE,
                       ORIGINAL_DENSITIES_DIR, ORIGINAL_IMAGES_DIR,
                       POINTS_COUNT_FILE, PREPROCESSED_DENSITIES_DIR,
                       PREPROCESSED_IMAGE_FEATURES_DIR, RESIZED_DENSITIES_DIR,
                       RESIZED_IMAGES_DIR, SPLIT_DIR, TEST, TRAIN, VAL)
from model import Resnet50FPN
from utils import Transform, extract_features

parser = argparse.ArgumentParser(description='Few Show Counting Preprocessing')
parser.add_argument('-sz', '--size', type=int, default=None)
parser.add_argument('-r', '--use-resized',
                    action='store_true', default=False)
parser.add_argument('-ib', '--use-interpolated-bboxes',
                    action='store_true', default=False)
parser.add_argument('-n', '--normalization', type=str,
                    choices=['box_max', 'image_max'], default=None)

args = parser.parse_args()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.use_resized:
        IMAGES_DIR = RESIZED_IMAGES_DIR
        DENSITIES_DIR = RESIZED_DENSITIES_DIR
        with open(BBOXES_COORDS_FILE) as f:
            bboxes_coords_dict = json.load(f)
    else:
        IMAGES_DIR = ORIGINAL_IMAGES_DIR
        DENSITIES_DIR = ORIGINAL_DENSITIES_DIR
        with open(ANNOTATION_DIR) as f:
            annotations = json.load(f)

    preprocessed_image_features_dir = f'{PREPROCESSED_IMAGE_FEATURES_DIR}_{args.normalization}_normalized' \
                                      if args.normalization else f'{PREPROCESSED_IMAGE_FEATURES_DIR}'
    preprocessed_densities_dir = f'{PREPROCESSED_DENSITIES_DIR}_{args.normalization}_normalized' \
                                 if args.normalization else f'{PREPROCESSED_DENSITIES_DIR}'

    for _dir in [preprocessed_image_features_dir, preprocessed_densities_dir]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    with open(SPLIT_DIR) as f:
        data = json.load(f)

    resnet50_conv = Resnet50FPN().to(device)
    resnet50_conv.eval()

    points_count_dict: Dict[str, int] = {}

    for split in [TRAIN, VAL, TEST]:
        print(f'=== Preprocessing {split} dataset... ===')

        image_features_split_dir = f'{preprocessed_image_features_dir}/{split}'
        densities_split_dir = f'{preprocessed_densities_dir}/{split}'

        for _dir in [image_features_split_dir, densities_split_dir]:
            if not os.path.exists(_dir):
                os.mkdir(_dir)

        image_ids = data[split]
        if args.size:
            image_ids = image_ids[:args.size]

        for image_id in image_ids:
            image_num = image_id.split('.jpg')[0]

            print(f'Preprocessing {image_id} ...')

            if args.use_resized:
                bboxes = bboxes_coords_dict[image_id]
            else:
                bboxes = []
                anno = annotations[image_id]
                example_coordinates = anno['box_examples_coordinates']
                for coord in example_coordinates:
                    x_min, y_min = coord[0]  # top-left
                    x_max, y_max = coord[2]  # bottom-right
                    bboxes.append([x_min, y_min, x_max, y_max])

            # Load (resized) image
            image_path = '{}/{}.jpg'.format(IMAGES_DIR, image_num)
            image = Image.open(image_path)
            image.load()

            # Load (resized) density map
            density_path = '{}/{}.npy'.format(DENSITIES_DIR, image_num)
            density = np.load(density_path).astype('float32')

            # proportionally resize image, bboxes, density
            image, bboxes, density = Transform({'image': image,
                                                'bboxes': bboxes,
                                                'density': density})
            gt_count = round(torch.sum(density).item())
            points_count_dict[image_id] = gt_count

            with torch.no_grad():
                features = extract_features(resnet50_conv, image.unsqueeze(
                    0), bboxes, use_interpolated_bboxes=args.use_interpolated_bboxes,
                    normalization=args.normalization)

            image_num = image_id.split('.jpg')[0]

            np.save(f'{image_features_split_dir}/{image_num}.npy',
                    features.cpu())
            np.save(f'{densities_split_dir}/{image_num}.npy', density.cpu())

    with open(POINTS_COUNT_FILE, 'w') as f:
        f.write(json.dumps(points_count_dict))
