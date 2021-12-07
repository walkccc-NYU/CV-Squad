import json
from typing import List

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from constants import (IMAGE_COORDS_FILE, PREPROCESSED_DENSITIES_DIR,
                       PREPROCESSED_IMAGE_FEATURES_DIR, SPLIT_DIR)

with open(SPLIT_DIR) as f:
    data = json.load(f)


#with open(IMAGE_COORDS_FILE) as f:
#    image_coords_dict = json.load(f)


class MyDataset(Dataset):
    def __init__(self, split, indices: List[int], normalization=None):
        preprocessed_image_features_dir = f'{PREPROCESSED_IMAGE_FEATURES_DIR}_{normalization}_normalized' \
                                          if normalization else f'{PREPROCESSED_IMAGE_FEATURES_DIR}'
        preprocessed_densities_dir = f'{PREPROCESSED_DENSITIES_DIR}_{normalization}_normalized' \
                                     if normalization else f'{PREPROCESSED_DENSITIES_DIR}'

        self.image_feature_paths = [f'{preprocessed_image_features_dir}/{split}/{i}.npy'
                                    for i in indices]
        self.density_paths = [f'{preprocessed_densities_dir}/{split}/{i}.npy'
                              for i in indices]
        #self.image_coords = [image_coords_dict[f'{i}.jpg'] for i in indices]
        assert len(self.image_feature_paths) == len(self.density_paths)
        print(f'Total data in {split} split: {len(self.density_paths)}')

    def __getitem__(self, index: int):
        image_feature_path = self.image_feature_paths[index]
        density_path = self.density_paths[index]
        #image_coord = self.image_coords[index]
        image_coord = None
        return np.load(image_feature_path).astype('float32'), \
            np.load(density_path).astype('float32'), \
            image_coord

    def __len__(self):
        return len(self.image_feature_paths)


def collate_fn(batch):
    image_features = []
    densities = []
    image_coords = []

    for image_feature, density, image_coord in batch:
        image_features.append(torch.Tensor(image_feature))
        densities.append(torch.Tensor(density))
        image_coords.append(image_coord)

    split_size = [i.shape[0] for i in image_features]
    return [torch.cat(image_features, dim=0), densities, image_coords, split_size]
