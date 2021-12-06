import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def __get_conved_feature(image, conv, normalization):
    # Pad image so that it won't change size after convoluted
    h, w = conv.shape[2], conv.shape[3]

    padded = F.pad(
        image, (int(w / 2), int((w - 1) / 2), int(h / 2), int((h - 1) / 2)))
    conved_feature = F.conv2d(padded, conv)
    if normalization == 'box_max':
        box_max = conved_feature.max(3).values.max(2).values.max(1).values
        conved_feature = conved_feature / (box_max + 1e-8)
    return conved_feature


def extract_features(
        feature_model: torch.nn.Module,
        image: Tensor,  # [N, C, H, W] = [1, 3, 384, 408]
        bboxes: Tensor,  # [M, 4] = [1, 3, 4]
        use_interpolated_bboxes=False,
        normalization=None):
    # Get features for the examples (N * M) * C * h * w
    # features_dict['map3'].shape = torch.Size([1, 512, 48, 51])
    # features_dict['map4'].shape = torch.Size([1, 1024, 24, 26])
    features_dict: Dict[str, Tensor] = feature_model(image)
    all_feature_scaled = []

    for feature_key, feature in features_dict.items():
        if feature_key == 'map3':
            scaling = 8.0
        else:  # keys == 'map4'
            scaling = 16.0
        B = bboxes / scaling  # scaled bboxes
        B[:, :2] = torch.floor(B[:, :2])
        B[:, 2:] = torch.ceil(B[:, 2:])
        # make sure exemplars don't go out of bound
        B[:, :2] = torch.clamp_min(B[:, :2], 0)
        B[:, 2] = torch.clamp_max(B[:, 2], feature.shape[-1])
        B[:, 3] = torch.clamp_max(B[:, 3], feature.shape[-2])
        B = B.to(torch.int16)

        # interpolate all feature_bbox to (max_h, max_w)
        max_h = max(B[:, 3] - B[:, 1] + 1)
        max_w = max(B[:, 2] - B[:, 0] + 1)

        # feature that bound by bboxes
        feature_bboxes = []
        conved_features = []
        for x_min, y_min, x_max, y_max in B:
            feature_bbox = feature[:, :, y_min:y_max, x_min:x_max]
            if use_interpolated_bboxes:
                interpolated = F.interpolate(feature_bbox, size=(
                    max_h, max_w), mode='bilinear', align_corners=False)
                feature_bboxes.append(interpolated)
            else:
                feature_bboxes.append(feature_bbox)
                conved_features.append(
                    __get_conved_feature(feature, feature_bbox, normalization))
        if use_interpolated_bboxes:
            feature_bboxes = torch.cat(feature_bboxes, dim=0)
            conved_feature = __get_conved_feature(
                feature, feature_bboxes, normalization)
        else:
            conved_feature = torch.cat(conved_features, dim=1)
            if normalization == 'image_max':
                conved_feature = conved_feature / (conved_feature.max() + 1e-8)

        # [M, N, H, W]
        feature_scaled = conved_feature.permute([1, 0, 2, 3])

        for scale in [0.9, 1.1]:
            h = max(1, math.ceil(max_h * scale))
            w = max(1, math.ceil(max_w * scale))
            if use_interpolated_bboxes:
                interpolated = F.interpolate(feature_bboxes, size=(
                    h, w), mode='bilinear', align_corners=False)
                conved_feature = __get_conved_feature(
                    feature, interpolated, normalization).permute([1, 0, 2, 3])
            else:
                conved_features = []
                for feature_bbox in feature_bboxes:
                    interpolated = F.interpolate(feature_bbox, size=(
                        h, w), mode='bilinear', align_corners=False)
                    conved_features.append(__get_conved_feature(
                        feature, interpolated, normalization))
                conved_feature = torch.cat(
                    conved_features, dim=1).permute([1, 0, 2, 3])
                if normalization == 'image_max':
                    conved_feature = conved_feature / \
                        (conved_feature.max() + 1e-8)
            feature_scaled = torch.cat(
                [feature_scaled, conved_feature], dim=1)

        if all_feature_scaled:
            h = all_feature_scaled[-1].shape[2]
            w = all_feature_scaled[-1].shape[3]
            feature_scaled = F.interpolate(
                feature_scaled, size=(h, w), mode='bilinear', align_corners=False)
        all_feature_scaled.append(feature_scaled)
    return torch.cat(all_feature_scaled, dim=1)


class ResizeImage:
    """
    Resize an image to (H', W') if either the height (H) or width (W) of it
    exceed a specific value (MAX_SIZE). After resizing:
        (1) max(H', W') <= MAX_SIZE
        (2) Both H' and W' are divisible by 8
        (3) The aspect ratio of H and W is preserved
    """

    def __init__(self, MAX_SIZE=1504):
        self.MAX_SIZE = MAX_SIZE

    def __call__(self, sample: Dict) -> Tuple:
        image: PIL.Image = sample['image']
        bboxes: List[List[int]] = sample['bboxes']
        is_trained: bool = 'density' in sample
        if is_trained:
            density = sample['density']
        W, H = image.size

        scale_factor: float = max(H, W) / float(self.MAX_SIZE) \
            if max(H, W) > self.MAX_SIZE else 1.0

        H_divisible_by_8 = 8 * int(H / scale_factor / 8)
        W_divisible_by_8 = 8 * int(W / scale_factor / 8)
        shrink_ratio: float = H_divisible_by_8 / H

        image = transforms.Resize((H_divisible_by_8, W_divisible_by_8))(image)
        if is_trained:
            original_density_sum: float = np.sum(density)
            density = cv2.resize(density, (W_divisible_by_8, H_divisible_by_8))
            new_density_sum: float = np.sum(density)
            if new_density_sum > 0:
                # make sure sum(density) remain unchanged
                density *= (original_density_sum / new_density_sum)

        assert image.size[0] % 8 == 0 and image.size[1] % 8 == 0
        assert density.shape[0] % 8 == 0 and density.shape[1] % 8 == 0

        for i in range(len(bboxes)):
            for j in range(len(bboxes[i])):
                bboxes[i][j] *= shrink_ratio

        if is_trained:
            return Normalize(image).cuda(), \
                torch.Tensor(bboxes).cuda(), \
                torch.from_numpy(density).unsqueeze(0).unsqueeze(0).cuda()
        else:
            return Normalize(image).cuda(), \
                torch.Tensor(bboxes).cuda()


Normalize = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])
Transform = transforms.Compose([ResizeImage()])
