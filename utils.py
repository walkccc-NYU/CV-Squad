import math
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms

MAPS = ['map3', 'map4']
SCALES = [0.9, 1.1]
MIN_HW = 384
MAX_SIZE = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


def select_exemplar_rois(image):
    all_rois = []

    print("Press 'q' or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar, 'space' to save.")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('n') or key == '\r':
            rect = cv2.selectROI('image', image, False, False)
            x1 = rect[0]
            y1 = rect[1]
            x2 = x1 + rect[2] - 1
            y2 = y1 + rect[3] - 1

            all_rois.append([y1, x1, y2, x2])
            for rect in all_rois:
                y1, x1, y2, x2 = rect
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            print(
                "Press q or Esc to quit. Press 'n' and then use mouse drag to draw a new examplar")

    return all_rois


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def PerturbationLoss(output, boxes, sigma=8, use_gpu=True):
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            out = output[:, :, y1:y2, x1:x2]
            GaussKernel = matlab_style_gauss2D(
                shape=(out.shape[2], out.shape[3]), sigma=sigma)
            GaussKernel = torch.from_numpy(GaussKernel).float()
            if use_gpu:
                GaussKernel = GaussKernel.cuda()
            Loss += F.mse_loss(out.squeeze(), GaussKernel)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        out = output[:, :, y1:y2, x1:x2]
        Gauss = matlab_style_gauss2D(
            shape=(out.shape[2], out.shape[3]), sigma=sigma)
        GaussKernel = torch.from_numpy(Gauss).float()
        if use_gpu:
            GaussKernel = GaussKernel.cuda()
        Loss += F.mse_loss(out.squeeze(), GaussKernel)
    return Loss


def MincountLoss(output, boxes, use_gpu=True):
    ones = torch.ones(1)
    if use_gpu:
        ones = ones.cuda()
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            X = output[:, :, y1:y2, x1:x2].sum()
            if X.item() <= 1:
                Loss += F.mse_loss(X, ones)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        X = output[:, :, y1:y2, x1:x2].sum()
        if X.item() <= 1:
            Loss += F.mse_loss(X, ones)
    return Loss


def __get_conved_feature(image, conv):
    # Pad image so that it won't change size after convoluted
    h, w = conv.shape[2], conv.shape[3]
    padded = F.pad(
        image, (int(w / 2), int((w - 1) / 2), int(h / 2), int((h - 1) / 2)))
    return F.conv2d(padded, conv)


def extract_features(
        feature_model: torch.nn.Module,
        image: Tensor,  # [N, C, H, W] = [1, 3, 384, 408]
        all_bboxes: Tensor,  # [N, M, 4] = [1, 3, 4]
        feat_map_keys=['map3', 'map4'],
        exemplar_scales=[0.9, 1.1]):
    # Get features for the examples (N * M) * C * h * w
    # features_dict['map3'].shape = torch.Size([1, 512, 48, 51])
    # features_dict['map4'].shape = torch.Size([1, 1024, 24, 26])
    features_dict: Dict[str, Tensor] = feature_model(image)
    bboxes = all_bboxes[0]
    start = time.time()

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
        for x_min, y_min, x_max, y_max in B:
            feature_bbox = feature[:, :, y_min:y_max, x_min:x_max]
            interpolated = F.interpolate(
                feature_bbox, size=(max_h, max_w), mode='bilinear')
            feature_bboxes.append(interpolated)
        feature_bboxes = torch.cat(feature_bboxes, dim=0)
        conved_feature = __get_conved_feature(feature, feature_bboxes)

        # [M, N, H, W]
        feature_scaled = conved_feature.permute([1, 0, 2, 3])

        # computing conved_feature for scales 0.9 and 1.1
        for scale in exemplar_scales:
            h = max(1, math.ceil(max_h * scale))
            w = max(1, math.ceil(max_w * scale))
            interpolated = F.interpolate(
                feature_bboxes, size=(h, w), mode='bilinear')
            conved_feature = __get_conved_feature(
                feature, interpolated).permute([1, 0, 2, 3])
            feature_scaled = torch.cat(
                [feature_scaled, conved_feature], dim=1)

        if all_feature_scaled:
            h = all_feature_scaled[-1].shape[2]
            w = all_feature_scaled[-1].shape[3]
            feature_scaled = F.interpolate(
                feature_scaled, size=(h, w), mode='bilinear')
        all_feature_scaled.append(feature_scaled)
    all_feature_scaled = torch.cat(all_feature_scaled, dim=1)
    return all_feature_scaled.unsqueeze(0)


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
        scale_factor = 1
        W, H = image.size

        if max(H, W) > self.MAX_SIZE:
            scale_factor: float = max(H, W) / float(self.MAX_SIZE)
            H = 8 * int(H / scale_factor / 8)
            W = 8 * int(W / scale_factor / 8)
            image = transforms.Resize((H, W))(image)
            if is_trained:
                original_density_sum: float = np.sum(density)
                density = cv2.resize(density, (H, W))
                new_density_sum: float = np.sum(density)
                if new_density_sum > 0:
                    # make sure sum(density) remain unchanged
                    density *= (original_density_sum / new_density_sum)

        for i in range(len(bboxes)):
            for j in range(len(bboxes[i])):
                bboxes[i][j] //= scale_factor

        if is_trained:
            return Normalize(image).cuda(), \
                torch.Tensor(bboxes).cuda(), \
                torch.from_numpy(density).unsqueeze(0).unsqueeze(0).cuda()
        else:
            return Normalize(image).cuda(), \
                torch.Tensor(bboxes).cuda()


Normalize = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])
Transform = transforms.Compose([ResizeImage(MAX_SIZE)])
