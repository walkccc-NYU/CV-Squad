import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from constants import (LOGS_DIR, N_CHANNELS, ORIGINAL_DENSITIES_DIR,
                       ORIGINAL_IMAGES_DIR, PREPROCESSED_DENSITIES_DIR,
                       PREPROCESSED_IMAGE_FEATURES_DIR, TEST, TRAIN, VAL)
from model import CountRegressor, CountRegressorPaper
from visualize_resize import get_original_bboxes, mark_image_with_bboxes

parser = argparse.ArgumentParser(description='Few Shot Counting Demo')
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id')
parser.add_argument('-bn', '--use-batch-normalization',
                    action='store_true', default=False)
parser.add_argument('-m', '--model-path', type=str,
                    default=f'{LOGS_DIR}/FamNet_align.pth')
parser.add_argument('-i', '--image-num', type=int, default='2')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

count_regressor = CountRegressor(
    N_CHANNELS, pool='max', use_bn=args.use_batch_normalization).to(device)
count_regressor.load_state_dict(torch.load(args.model_path))
count_regressor.eval()

count_regressor_paper = CountRegressorPaper(N_CHANNELS, pool='max').to(device)
count_regressor_paper.load_state_dict(
    torch.load(f'{LOGS_DIR}/FamNet_paper.pth'))
count_regressor_paper.eval()


def __get_split() -> str:
    for split in [TRAIN, VAL, TEST]:
        image_feature_path = f'{PREPROCESSED_IMAGE_FEATURES_DIR}/{split}/{args.image_num}.npy'
        if os.path.exists(image_feature_path):
            return split
    os.abort()


if __name__ == '__main__':
    image_id = f'{args.image_num}.jpg'
    split: str = __get_split()

    image_path = '{}/{}'.format(ORIGINAL_IMAGES_DIR, image_id)
    image = cv2.imread(image_path)
    bboxes = get_original_bboxes(image_id)
    mark_image_with_bboxes(image, bboxes)

    density_path = '{}/{}.npy'.format(ORIGINAL_DENSITIES_DIR, args.image_num)
    original_density = np.load(density_path)

    image_feature_align_path = f'{PREPROCESSED_IMAGE_FEATURES_DIR}/{split}/{args.image_num}.npy'
    image_feature_align = torch.Tensor(
        np.load(image_feature_align_path).astype('float32'))

    image_feature_ib_path = f'{PREPROCESSED_IMAGE_FEATURES_DIR}_ib/{split}/{args.image_num}.npy'
    image_feature_ib = torch.Tensor(
        np.load(image_feature_ib_path).astype('float32'))

    num_boxes = image_feature_align.shape[0]

    fig, axs = plt.subplots(num_boxes + 1, 4)
    fig.suptitle(image_id)

    predict_densities_align = count_regressor(
        torch.cat([image_feature_align], dim=0).to(device), [num_boxes])
    predict_densities_ib = count_regressor_paper(
        torch.cat([image_feature_ib], dim=0).to(device), [num_boxes])

    fig.suptitle(image_id)

    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original image')

    axs[0, 1].imshow(original_density)
    axs[0, 1].set_title('Original density')

    axs[0, 2].imshow(predict_densities_ib[0][0].cpu().detach())
    axs[0, 2].set_title('Predict density (paper)')

    axs[0, 3].imshow(predict_densities_align[0][0][0].cpu().detach())
    axs[0, 3].set_title('Predict density (us)')

    for i in range(num_boxes):
        axs[i + 1][0].imshow(image_feature_ib[i][0])
        axs[i + 1][1].imshow(image_feature_ib[i][3])
        axs[i + 1][2].imshow(image_feature_align[i][0])
        axs[i + 1][3].imshow(image_feature_align[i][3])

    plt.show()
