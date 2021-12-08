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

    fig, axs = plt.subplots(2, num_boxes + 2)
    fig.suptitle(image_id)

    predict_densities_align = count_regressor(
        torch.cat([image_feature_align], dim=0).to(device), [num_boxes])
    predict_densities_ib = count_regressor_paper(image_feature_ib
                                                 torch.cat([image_feature_ib], dim=0).to(device), [num_boxes])

    fig.suptitle('Visualize feature maps')

    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Original image')

    axs[0, 1].imshow(original_density)
    axs[0, 1].set_title('Original density')

    axs[1, 1].imshow(predict_densities[0][0][0].cpu().detach())
    axs[1, 1].set_title('Predict density')
    for i in range(num_boxes):
        axs[0, i + 2].imshow(image_feature[i][0])
        axs[1, i + 2].imshow(image_feature[i][3])
        axs[0, i + 2].set_title(f'Box {i} (Map 3)')
        axs[1, i + 2].set_title(f'Box {i} (Map 4)')

    plt.show()
