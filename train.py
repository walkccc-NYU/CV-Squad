"""
Training Code for Learning To Count Everything, CVPR 2021
Authors: Viresh Ranjan, Udbhav, Thu Nguyen, Minh Hoai
"""
import argparse
import json
import os
import random
import time
from os.path import exists, join
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

from model import CountRegressor, Resnet50FPN, weights_normal_init
from utils import MAPS, SCALES, Transform, extract_features

dirname = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description='Few Shot Counting')
parser.add_argument('-ts', '--test-split', type=str, default='val',
                    choices=['train', 'test', 'val'], help='what data split to evaluate on')
parser.add_argument('-ep', '--epochs', type=int, default=1,
                    help='number of training epochs')
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id')
parser.add_argument('-lr', '--learning-rate', type=float,
                    default=1e-5, help='learning rate')
parser.add_argument('--debug-mode', action='store_true', default=False)
parser.add_argument('-sz', '--size', type=int, default=1)
args = parser.parse_args()

DATA_DIR = './data'
LOGS_DIR = './logsSave'

# Constant directories
anno_file = f'{DATA_DIR}/annotation_FSC147_384.json'
images_dir = f'{DATA_DIR}/images_384_VarV2'
gt_density_map_adaptive_dir = f'{DATA_DIR}/gt_density_map_adaptive_384_VarV2'

if not exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.MSELoss().to(device)

resnet50_conv = Resnet50FPN().to(device)
resnet50_conv.eval()

N_SCALES = 3  # [1.0, 0.9, 1.1]
N_FEATURES = 2  # [map3, map4]
regressor = CountRegressor(N_SCALES * N_FEATURES, pool='mean').to(device)
weights_normal_init(regressor, dev=0.001)
regressor.train()

optimizer = optim.Adam(regressor.parameters(), lr=args.learning_rate)

with open(anno_file) as f:
    annotations = json.load(f)

with open(f'{DATA_DIR}/Train_Test_Val_FSC_147.json') as f:
    data = json.load(f)


def train():
    print('Training on FSC147 train set data...')

    image_ids: List[str] = data['train']  # ['2.jpg', '3.jpg', ...]
    if args.debug_mode:
        image_ids = image_ids[:args.size]
    random.shuffle(image_ids)
    train_mae: float = 0
    train_rmse: float = 0
    train_loss: float = 0

    pbar = tqdm(image_ids)
    for i, image_id in enumerate(pbar):
        anno = annotations[image_id]
        example_coordinates: List[List[int]] = anno['box_examples_coordinates']
        bboxes: List[List[int]] = []  # coordinates of 3 bounding boxes

        for coord in example_coordinates:
            x_min, y_min = coord[0]  # top-left
            x_max, y_max = coord[2]  # bottom-right
            bboxes.append([x_min, y_min, x_max, y_max])

        # load image
        image_path = '{}/{}'.format(images_dir, image_id)
        image = Image.open(image_path)
        image.load()

        # load density map
        density_path = '{}/{}.npy'.format(gt_density_map_adaptive_dir,
                                          image_id.split('.jpg')[0])
        density = np.load(density_path).astype('float32')

        # proportionally resize image, bboxes, density
        image, bboxes, density = Transform({'image': image,
                                            'bboxes': bboxes,
                                            'density': density})

        with torch.no_grad():
            features = extract_features(
                resnet50_conv,
                image.unsqueeze(0),
                bboxes.unsqueeze(0),
                MAPS,
                SCALES)
        features.requires_grad = True
        optimizer.zero_grad()
        output = regressor(features)

        # if image size isn't divisible by 8, gt size is slightly different from output size
        if output.shape[2] != density.shape[2] or output.shape[3] != density.shape[3]:
            orig_count = density.sum().detach().item()
            density = F.interpolate(density, size=(
                output.shape[2], output.shape[3]), mode='bilinear')
            new_count = density.sum().detach().item()
            if new_count > 0:
                density = density * (orig_count / new_count)
        loss = criterion(output, density)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_cnt = torch.sum(output).item()
        gt_cnt = torch.sum(density).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2
        pbar.set_description('actual-predicted: {:6.1f}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f} Best VAL MAE: {:5.2f}, RMSE: {:5.2f}'.format(
            gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), train_mae / (i + 1), (train_rmse / (i + 1))**0.5, best_mae, best_rmse))
        print('')
    train_loss = train_loss / len(image_ids)
    train_mae = (train_mae / len(image_ids))
    train_rmse = (train_rmse / len(image_ids))**0.5
    return train_loss, train_mae, train_rmse


def validation():
    print('Evaluation on {} data'.format(args.test_split))

    image_ids = data['val']   # ['190.jpg', '191.jpg', ...]
    if args.debug_mode:
        image_ids = image_ids[:args.size]

    SAE = 0  # sum of absolute errors
    SSE = 0  # sum of square errors

    pbar = tqdm(image_ids)
    for i, image_id in enumerate(pbar):
        anno = annotations[image_id]
        example_coordinates: List[List[int]] = anno['box_examples_coordinates']
        dots = np.array(anno['points'])
        bboxes: List[List[int]] = []  # coordinates of 3 bounding boxes

        for coord in example_coordinates:
            x_min, y_min = coord[0]  # top-left
            x_max, y_max = coord[2]  # bottom-right
            bboxes.append([x_min, y_min, x_max, y_max])

        # load image
        image_path = '{}/{}'.format(images_dir, image_id)
        image = Image.open(image_path)
        image.load()

        # proportionally resize image, bboxes
        image, bboxes = Transform({'image': image, 'bboxes': bboxes})

        with torch.no_grad():
            features = extract_features(
                resnet50_conv,
                image.unsqueeze(0),
                bboxes.unsqueeze(0),
                MAPS,
                SCALES)

        gt_cnt = dots.shape[0]
        output = regressor(features)
        pred_cnt = output.sum().item()
        err = abs(gt_cnt - pred_cnt)
        SAE += err
        SSE += err**2

        pbar.set_description('{:<8}: actual-predicted: {:6d}, {:6.1f}, error: {:6.1f}. Current MAE: {:5.2f}, RMSE: {:5.2f}'.format(
            image_id, gt_cnt, pred_cnt, abs(pred_cnt - gt_cnt), SAE / (i + 1), (SSE / (i + 1))**0.5))
        print('')

    print('On validation data, MAE: {:6.2f}, RMSE: {:6.2f}'.format(
        SAE / (i + 1), (SSE / (i + 1))**0.5))
    return SAE / len(image_ids), (SSE / len(image_ids))**0.5


best_mae, best_rmse = 1e7, 1e7
stats = []
for epoch in range(0, args.epochs):
    start: int = time.time()
    regressor.train()
    train_loss, train_mae, train_rmse = train()
    regressor.eval()
    val_mae, val_rmse = validation()
    stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))
    stats_file = join(LOGS_DIR, 'stats' + '.txt')
    with open(stats_file, 'w') as f:
        for s in stats:
            f.write('%s\n' % ','.join([str(x) for x in s]))
    if best_mae >= val_mae:
        print(f"=== Val MAE ({val_mae}) <= Best MAE ({best_mae}) ===")
        best_mae = val_mae
        best_rmse = val_rmse
        model_name = LOGS_DIR + '/' + 'FamNet.pth'
        print(f"=== Write {model_name} ===")
        torch.save(regressor.state_dict(), model_name)

    epoch_duration: int = time.time() - start

    print('Epoch {}, Avg. Epoch Loss: {} Train MAE: {} Train RMSE: {} Val MAE: {} Val RMSE: {} Best Val MAE: {} Best Val RMSE: {} '.format(
        epoch+1,  stats[-1][0], stats[-1][1], stats[-1][2], stats[-1][3], stats[-1][4], best_mae, best_rmse))
    print(f'{epoch_duration // 60}m : {epoch_duration % 60}s')
