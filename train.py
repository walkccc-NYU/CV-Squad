"""
Training Code for Learning To Count Everything, CVPR 2021
Authors: Viresh Ranjan, Udbhav, Thu Nguyen, Minh Hoai
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import LOGS_DIR, N_CHANNELS, SPLIT_DIR
from dataset import collate_fn, train_set, val_set
from model import CountRegressor, weights_normal_init

parser = argparse.ArgumentParser(description='Few Shot Counting')
parser.add_argument('-ep', '--epochs', type=int, default=1,
                    help='number of training epochs')
parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU id')
parser.add_argument('-lr', '--learning-rate', type=float,
                    default=1e-5, help='learning rate')
parser.add_argument('-bs', '--batch-size', type=int, default=1)
parser.add_argument('-sz', '--size', type=int, default=None)
parser.add_argument('-li', '--log-interval', type=int, default=1)
parser.add_argument('-r', '--use-resize',
                    action='store_true', default=False)
args = parser.parse_args()

train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size,
                          shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate_fn)


with open(SPLIT_DIR) as f:
    data = json.load(f)

if not os.path.exists(LOGS_DIR):
    os.mkdir(LOGS_DIR)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

count_regressor = CountRegressor(N_CHANNELS, pool='mean').to(device)
weights_normal_init(count_regressor, dev=0.001)
optimizer = torch.optim.Adam(
    count_regressor.parameters(), lr=args.learning_rate)
criterion = torch.nn.MSELoss().to(device)

min_mae, min_rmse = 1e7, 1e7


def train(epoch: int):
    print('Training on FSC147 train set data...')

    total_loss = 0.0
    sum_absolute_error = 0.0
    sum_square_error = 0.0

    total_images: int = len(train_loader) * train_loader.batch_size
    pbar = tqdm(train_loader, miniters=train_loader.batch_size)
    for i, (image_features, densities, image_coords, split_size) in enumerate(pbar):
        if args.size and i == args.size:
            break
        image_features = image_features.to(device)
        predict_densities = count_regressor(image_features, split_size)

        optimizer.zero_grad()
        batch_loss = 0

        # Accumulate count error
        for predict_density, density, image_coord in zip(predict_densities, densities, image_coords):
            x_min, y_min, x_max, y_max = image_coord
            target_predict_density = predict_density[0][0]
            target_density = density[0][0].to(device)
            if args.use_resize:
                target_predict_density = target_predict_density[y_min:y_max, x_min:x_max]
                target_density = target_density[y_min:y_max, x_min:x_max]

            batch_loss += criterion(target_predict_density, target_density)

            # Caculate count error
            pred_count = torch.sum(target_predict_density).item()
            gt_count = torch.sum(target_density).item()
            count_error = abs(pred_count - gt_count)

            # Accumulate count error
            sum_absolute_error += count_error
            sum_square_error += count_error**2

        n_images: int = (i + 1) * train_loader.batch_size
        batch_loss.backward()
        total_loss += batch_loss.item()
        optimizer.step()

        if i % args.log_interval == 0:
            pbar.set_description('{:>7} Epoch: {} | {:>4}/{:>4} = {:>3}% | gt-predict: {:6.1f}, {:6.1f} | err: {:6.1f} | MAE: {:6.2f} | RMSE: {:6.2f} | Avg Loss: {:6.8f}\n'.format(
                "[TRAIN]", epoch,
                n_images, total_images, (n_images * 100 // total_images),
                gt_count, pred_count, count_error,
                sum_absolute_error / n_images,
                (sum_square_error / n_images)**0.5,
                total_loss / n_images))

    return total_loss / n_images, \
        sum_absolute_error / n_images, \
        (sum_square_error / n_images)**0.5


def validation(epoch: int):
    print('\nEvaluating on validation data...')

    sum_absolute_error = 0.0
    sum_square_error = 0.0

    total_images: int = len(val_loader) * val_loader.batch_size
    pbar = tqdm(val_loader, miniters=val_loader.batch_size)
    for i, (image_features, densities, image_coords, split_size) in enumerate(pbar):
        if args.size and i == args.size:
            break
        image_features = image_features.to(device)
        predict_densities = count_regressor(image_features, split_size)

        # Accumulate count error
        for predict_density, density, image_coord in zip(predict_densities, densities, image_coords):
            x_min, y_min, x_max, y_max = image_coord
            target_predict_density = predict_density[0][0]
            target_density = density[0][0].to(device)
            if args.use_resize:
                target_predict_density = target_predict_density[y_min:y_max, x_min:x_max]
                target_density = target_density[y_min:y_max, x_min:x_max]

            # Caculate count error
            pred_count = torch.sum(target_predict_density).item()
            gt_count = torch.sum(target_density).item()
            count_error = abs(pred_count - gt_count)

            # Accumulate count error
            sum_absolute_error += count_error
            sum_square_error += count_error**2

        n_images: int = (i + 1) * val_loader.batch_size
        if i % args.log_interval == 0:
            pbar.set_description('{:>7} Epoch: {} | {:>4}/{:>4} = {:>3}% | gt-predict: {:6.1f}, {:6.1f} | err: {:6.1f} | MAE: {:6.2f} | RMSE: {:6.2f} | Min MAE: {:6.2f} | Min RMSE: {:6.2f}\n'.format(
                "[VAL]", epoch,
                n_images, total_images, (n_images * 100 // total_images),
                gt_count, pred_count, count_error,
                sum_absolute_error / n_images,
                (sum_square_error / n_images)**0.5,
                min_mae, min_rmse))
    return sum_absolute_error / n_images, (sum_square_error / n_images)**0.5


if __name__ == '__main__':
    stats = []

    for epoch in range(1, args.epochs + 1):
        start: int = time.time()
        count_regressor.train()
        train_loss, train_mae, train_rmse = train(epoch)

        count_regressor.eval()
        val_mae, val_rmse = validation(epoch)
        stats.append((train_loss, train_mae, train_rmse, val_mae, val_rmse))

        if val_mae <= min_mae:
            print('\nVal MAE ({:6.2f}) <= Min MAE ({:6.2f})'.format(
                val_mae, min_mae))
            min_mae = val_mae
            min_rmse = val_rmse
            model_name = f'{LOGS_DIR}/FamNet.pth'
            print(f'Write {model_name}...\n')
            torch.save(count_regressor.state_dict(), model_name)

        epoch_duration = int(time.time() - start)

        print('Epoch {} ({}m : {}s), Avg. Epoch Loss: {:6.6f}'.format(
            epoch, epoch_duration // 60, epoch_duration % 60, train_loss))
        print('{:>8} | MAE: {:6.2f} | RMSE: {:6.2f}'.format(
            'Train', train_mae, train_rmse))
        print('{:>8} | MAE: {:6.2f} | RMSE: {:6.2f}'.format(
            'Val', val_mae, val_rmse))
        print('{:>8} | MAE: {:6.2f} | RMSE: {:6.2f}'.format(
            'Min Val', min_mae, min_rmse))
        print()

        # Eager logging
        with open(f'{LOGS_DIR}/lr{str(args.learning_rate)}_bs{train_loader.batch_size}_ep{args.epochs}.txt', 'w') as f:
            f.write(
                'Epoch | Train Loss, Train MAE, Train RMSE, Val MAE, VAL RMSE\n')
            for i, stat in enumerate(stats):
                s = '{:.8f}'.format(
                    stat[0]) + ', ' + ', '.join(['{:4.2f}'.format(x) for x in stat[1:]])
                f.write('{:>5} | {}'.format(i + 1, s))
                f.write('\n')
