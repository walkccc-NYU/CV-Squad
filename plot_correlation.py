import argparse
import json
import math
from typing import List, Tuple

import matplotlib.pyplot as plt

from constants import CORRELATION_FILE, SPLIT_DIR, TRAIN, VAL

parser = argparse.ArgumentParser(description='Few Shot Counting Correlation')
parser.add_argument('-s', '--split', type=str,
                    choices=[TRAIN, VAL], default=None)

args = parser.parse_args()

with open(f'{CORRELATION_FILE}') as f:
    correlation_dict = json.load(f)

with open(SPLIT_DIR) as f:
    data = json.load(f)

# if args.split == TRAIN:
#     image_ids = data[TRAIN]
# elif args.split == VAL:
#     image_ids = data[VAL]
# else:
#     image_ids = data[TRAIN] + data[VAL]


def __show_map3_idAvgStdBoxsized(image_ids: List[str], sorted_by_box: bool, ax, xlabel: str):
    map3_idAvgStdBoxsizes: List[Tuple[str, float, float]] = []
    for image_id in image_ids:
        stats = correlation_dict[image_id]
        map3_sum = stats['map3_sum']
        map3_squareSum = stats['map3_squareSum']
        map3_bbox_size = stats['bbox_sizes']['map3'][0]
        box_h, box_w = map3_bbox_size
        box_count = stats['box_count']
        H = stats['H']
        W = stats['W']
        N = box_count * H * W
        map3_avg = map3_sum / N
        map3_std = math.sqrt(map3_squareSum / N - map3_avg**2)
        map3_idAvgStdBoxsizes.append(
            (image_id, map3_avg, map3_std, box_h * box_w))
    if sorted_by_box:
        map3_idAvgStdBoxsizes.sort(key=lambda x: x[3])
    else:  # sorted by avg
        map3_idAvgStdBoxsizes.sort(key=lambda x: x[1])

    image_ids = []
    avgs = []
    stds = []
    boxsizes = []

    # N = 5
    # print(f'Top {N} correlations in {args.split}')
    # for image_id, avg, std, boxsize in map3_idAvgStdBoxsizes[-N:]:
    #     print(image_id, correlation_dict[image_id]['bbox_sizes'])

    for image_id, avg, std, boxsize in map3_idAvgStdBoxsizes:
        image_ids.append(image_id)
        avgs.append(avg)
        stds.append(std)
        boxsizes.append(boxsize)

    # ax.errorbar(image_ids, avgs, stds, linestyle='None', marker='^')
    ax.plot(image_ids, avgs)
    ax.tick_params(axis='x', which='both', bottom=False,
                   top=False, labelbottom=False)
    ax.set_xlabel(xlabel)
    # ax.set_ylabel('Correlation average')
    ax.set_yticks(range(1000, int(max(avgs)), 1000))
    print(f'{max(avgs) = }')


fig, axs = plt.subplots(2, 2)
__show_map3_idAvgStdBoxsized(data[TRAIN], False, axs[0, 0], '(a)')
__show_map3_idAvgStdBoxsized(data[VAL], False, axs[0, 1], '(b)')
__show_map3_idAvgStdBoxsized(data[TRAIN] + data[VAL], False, axs[1, 0], '(c)')
__show_map3_idAvgStdBoxsized(data[TRAIN] + data[VAL], True, axs[1, 1], '(d)')

fig.suptitle('Correlation values')
plt.show()
