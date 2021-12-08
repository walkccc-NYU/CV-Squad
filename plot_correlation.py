import json
import math
from typing import List, Tuple

import matplotlib.pyplot as plt

from constants import CORRELATION_FILE

with open(CORRELATION_FILE) as f:
    correlation_dict = json.load(f)


map3_idAvgStds: List[Tuple[str, float, float]] = []

for image_id, stats in correlation_dict.items():
    map3_sum = stats['map3_sum']
    map3_squareSum = stats['map3_squareSum']
    box_count = stats['box_count']
    H = stats['H']
    W = stats['W']
    N = box_count * H * W
    map3_avg = map3_sum / N
    map3_std = math.sqrt(map3_squareSum / N - map3_avg**2)
    map3_idAvgStds.append((image_id, map3_avg, map3_std))


map3_idAvgStds.sort(key=lambda x: x[1])

image_ids = []
avgs = []
stds = []

for image_id, avg, std in map3_idAvgStds[-5:]:
    print(image_id, correlation_dict[image_id]['bbox_sizes'])

for image_id, avg, std in map3_idAvgStds:
    image_ids.append(image_id)
    avgs.append(avg)
    stds.append(std)

plt.errorbar(image_ids, avgs, stds, linestyle='None', marker='^')
# plt.show()
