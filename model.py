from typing import List

import torch
import torch.nn as nn
import torchvision


class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        feature_3 = self.conv3(x)
        feature_4 = self.conv4(feature_3)
        return {'map3': feature_3,
                'map4': feature_4}


class Conv(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, padding=0, use_bn=False):
        super(Conv, self).__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        return x


class CountRegressor(nn.Module):
    def __init__(self, in_planes: int, pool='mean', use_bn=False):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = Conv(in_planes, 98, 7, padding=3, use_bn=use_bn)
        self.conv2 = Conv(98, 64, 5, padding=2, use_bn=use_bn)
        self.conv3 = Conv(64, 32, 3, padding=1, use_bn=use_bn)
        self.conv4 = Conv(32, 32, 1, use_bn=use_bn)
        # do not use batch normalization for last layer
        self.conv5 = Conv(32, 1, 1, use_bn=False)

    def forward(self, x: torch.Tensor, split_size: List[int]):
        # x = (x.permute(1, 2, 3, 0) /
        #      (x.max(3).values.max(2).values.max(1).values + 1e-8)).permute(3, 0, 1, 2)
        x = x / (x.max() + 1e-8)
        x = self.upsampling(self.conv1(x))
        x = self.upsampling(self.conv2(x))
        x = self.upsampling(self.conv3(x))
        x = self.conv4(x)
        x = self.conv5(x)
        y = torch.split(x, split_size)
        if self.pool == 'mean':
            return [torch.mean(group, dim=0, keepdim=True) for group in y]
        else:  # self.pool == 'max'
            return [torch.max(group, dim=0, keepdim=False) for group in y]


class CountRegressorPaper(nn.Module):
    def __init__(self, in_planes: int, pool='mean'):
        super(CountRegressorPaper, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(in_planes, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def forward(self, im, split_size):
        x = self.regressor(im.squeeze(0))
        if self.pool == 'mean':
            return torch.mean(x, dim=0, keepdim=True)
        else:
            return torch.max(x, dim=0, keepdim=True)[0]


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)
