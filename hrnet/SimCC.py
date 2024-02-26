import os
import logging
import torch
import torch.nn as nn
from torch.nn import functional as nn_fn
from torchvision import models
from einops import rearrange, repeat
from yacs.config import CfgNode as CN

logger = logging.getLogger(__name__)
BN_MOMENTUM = 0.1


# 中间层特征提取
class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


def deconv(in_cs, out_cs):
    return nn.Sequential(
        nn.ConvTranspose2d(in_cs, out_cs, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
        nn.BatchNorm2d(out_cs),
        nn.ReLU(inplace=True),
    )


class SimCC(nn.Module):
    """docstring for DeepPose"""

    def __init__(self, w, h, nJoints, extract_list, model_path, device, train=1):
        super(SimCC, self).__init__()
        self.w = w
        self.h = h
        self.nJoints = nJoints
        self.extract_list = extract_list
        self.model_path = model_path
        self.device = device
        # 加载resnet
        self.resnet = models.resnet50(pretrained=False)
        self.Up1 = deconv(2048, 256)  # size: 20*15--40*30
        self.Up2 = deconv(256, 256)  # size: 40*30--80*60
        self.Up3 = deconv(256, 256)  # size: 80*60--160*120
        self.outConv = nn.Conv2d(256, self.nJoints, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

        self.x_vector = nn.Linear((self.w // 8) * (self.h // 8), w)
        self.y_vector = nn.Linear((self.w // 8) * (self.h // 8), h)

    def forward(self, x):
        f = self.SubResnet(x)
        # print(len(f), f[0].shape)
        f = self.Up1(f[0])
        f = self.Up2(f)
        f = self.Up3(f)
        f = self.outConv(f)
        f = nn_fn.adaptive_avg_pool2d(f, (self.w // 8, self.h // 8))
        f = f.reshape((f.shape[0], self.nJoints, -1))
        pred_x = self.x_vector(f)
        pred_y = self.y_vector(f)
        return pred_x, pred_y


class SimCC_r34(nn.Module):
    """docstring for DeepPose"""

    def __init__(self, w, h, nJoints, extract_list, model_path, device, train=1):
        super(SimCC_r34, self).__init__()
        self.w = w
        self.h = h
        self.nJoints = nJoints
        self.extract_list = extract_list
        self.model_path = model_path
        self.device = device
        # 加载resnet
        self.resnet = models.resnet34(pretrained=False)
        self.Up1 = deconv(512, 128)  # size: 20*15--40*30
        self.Up2 = deconv(128, 64)  # size: 40*30--80*60
        # self.Up3 = deconv(128, 128)  # size: 80*60--160*120
        self.outConv = nn.Conv2d(64, self.nJoints, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

        self.x_vector = nn.Linear((self.w // 8) * (self.h // 8), w)
        self.y_vector = nn.Linear((self.w // 8) * (self.h // 8), h)

    def forward(self, x):
        f = self.SubResnet(x)
        f = self.Up1(f[0])
        f = self.Up2(f)
        # f = self.Up3(f)
        f = self.outConv(f)
        # f = nn_fn.adaptive_avg_pool2d(f, (self.w//8, self.h//8))
        f = f.reshape((f.shape[0], self.nJoints, -1))
        pred_x = self.x_vector(f)
        pred_y = self.y_vector(f)
        return pred_x, pred_y


class SimCC_v2(nn.Module):
    """docstring for SimCC"""

    def __init__(self, w, h, nJoints, extract_list, model_path, device, train=1):
        super(SimCC_v2, self).__init__()
        self.w = w
        self.h = h
        self.nJoints = nJoints
        self.extract_list = extract_list
        self.model_path = model_path
        self.device = device
        # 加载resnet
        self.resnet = models.resnet50(pretrained=False)
        # self.Up1 = deconv(2048, 256)  # size: 20*15--40*30
        # self.Up2 = deconv(256, 256)  # size: 40*30--80*60
        # self.Up3 = deconv(256, 256)  # size: 80*60--160*120
        self.outConv = nn.Conv2d(256, self.nJoints, kernel_size=(1, 1), stride=(1, 1))
        if train:
            nn.init.normal_(self.outConv.weight, std=0.001)
            nn.init.constant_(self.outConv.bias, 0)
            self.resnet.load_state_dict(torch.load(self.model_path, map_location=self.device), strict=False)
        self.SubResnet = FeatureExtractor(self.resnet, self.extract_list)  # 提取resnet层

        self.x_vector = nn.Linear((self.w // 8) * (self.h // 8), w)
        self.y_vector = nn.Linear((self.w // 8) * (self.h // 8), h)

    def forward(self, x):
        f = self.SubResnet(x)
        print(len(f))
        # f = self.Up1(f[0])
        # f = self.Up2(f)
        # f = self.Up3(f)
        f = self.outConv(f)
        f = nn_fn.adaptive_avg_pool2d(f, (self.w // 8, self.h // 8))
        f = f.reshape((f.shape[0], self.nJoints, -1))
        pred_x = self.x_vector(f)
        pred_y = self.y_vector(f)
        return pred_x, pred_y


class SimCC_hr(nn.Module):
    """docstring for SimCC"""

    def __init__(self, w, h, nJoints, hrnet):
        super(SimCC_hr, self).__init__()
        self.w = w
        self.h = h
        self.nJoints = nJoints
        self.hrnet = hrnet
        # self.x_vector = nn.Linear((self.w//8) * (self.h//8), w)
        # self.y_vector = nn.Linear((self.w//8) * (self.h//8), h)
        self.x_vector = nn.Linear((self.w // 4) * (self.h // 4), w)
        self.y_vector = nn.Linear((self.w // 4) * (self.h // 4), h)

    def forward(self, x):
        f = self.hrnet(x)
        # f = nn_fn.adaptive_avg_pool2d(f, (self.w//8, self.h//8))
        f = f.reshape((f.shape[0], self.nJoints, -1))
        pred_x = self.x_vector(f)
        pred_y = self.y_vector(f)
        return pred_x, pred_y


class SimCC_hrv2(nn.Module):
    """docstring for SimCC"""

    def __init__(self, w, h, nJoints, hrnet):
        super(SimCC_hrv2, self).__init__()
        self.w = w
        self.h = h
        self.nJoints = nJoints
        self.hrnet = hrnet
        # self.x_vector = nn.Linear((self.w//8) * (self.h//8), w)
        # self.y_vector = nn.Linear((self.w//8) * (self.h//8), h)
        self.x_vector1 = nn.Linear((self.w // 4) * (self.h // 4), (self.w // 8) * (self.h // 8))
        self.y_vector1 = nn.Linear((self.w // 4) * (self.h // 4), (self.w // 8) * (self.h // 8))
        self.x_vector2 = nn.Linear((self.w // 8) * (self.h // 8), w * 2)
        self.y_vector2 = nn.Linear((self.w // 8) * (self.h // 8), h * 2)
        self.x_vector3 = nn.Linear(w * 2, w)
        self.y_vector3 = nn.Linear(h * 2, h)

    def forward(self, x):
        f = self.hrnet(x)
        # f = nn_fn.adaptive_avg_pool2d(f, (self.w//8, self.h//8))
        f = f.reshape((f.shape[0], self.nJoints, -1))
        f_x = self.x_vector1(f)
        f_y = self.y_vector1(f)
        f_x = self.x_vector2(f_x)
        f_y = self.y_vector2(f_y)
        pred_x = self.x_vector3(f_x)
        pred_y = self.y_vector3(f_y)
        return pred_x, pred_y


class SimCC_hrv3(nn.Module):
    """docstring for SimCC"""

    def __init__(self, w, h, nJoints, hrnet):
        super(SimCC_hrv3, self).__init__()
        self.w = w
        self.h = h
        self.nJoints = nJoints
        self.hrnet = hrnet
        self.x_vector2 = nn.Linear((self.w // 4) * (self.h // 4), w * 2)
        self.y_vector2 = nn.Linear((self.w // 4) * (self.h // 4), h * 2)
        self.x_vector3 = nn.Linear(w * 2, w)
        self.y_vector3 = nn.Linear(h * 2, h)

    def forward(self, x):
        f0 = self.hrnet(x)
        # f = nn_fn.adaptive_avg_pool2d(f, (self.w//8, self.h//8))
        f = f0.reshape((f0.shape[0], self.nJoints, -1))
        f_x = self.x_vector2(f)
        f_y = self.y_vector2(f)
        pred_x = self.x_vector3(f_x)
        pred_y = self.y_vector3(f_y)
        return pred_x, pred_y, f0
