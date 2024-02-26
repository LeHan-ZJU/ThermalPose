# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: MyPoseNet
# @Author : Hanle
# @E-mail : hanle@zju.edu.cn
# @Date   : 2021-06-17
# --------------------------------------------------------
"""

import csv
import re
import os
import cv2
import math
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging
from PIL import Image


def rotate_img(image, angle, center):  # 对输入图像以点center为中心，旋转angle度
    # 构造变换矩阵
    # print('center:', center, 'angle:', angle)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 应用变换矩阵到图像
    rotated_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return rotated_img


def CenterLabelHeatMap(img_width, img_height, c_x, c_y, sigma):  # 根据关键点的坐标生成heatmap
    if max(c_x, c_y) == 0:
        return np.zeros([img_width, img_height])
    else:
        X1 = np.linspace(1, img_width, img_width)
        Y1 = np.linspace(1, img_height, img_height)
        [X, Y] = np.meshgrid(Y1, X1)
        X = X - c_x
        Y = Y - c_y
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma * sigma
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        heatmap = heatmap * 255
        return heatmap


def CenterLabelHeatMapResize(img_height, img_width, c_x, c_y, resize_h, resize_w, sigma):   # 根据关键点的坐标生成heatmap
    if max(c_x, c_y) == 0:
        return np.zeros([resize_h, resize_w])
    else:
        c_x = int(c_x * (resize_w / img_width))
        c_y = int(c_y * (resize_h / img_height))
        # sigma = max(int(sigma * (resize_h / img_height)), 1)

        Y1 = np.linspace(1, resize_w, resize_w)
        X1 = np.linspace(1, resize_h, resize_h)
        [X, Y] = np.meshgrid(Y1, X1)
        X = X - c_x
        Y = Y - c_y
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma * sigma
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        heatmap = heatmap * 255
        return heatmap


def CenterLabelHeatMapResize2(img_height, img_width, c_x, c_y, resize_h, resize_w, sigma):   # 先画heatmap，然后再resize
    if max(c_x, c_y) == 0:
        return np.zeros([resize_h, resize_w])
    else:
        # c_x = int(c_x * (resize_w / img_width))
        # c_y = int(c_y * (resize_h / img_height))
        # sigma = max(int(sigma * (resize_h / img_height)), 1)

        # Y1 = np.linspace(1, resize_w, resize_w)
        # X1 = np.linspace(1, resize_h, resize_h)
        Y1 = np.linspace(1, img_width, img_width)
        X1 = np.linspace(1, img_height, img_height)
        [X, Y] = np.meshgrid(Y1, X1)
        X = X - c_x
        Y = Y - c_y
        D2 = X * X + Y * Y
        E2 = 2.0 * sigma * sigma
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        heatmap = heatmap * 255
        heatmap = cv2.resize(heatmap, (resize_w, resize_h))
        return heatmap


def RealtionLabelHeatMap(img_height, img_width, c_x1, c_y1, c_x2, c_y2, line_thickness, blur_sigma):
    # 根据关键点的坐标生成relation-heatmap
    relationmap = np.zeros([img_height, img_width])

    relationmap = cv2.line(relationmap, (c_x1, c_y1), (c_x2, c_y2), line_thickness)
    relationmap = cv2.GaussianBlur(relationmap, ksize=(blur_sigma, blur_sigma),
                                       sigmaX=blur_sigma, sigmaY=blur_sigma)
    relationmap = relationmap  # * 255
    return relationmap


def find_edgePoints(num, points_all, w, h):
    points = []
    # print(points_all)
    for sort in range(num):
        if min(points_all[sort, :]) > 0:
            points.append(points_all[sort, :])
    points = np.array(points)
    # print(points.shape)
    if len(points) == 0:
        return np.array([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])
    else:
        list_x = list(points[:, 0])
        list_y = list(points[:, 1])
        sort_min_x = list_x.index(min(list_x))
        sort_min_y = list_y.index(min(list_y))
        sort_max_x = list_x.index(max(list_x))
        sort_max_y = list_y.index(max(list_y))
        point_lu = [points[sort_min_x, 0], points[sort_min_y, 1]]  # 左上
        point_rd = [points[sort_max_x, 0], points[sort_max_y, 1]]  # 右下
        point_ru = [points[sort_max_x, 0], points[sort_min_y, 1]]  # 右上
        point_ld = [points[sort_min_x, 0], points[sort_max_y, 1]]  # 左下
        # print(point_lu, point_ru, point_rd, point_ld)

        # area = np.array([point_lu, point_ru, point_rd, point_ld, point_lu])
        return np.array([point_lu, point_ru, point_rd, point_ld, point_lu])


def object_area(num, points_all):
    # fuction: 根据输入的一组关键点集合，先挑选出其中所有不全为0的点，然后两两组合

    # 选出所有不为0的点
    points = []
    # print(points_all)
    for sort in range(num):
        if min(points_all[sort, :]) > 0:
            points.append(points_all[sort, :])
    points = np.array(points)
    # print(points)

    # 建立连接关系
    relation = []
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            relation.append([i, j])
    relation = np.array(relation)
    # print(relation)

    area = []
    for i in range(len(relation)):
        area.append(points[relation[i, 0], :])
        area.append(points[relation[i, 1], :])

    return np.array(area)


def LocationMapResize(img_height, img_width, points_all, resize_h, resize_w, coco):   # 根据关键点的坐标生成heatmap
    num = points_all.shape
    locationmap = np.zeros([int(resize_h), int(resize_w), 3])  # 初始化定位标签
    locationmap = locationmap.astype(np.uint8)
    Points = []
    for i in range(num[1]):
        c_x = points_all[0, i, 0]
        c_y = points_all[0, i, 1]
        # if min(c_x, c_y) > 0:
        c_x = int(c_x * (resize_w / img_width))
        c_y = int(c_y * (resize_h / img_height))
        Points.append([c_x, c_y])
    Points = np.array(Points)

    if coco == 1:
        # area = find_edgePoints(num[1], Points)
        area = object_area(num[1], Points)
        area = area.astype(int)
        if len(area) > 0:
            cv2.fillConvexPoly(locationmap, area, (255, 255, 255))
            locationmap = cv2.GaussianBlur(locationmap[:, :, 0], (45, 45), 25)  # kernel_size=(45, 45)， sigma=45
        else:
            locationmap = locationmap[:, :, 0]

    else:
        area = np.array([Points[0, :], Points[2, :], Points[5, :], Points[3, :], Points[1, :], Points[4, :]])
        area = area.astype(int)
        cv2.fillConvexPoly(locationmap, area, (255, 255, 255))
        locationmap = cv2.GaussianBlur(locationmap[:, :, 0], (25, 25), 25)  # kernel_size=(45, 45)， sigma=45

    # print('2:', locationmap.shape)
    # cv2.imshow('map', locationmap)
    # cv2.waitKey(0)
    locationmap = cv2.resize(locationmap, (int(resize_w / 16), int(resize_h / 16)))
    locationmap = np.expand_dims(locationmap, axis=0)
    locationmap = np.array(locationmap/255)
    # print('3:', locationmap.shape)

    return locationmap


def cal_angle(v1, v2):  # 计算向量夹角
    pi = 3.1415
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = math.sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * math.sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (math.acos(cos) / pi) * 180


def cal_norm(v):  # 计算向量模长度
    return math.sqrt(pow(v[0], 2) + pow(v[1], 2))


def normalization_vec(v):  # 一维向量归一化
    v = np.array(v)
    return v/max(v)


def gen_kernel(size):
    Kernel = np.zeros([size, size], np.float32) / 225
    for i in range(size):
        for j in range(size):
            if math.sqrt(pow((i-size/2), 2) + pow((j-size/2), 2)) <= (size/2):
                Kernel[i, j] = 1 / math.sqrt((math.sqrt(pow((i-size/2), 2) + pow((j-size/2), 2))))
    # print(Kernel)
    # print(sum(sum(Kernel)))
    Kernel = Kernel / sum(sum(Kernel))
    return Kernel


def LocationMapResize_new(img_height, img_width, labels, index, resize_h, resize_w, searchCont, sort0, num_points, coco, num_up):
    #  LocationMapResize(h, w, self.labels, idx, self.resize_h, self.resize_w, searchContext, self.sort0)
    sort = sort0[:]
    keypoints = []
    s = 0
    start = 0
    if coco == 1:
        start = 1

    for j in range(start, num_points):
        # print('j0:', j)
        # print(sort)
        point = labels[index][3 + j]
        numList2 = [m.start() for m in re.finditer(searchCont, point)]
        p_x = int(float(point[0:numList2[0]]))
        p_y = int(float(point[numList2[0] + 1:numList2[1]]))
        if min(p_x, p_y) > 0:
            keypoints.append([p_x, p_y])
        else:
            s = s + 1
            if (j - s) in sort:
                sort.remove(j - s)
            for n in range(len(sort)):
                if sort[n] > j - s:
                    sort[n] = sort[n] - 1

    keypoints = np.array(keypoints)
    # print('keyp:', keypoints.shape, len(keypoints), keypoints)
    # print(sort)

    if len(keypoints) == 1:
        locationmap = CenterLabelHeatMap(img_width, img_height, keypoints[0, 0], keypoints[0, 1], 5)
    elif len(keypoints) == 2:
        locationmap = RealtionLabelHeatMap(img_height, img_width, keypoints[0, 0], keypoints[0, 1],
                                           keypoints[1, 0], keypoints[1, 1], 3, 5)
    elif len(keypoints) == 0:
        locationmap = np.zeros([img_height, img_width])
        locationmap = locationmap.astype(np.uint8)
    else:
        points_chain = sort[0:2]
        # print('point_chain:', points_chain)
        # print(points_chain[-2])
        area = []
        p1 = keypoints[points_chain[-2], :]
        p2 = keypoints[points_chain[-1], :]
        area.append(p1)
        area.append(p2)
        # cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 2)
        # cv2.imshow('img_new', img)
        # cv2.waitKey(0)

        locationmap = np.zeros([img_height, img_width, 3])  # 初始化定位标签
        locationmap = locationmap.astype(np.uint8)

        candicate = sort[2:]
        for i in range(len(keypoints) - 2):
            # print('i:', i)
            v1 = keypoints[points_chain[-2]] - keypoints[points_chain[-1], :]
            angles = []
            norms = []
            for j in range(len(candicate)):
                # print('j', candicate[j])
                v2 = keypoints[candicate[j], :] - keypoints[points_chain[-1], :]
                angles.append(cal_angle(v1, v2))
                norms.append(cal_norm(v2))
            # print(angles, norms)
            angles_norm = normalization_vec(angles)
            norms_norm = normalization_vec(norms)
            score = 0.8 * angles_norm + 0.2 * norms_norm
            # print(angles_norm, norms_norm, score)

            # del str[1]  # 删除该位置的值
            # 删除并在新位置插入最大分数点序号
            index_max = np.argmax(np.array(score))
            # print('index_max:', index_max)
            points_chain.append(candicate[index_max])
            candicate.remove(candicate[index_max])
            if i == 0:
                candicate.append(sort[0])
            # sort0.insert(i+2, index_max)
            # print('candicate:', candicate)
            # print('points_chain:', points_chain)

            p1 = keypoints[points_chain[-2], :]
            p2 = keypoints[points_chain[-1], :]
            area.append(p2)
            # cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0), 2)
            # cv2.imshow('img_new', img)
            # cv2.waitKey(0)

            if points_chain[-1] == sort[0]:
                break

        area = np.array(area)
        area = area.astype(int)
        cv2.fillConvexPoly(locationmap, area, (255, 255, 255))
        kernel = gen_kernel(51)
        locationmap = cv2.filter2D(locationmap, -1, kernel=kernel)
        locationmap = locationmap[:, :, 0]
    # cv2.imshow('map', locationmap)
    # cv2.waitKey(0)
    # print('location:', locationmap.shape)
    # locationmap = cv2.resize(locationmap, (int(resize_w / 16), int(resize_h / 16)))
    locationmap = cv2.resize(locationmap, (int((resize_w / 32) * 2 ** num_up), int((resize_h / 32) * 2 ** num_up)))
    locationmap = np.expand_dims(locationmap, axis=0)
    locationmap = np.array(locationmap / 255)
    return locationmap


def AugImg(Img, resize):
    # 对图像按目标尺寸进行resize和padding，并且将标签也进行对应变换 resize = [H, W]
    h, w, _ = Img.shape
    if max([h, w]) == w:   # 按w对齐
        res_2 = int(h * (resize[1] / w))
        Img = cv2.resize(Img, (resize[1], res_2))
        padding_l = int((resize[0] - res_2) / 2)
        if padding_l > 0:
            Img = cv2.copyMakeBorder(Img, padding_l, padding_l, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # 上下填充0

    else:   # 按h对齐
        res_2 = int(w * (resize[0] / h))
        Img = cv2.resize(Img, (res_2, resize[0]))
        padding_l = int((resize[1] - res_2) / 2)
        if padding_l > 0:
            Img = cv2.copyMakeBorder(Img, 0, 0, padding_l, padding_l, cv2.BORDER_CONSTANT, value=(0, 0, 0))  # 左右填充0

    return Img


def Trans_point(p, h, w, resize):
    if max([h, w]) == w:  # 按w对齐
        res_2 = int(h * (resize[1] / w))
        padding_l = int((resize[0] - res_2) / 2)
        p[0] = int(p[0] * (resize[1] / w))  # x
        p[1] = int(p[1] * (resize[1] / w) + padding_l)  # y
    else:  # 按h对齐
        res_2 = int(w * (resize[0] / h))
        padding_l = int((resize[1] - res_2) / 2)
        p[0] = int(p[0] * (resize[0] / h) + padding_l)  # x
        p[1] = int(p[1] * (resize[0] / h))  # y

    return p


class DatasetPoseCSV_pad(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, scale, num_points, Aug):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.num_points = num_points
        self.Aug = Aug

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)

            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:
                self.name_list.append(img_sort)
                points_num = int(self.labels[img_sort][1])  # 当前图像中的人数
                img_sort = img_sort+points_num
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis = -1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):

        idx = int(self.name_list[i])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        # print(img_file)

        img = Image.open(img_file)
        if self.Aug == 1:    # 判断是否加数据增广，若Aug=1则加增广
            RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
            if RandomRate[0] == 1:
                img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape
        img = AugImg(img0, [self.resize_h, self.resize_w])
        img = self.preprocess(self.resize_w, self.resize_h, img0, 1)
        searchContext = "_"

        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point
                sigma = 1
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

        # shows = np.zeros([self.resize_h, self.resize_w, 3])
        # for kk in range(3):
        #     shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
        # cv2.imshow('image', shows)
        # cv2.waitKey(0)

        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        heatmaps = np.array(heatmaps)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)  # ,
        }


class DatasetPoseCSV(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, scale, num_points, Aug):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.num_points = num_points
        self.Aug = Aug

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)

            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:
                self.name_list.append(img_sort)
                points_num = int(self.labels[img_sort][1])  # 当前图像中的人数
                img_sort = img_sort+points_num
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis = -1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):

        idx = int(self.name_list[i])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图

        img = Image.open(img_file)
        if self.Aug == 1:    # 判断是否加数据增广，若Aug=1则加增广
            RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
            if RandomRate[0] == 1:
                img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
        img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        h, w, _ = img0.shape
        img = self.preprocess(self.resize_w, self.resize_h, img0, 1)
        searchContext = "_"

        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                points_all[k, n, :] = point
                sigma = 2
                heatmap0 = CenterLabelHeatMapResize(h, w, point[0], point[1], self.resize_h, self.resize_w, sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

        # shows = np.zeros([self.resize_h, self.resize_w, 3])
        # for kk in range(3):
        #     shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
        # cv2.imshow('image', shows)
        # cv2.waitKey(0)

        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        heatmaps = np.array(heatmaps)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor)  # ,
        }


class DatasetStage2_single(Dataset):
    def __init__(self, resize_w, resize_h, imgs_dir, csv_path, num_points, aug, scale, angle):
        self.resize_w = resize_w
        self.resize_h = resize_h
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.scale = scale
        self.angle = angle
        self.num_points = num_points
        self.Aug = aug

        # 读标签
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            self.labels = list(reader)
            self.name_list = []   # 存放每个图片第一行标签在labels中的行序号，长度为图像数
            img_sort = 1
            trainset_num = len(self.labels)

            while img_sort < trainset_num:  # 4198:  # len(self.labels):  #
                self.name_list.append(img_sort)
                img_sort = img_sort + 1
            logging.info(f'Creating dataset with {len(self.name_list)} examples')

    def __len__(self):
        return len(self.name_list)

    @classmethod
    def preprocess(cls, resize_w, resize_h, pil_img, trans):

        if trans == 1:
            pil_img = cv2.resize(pil_img, (resize_w, resize_h))
            img_nd = np.array(pil_img)
            if len(img_nd.shape) == 2:
                img_nd0 = img_nd
                img_nd = np.expand_dims(img_nd0, axis=2)
                img_nd = np.concatenate([img_nd, img_nd, img_nd], axis=-1)
            img_nd = img_nd.transpose((2, 0, 1))  # 如果输入是img，则HWC to CHW
        else:
            img_nd = pil_img.transpose((2, 0, 1))
        if img_nd.max() > 1:   # 归一化
            img_nd = img_nd / 255
        return img_nd

    def __getitem__(self, i):
        idx = int(self.name_list[i])
        people_num = int(self.labels[idx][1])  # 当前图像中的人数
        img_name = self.labels[idx][0]
        img_file = self.imgs_dir + self.labels[idx][0]  # 获取图像名，读图
        numList = [m.start() for m in re.finditer('/', img_name)]
        if img_file[-9:-5] == 'flip':
            if img_name[1: numList[1]] in (['maze', 'minefield', 'treadmill']):
                img = Image.open(img_file[:-10] + '.bmp')
            else:
                img = Image.open(img_file[:-10] + '.jpg')
            if self.Aug == 1:
                RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
                if RandomRate[0] == 1:
                    img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                    img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
            img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = img0.shape
            # img0 = cv2.flip(img0, int(img_file[-5]))
            img0 = rotate_img(img0, 180, (w / 2, h / 2))  # 旋转增广

        else:
            img = Image.open(img_file)
            if self.Aug == 1:
                RandomRate = random.sample(range(2), 1)  # 百分之五十的概率进行数据增广
                if RandomRate[0] == 1:
                    img1 = transforms.ColorJitter(brightness=0.5)(img)  # 随机从0~2之间的亮度变化
                    img = transforms.ColorJitter(contrast=1)(img1)  # 随机从0~2之间的对比度
            img0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            h, w, _ = img0.shape

        img = AugImg(img0, [self.resize_h, self.resize_w])
        # flip_type = 1 - random.sample(range(3), 1)[0]  # 随机生成-1到1之间的三个整数
        # img_flip = cv2.flip(img, flip_type)
        # print('angle:', self.angle)

        img = self.preprocess(self.resize_w, self.resize_h, img, 1)
        searchContext = "_"

        # 初始化heatmap
        heatmaps = np.zeros([int(self.resize_h/self.scale), int(self.resize_w/self.scale), self.num_points])
        points_all = np.zeros([people_num, self.num_points, 2])  # 存放当前图中的所有人的所有关键点
        for k in range(people_num):
            index = idx + k
            for n in range(self.num_points):
                numList = [m.start() for m in re.finditer(searchContext, self.labels[index][n + 3])]
                point = [int(self.labels[index][n + 3][0:numList[0]]),
                         int(self.labels[index][n + 3][numList[0] + 1:numList[1]])]
                point = Trans_point(point, h, w, [self.resize_h, self.resize_w])
                points_all[k, n, :] = point

                sigma = 3
                heatmap0 = CenterLabelHeatMap(self.resize_h, self.resize_w, point[0], point[1], sigma)
                heatmap = cv2.resize(heatmap0, (int(self.resize_w / self.scale), int(self.resize_h / self.scale)))
                heatmaps[:, :, n] = heatmaps[:, :, n] + heatmap

                # heatmap_flip = cv2.flip(heatmap, flip_type)
                # heatmaps_flip[:, :, n] = heatmaps_flip[:, :, n] + heatmap_flip

                # savename = 'E:/Codes/Mine/RatPose/data/label_'+ str(i) + '_' + str(j) + '.jpg'
                # cv2.imwrite(savename, heatmap0*255 + img[0, :, :]*255)  # heatmap + img_show*128)
            # shows = np.zeros([self.resize_h, self.resize_w, 3])
            # for kk in range(3):
            #     shows[:, :, kk] = heatmap0 + img[kk, :, :] * 0.3
            # cv2.imshow('image', shows)
            # cv2.waitKey(0)

        heatmaps = self.preprocess(self.resize_w, self.resize_h, heatmaps, 0)
        heatmaps = np.array(heatmaps)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'heatmap': torch.from_numpy(heatmaps).type(torch.FloatTensor),
        }
