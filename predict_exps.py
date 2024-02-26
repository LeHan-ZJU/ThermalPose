import argparse
import logging
import os
import re
import cv2
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
from PIL import Image
import matplotlib.pyplot as plt

from hrnet.hrnet_Rat import HRNet
from hrnet.HourGlassNet import Bottleneck, HourglassNet
from hrnet.DeepLabCut import DeepLabCut
from Models.RatNetAttention_DOConv import Net_ResnetAttention_DOConv
from utils.dataset_csv import DatasetPoseCSV, AugImg
from Eval.PCK import keypoint_pck_accuracy
from Eval.PCK_Mine import PCK_metric, PCK_metric_box
from Eval.CalAP import cal_AP, cal_OKS
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
curve_name = 'DLC_wb_HRNet32'

resize_w = 640
resize_h = 480
extract_list = ["layer4"]

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m',
                        default='/8T/hanle/Models/ThermalPose/TrainedModels/ThermalImgs_Color/debug_hr32_pad/CP_epoch200.pth',
                        metavar='FILE', help="Specify the file in which the model is stored")
    parser.add_argument('-l', '--label_dir',
                        default='/8T/hanle/Datasets/Search/labels/ThermalImgs_Color_test.csv',
                        metavar='LABEL', nargs='+', help='path of labels')
    parser.add_argument('-i', '--img_dir', default='/8T/hanle/Datasets/Search/imgs/',
                        metavar='INPUT', nargs='+', help='filenames of input images')
    parser.add_argument('-o', '--output',  default='/8T/hanle/Models/ThermalPose/TrainedModels/ThermalImgs_Color/debug_hr32_pad/results/',
                        metavar='OUTPUT', nargs='+', help='Filenames of ouput images')
    parser.add_argument('-c', '--channel', default=14,
                        metavar='CHANNEL', nargs='+', help='Number of keypoints')
    parser.add_argument('-n', '--num_points', default=14,
                        metavar='num_points', nargs='+', help='Number of keypoints')
    parser.add_argument('--scale', '-s', type=float, help="Scale factor for the input images", default=1)
    return parser.parse_args()


def predict_img(net,
                full_img,
                device,
                resize_w,
                resize_h,
                scale_factor=1):
    net.eval()

    # padding
    img = AugImg(full_img, [resize_h, resize_w])
    img = DatasetPoseCSV.preprocess(resize_w, resize_h, full_img, 1)
    # print('img：', img.shape)
    # show = img
    # cv2.imshow('image', show[0, :, :])
    # cv2.waitKey(0)
    img = torch.from_numpy(img)  # self.resize_w, self.resize_h, img0, self.scale, 1
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output1 = net(img)
        probs = F.softmax(output1, dim=1)
        probs = probs.squeeze(0)
        output1 = probs.cpu()

        # output2 = output2.squeeze(0)
        # output2 = output2.cpu()

    # print('output:', output1.shape, 'output2:', output2.shape)

    return output1#, output2  # > out_threshold


def heatmap_to_points(Img, heatmap, numPoints, ori_W, ori_H):
    Img = cv2.resize(Img, (ori_W, ori_H))
    keyPoints = np.zeros([numPoints, 2])  # 第一行为y（对应行数），第二行为x（对应列数）
    # Img = cv2.resize(Img, (resize_w // 4, resize_h // 4))
    for j in range(numPoints):
        hm = cv2.resize(heatmap[j, :, :], (ori_W, ori_H))
        center = np.unravel_index(np.argmax(hm), hm.shape)   # 寻找最大值点
        # print('center:', center)
        keyPoints[j, 0] = center[1]   # X
        keyPoints[j, 1] = center[0]   # Y
        cv2.circle(Img, (center[1], center[0]), 1, (0, 0, 255), 2)  # 画出heatmap点重心    img = img*0.3    print(keyPoints)
    return Img, keyPoints


def draw_relation(Img, allPoints, relations):
    for k in range(len(relations)):
        c_x1 = int(allPoints[relations[k][0]-1, 0])
        c_y1 = int(allPoints[relations[k][0]-1, 1])
        c_x2 = int(allPoints[relations[k][1]-1, 0])
        c_y2 = int(allPoints[relations[k][1]-1, 1])
        # print('p1p2:', c_x1, c_y1, c_x2, c_y2)
        cv2.line(Img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
    return Img


def draw_relation2(Img, allPoints, relations):  # 640*480
    # Img = cv2.resize(Img, (640, 480))
    for k in range(len(relations)):
        c_x1 = int(allPoints[relations[k][0]-1, 0] * (W/resize_w))
        c_y1 = int(allPoints[relations[k][0]-1, 1] * (H/resize_h))
        c_x2 = int(allPoints[relations[k][1]-1, 0] * (W/resize_w))
        c_y2 = int(allPoints[relations[k][1]-1, 1] * (H/resize_h))
        cv2.line(Img, (c_x1, c_y1), (c_x2, c_y2), (0, 255, 0), 2)
    r = np.arange(50, 255, int(205/len(allPoints)))

    for j in range(len(allPoints)):  # 在原图中画出关键点
        cv2.circle(Img, (int(allPoints[j, 0] * (W/resize_w)), int(allPoints[j, 1] * (H/resize_h))), 2,
                   [int(r[len(allPoints) - j]), 20, int(r[j])], 2)
    return Img


def draw_maps(Img, heatmaps, num_points):
    h, w, c = Img.shape
    img = Img * 0.2
    for i in range(num_points):
        hp = heatmaps[i, :, :]
        NormMap = (hp - np.mean(hp)) / (np.max(hp) - np.mean(hp))
        map = np.round(NormMap*100)
        map = cv2.resize(map, (w, h))

        img[:, :, 1] = map + img[:, :, 1]
        cv2.imshow('hp', np.uint8(img))
        cv2.waitKey(0)



def read_labels(label_dir):
    # 读标签
    with open(label_dir, 'r') as f:
        reader = csv.reader(f)
        labels = list(reader)
        imgs_num = len(labels)
    print('imgs_num:', imgs_num)
    return labels, imgs_num


if __name__ == "__main__":
    args = get_args()
    in_files = args.img_dir
    out_files = args.output
    num_points = args.channel
    num_points2 = args.num_points
    isExists = os.path.exists(out_files)
    if not isExists:  # 判断结果
        os.makedirs(out_files)

    labels, imgs_num = read_labels(args.label_dir)  # 读取标签

    # keypoints = ['rRP', 'lRP', 'rFP', 'lFP', 'tail_root', 'head']
    # relation = [[1, 4], [1, 5], [2, 3], [2, 5], [3, 6], [4, 6]]
    if args.channel == 14:
        keypoints = ['Right_ankle', 'Right_knee', 'Right_hip', 'Left_hip', 'Left_knee', 'Left_ankle',  'Right_wrist',
              'Right_elbow', 'Right_shoulder', 'Left_shoulder', 'Left_elbow', 'Left_wrist', 'Neck', 'Head_top']
        relation = [[1, 2], [2, 3], [3, 9], [4, 5], [4, 10], [5, 6], [7, 8], [8, 9],[9, 13], [10, 11], [10, 13], 
                    [11, 12], [13, 14]]
    # 构建网络
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info(f'Using device {device}')

    net = HRNet(c=32, n_channels=3, nof_joints=num_points, bn_momentum=0.1)
    # net = HourglassNet(Bottleneck, num_stacks=4, num_blocks=4, num_classes=num_points)
    # net = DeepLabCut(nJoints=num_points, extract_list=extract_list, model_path=None, device=device, train=0)
    # net = Net_ResnetAttention_DOConv(args.model, extract_list, device, train=False, n_channels=3, nof_joints=num_points)
    logging.info("Loading model {}".format(args.model))
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    logging.info("Model loaded !")

    # # stage2阶段先加载RatNet在前面训练好的权重，然后再加载GAN训练的resnetnet部分权重，再将resnet冻结
    # net.SubResnet.load_state_dict(torch.load(args.path_backbone, map_location=device), strict=False)
    # print('Pretrained generator weights have been loaded!')

    points_all_gt = np.zeros([imgs_num-1, num_points2 + 2, 2])     # 存放所有人的所有关键点标签
    points_all_pred = np.zeros([imgs_num-1, num_points, 2])   # 存放检测到的所有关键点
    time_all = 0
    for k in range(1, imgs_num):
        # 读取图像
        img_path = labels[k][0]
        fn = os.path.join(in_files + img_path)
        if k % 200 == 0:
            print(k, fn)
        img0 = cv2.imread(fn)
        # img = Image.open(fn)
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        H, W, C = img0.shape

        # 网络预测
        time_start = time.time()
        heatmap = predict_img(net=net,
                              full_img=img0,
                              scale_factor=args.scale,
                              device=device,
                              resize_w=resize_w,
                              resize_h=resize_h)
        # print('heatmap:', heatmap.shape)
        heatmap = heatmap.numpy()
        time_end = time.time()
        time_all = time_all + (time_end-time_start)
        # print('time:', time_end-time_start)
        img = cv2.resize(img0, (resize_w, resize_h))
        # print('heatmap:', heatmap.shape)
        # draw_maps(img, heatmap, num_points)


        img, keyPoints = heatmap_to_points(img, heatmap, num_points, resize_w, resize_h)
        # img = draw_relation(img, keyPoints, relation)
        img = draw_relation2(img0, keyPoints, relation)

        # 保存图像
        # searchContext1 = '/'
        # numList = [m.start() for m in re.finditer(searchContext1, labels[k][0])]
        # save_name = out_files + img_path[numList[1] + 1:]
        # print(save_name)
        # cv2.imwrite(save_name, img)
        searchContext1 = '/'
        numList = [m.start() for m in re.finditer(searchContext1, labels[k][0])]
        if len(numList) == 3:
            save_name = out_files + img_path[numList[-3] + 1:numList[-2]] + '_' + img_path[numList[-2] + 1:numList[-1]] + '_' + img_path[numList[-1] + 1:]
        else:
            save_name = out_files + img_path[numList[-2] + 1:numList[-1]] + '_' + img_path[numList[-1] + 1:]
        cv2.imwrite(save_name, img)

        # 保存feature
        # index = filename.index('.')
        # save_feature_name = out_files + 'TestData_Targets_' + filename[0:index] + '.npy'
        # print(save_feature_name)
        # np.save(save_feature_name, feature)

        # 读取标签，评估精度
        searchContext2 = '_'
        for n in range(num_points2):
            numList = [m.start() for m in re.finditer(searchContext2, labels[k][n + 3])]
            point = [int(labels[k][n + 3][0:numList[0]]),
                     int(labels[k][n + 3][numList[0] + 1:numList[1]])]
            # resize
            point_resize = point
            point_resize[0] = point[0] * (resize_w / W)
            point_resize[1] = point[1] * (resize_h / H)
            points_all_gt[k-1, n, :] = point_resize

        # 读取box信息
        numList = [m.start() for m in re.finditer(searchContext2, labels[k][2])]
        box = [int(labels[k][2][0:numList[0]]), int(labels[k][2][numList[0] + 1:numList[1]]),
                int(labels[k][2][numList[1] + 1:numList[2]]), int(labels[k][2][numList[2] + 1:])]
        box_resize = []
        box_resize.append(box[0] * (resize_w / W))
        box_resize.append(box[1] * (resize_h / H))
        box_resize.append(box[2] * (resize_w / W))
        box_resize.append(box[3] * (resize_h / H))
        box_resize = np.array(box_resize)
        points_all_gt[k - 1, num_points2, :] = box_resize[0:2]
        points_all_gt[k - 1, num_points2 + 1, :] = box_resize[2:4]

        points_all_pred[k-1, :, :] = keyPoints
        # print('pred:', points_all_pred[k, :, :], '   gt:', points_all_gt[k, :, :])

    time_mean = time_all/imgs_num
    print('mean time:', time_mean)
    # 保存标签和结果
    Save_result = out_files + 'results.mat'
    sio.savemat(Save_result, {'points_all_gt': points_all_gt, 'points_all_pred': points_all_pred})

    # pck_mine
    pred_name = out_files + 'points_all_pred.npy'
    gt_name = out_files + 'points_all_gt.npy'
    np.save(pred_name, points_all_pred)
    np.save(gt_name, points_all_gt)

    thr = np.linspace(0, 1, 101)
    mean_all = np.ones(101)
    for i in range(101):
        mean_points, var_points, mean_all[i], var_all = PCK_metric_box(points_all_pred[:, 0:num_points2, :], points_all_gt, num_points2, num_points2+1, thr[i])
        if thr[i] == 0.2:
            print('pck_points_mean:', mean_points)
            print('pck_points_val:', var_points)
            print('pck_all_mean:', mean_all[i], '    pck_all_val:', var_all)
    np.save('./PCK_mean_' + curve_name + '.npy', mean_all)
    plt.plot(thr, mean_all, color='b')  # 绘制loss曲线
    plt.xlabel("normalized distance", fontsize=12)
    plt.ylabel("PCK", fontsize=12)
    plt.savefig('./PCK_curve_' + curve_name + '.jpg')
    # print('PCK_mean_all:', mean_all)

    # 计算AP值
    OKS = cal_OKS(points_all_pred[:, 0:num_points2, :], points_all_gt, sigmas=0.06)
    AP, AP50, AP75 = cal_AP(OKS)
    print('AP:', AP, '  AP50:', AP50, ' AP75:', AP75, 'OKS:', np.mean(OKS))
