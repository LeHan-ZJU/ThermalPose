import numpy as np
import glob
import os
import cv2


path_pred = '/results_location_new/Debug1_aug/2stage640_2023/points_all_pred.npy'
path_gt = '/results_location_new/Debug1_aug/2stage640_2023/points_all_gt.npy'

# oks = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # person
# oks = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]  # person
# sigmas = np.array([0.75, 0.75])/10.0

def compute_kpts_oks(dt_kpts, gt_kpts, area, variances):
    """
    this function only works for computing oks with keypoints，
    :param dt_kpts: 模型输出的一组关键点检测结果　dt_kpts.shape=[3,14],dt_kpts[0]表示14个横坐标值，dt_kpts[1]表示14个纵坐标值，dt_kpts[3]表示14个可见性，
    :param gt_kpts:　groundtruth的一组关键点标记结果　gt_kpts.shape=[3,14],gt_kpts[0]表示14个横坐标值，gt_kpts[1]表示14个纵坐标值，gt_kpts[3]表示14个可见性，
    :param area:　groundtruth中当前一组关键点所在人检测框的面积
    :return:　两组关键点的相似度oks
    """
    g = np.array(gt_kpts).reshape(3, -1)
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    assert(np.count_nonzero(vg > 0) > 0)
    d = np.array(dt_kpts).reshape(3, -1)
    xd = d[0::3]
    yd = d[1::3]
    dx = xd - xg
    dy = yd - yg
    # e = (dx**2 + dy**2) /variances/ (area+np.spacing(1)) / 2  # 加入np.spacing()防止面积为零
    e = dx ** 2 + dy ** 2
    e = e / variances
    e = e / (area+np.spacing(1))
    e = e / 2

    e = e[vg > 0]
    return np.sum(np.exp(-e)) / e.shape[0]


def cal_OKS(points_pred, points_gt, sigmas):
    variances = (sigmas * 2) ** 2
    n_i, n_p, _ = points_pred.shape
    OKS_eval = np.zeros([n_i])

    for i in range(n_i):
        pred_temp = np.ones([3, n_p])
        gt_temp = np.ones([3, n_p])
        # print(points_gt[i, :, :])

        # 计算area 基于关键点的最小外接矩形
        # max_x = max(points_gt[i, :, 0])
        # max_y = max(points_gt[i, :, 1])
        # min_x = min(points_gt[i, :, 0])
        # min_y = min(points_gt[i, :, 1])
        # area = (max_x - min_x) * (max_y - min_y)
        # 计算area 基于boundingbox
        area = np.abs(points_gt[i, n_p, 0] - points_gt[i, n_p + 1, 0]) * \
                np.abs(points_gt[i, n_p, 1] - points_gt[i, n_p + 1, 1])

        pred_temp[0:2, :] = points_pred[i, :, :].T
        gt_temp[0:2, :] = points_gt[i, 0:n_p, :].T
        for p in range(n_p):
            if max(points_gt[i, p, :]) == 0:
                pred_temp[2, p] = 0
                gt_temp[2, p] = 0
        OKS_eval[i] = compute_kpts_oks(pred_temp, gt_temp, area, variances)
    return OKS_eval


def cal_OKS_v2(points_pred, points_gt, sigmas):
    n_i, n_p, _ = points_pred.shape
    OKS_eval = np.zeros([n_i])

    for i in range(n_i):
        pred_temp = np.ones([3, n_p])
        gt_temp = np.ones([3, n_p])
        # print(points_gt[i, :, :])

        # 计算area 基于boundingbox
        area = np.abs(points_gt[i, n_p, 0] - points_gt[i, n_p + 1, 0]) * \
                np.abs(points_gt[i, n_p, 1] - points_gt[i, n_p + 1, 1])
        # print(area)

        pred_temp[0:2, :] = points_pred[i, :, :].T
        gt_temp[0:2, :] = points_gt[i, 0:n_p, :].T
        for p in range(n_p):
            variances = (sigmas[p] * 2) ** 2
            if max(points_gt[i, p, :]) == 0:
                pred_temp[2, p] = 0
                gt_temp[2, p] = 0
        OKS_eval[i] = compute_kpts_oks(pred_temp, gt_temp, area, variances)
    return OKS_eval


def cal_AP(oks):
    t = np.linspace(0.50, 0.95, 10)
    ap_all = np.zeros(10)
    for k in range(len(oks)):
        for j in range(10):
            if oks[k] >= t[j]:
                ap_all[j] = ap_all[j] + 1
    ap50 = ap_all[0] / len(oks)
    ap75 = ap_all[5] / len(oks)
    ap = np.mean(ap_all) / len(oks)
    return ap, ap50, ap75


if __name__ == "__main__":
    sigmas = 0.05
    points_pred = np.load(path_pred)
    points_gt = np.load(path_gt)
    # print(points_pred.shape, points_gt.shape)
    OKS = cal_OKS(points_pred, points_gt, sigmas)
    # print(OKS)
    AP, AP50, AP75 = cal_AP(OKS)
    print('AP:', AP, '  AP50:', AP50, ' AP75:', AP75, 'OKS:', np.mean(OKS))