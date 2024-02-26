import re
import csv
import random
import numpy as np


def keypoint_pck_accuracy(pred, gt, mask, thr, normalize):
    """Calculate the pose accuracy of PCK for each individual keypoint and the
    averaged accuracy across all keypoints for coordinates.

    Note:
        PCK metric measures accuracy of the localization of the body joints.
        The distances between predicted positions and the ground-truth ones
        are typically normalized by the bounding box size.
        The threshold (thr) of the normalized distance is commonly set
        as 0.05, 0.1 or 0.2 etc.

        batch_size: N
        num_keypoints: K

    Args:
        pred (np.ndarray[N, K, 2]): Predicted keypoint location.
        gt (np.ndarray[N, K, 2]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        thr (float): Threshold of PCK calculation.
        normalize (np.ndarray[N, 2]): Normalization factor for H&W.

    Returns:
        tuple: A tuple containing keypoint accuracy.

        - acc (np.ndarray[K]): Accuracy of each keypoint.
        - avg_acc (float): Averaged accuracy across all keypoints.
        - cnt (int): Number of valid keypoints.
    """
    distances = _calc_distances(pred, gt, mask, normalize)

    acc = np.array([_distance_acc(d, thr) for d in distances])
    valid_acc = acc[acc >= 0]
    cnt = len(valid_acc)
    avg_acc = valid_acc.mean() if cnt > 0 else 0
    var_acc = valid_acc.var() if cnt > 0 else 0
    return acc, avg_acc, var_acc, cnt


def _calc_distances(preds, targets, mask, normalize):
    """Calculate the normalized distances between preds and target.

    Note:
        batch_size: N
        num_keypoints: K
        dimension of keypoints: D (normally, D=2 or D=3)

    Args:
        preds (np.ndarray[N, K, D]): Predicted keypoint location.
        targets (np.ndarray[N, K, D]): Groundtruth keypoint location.
        mask (np.ndarray[N, K]): Visibility of the target. False for invisible
            joints, and True for visible. Invisible joints will be ignored for
            accuracy calculation.
        normalize (np.ndarray[N, D]): Typical value is heatmap_size

    Returns:
        np.ndarray[K, N]: The normalized distances.
          If target keypoints are missing, the distance is -1.
    """
    N, K, _ = preds.shape
    distances = np.full((N, K), -1, dtype=np.float32)
    print(distances, mask)
    # handle invalid values
    normalize[np.where(normalize <= 0)] = 1e6
    # distances[mask] = np.linalg.norm(
    #     ((preds - targets) / normalize[:, None, :])[mask], axis=-1)
    distances = np.linalg.norm(
        ((preds - targets) / normalize[:, None, :]), axis=-1)
    return distances.T


def _distance_acc(distances, thr=0.5):
    """Return the percentage below the distance threshold, while ignoring
    distances values with -1.

    Note:
        batch_size: N
    Args:
        distances (np.ndarray[N, ]): The normalized distances.
        thr (float): Threshold of the distances.

    Returns:
        float: Percentage of distances below the threshold.
          If all target keypoints are missing, return -1.
    """
    distance_valid = distances != -1
    num_distance_valid = distance_valid.sum()
    if num_distance_valid > 0:
        return (distances[distance_valid] < thr).sum() / num_distance_valid
    return -1


if __name__ == '__main__':
    num_points = 10
    label_dir = "G:/Data/RatPose/RatData2/labels/batch1/RatPose2_batch1_new.csv"

    # 读t标签
    with open(label_dir, 'r', encoding='gb18030') as f:
        reader = csv.reader(f)
        name_list = list(reader)
        print('len', len(name_list))
        labels = []  # 存放每个图片第一行标签在labels中的行序号，长度为图像数
        img_sort = 1
        trainset_num = len(name_list)
        while img_sort < trainset_num:
            labels.append(name_list[img_sort])

            img_sort = img_sort + 1

    Json_results = []
    # 获取box
    for i in range(img_sort - 1):
        # print(labels[i])
        image_id = labels[i][0]
        people_num = int(labels[i][1])  # 当前图像中的人数
        # # img_file = imgs_dir + labels[idx][0]  # 获取图像名，读图
        # # print(img_file)
        # index = labels[idx][0].index('.')
        searchContext = "_"
        #
        points_all_gt = np.zeros([1, num_points, 2])  # 存放当前图中的所有人的所有关键点
        points_all_pred = np.zeros([1, num_points, 2])

        for n in range(num_points):
            numList = [m.start() for m in re.finditer(searchContext, labels[i][n + 3])]
            point = [float(labels[i][n + 3][0:numList[0]]),
                     int(labels[i][n + 3][numList[0] + 1:numList[1]])]
            points_all_gt[0, n, :] = point
            points_all_pred[0, n, 0] = point[0] + random.uniform(0, 2)   # 叠加1-10之间的随机数
            points_all_pred[0, n, 1] = point[1] + random.uniform(0, 5)  # 叠加1-10之间的随机数

        # print(image_id, points_all)

    mask = np.ones([1, num_points])
    thr = 0.5
    normalize = np.full([num_points, 2],  5, dtype=np.float32)

    acc, avg_acc, var_acc, cnt = keypoint_pck_accuracy(points_all_pred, points_all_gt, mask, thr, normalize)
    print('acc:', acc)
    print('avg_acc:', avg_acc)
    print('cnt:', cnt)