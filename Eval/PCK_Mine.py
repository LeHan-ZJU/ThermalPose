import numpy as np


def PCK_metric(pred, gt, sort1, sort2, percent):
    num_imgs, num_points, _ = pred.shape
    results = np.full((num_imgs, num_points), 0, dtype=np.float32)
    thrs = []

    for i in range(num_imgs):
        thr = cal_distance(gt[i, sort1, :], gt[i, sort2, :]) * percent
        thrs.append(thr)
        # thr = 20
        for j in range(num_points):
            distance = cal_distance(pred[i, j, :], gt[i, j, :])
            if distance <= thr:
                results[i, j] = 1

    thrs = np.array(thrs)
    print('mean:', np.mean(thrs))
    # 计算均值
    mean_points = np.mean(results, axis=0)  # 计算每个点的pck均值
    mean_all = np.mean(mean_points)         # 计算所有点的pck均值

    # 计算方差
    var_points = np.zeros([1, num_points])
    for k in range(num_points):             # 计算每个关键点的方差
        var_points[0, k] = np.var(results[:, k])

    results_reshape = results.reshape([1, num_imgs * num_points])
    var_all = np.var(results_reshape)       # 计算所有点的方差

    return mean_points, var_points, mean_all, var_all


def PCK_metric_box(pred, gt, sort1, sort2, percent):
    num_imgs, num_points, _ = pred.shape
    results = np.full((num_imgs, num_points), 0, dtype=np.float32)
    thrs = []

    for i in range(num_imgs):
        thr = find_length(gt[i, sort1, :], gt[i, sort2, :]) * percent
        thrs.append(thr)
        # thr = 20
        for j in range(num_points):
            if max(gt[i, j, :]) == 0:   # 判断标签中该点是否存在
                distance = 0
            else:
                distance = cal_distance(pred[i, j, :], gt[i, j, :])
            if distance <= thr:
                results[i, j] = 1

    thrs = np.array(thrs)
    # print(thrs)
    # print('mean:', np.mean(thrs))
    # 计算均值
    mean_points = np.mean(results, axis=0)  # 计算每个点的pck均值
    mean_all = np.mean(mean_points)         # 计算所有点的pck均值

    # 计算方差
    var_points = np.zeros([1, num_points])
    for k in range(num_points):             # 计算每个关键点的方差
        var_points[0, k] = np.var(results[:, k])

    results_reshape = results.reshape([1, num_imgs * num_points])
    var_all = np.var(results_reshape)       # 计算所有点的方差

    return mean_points, var_points, mean_all, var_all


def cal_distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[0]-p2[0])**2)


def find_length(p1, p2):
    l1 = np.abs(p1[0] - p2[0])
    l2 = np.abs(p1[1] - p2[1])
    return max(l1, l2)


def PCK_metric0(pred, gt, thr):
    # ## pred:[n, k, 2], n is the num of people, k is the number of keypoints
    # ## gt:[n, k, 2]
    # ## thr = 0.2*length_body (or thr = 0.5*length_head)
    num_imgs, num_points, _ = pred.shape
    results = np.full((num_imgs, num_points), 0, dtype=np.float32)

    for i in range(num_imgs):

        for j in range(num_points):
            distance = cal_distance(pred[i, j, :], gt[i, j, :])
            if distance <= thr:
                results[i, j] = 1

    # 计算均值
    mean_points = np.mean(results, axis=0)  # 计算每个点的pck均值
    mean_all = np.mean(mean_points)         # 计算所有点的pck均值

    return mean_points, var_points, mean_all, var_all


if __name__ == '__main__':
    # gt_file = "E:/Codes/Mine/RatPose_paper/results_location/Debug1/points_all_gt.npy"
    # pred_file = "E:/Codes/Mine/RatPose_paper/results_location/Debug1/points_all_pred.npy"
    gt_file = "E:/Codes/Mine/RatPose_paper/results_coco/Debug_ResnetADOConv_crop/2stage/points_all_gt.npy"
    pred_file = "E:/Codes/Mine/RatPose_paper/results_coco/Debug_ResnetADOConv_crop/2stage/points_all_pred.npy"

    pred_data = np.load(pred_file)
    gt_data = np.load(gt_file)
    print(pred_data.shape, gt_data.shape)
    mean_points, var_points, mean_all, var_all = PCK_metric_box(pred_data, gt_data, 4, 5, 0.3)
    print('pck_points_mean:', mean_points)
    print('pck_points_val:', var_points)
    print('pck_all_mean:', mean_all, '    pck_all_val:', var_all)