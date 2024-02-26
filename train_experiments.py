import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist

from Models.eval_pose import eval_net
from hrnet.hrnet_Rat import HRNet
from hrnet.HourGlassNet import Bottleneck, HourglassNet
from hrnet.DeepLabCut import DeepLabCut, DeepLabCut2
from Models.RatNetAttention_DOConv import Net_ResnetAttention_DOConv

from utils.dataset_csv import DatasetPoseCSV, DatasetStage2_single, DatasetPoseCSV_pad
from torch.utils.data import DataLoader, random_split

torch.distributed.init_process_group(backend="nccl")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
# Rat

# 206服务器 rat-10points
# dir_img = '/mnt/8T1/hanle/RatPose/all/'
# dir_label = '/mnt/8T1/hanle/RatPose/label/RatPoseLabels/VerticalTilt_trainval.csv.csv'
# dir_label = '/mnt/8T1/hanle/RatPose/label/RatPoseLabels/RatPoseAll_indoor_trainval.csv'
# dir_checkpoint = './TrainedModels_UDAPose/Debug_HRNet48_Vertical/'
# dir_checkpoint = './TrainedModels_UDAPose/Debug_HRNet48_Indoor/'

# 200
dir_img = '/8T/hanle/Datasets/RatPose/all/'
dir_label = '/8T/hanle/Datasets/RatPose/label/RatPoseLabels/Data_v34_crop_trainval.csv'
# dir_checkpoint = './TrainedModels_location_v34/8points/train1/'
dir_checkpoint = '/8T/hanle/Models/Ratpose/TrainedModels_exps_v34_revise/DeepLabCut2_crop/'
num_points = 10

# relation = [[1, 4], [1, 5], [2, 3], [2, 5], [3, 6], [4, 6]]
# num_points = 10

resize_w = 640
resize_h = 480  # 128 64 32 16 8
extract_list = ["layer4"]

isExists = os.path.exists(dir_checkpoint)
if not isExists and dist.get_rank() == 0:  # 判断结果
    os.makedirs(dir_checkpoint)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target heatmaps',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-a', '--augment', dest='augment', type=float, default=0,
                        help='Whether to add data to augment')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=200,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=15,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str,  default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=4,
                        help='the ratio between img and GT')
    parser.add_argument('-w', '--weight', dest='weight', type=float, default=0,
                        help='the weight of similar loss')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-p', '--path_backbone', dest='path_backbone', type=str,  # default=False,
                        # default='/mnt/8T1/hanle/BackBone/hrnet_w48-8ef0771d.pth',
                        # default='/8T/hanle/Backbone/pytorch/HRNet/hrnet_w48-8ef0771d.pth',
                        default='/8T/hanle/Backbone/pytorch/Resnet/resnet50-19c8e357.pth',  # resnet50
                        help='the path of backbone')
    parser.add_argument("--local_rank", type=int, default=1,
                        help="number of cpu threads to use during batch generation")
    return parser.parse_args()


def train_net(net,
              device,
              epochs=30,
              batch_size=4,
              lr=0.001,
              weight=0.01,
              val_percent=0.1,
              save_cp=True,
              img_scale=1,
              augment=0):   # scale是输入与输出的边长比

    # dataset = DatasetPoseRat_transed(resize_w, resize_h, dir_img, dir_label, img_scale)
    # dataset = DatasetPoseCSV(resize_w, resize_h, dir_img, dir_label, img_scale, num_points, augment)
    dataset = DatasetPoseCSV_pad(resize_w, resize_h, dir_img, dir_label, img_scale, num_points, augment)
    # dataset = DatasetStage2_single(resize_w, resize_h, dir_img, dir_label, num_points, augment, img_scale, angle=180)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    # 并行
    train_sampler = torch.utils.data.distributed.DistributedSampler(train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=40,
                                               pin_memory=True, sampler=train_sampler)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=40,
                                             pin_memory=True, sampler=val_sampler)
    # train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  # if net.n_classes > 1 else 'max', patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)   # 每隔50个epoch降一次学习率（*0.1）

    criterion = nn.MSELoss()
    # criterion2 = CosSimlarLoss(int(resize_w / img_scale), int(resize_h / img_scale), relation)
    # criterion2 = SimlarLoss(int(resize_w / img_scale), int(resize_h / img_scale))
    # criterion3 = PointsLoss(int(resize_w / img_scale), int(resize_h / img_scale))

    loss_all = np.zeros([4, epochs])   # 4*epochs大小的矩阵，第一行存epochs序号数，第二行为每个epoch的总loss，第三行为mse loss，第四行为similarloss
    for epoch in range(epochs):
    # for epoch in range(50, epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_heatmaps = batch['heatmap']

                imgs = imgs.to(device=device, dtype=torch.float32)
                heatmap_type = torch.float32  # if net.n_classes == 1 else torch.long
                true_heatmaps = true_heatmaps.to(device=device, dtype=heatmap_type)

                heatmaps_pred = net(imgs)
                # print('true_heatmaps:', true_heatmaps.shape)
                # print('pred_heatmaps:', heatmaps_pred.shape)

                loss_mse = criterion(heatmaps_pred, true_heatmaps)
                # loss_similar = weight * criterion2(heatmaps_pred, true_heatmaps)
                # loss_similar = weight * criterion3(heatmaps_pred, true_heatmaps)   # PointLoss
                # print('loss:', loss_mse, loss_similar)
                loss = loss_mse  # + loss_similar

                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})  # , 'loss_mse (batch)': loss_mse.item(),
                                    # 'loss_similar (batch)': loss_similar.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    # val_score=0

                    val_score = eval_net(net, val_loader, device)
                    # scheduler.step(val_score)

                    logging.info('Validation cross entropy: {}'.format(val_score))

        scheduler.step()
        print('epoch:', epoch + 1, ' loss:', loss.item())
        loss_all[0, epoch] = epoch + 1
        loss_all[1, epoch] = loss.item()
        # loss_all[2, epoch] = loss_mse.item()
        # loss_all[3, epoch] = loss_similar.item()

        if dist.get_rank() == 0 and (epoch + 1) % 10 == 0:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.module.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        
        # 绘制loss曲线
        plt.plot(loss_all[0, :epoch], loss_all[1, :epoch], color='b')  # 绘制loss曲线
        plt.xlabel("epochs", fontsize=12)
        plt.ylabel("loss", fontsize=12)
        plt.savefig(dir_checkpoint + 'loss.jpg')
        save_path_loss = dir_checkpoint + 'loss.npy'
        np.save(save_path_loss, loss_all)
        plt.close()

    return loss_all


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    # 并行
    local_rank = torch.distributed.get_rank()
    print('local_rank:', local_rank)
    torch.cuda.set_device(local_rank)
    global device
    device = torch.device("cuda", local_rank)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    logging.info(f'Using device {device}')
    print(args.path_backbone)
    print('input_size:', resize_w, resize_h, ';  Augment:', args.augment)
    print('lr:', args.lr, ';  batch_size:', args.batchsize, ';  the weight of similar loss:', args.weight)
    print('trainset:', dir_label)

    # 构建网络
    # net = HRNet(c=32, n_channels=3, nof_joints=num_points, bn_momentum=0.1)
    # net = HourglassNet(Bottleneck, num_stacks=4, num_blocks=4, num_classes=num_points)
    net = DeepLabCut2(nJoints=num_points, extract_list=extract_list, model_path=args.path_backbone, device=device, train=True)
    # net = Net_ResnetAttention_DOConv(args.path_backbone, extract_list, device, train=True, n_channels=3, nof_joints=num_points)
    # nJoints, extract_list, model_path, device, train=1)

    # stat(net, input_size=(3, 320, 256))

    if args.load:
        print(args.load)
        net.load_state_dict(
            torch.load(args.load, map_location=device), strict=False   # strict，该参数默认是True，表示预训练模型的层和自己定义的网络结构层严格对应相等（比如层名和维度）
        )
        logging.info(f'Model loaded from {args.load}')
        print('Pretrained weights have been loaded!')
    else:
        print('No pretrained models have been loaded except the backbone!')

    net.to(device=device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank,
                                                    find_unused_parameters=True)

    try:
        loss_all = train_net(net=net,
                             device=device,
                             epochs=args.epochs,
                             batch_size=args.batchsize,
                             lr=args.lr,
                             weight=args.weight,
                             val_percent=args.val / 100,
                             img_scale=args.scale,
                             augment=args.augment)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

    # 绘制loss曲线
    plt.plot(loss_all[0, :], loss_all[1, :], color='b')  # 绘制loss曲线
    plt.xlabel("epochs", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.savefig(dir_checkpoint + 'loss.jpg')
    save_path_loss = dir_checkpoint + 'loss.npy'
    np.save(save_path_loss, loss_all)
