# nohup python train_resnet3Sim_linux.py  >> /8T/hanle/Models/Ratpose/TrainedModels_exps/BoneLoss/Indoor_Bonev0_w0001/train.log 2>&1 &
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# nohup python -m torch.distributed.launch --nproc_per_node=2 train_resnet3Sim_parallel.py >> /8T/hanle/Models/Ratpose/TrainedModels_exps_zhanghan/debug_Ours/train.log 2>&1 &
nohup python train_experiments_linux.py >> /8T/hanle/Models/ThermalPose/TrainedModels/ThermalImgs_Color/debug_hr32_pad/train.log 2>&1 &