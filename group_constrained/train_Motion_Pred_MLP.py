import argparse
import datetime
import time
from pathlib import Path
import torch
import misc as utils
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
base_path = "/media/disk1/yjt/lzl/motion_pred"

import sys
sys.path.append(base_path)
from group_constrained.Motion_Pred_MLP import build as build_motion_pred_model
import numpy as np
import random
import os
import torch.nn.functional as F
from group_constrained.utils import convert_boxes, calculate_ciou
from collections import deque

def get_args_parser():
    parser = argparse.ArgumentParser('train motion pred', add_help=False)
    parser.add_argument('--model_motion_pred_lr', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--baseroot', default='/media/disk1/yjt/lzl/motion_pred',
                        help='path where to save, empty for no saving')
    parser.add_argument('--output_dir', default='/media/disk1/yjt/lzl/motion_pred/group_constrained/output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume_model_motion_pred', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--motion_pred_layer', default='MP', type=str,
                        help="")
    parser.add_argument('--merger_dropout', type=float, default=0.1)
    # 使用过去k帧的偏移预测下一帧的偏移
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--d_model', default=128, type=int)

    parser.add_argument('--seed', default=8, type=int)
    return parser

class MotionPredDatasetTrain(Dataset):
    def __init__(self, data_all): # data_all是一个lsit, 存放所有数据， 里面元素为 (x,y)
        self.data_all = data_all

    def __getitem__(self, idx):
        return self.data_all[idx]

    def __len__(self):
        return len(self.data_all)

class CIoULoss(torch.nn.Module):
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, pred_boxes, target_boxes):
        ciou = calculate_ciou(pred_boxes, target_boxes)
        ciou_loss = 1 - ciou.mean()

        return ciou_loss

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(args)
    # 设置随机化种子
    # seed_value = 8   # 设定随机数种子
    seed_value = args.seed   # 设定随机数种子
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed_value)     # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)

    hidden_dim = args.hidden_dim
    d_model = args.d_model

    model_mp = build_motion_pred_model(args, args.motion_pred_layer, d_model, hidden_dim, args.k)
    model_mp.to(device)
    model_without_ddp_mp = model_mp

    if args.distributed:
        model_mp = torch.nn.parallel.DistributedDataParallel(model_mp, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp_mp = model_mp.module

    if args.sgd:
        optimizer_mp = torch.optim.SGD(model_mp.parameters(), lr=args.model_motion_pred_lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        print(f"args.weight_decay:{args.weight_decay}")
        optimizer_mp = torch.optim.AdamW(model_mp.parameters(), lr=args.model_motion_pred_lr, weight_decay=args.weight_decay)
    lr_scheduler_mp = torch.optim.lr_scheduler.StepLR(optimizer_mp, args.lr_drop)

    if args.resume_model_motion_pred: # 接着训练
        checkpoint_mp = torch.load(args.resume_model_motion_pred, map_location='cpu')
        missing_keys_mp, unexpected_keys_mp = model_without_ddp_mp.load_state_dict(checkpoint_mp['model'], strict=False)  # save的是model_without_ddp_ff.state_dict, 这样load权重后，model_ff也会加载权重
        unexpected_keys_mp = [k for k in unexpected_keys_mp if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys_mp) > 0:
            print('Missing Keys: {}'.format(missing_keys_mp))
        if len(unexpected_keys_mp) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys_mp))

        # load之前训练的model接着训练
        import copy
        p_groups = copy.deepcopy(optimizer_mp.param_groups)
        optimizer_mp.load_state_dict(checkpoint_mp['optimizer'])
        for pg, pg_old in zip(optimizer_mp.param_groups, p_groups):
            pg['lr'] = pg_old['lr']
            pg['initial_lr'] = pg_old['initial_lr']
        lr_scheduler_mp.load_state_dict(checkpoint_mp['lr_scheduler'])
        args.override_resumed_lr_drop = True
        if args.override_resumed_lr_drop:
            print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
            lr_scheduler_mp.step_size = args.lr_drop
            lr_scheduler_mp.base_lrs = list(map(lambda group: group['initial_lr'], optimizer_mp.param_groups))
        lr_scheduler_mp.step(lr_scheduler_mp.last_epoch)
        args.start_epoch = checkpoint_mp['epoch'] + 1

    print("Start training")
    start_time = time.time()

    writer_loss = SummaryWriter(f'{args.output_dir}/loss_log')
    train_seqs = ["MOT17-02-FRCNN",
                      "MOT17-04-FRCNN",
                      "MOT17-05-FRCNN",
                      "MOT17-09-FRCNN",
                      "MOT17-10-FRCNN",
                      "MOT17-11-FRCNN",
                      "MOT17-13-FRCNN"]
    obts_person = {
        "MOT17-02-FRCNN":{}, # 存 轨迹id:[]   轨迹id对应的框，按帧序列排好
        "MOT17-04-FRCNN":{},
        "MOT17-05-FRCNN":{},
        "MOT17-09-FRCNN":{},
        "MOT17-10-FRCNN":{},
        "MOT17-11-FRCNN":{},
        "MOT17-13-FRCNN":{}
    }

    # {seq : { frame_id : { track_id : [center_x, center_y, w, h] } } }
    for train_seq in train_seqs:
        f_r = open(f"{args.baseroot}/datasets/MOT17/train/{train_seq}/gt/gt_train_half.txt", mode="r")
        for line in f_r:
            # gt中每条轨迹的框 都是按帧 依次排列好的， 依次append就行了
            lines = line.split(",")
            frame_id = int(lines[0])
            track_id = int(lines[1])
            tlx = float(lines[2])
            tly = float(lines[3])
            w = float(lines[4])
            h = float(lines[5])
            center_x = tlx + 0.5 * w
            center_y = tly + 0.5 * h
            if frame_id not in obts_person[train_seq].keys():
                obts_person[train_seq][frame_id] = {}
            obts_person[train_seq][frame_id][track_id] = [center_x, center_y, w, h]
    # 将轨迹封装成 datasets: x y 的形式，
    # 使用 t帧的center_x, center_y, w, h, 以及t-1到t的(center_x, center_y, w, h)偏移作为x，
    # t到t+1的(center_x, center_y, w, h)偏移, 以及t+1帧的center_x, center_y, w, h, 作为y
    # 测试时，第一帧就不预测，直接拿当前帧的检测去匹配，     或者训练、测试把t=0时 第一帧 的偏移设置为(0,0,0,0)
    # 对于第一帧、或者轨迹的开始帧， 前一帧到当前帧的偏移全设置为0，   最后一帧，计算目标的预测值后 没有真实标注了，对轨迹的最后一帧的目标不进行训练
    data_all = []
    for train_seq_name, train_seq_data in obts_person.items():
        track_id_data = {}  # 存的是每条轨迹的连续帧的偏移量，例如i->[], 的第一个元素表示轨迹i的第一帧到第二帧的偏移量
        for frame_id, frame_data in train_seq_data.items():
            data_x = [] # M x 8:  M条轨迹
            data_y = []
            if frame_id == 1:
                for track_id, track_data in frame_data.items():
                    track_data_now_frame = track_data.copy()
                    # 初始帧 上一帧到初始帧的偏移设置为0
                    track_id_data[track_id] = deque([[0, 0, 0, 0]]*args.k, maxlen=args.k)
                    track_data.extend(track_id_data[track_id].copy())
                    data_x.append(track_data.copy())
                    if frame_id+1 not in train_seq_data.keys() or track_id not in train_seq_data[frame_id+1]:
                        # 目标在下一帧不存在，没有真实标注了，对轨迹的最后一帧的目标不进行训练
                        data_y.append([False]*8)
                    else:
                        # 该轨迹在下一帧的位置
                        track_data_next_frame = train_seq_data[frame_id+1][track_id]
                        # 当前帧到下一帧的偏移
                        deta = [a-b for a,b in zip(track_data_now_frame, track_data_next_frame)]
                        deta.extend(track_data_next_frame)
                        data_y.append(deta.copy())
            else:
                for track_id, track_data in frame_data.items():
                    track_data_now_frame = track_data.copy()
                    if frame_id-1 not in train_seq_data.keys() or track_id not in train_seq_data[frame_id-1]:
                        # 目标的前一帧不存在，说明这是轨迹的初始帧
                        track_id_data[track_id] = deque([[0, 0, 0, 0]]*args.k, maxlen=args.k)
                        track_data.extend(track_id_data[track_id].copy())
                        data_x.append(track_data.copy())
                    else:
                        # 该轨迹在上一帧的位置
                        track_data_pre_frame = train_seq_data[frame_id-1][track_id]
                        # 上一帧到当前帧的偏移
                        deta = [a-b for a,b in zip(track_data_pre_frame, track_data_now_frame)]
                        track_id_data[track_id].append(deta.copy())
                        track_data.extend(track_id_data[track_id].copy())
                        data_x.append(track_data.copy())
                    if frame_id+1 not in train_seq_data.keys() or track_id not in train_seq_data[frame_id+1]:
                        # 目标在下一帧不存在，没有真实标注了，对轨迹的最后一帧的目标不进行训练
                        data_y.append([False]*8)
                    else:
                        # 该轨迹在下一帧的位置
                        track_data_next_frame = train_seq_data[frame_id+1][track_id]
                        # 当前帧到下一帧的偏移
                        deta = [a-b for a,b in zip(track_data_now_frame, track_data_next_frame)]
                        deta.extend(track_data_next_frame)
                        data_y.append(deta.copy())
            data_all.append((data_x, data_y))
    # 现在 data_x 缓存的是过去轨迹的k个偏移量，最多k个偏移，如果过去没有k帧，则只缓存过去能有的全部偏移
    # data_all 总共有两千多帧， 每帧里目标数量不一样， 不方便使用多batchsize训练， batchsize设置为1
    # motion_pred_dataset = MotionPredDatasetTrain(data_all)
    #创建一个dataloader,设置批大小为batch_size，每一个epoch重新洗牌，不进行多进程读取机制，不舍弃不能被整除的批次
    # motion_pred_dataloader = DataLoader(dataset=motion_pred_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

    losses_min = float("inf")
    ciou_loss = CIoULoss()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_loss = 0
        if epoch != 0 and epoch % args.lr_drop == 0:
            optimizer_mp.param_groups[0]["lr"] *= 0.1

        # random.shuffle(data_all)
        for data_x, data_y in data_all:
            # 当前目标的 center_x, center_y, w, h            Nx4
            data_x_1 = []
            for item in data_x:
                data_x_1.append(item[:4])
            # N个目标过去K帧的偏移        Nxkx4
            data_x_2 = []
            for item in data_x:
                data_x_2.append(item[4:])
            data_x_1 = torch.tensor(data_x_1, dtype=torch.float32).unsqueeze(0).to(device)   # 1xNx4
            data_x_2 = torch.tensor(data_x_2, dtype=torch.float32).unsqueeze(0).to(device)   # 1xNxkx4
            # 预测的是当前帧的目标到下一帧的偏移量, deta_pred维度：目标数, 4 (center_x, center_y, w, h)偏移
            P_offs = model_mp(data_x_1, data_x_2)
            # 根据P_offs 计算两个loss:   iou loss 和 偏移量的mse/mae loss ?
            # (将iou loss 替换为giou\ciou loss, 以及添加mse/mae loss倒是不急， 先将iou loss 跑通作为baseline， 在此基础上添加多尺度卷积、时序偏移的融合，验证有效性)

            # False表示目标在下一帧消失了，不对其计算loss，先筛选掉data_y中的False， 把data_y中false对应索引的 P_offs 删除，然后一一计算loss
            data_y = torch.tensor(data_y, dtype=torch.float32).to(device)
            row_has_false = torch.all(data_y == False, dim=1)
            data_y = data_y[~row_has_false]
            coordinate_at_frame_t1 = data_y[:, 4:]
            target_boxes = convert_boxes(coordinate_at_frame_t1)
            P_offs = P_offs.squeeze(0)[~row_has_false]

            # ciou loss
            coordinate_at_frame_t  = data_x_1[:, :, 0:4].squeeze(0)[~row_has_false] # 目标在第t帧的位置
            pred_coordinate_at_frame_t1 = coordinate_at_frame_t + P_offs  # 目标在第t帧的位置 加上预测的偏置，得到预测的在下一帧的位置，
            # 数据格式是(center_x, center_y, w, h)  转变为 (x_min, y_min, x_max, y_max)
            pred_boxes = convert_boxes(pred_coordinate_at_frame_t1)
            if pred_boxes.shape[0] > 0:
                # pred_boxes.shape[0] == 0时，表示当前帧时最后一帧，下一帧全部为False,  pred_boxes/target_boxes 维度都为(0,4) 不能计算loss, 否则loss出现NAN
                loss = ciou_loss(pred_boxes, target_boxes)
                epoch_loss += loss.item()
                optimizer_mp.zero_grad()
                loss.backward()
                optimizer_mp.step()

        f_1 = open(f"{args.output_dir}/train.txt", mode="a+")
        f_1.write(f"epoch:{epoch},   轨迹预测的 ciou loss :{epoch_loss}\n")
        f_1.close()

        if epoch_loss  < losses_min:
            losses_min = epoch_loss

            filename=f'{args.output_dir}/model_best.pth'
            utils.save_on_master({
                    'model': model_without_ddp_mp.state_dict(),
                    'optimizer': optimizer_mp.state_dict(),
                    'lr_scheduler': lr_scheduler_mp.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, filename)
        writer_loss.add_scalar('epoch_loss', epoch_loss , epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('train motion pred', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)