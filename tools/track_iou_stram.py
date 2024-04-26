import argparse
import os
import sys
import os.path as osp
import cv2
import numpy as np
import torch

sys.path.append('.')

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking

from tracker.tracking_utils.timer import Timer
from tracker.bot_sort import BoTSORT

from tools.models.stram import build as build_model_st
from tools.utils.tool import load_model

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Global
trackerTimer = Timer()
timer = Timer()


def make_parser():
    parser = argparse.ArgumentParser("BoT-SORT Tracks For Evaluation!")

    parser.add_argument("path", help="path to dataset under evaluation, currently only support MOT17 and MOT20.")
    parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
    parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
    parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

    # Detector
    parser.add_argument("--device", default="gpu", type=str, help="device to run our model, can either be cpu or gpu")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.75, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6, help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')

    # CMC
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow | orb | ecc | none")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')

    parser.add_argument('--merger_dropout', type=float, default=0.1)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument("--pretrained", type=str, default="/media/disk1/yjt/lzl/RATracker/pretrained/mot17_half_stram.pth", help=' ')
    parser.add_argument('--mark_ration', type=float, default=0.2)
    parser.add_argument('--weight_t', type=float, default=0.1)
    parser.add_argument('--weight_s', type=float, default=0.1)
    parser.add_argument('--weight_st', type=float, default=0.5)
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            device=torch.device("cpu"),
            fp16=False
    ):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        if img is None:
            raise ValueError("Empty image: ", img_info["file_name"])

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img)
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)

        return outputs, img_info


def image_track(predictor, vis_folder, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    if args.ablation:
        files = files[len(files) // 2 + 1:]

    num_frames = len(files)

    # Tracker
    tracker = BoTSORT(args, frame_rate=args.fps)

    results = []

    model = build_model_st(args)
    model.to(args.device)
    model = load_model(model, args.pretrained)
    model.eval()
    
    for frame_id, img_path in enumerate(files, 1):

        # Detect objects
        # outputs, img_info = predictor.inference(img_path, timer)
        # scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

        seq_name = img_path.split("/")[-3]
        img_name = img_path.split("/")[-1].split(".")[0]

        img_info = {}
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width

        with open(f"/media/disk1/yjt/lzl/motion_pred/save_mot17_val_detections/{seq_name}/{img_name}.txt", "r") as file:
            lines = file.readlines()
        # 提取每行的值，并存储为NumPy数组
        data = []
        for line in lines:
            values = line.strip().split(',')
            row = [float(x) for x in values]
            data.append(row)
        # 转换为NumPy数组
        outputs = np.array(data)

        scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))

        # if outputs[0] is not None:
        #     outputs = outputs[0].cpu().numpy()
        if outputs is not None:
            detections = outputs[:, :7]
            detections[:, :4] /= scale

            trackerTimer.tic()
            online_targets = tracker.update_iou_stram(detections, None, model)
            trackerTimer.toc()

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            # online_im = plot_tracking(
            #     img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            # )
        else:
            timer.toc()
            # online_im = img_info['raw_img']

        # if args.save_frames:
        #     save_folder = osp.join(vis_folder, args.name)
        #     os.makedirs(save_folder, exist_ok=True)
        #     cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {}/{} ({:.2f} fps)'.format(frame_id, num_frames, 1. / max(1e-5, timer.average_time)))

    res_file = osp.join(vis_folder, args.name + ".txt")

    with open(res_file, 'w') as f:
        f.writelines(results)
    logger.info(f"save results to {res_file}")

    return tracker.total_iou_i, tracker.count_i

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_results")
    os.makedirs(vis_folder, exist_ok=True)

    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if args.ckpt is None:
        ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
    else:
        ckpt_file = args.ckpt
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    predictor = Predictor(model, exp, args.device, args.fp16)

    return image_track(predictor, "/media/disk1/yjt/lzl/motion_pred/mot_metrix/pred_kf", args)


if __name__ == "__main__":
    args = make_parser().parse_args()

    data_path = args.path
    fp16 = args.fp16
    device = args.device

    if args.benchmark == 'MOT20':
        train_seqs = [1, 2, 3, 5]
        test_seqs = [4, 6, 7, 8]
        seqs_ext = ['']
        MOT = 20
    elif args.benchmark == 'MOT17':
        train_seqs = [2, 4, 5, 9, 10, 11, 13]
        test_seqs = [1, 3, 6, 7, 8, 12, 14]
        seqs_ext = ['FRCNN']
        MOT = 17
    else:
        raise ValueError("Error: Unsupported benchmark:" + args.benchmark)

    ablation = False
    if args.split_to_eval == 'train':
        seqs = train_seqs
    elif args.split_to_eval == 'val':
        seqs = train_seqs
        ablation = True
    elif args.split_to_eval == 'test':
        seqs = test_seqs
    else:
        raise ValueError("Error: Unsupported split to evaluate:" + args.split_to_eval)

    mainTimer = Timer()
    mainTimer.tic()

    # 统计已经匹配上的目标，预测的位置和检测的位置间的IoU
    total_iou = 0
    count = 0
    for ext in seqs_ext:
        for i in seqs:
            if i < 10:
                seq = 'MOT' + str(MOT) + '-0' + str(i)
            else:
                seq = 'MOT' + str(MOT) + '-' + str(i)

            if ext != '':
                seq += '-' + ext

            args.name = seq

            args.ablation = ablation
            args.mot20 = MOT == 20
            args.fps = 30
            args.device = device
            args.fp16 = fp16
            args.batch_size = 1
            args.trt = False

            split = 'train' if i in train_seqs else 'test'
            args.path = data_path + '/' + split + '/' + seq + '/' + 'img1'

            if args.default_parameters:

                if MOT == 20:  # MOT20
                    args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_mot20_ch.py'
                    args.ckpt = r'./pretrained/bytetrack_x_mot20.tar'
                    args.match_thresh = 0.7
                else:  # MOT17
                    if ablation:
                        args.exp_file = r'./yolox/exps/example/mot/yolox_x_ablation.py'
                        args.ckpt = r'./pretrained/bytetrack_ablation.pth.tar'
                    else:
                        args.exp_file = r'./yolox/exps/example/mot/yolox_x_mix_det.py'
                        args.ckpt = r'./pretrained/bytetrack_x_mot17.pth.tar'

                exp = get_exp(args.exp_file, args.name)

            else:
                exp = get_exp(args.exp_file, args.name)

            exp.test_conf = max(0.001, args.track_low_thresh - 0.01)
            total_iou_i, count_i = main(exp, args)
            total_iou += total_iou_i
            count += count_i

    # 统计性能
    # 评估跟踪性能
    import os
    from evaluation import Evaluator
    import motmetrics as mm
    def eval_seq(seq_num, gt_path, pred_path):
        result_filename = os.path.join(pred_path, f'{seq_num}.txt') # predict result
        evaluator = Evaluator(gt_path, seq_num)
        accs = evaluator.eval_file(result_filename)
        return accs
    accs = []
    seqs = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN", "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
    for seq_num in seqs:
        accs.append(eval_seq(seq_num, "/media/disk1/yjt/lzl/motion_pred/mot_metrix/gt", "/media/disk1/yjt/lzl/motion_pred/mot_metrix/pred_kf"))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        # formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)

    print(f"统计已经匹配上的匹配对,预测位置和检测位置间的iou:{total_iou/count}")

    # fw = open(f"/media/disk1/lzl/TransTrack/Temporal_Spatial.txt", mode="a+")
    # # fw.write(f"w_model_iou_1_S:{args.w_model_iou_1_S} w_model_iou_1_T:{args.w_model_iou_1_T}  w_model_iou_1_S_T:{args.w_model_iou_1_S_T} thresh_1:{args.thresh_1}\n")
    # fw.write(f"max_age: {args.max_age}\n")
    # fw.write(strsummary)
    # fw.write("\n")
    # fw.close()

    mainTimer.toc()
    print("TOTAL TIME END-to-END (with loading networks and images): ", mainTimer.total_time)
    print("TOTAL TIME (Detector + Tracker): " + str(timer.total_time) + ", FPS: " + str(1.0 /timer.average_time))
    print("TOTAL TIME (Tracker only): " + str(trackerTimer.total_time) + ", FPS: " + str(1.0 / trackerTimer.average_time))
