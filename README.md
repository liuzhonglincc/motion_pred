下载MOT17
```
<datasets>
      │
      ├── MOT17
      │      ├── train
      │      └── test    
```

[下载模型和数据](https://pan.baidu.com/s/1bpvctUhn2X2LdwMaRBwIqw?pwd=7712)

将dancetrack_yolo_det、save_mot17_val_detections、pretrained放在根目录motion_pred

将ByteTrack_pretrained/bytetrack_model.pth放在motion_pred/ByteTrack/pretrained目录



cd motion_pred/group_constrained

训练 group：bash train_Motion_Pred_Group.sh
训练 mlp: bash train_Motion_Pred_MLP.sh

table 1
```
python3 tools/track_iou_no_motion.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --track_high_thresh 0.7
python3 tools/track_iou.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --track_high_thresh 0.7
python3 tools/track_iou_pred_model.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --seed 153 --track_high_thresh 0.7
python3 tools/track_iou_pred_model_mlp.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --seed 153 --track_high_thresh 0.7
```
table 3
```
python3 tools/track_iou_no_motion_SORT.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse
python3 tools/track_iou_SORT.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse
python3 tools/track_iou_pred_model_SORT.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --seed 153 --match_thresh 0.69
python3 tools/track_iou_pred_model_mlp_SORT.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --seed 153
python3 tools/track_iou_no_motion.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse
python3 tools/track_iou.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --track_high_thresh 0.61
python3 tools/track_iou_pred_model.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --seed 153 --track_high_thresh 0.57
python3 tools/track_iou_pred_model_mlp.py datasets/MOT17 --default-parameters --benchmark "MOT17" --eval "val" --fp16 --fuse --seed 153
```
table 2
```
KF:
python ByteTrack/tools/track.py -f ByteTrack/exps/example/dancetrack/yolox_x.py -c ByteTrack/pretrained/bytetrack_model.pth.tar -b 1 -d 1 --fp16 --fuse
no motion:
python ByteTrack/tools/track_no_motion.py -f ByteTrack/exps/example/dancetrack/yolox_x.py -c ByteTrack/pretrained/bytetrack_model.pth.tar -b 1 -d 1 --fp16 --fuse
our_pred_model
python ByteTrack/tools/track_our_pred_model.py -f ByteTrack/exps/example/dancetrack/yolox_x.py -c ByteTrack/pretrained/bytetrack_model.pth.tar -b 1 -d 1 --fp16 --fuse
mlp
python ByteTrack/tools/track_our_pred_model_mlp.py -f ByteTrack/exps/example/dancetrack/yolox_x.py -c ByteTrack/pretrained/bytetrack_model.pth.tar -b 1 -d 1 --fp16 --fuse
```
然后运行
```
python3 TrackEval/scripts/run_mot_challenge.py --SPLIT_TO_EVAL val  --METRICS HOTA CLEAR Identity  --GT_FOLDER dancetrack/val --SEQMAP_FILE dancetrack/val_seqmap.txt --SKIP_SPLIT_FOL True   --TRACKERS_TO_EVAL '' --TRACKER_SUB_FOLDER ''  --USE_PARALLEL True --NUM_PARALLEL_CORES 8 --PLOT_CURVES False --TRACKERS_FOLDER YOLOX_outputs/yolox_x/track_results
```