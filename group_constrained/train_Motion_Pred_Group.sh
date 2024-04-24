export CUDA_VISIBLE_DEVICES=0
export seed=153
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 2064 \
    --use_env train_Motion_Pred_Group.py \
    --model_motion_pred_lr 1e-3 \
    --batch_size 1 \
    --weight_decay 1e-4 \
    --epochs 30 \
    --lr_drop 5 \
    --baseroot '/media/disk1/yjt/lzl/motion_pred' \
    --output_dir "/media/disk1/yjt/lzl/motion_pred/group_constrained/group_seed_$seed" \
    --motion_pred_layer 'MP' \
    --merger_dropout 0.1 \
    --hidden_dim 256 \
    --d_model 128 \
    --k 4 \
    --seed $seed


