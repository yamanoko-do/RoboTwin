# LingbotDepth DP 训练

## 准备训练数据

```bash
bash process_data.sh move_playingcard_away demo_clean 50
```

## 抽取 depth 特征

```bash
python extract_lingbot_features.py --zarr_path ./data/move_playingcard_away-demo_clean-50.zarr --gpus "0,1,2,3,4,5,6,7"
```

训练 config 来自 `diffusion_policy/config`，其中进一步读取了 `diffusion_policy/config/task/only_RGB_actdim14.yaml`。

## 训练策略

```bash
# 仅 RGB 观察
bash train_eval_loop.sh cfg=train_cfg_rgb_only_spatial.yaml

# 三个视角 RGB+Depth
bash train_eval_loop.sh cfg=train_cfg_threeview_spatial.yaml

# 仅第一人称视角 RGB+Depth
bash train_eval_loop.sh cfg=train_cfg_firstview_spatial.yaml

# 无深度信息 baseline
bash train_eval_loop.sh cfg=train_cfg_baseline.yaml
```

## 手动评估

```bash
sendwechat "0.05height mono" -- bash eval.sh \
  --task_name move_playingcard_away \
  --task_config demo_clean \
  --seed 50 \
  --gpu_id 0 \
  --ckpt_path /mnt/workspace/yama/RoboTwin/policy/DP/data/outputs/20260509_baseline_resnet18/checkpoints/move_playingcard_away-demo_clean-50-42/10000.ckpt \
  --test_num 100
```
