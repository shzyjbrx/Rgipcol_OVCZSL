#!/bin/bash
#SBATCH --job-name=ovczsl-cgqa
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# 1. 修改日志输出路径为 cgqa 专用目录
# 请确保已创建 logs/cgqa/train 目录
#SBATCH --output=logs/cgqa/train/CLIPL+FT-%j.out
#SBATCH --error=logs/cgqa/train/CLIPL+FT-%j.err

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# --- 1. 数据集配置 ---
# 指向您本地标准 C-GQA 数据集的图片根目录
DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/cgqa"
CONFIG="config/cgqa.yml"

# 如果使用单卡，N_GPU 设为 1；当前申请了 4 卡，保持 4
N_GPU=4

echo ">>> Start Training OV-CZSL (CLIP+FT+NEL) | Dataset: C-GQA | Config: ${CONFIG} | GPUs: ${N_GPU} <<<"

export OMP_NUM_THREADS=1

# --- 2. 训练启动命令 ---
# 注意：C-GQA 数据集较大，建议初始 Epoch 设为 15-20 以观察收敛情况
python train.py --cfg ${CONFIG} \
    DATASET.root_dir ${DATA_ROOT} \
    DISTRIBUTED.world_size ${N_GPU} \
    TRAIN.checkpoint_dir "checkpoints/cgqa/CLIPL_FT" \
    TRAIN.log_dir "tensorboards/cgqa/CLIPL_FT" \
    TRAIN.clip_type "ViT-L/14" \
    MODEL.extra_pair_loss_ratio 0.0 \
    MODEL.extra_attr_loss_ratio 0.0 \
    MODEL.extra_obj_loss_ratio 0.0 \
    MODEL.use_extra_pair_loss False \
    TRAIN.finetune_backbone True \
    TRAIN.lr_encoder 1e-6 \
    TRAIN.lr 1e-5 \
    TRAIN.batch_size 16 \
    TRAIN.max_epoch 20