#!/bin/bash
#SBATCH --job-name=ovczsl-test
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# 1. 修改日志输出路径
#SBATCH --output=logs/mit/train/CLIPL+FT-%j.out
#SBATCH --error=logs/mit/train/CLIPL+FT-%j.err

module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
# 💡 强制设置终端为 UTF-8 编码，彻底消灭中文和 Emoji 乱码
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# --- 1. 数据集配置 ---
# 指向您本地标准 MIT-States 数据集的图片根目录
DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/mit-states"
CONFIG="config/mit.yml"

# 如果使用单卡，N_GPU 设为 1；当前申请了 4 卡，保持 4
N_GPU=4

echo ">>> Start Fast Testing OV-CZSL | Dataset: MIT-States | Config: ${CONFIG} | GPUs: ${N_GPU} <<<"

export OMP_NUM_THREADS=1

# --- 2. 训练启动命令 ---
# 修改脚本最后一行
python train.py --cfg ${CONFIG} \
    DATASET.root_dir ${DATA_ROOT} \
    DISTRIBUTED.world_size ${N_GPU} \
    TRAIN.checkpoint_dir "checkpoints/mit/CLIPL_FT" \
    TRAIN.log_dir "tensorboards/mit/CLIPL_FT" \
    TRAIN.clip_type "ViT-L/14" \
    MODEL.extra_pair_loss_ratio 0.0 \
    MODEL.extra_attr_loss_ratio 0.0 \
    MODEL.extra_obj_loss_ratio 0.0 \
    MODEL.use_extra_pair_loss False \
    TRAIN.finetune_backbone True \
    TRAIN.lr_encoder 1e-6 \
    TRAIN.lr 1e-5 \
    TRAIN.batch_size 64 \
    TRAIN.max_epoch 20