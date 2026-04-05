#!/bin/bash
#SBATCH --job-name=ovczsl_step1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/llm/gen_neighbors_%j.out

set -e

# 1. 环境准备 (根据你的路径修改)
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

export HF_ENDPOINT=https://hf-mirror.com

# 2. 参数设置
DATASET="mit-states"
DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/${DATASET}"
SAVE_DIR="llm_nel_gen/${DATASET}_neighbors"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"

echo "Starting OVCZSL Neighborhood Generation..."
echo "Model: $MODEL_ID | Dataset: $DATASET"

# 3. 执行
python -u generate_ovczsl_neighbors.py \
    --data_root ${DATA_ROOT} \
    --save_dir ${SAVE_DIR} \
    --model_id ${MODEL_ID}

echo "✅ Step 1 complete. Files are in ${SAVE_DIR}"