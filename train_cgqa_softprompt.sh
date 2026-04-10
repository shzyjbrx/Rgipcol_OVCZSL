#!/bin/bash
#SBATCH --job-name=cgqa-softprompt
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=logs/cgqa/Adapter/CLIPL-adapter-%j.out
#SBATCH --error=logs/cgqa/Adapter/CLIPL-adapter-%j.err

# 环境清理与加载
module purge
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
source ~/.bashrc
source activate /home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/miniconda3/envs/recAtk/xuan-czsl-py38

# 分布式与加速配置
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export OMP_NUM_THREADS=1

# 路径配置 (请确保 C-GQA 路径正确)
DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/cgqa"
CONFIG="config/cgqa_softprompt.yml"
N_GPU=4

echo "=========================================================="
echo "  三分支软提示微调 | C-GQA | GPUs: ${N_GPU}"
echo "  CLIP 编码器：完全冻结"
echo "=========================================================="

# 启动训练
# 注意：C-GQA 数据量大，如果显存紧张，请将 batch_size 调至 128
python train.py --cfg ${CONFIG} \
    DATASET.root_dir         ${DATA_ROOT}               \
    DISTRIBUTED.world_size   ${N_GPU}                   \
    TRAIN.checkpoint_dir     "checkpoints/cgqa/CLIPL_SoftPT/${SLURM_JOB_ID}" \
    TRAIN.log_dir             "tensorboards/cgqa/CLIPL_SoftPT/${SLURM_JOB_ID}" \
    TRAIN.batch_size         128                        \
    TRAIN.lr                 0.002                       \
    TRAIN.max_epoch          20                         \
    MODEL.n_ctx              16                          \
    MODEL.adapter_reduction  4                          \
    TRAIN.lr_decay_milestones    [6,12]                \
    TRAIN.wd                 1e-3                      \
    MODEL.w_loss_attr        0.5                        \
    MODEL.w_loss_obj         0.3

echo "=========================================================="
echo "  C-GQA 训练完成"
echo "=========================================================="