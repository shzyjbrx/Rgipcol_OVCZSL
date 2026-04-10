#!/bin/bash
#SBATCH --job-name=mit-softprompt
#SBATCH --partition=gpu_mem
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/mit/Adapter/CLIPL_Adapter-%j.out
#SBATCH --error=logs/mit/Adapter/CLIPL_Adapter-%j.err

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
export OMP_NUM_THREADS=1

DATA_ROOT="/home/bingxing2/home/scx6d4e/run/xuanzhenzhen/Base/data/mit-states"
CONFIG="config/mit_softprompt.yml"
N_GPU=4

echo "=========================================================="
echo "  三分支软提示微调 | MIT-States | GPUs: ${N_GPU}"
echo "  CLIP 编码器：完全冻结"
echo "  NEL：关闭"
echo "=========================================================="

python train.py --cfg ${CONFIG} \
    DATASET.root_dir         ${DATA_ROOT}              \
    DISTRIBUTED.world_size   ${N_GPU}                  \
    TRAIN.checkpoint_dir     "checkpoints/mit/CLIPL_SoftPT/${SLURM_JOB_ID}" \
    TRAIN.log_dir            "tensorboards/mit/CLIPL_SoftPT/${SLURM_JOB_ID}" \
    TRAIN.batch_size         256                        \
    TRAIN.lr                 2e-3                       \
    TRAIN.max_epoch          20                        \
    MODEL.adapter_reduction  8                          \
    MODEL.n_ctx              16                         \
    MODEL.w_loss_attr        0.3                        \
    MODEL.w_loss_obj         0.3

echo "=========================================================="
echo "  训练完成"
echo "=========================================================="