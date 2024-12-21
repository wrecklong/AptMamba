#!/bin/bash
#SBATCH -o /cluster/home/guosun/shangye/MambaVision/logs/job.%j.out
#SBATCH -e /cluster/home/guosun/shangye/MambaVision/logs/job.%j.err
#SBATCH --time 96:00:00
#SBATCH -J MambaVision_T_pruning
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=8000
#SBATCH --gres=gpumem:24GB
#SBATCH --gpus=rtx:4090:8

module load  stack/2024-06  intel-oneapi-compilers/2023.2.0
module load pigz/2.7-q5vkqhu

rsync -aq ./ ${TMPDIR}

cd $TMPDIR

mkdir datasets
tar -I pigz -xf /cluster/work/cvl/guosun/shangye/datasets01/imagenet_full_size.tar.gz -C ${TMPDIR}/datasets

export PYTHONPATH=
source  /cluster/home/guosun/envir/shangye/MambaND/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
                    --use_env main.py   --epochs=100 \
                    --model mamba_vision_T_pruning  \
                    --teacher-path /cluster/work/cvl/guosun/shangye/pretrained/mambavision_tiny_1k.pth.tar \
                    --distillation-type none --batch-size 128 --lr 0.000005 --min-lr 0.000001 --drop-path 0.2 --drop 0.1  \
                    --weight-decay 0.1  --num_workers 32 --data-path ${TMPDIR}/datasets/imagenet_full_size/  \
                    --output_dir /cluster/work/cvl/guosun/shangye/output/MambaVision/mamba_vision_T_pruning_300einit_100e_batch_size128_p0.7_lr0.000005_min_lr1e-6_decoder_pruning_loss3stage_weight0.1_mse_weight0.02_sort_keep_policy_pretrain_mae_not_restore_value \
                    --base_rate 0.7  --mse_weight 0.02 --pruning_weight 0.1 --pretrained-mae-path /cluster/work/cvl/guosun/shangye/output/pretrain_mae_vim/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_clstok_div2_mae_300e/checkpoint.pth \
                    --no_amp --if_continue_inf

